"""Publish a locally-trained, merged model to the Hugging Face Hub.

Training is done locally (the user has the GPU + HF token); there is no CI.
Each model lives under ``outputs/<name>/`` with a ``merged_16bit/`` dir (full
fp16 model + tokenizer + prompt_config.json) and a ``gguf/`` dir (Q4_K_M GGUF).
This uploads a chosen model and writes a correct, license-aware model card.

The HF token is read from the ``HF_TOKEN`` env var and never printed.

Examples:
    uv run publish outputs/lfm2.5-350m/merged_16bit <user>/lfm2.5-350m-promql --gguf outputs/lfm2.5-350m/gguf
    uv run publish outputs/qwen3-0.6b/merged_16bit  <user>/qwen3-0.6b-promql  --gguf outputs/qwen3-0.6b/gguf
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .dataset import load_prompt_config


def derive_repo_url(override: str | None) -> str | None:
    """The source GitHub repo URL: explicit override, else the git origin
    remote normalized to an https URL. None if undiscoverable."""
    if override:
        return override
    try:
        url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    if not url:
        return None
    # git@github.com:owner/repo.git -> https://github.com/owner/repo
    if url.startswith("git@"):
        host, path = url[4:].split(":", 1)
        url = f"https://{host}/{path}"
    return url[:-4] if url.endswith(".git") else url

# Map base-model org prefix -> license metadata. ``note`` (when present) is
# appended to the card body to spell out non-standard terms.
LICENSE_MAP = {
    "LiquidAI": {
        "license": "other",
        "license_name": "lfm-open-license-v1.0",
        "license_link": "https://www.liquid.ai/lfm-license",
        "note": (
            "This model is a derivative of a LiquidAI LFM2.5 base model and is "
            "distributed under the LFM Open License v1.0. The license is free to "
            "use for entities under $10M USD annual revenue; a commercial license "
            "from Liquid AI is required above that threshold. This derivative "
            "retains LiquidAI's copyright and license notices."
        ),
    },
    "Qwen": {"license": "apache-2.0"},
    "HuggingFaceTB": {"license": "apache-2.0"},
    "h2oai": {"license": "apache-2.0"},
}

DEFAULT_LICENSE = {"license": "apache-2.0"}


def derive_base_model(model_dir: Path, override: str | None) -> str | None:
    """Resolve the base model name: explicit override, else the sibling LoRA
    adapter config, else this dir's config.json. None if undiscoverable."""
    if override:
        return override
    adapter = model_dir.parent / "lora_adapter" / "adapter_config.json"
    if adapter.exists():
        name = json.loads(adapter.read_text()).get("base_model_name_or_path")
        if name:
            return name
    config = model_dir / "config.json"
    if config.exists():
        cfg = json.loads(config.read_text())
        name = cfg.get("_name_or_path") or cfg.get("base_model_name_or_path")
        if name:
            return name
    return None


def derive_license(base_model: str) -> dict:
    """License metadata for a base model name via its org prefix."""
    org = base_model.split("/", 1)[0] if "/" in base_model else ""
    info = LICENSE_MAP.get(org)
    if info is None:
        print(
            f"WARNING: could not determine license for base model '{base_model}'; "
            "defaulting to apache-2.0. Verify the base model's license manually.",
            file=sys.stderr,
        )
        return dict(DEFAULT_LICENSE)
    return info


def build_card(model_dir: Path, repo_id: str, base_model: str,
               lic: dict, has_gguf: bool, repo_url: str | None = None) -> str:
    """Render the README.md model card (YAML frontmatter + body)."""
    prompt = load_prompt_config(model_dir)
    system_prompt = prompt["system_prompt"]
    input_label = prompt["input_label"]

    fm = ["---", f"license: {lic['license']}"]
    if lic["license"] == "other":
        fm.append(f"license_name: {lic['license_name']}")
        fm.append(f"license_link: {lic['license_link']}")
    fm += [
        f"base_model: {base_model}",
        "library_name: transformers",
        "tags:",
        "- lora",
        "- qlora",
        "- fine-tuned",
        "- text-generation",
        "pipeline_tag: text-generation",
        "---",
    ]

    name = repo_id.split("/", 1)[-1]
    body = [
        "",
        f"# {name}",
        "",
        f"A small language model fine-tuned with QLoRA SFT on top of "
        f"`{base_model}` for an instruction -> rewrite task.",
        "",
    ]
    if repo_url:
        body += [f"Training code, data, and benchmarks: [{repo_url}]({repo_url})", ""]
    body += [
        "## Prompt format",
        "",
        f"- System prompt: `{system_prompt}`",
        f"- Input label: `{input_label}`",
        "",
        "## Usage (transformers)",
        "",
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f'model_id = "{repo_id}"',
        "tokenizer = AutoTokenizer.from_pretrained(model_id)",
        "model = AutoModelForCausalLM.from_pretrained(model_id, device_map=\"auto\")",
        "",
        "messages = [",
        f'    {{"role": "system", "content": "{system_prompt}"}},',
        f'    {{"role": "user", "content": "<your instruction>\\n\\n{input_label}:\\n<your input>"}},',
        "]",
        "inputs = tokenizer.apply_chat_template(",
        "    messages, add_generation_prompt=True, return_tensors=\"pt\"",
        ").to(model.device)",
        "out = model.generate(inputs, max_new_tokens=256)",
        "print(tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))",
        "```",
        "",
    ]
    if has_gguf:
        body += [
            "A Q4_K_M GGUF quantization is included for use with llama.cpp / Ollama.",
            "",
        ]
    if lic.get("note"):
        body += ["## License", "", lic["note"], ""]

    return "\n".join(fm + body)


def gguf_files(gguf_path: Path) -> list[Path]:
    """The GGUF file(s) to upload: a single file, or every *.gguf in a dir."""
    if gguf_path.is_file():
        return [gguf_path]
    if gguf_path.is_dir():
        return sorted(gguf_path.glob("*.gguf"))
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="publish",
        description="Publish a merged, locally-trained model to the Hugging Face Hub.",
    )
    parser.add_argument("model_dir", help="merged model dir, e.g. outputs/lfm2.5-350m/merged_16bit")
    parser.add_argument("repo_id", help="target HF repo, e.g. username/lfm2.5-350m-promql")
    parser.add_argument("--gguf", help="GGUF file or dir (uploads any *.gguf inside a dir)")
    parser.add_argument("--base-model", dest="base_model", help="base model name for license/metadata")
    parser.add_argument("--repo-url", dest="repo_url",
                        help="source repo URL for the card (default: git origin remote)")
    parser.add_argument("--private", action="store_true", help="create the repo private")
    parser.add_argument("--dry-run", action="store_true", help="do everything except create/upload")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        parser.error(f"model_dir not found: {model_dir}")

    base_model = derive_base_model(model_dir, args.base_model)
    if not base_model:
        parser.error(
            "could not determine the base model; pass --base-model NAME "
            "(no adapter_config.json or config.json hint found)"
        )

    lic = derive_license(base_model)
    repo_url = derive_repo_url(args.repo_url)

    ggufs: list[Path] = []
    if args.gguf:
        gguf_path = Path(args.gguf)
        ggufs = gguf_files(gguf_path)
        if not ggufs:
            parser.error(f"no GGUF file(s) found at: {gguf_path}")

    card = build_card(model_dir, args.repo_id, base_model, lic, bool(ggufs), repo_url)
    (model_dir / "README.md").write_text(card)

    url = f"https://huggingface.co/{args.repo_id}"

    if args.dry_run:
        print(f"[dry-run] base model: {base_model}")
        print(f"[dry-run] license:    {lic['license']}")
        print(f"[dry-run] would create repo: {args.repo_id} (private={args.private})")
        print(f"[dry-run] would upload folder: {model_dir}")
        for f in ggufs:
            print(f"[dry-run] would upload file:   {f}")
        print(f"[dry-run] target URL: {url}")
        print("\n----- model card -----\n")
        print(card)
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        parser.error("HF_TOKEN env var is not set")

    import huggingface_hub

    api = huggingface_hub.HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(model_dir), repo_id=args.repo_id)
    for f in ggufs:
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=args.repo_id,
        )

    print(url)


if __name__ == "__main__":
    main()
