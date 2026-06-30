"""Dataset loading and prompt formatting.

Nothing here is domain-specific: the system prompt and the input label live in
the training config (``configs/base.yaml`` -> ``dataset``), and the task itself
is whatever ``dataset.path`` points at. Swapping domains is a config + data
change, no code edits.
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, DatasetDict

# Written next to a saved adapter so inference can reconstruct the exact prompt
# the model was trained with, without needing the training config.
PROMPT_CONFIG_FILE = "prompt_config.json"

DEFAULT_INPUT_LABEL = "Query"
DEFAULT_SYSTEM_PROMPT = "Rewrite the input according to the instruction. Output only the result."


# Minimal ChatML shell for true base checkpoints that ship no chat_template
# at all (e.g. SmolLM2, danube3). Turns end on the tokenizer's own eos_token,
# not a fresh special token like <|im_end|> - that token is already
# well-calibrated from pretraining, so the model doesn't need extra capacity
# (modules_to_save) just to learn when to stop.
_FALLBACK_CHATML = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + eos_token + '\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"
)


def build_user(instruction: str, input_text: str, input_label: str = DEFAULT_INPUT_LABEL) -> str:
    """User-turn text: the instruction, optionally followed by the input."""
    if input_text and input_text.strip():
        return f"{instruction}\n\n{input_label}:\n{input_text}"
    return instruction


def ensure_chat_template(tokenizer) -> None:
    """Set a fallback chat template when the checkpoint ships none (true base
    models with no downstream chat fine-tuning, unlike LFM2.5/Qwen3 whose
    base checkpoints still carry a template). No-op otherwise."""
    if tokenizer.chat_template is None:
        tokenizer.chat_template = _FALLBACK_CHATML


def align_eos_token(model, tokenizer) -> None:
    """Some chat templates (e.g. Qwen3's ChatML) terminate turns with
    <|im_end|> while the tokenizer's default eos_token_id points elsewhere
    (e.g. <|endoftext|>) since the underlying checkpoint is a base model, not
    chat-tuned. Generation only stops at eos_token_id, so leaving it
    unaligned means the model never learns/signals a stop and rambles past
    the answer. No-op when the default eos already matches the template.
    """
    im_end_id = tokenizer.get_vocab().get("<|im_end|>")
    if im_end_id is None:
        return
    if tokenizer.eos_token_id != im_end_id:
        tokenizer.eos_token = "<|im_end|>"
    # model.generation_config is loaded independently of the tokenizer (from
    # the base repo's config, not the adapter dir, when loading a saved LoRA
    # adapter) so it can be stale even when the tokenizer above is already
    # correct - sync it unconditionally.
    if model.config.eos_token_id != im_end_id:
        model.config.eos_token_id = im_end_id
    if model.generation_config.eos_token_id != im_end_id:
        model.generation_config.eos_token_id = im_end_id


def save_prompt_config(out_dir: str | Path, system_prompt: str, input_label: str) -> None:
    """Record the prompt framing alongside an adapter so inference is exact."""
    payload = {"system_prompt": system_prompt, "input_label": input_label}
    (Path(out_dir) / PROMPT_CONFIG_FILE).write_text(json.dumps(payload, indent=2))


def load_prompt_config(model_dir: str | Path) -> dict:
    """Read a saved prompt config; fall back to defaults if absent."""
    path = Path(model_dir) / PROMPT_CONFIG_FILE
    if path.exists():
        return json.loads(path.read_text())
    return {"system_prompt": DEFAULT_SYSTEM_PROMPT, "input_label": DEFAULT_INPUT_LABEL}


def resolve_data(prefix: str | Path) -> tuple[Path, Path, dict]:
    """Map a data prefix (e.g. ``data/promql``) to its generated artifacts:
    ``(train.jsonl, eval.jsonl, prompt)``. Produced by ``gen-dataset``."""
    p = str(prefix)
    prompt_path = Path(f"{p}_prompt.json")
    prompt = json.loads(prompt_path.read_text()) if prompt_path.exists() else {
        "system_prompt": DEFAULT_SYSTEM_PROMPT, "input_label": DEFAULT_INPUT_LABEL,
    }
    return Path(f"{p}_train.jsonl"), Path(f"{p}_eval.jsonl"), prompt


def _format(example: dict, tokenizer, system_prompt: str, input_label: str) -> dict:
    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": build_user(example["instruction"],
                                                     example.get("input", ""), input_label)},
        {"role": "assistant", "content": example["output"]},
    ]
    # enable_thinking=False: no-op for templates without a reasoning mode (e.g.
    # LFM2.5); for hybrid-thinking templates (e.g. Qwen3) it skips the
    # <think></think> stub so train- and inference-time rendering match.
    return {"text": tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )}


def load_train(train_path: str | Path, tokenizer, prompt: dict, loader: dict) -> DatasetDict:
    """Build train/eval splits.

    train_path -- generated *_train.jsonl
    prompt     -- {system_prompt, input_label} (from the data dir's *_prompt.json)
    loader     -- generic knobs from the config's ``dataset`` section
    """
    system_prompt = prompt.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    input_label = prompt.get("input_label", DEFAULT_INPUT_LABEL)

    rows = [json.loads(l) for l in Path(train_path).read_text().splitlines() if l.strip()]
    ds = Dataset.from_list(rows)
    ds = ds.map(
        lambda ex: _format(ex, tokenizer, system_prompt, input_label),
        num_proc=loader.get("num_proc", 2),
        remove_columns=ds.column_names,
    )
    split = ds.train_test_split(test_size=loader.get("val_split", 0.1), seed=loader.get("seed", 42))
    return DatasetDict({"train": split["train"], "eval": split["test"]})
