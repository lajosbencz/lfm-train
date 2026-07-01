"""MLX backend: mlx-lm, Apple Silicon/Metal.

UNTESTED ON METAL - there is no Apple hardware in this environment. The mlx-lm
*API surface* used here (``mlx_lm.load``/``generate``/``convert``,
``mlx_lm.tuner.trainer.{TrainingArgs,train}``,
``mlx_lm.tuner.utils.{linear_to_lora_layers,build_schedule}``,
``mlx_lm.tuner.datasets.load_dataset``, the ``mlx_lm fuse`` CLI) was verified by
introspecting mlx-lm 0.31.3 source on Linux (mlx-lm is a pure-python wheel; only
its ``mlx`` dep is Darwin-gated). What remains unverified needs the Metal runtime
and is called out with ``# NOTE:``: ``mlx.optimizers.AdamW`` instantiation,
greedy-sampling default of ``generate``, the ``mlx.core`` peak-memory counter
name, and of course actual training/generation numerics.

Heavy imports (``mlx``, ``mlx_lm``) are function-local on purpose: importing
this module must not require mlx to be installed, so ``import lfm_train``
keeps working on non-Apple hosts (this repo is developed on Linux/CUDA).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from lfm_train import dataset
from lfm_train.backends.config_map import to_mlx_lora


class MlxBackend:
    name = "mlx"

    def train_sft(self, cfg: dict, train_path: str, prompt: dict) -> dict:
        import mlx.optimizers as optim
        from mlx_lm import load
        from mlx_lm.convert import convert
        from mlx_lm.tuner.datasets import load_dataset
        from mlx_lm.tuner.trainer import TrainingArgs, train
        from mlx_lm.tuner.utils import build_schedule, linear_to_lora_layers

        knobs = to_mlx_lora(cfg)
        name = cfg["model"]["name"]
        out = Path(cfg["training"]["output_dir"])
        out.mkdir(parents=True, exist_ok=True)

        # QLoRA: train against a 4-bit-quantized MLX copy of the base model,
        # cached under the output dir so repeat runs skip the conversion.
        if knobs["load_in_4bit"]:
            base_dir = out / "mlx_base_4bit"
            if not (base_dir / "config.json").exists():
                print(f"Converting {name} -> 4-bit MLX ({base_dir})")
                convert(name, mlx_path=str(base_dir), quantize=True, q_bits=4)
            base_path = str(base_dir)
        else:
            base_path = name

        print(f"Loading {base_path}")
        model, tokenizer = load(base_path)
        dataset.ensure_chat_template(tokenizer)
        # NOTE: mlx-lm's tokenizer wrapper has no torch-style generation_config
        # to keep in sync the way dataset.align_eos_token does for CUDA; mlx-lm
        # generation stops on the tokenizer's own eos_token_id, which the chat
        # template's rendered text (incl. </im_end> etc.) already accounts for.

        datasets = dataset.load_train(train_path, tokenizer, prompt, cfg.get("dataset", {}))
        print(f"Train: {len(datasets['train'])}  Eval: {len(datasets['eval'])}")

        # mlx-lm's own dataset loader expects on-disk train.jsonl/valid.jsonl
        # (one {"text": ...} object per line) in a directory; write the
        # pre-rendered chat-templated rows straight through.
        data_dir = Path(tempfile.mkdtemp(prefix="mlx_data_", dir=str(out)))
        _write_jsonl(data_dir / "train.jsonl", datasets["train"])
        _write_jsonl(data_dir / "valid.jsonl", datasets["eval"])

        # Verified against mlx-lm 0.31.3: load_dataset(args, tokenizer) reads only
        # args.{data,train,test,hf_dataset} and returns (train, valid, test).
        # It reads data_dir/{train,valid,test}.jsonl; a missing test.jsonl yields
        # an empty list, which is fine because test=False (load_dataset only
        # raises on empty test when args.test is truthy). create_dataset then
        # picks the dataset type by feature key: our rows expose only "text", so
        # (prompt/completion and messages absent) it selects TextDataset via the
        # text_feature="text" default. The feature attrs below are belt-and-suspenders.
        from types import SimpleNamespace
        args_ns = SimpleNamespace(
            data=str(data_dir), train=True, test=False, hf_dataset=None,
            prompt_feature="prompt", completion_feature="completion",
            chat_feature="messages", text_feature="text", mask_prompt=False,
        )
        train_set, val_set, _test_set = load_dataset(args_ns, tokenizer)

        # LoRA injection.
        num_layers = knobs["num_layers"]
        if num_layers is None:
            num_layers = _resolve_num_layers(model)
        linear_to_lora_layers(model, num_layers, knobs["lora"])

        n_train = len(datasets["train"])
        batch_size = knobs["batch_size"]
        grad_accum = knobs["grad_accumulation_steps"]
        if knobs["max_steps"] > 0:
            iters = knobs["max_steps"]
        else:
            steps_per_epoch = max(1, math.ceil(n_train / (batch_size * grad_accum)))
            iters = max(1, math.ceil(knobs["num_train_epochs"] * steps_per_epoch))

        # Cosine LR schedule with linear warmup, matching the CUDA path's
        # lr_scheduler_type="cosine" default. Anything else falls back to a
        # flat learning rate (mlx-lm's optimizers accept either a float or a
        # schedule callable).
        if knobs["lr_scheduler_type"] == "cosine" and iters > 0:
            schedule = build_schedule({
                "name": "cosine_decay",
                "arguments": [knobs["learning_rate"], iters],
                "warmup": knobs["warmup_steps"],
            })
        else:
            schedule = knobs["learning_rate"]
        optimizer = optim.AdamW(learning_rate=schedule)

        lora_dir = out / "lora_adapter"
        lora_dir.mkdir(parents=True, exist_ok=True)
        adapter_file = lora_dir / "adapters.safetensors"

        args = TrainingArgs(
            batch_size=batch_size,
            iters=iters,
            val_batches=25,
            max_seq_length=knobs["max_seq_length"],
            adapter_file=str(adapter_file),
            grad_checkpoint=knobs["grad_checkpoint"],
            grad_accumulation_steps=grad_accum,
        )

        import time
        t0 = time.perf_counter()
        train(model, optimizer, train_set, val_set, args=args)
        elapsed = time.perf_counter() - t0

        # mlx_lm.tuner.utils.load_adapters reads adapter_config.json alongside
        # the safetensors weights; write the metadata it (and our own
        # load_inference detection below) needs.
        adapter_config = {
            "fine_tune_type": "lora",
            "num_layers": num_layers,
            "lora_parameters": knobs["lora"],
            "base_model": name,
        }
        (lora_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))
        dataset.save_prompt_config(lora_dir, prompt.get("system_prompt", ""), prompt.get("input_label", "Query"))
        print(f"LoRA adapter   -> {lora_dir}")

        # Fuse into a standalone merged MLX model dir. No GGUF export here:
        # mlx-lm's GGUF conversion only covers Llama/Mistral/Mixtral
        # architectures, and this project's model zoo (LFM2, Qwen3, SmolLM2,
        # danube3) isn't in that list.
        merged_dir = out / "merged_mlx"
        # Verified against mlx-lm 0.31.3: `python -m mlx_lm` dispatches to
        # cli.main(), which registers a "fuse" subcommand whose parse_arguments
        # defines --model / --adapter-path / --save-path. Using the CLI (stable
        # contract) rather than internal fuse helpers whose names shift.
        fuse_cmd = [
            sys.executable, "-m", "mlx_lm", "fuse",
            "--model", base_path,
            "--adapter-path", str(lora_dir),
            "--save-path", str(merged_dir),
        ]
        print("Fusing adapter:", " ".join(fuse_cmd))
        subprocess.run(fuse_cmd, check=True)
        dataset.save_prompt_config(merged_dir, prompt.get("system_prompt", ""), prompt.get("input_label", "Query"))
        print(f"Merged model   -> {merged_dir}")

        return {
            "train_runtime": elapsed,
            "train_steps_per_second": iters / elapsed if elapsed else 0.0,
            "train_samples_per_second": (iters * batch_size) / elapsed if elapsed else 0.0,
        }

    def load_inference(self, path: str, max_seq_length: int = 512):
        from mlx_lm import load

        p = Path(path)
        if (p / "adapter_config.json").exists():
            # Adapter dir: needs the base model name it was trained from.
            cfg = json.loads((p / "adapter_config.json").read_text())
            base = cfg.get("base_model")
            if not base:
                raise ValueError(f"{p}/adapter_config.json missing 'base_model'; "
                                  "can't resolve the base checkpoint for this adapter")
            model, tokenizer = load(base, adapter_path=str(p))
        else:
            model, tokenizer = load(str(p))

        dataset.ensure_chat_template(tokenizer)
        return model, tokenizer

    def generate(self, model, tokenizer, messages: list[dict],
                 max_new_tokens: int = 128) -> str:
        from mlx_lm import generate

        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False, tokenize=False,
        )
        # NOTE: mlx_lm.generate is greedy by default when no sampler is given;
        # if the installed mlx_lm version requires an explicit sampler for
        # deterministic decoding, pass one via sampler=make_sampler(temp=0.0)
        # (mlx_lm.sample_utils) here.
        out = generate(model, tokenizer, prompt=text, max_tokens=max_new_tokens)
        return out.strip()

    def reset_peak_memory(self) -> None:
        import mlx.core as mx
        # NOTE: the Metal memory-stats API has moved between mx.metal.* and
        # mx.* across mlx releases; guard both spellings and no-op if absent.
        fn = getattr(getattr(mx, "metal", mx), "reset_peak_memory", None)
        if fn is not None:
            fn()

    def peak_memory_mb(self) -> float | None:
        import mlx.core as mx
        fn = getattr(getattr(mx, "metal", mx), "get_peak_memory", None)
        if fn is None:
            return None
        return fn() / 1e6


def _write_jsonl(path: Path, split) -> None:
    with path.open("w") as fh:
        for row in split:
            fh.write(json.dumps({"text": row["text"]}) + "\n")


def _resolve_num_layers(model) -> int:
    """mlx-lm models nest transformer layers either at ``model.layers`` or
    ``model.model.layers`` depending on the architecture wrapper; try both."""
    for candidate in (model, getattr(model, "model", None)):
        layers = getattr(candidate, "layers", None)
        if layers is not None:
            return len(layers)
    raise AttributeError("could not resolve transformer layer count from model "
                          "(.layers / .model.layers both missing)")
