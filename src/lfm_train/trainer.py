#!/usr/bin/env python3
"""
Entry point: uv run train --config configs/lfm2.5_1.2b.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from transformers import TrainerCallback


def _load_config(path: str) -> dict:
    base = Path(__file__).parents[2] / "configs" / "base.yaml"
    with open(base) as f:
        cfg = yaml.safe_load(f)
    if path:
        with open(path) as f:
            _deep_merge(cfg, yaml.safe_load(f))
    return cfg


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


class _MetricsCallback(TrainerCallback):
    """Captures per-step loss, LR, and GPU memory to a JSONL file."""

    def __init__(self, log_path: Path) -> None:
        self._fh = log_path.open("w")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        import torch
        row: dict = {
            "step":  state.global_step,
            "epoch": round(state.epoch or 0.0, 4),
        }
        for key in ("loss", "eval_loss", "learning_rate"):
            if key in logs:
                row[key] = logs[key]
        if torch.cuda.is_available():
            row["gpu_mem_mb"] = torch.cuda.memory_allocated() / 1e6
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def on_train_end(self, args, state, control, **kwargs):
        self._fh.close()


def train_sft(cfg: dict, train_path: str, prompt: dict) -> dict:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.recompile_limit = 64

    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    from lfm_train.dataset import (align_eos_token, ensure_chat_template,
                                   load_train, save_prompt_config)

    m = cfg["model"]
    l = cfg["lora"]
    t = cfg["training"]
    d = cfg["dataset"]

    print(f"Loading {m['name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=m["name"],
        max_seq_length=m["max_seq_length"],
        load_in_4bit=m.get("load_in_4bit", True),
        dtype=None,
        use_exact_model_name=True,
    )
    ensure_chat_template(tokenizer)
    align_eos_token(model, tokenizer)
    model = FastLanguageModel.get_peft_model(
        model,
        r=l["r"],
        lora_alpha=l["alpha"],
        lora_dropout=l["dropout"],
        bias=l["bias"],
        target_modules=l["target_modules"],
        # Full fine-tune (not LoRA-decomposed) of the LM head/embeddings when
        # configured. Needed for tied-embedding base checkpoints whose chat
        # special tokens (e.g. Qwen3's <|im_end|>) were never exercised
        # during pretraining: LoRA on attention/MLP alone can't reliably move
        # the frozen head's logit for a token it has no learned signal for.
        modules_to_save=l.get("modules_to_save"),
        # Required when modules_to_save touches a tied lm_head/embed_tokens
        # pair (e.g. Qwen3), or merge/GGUF export can desync the two copies.
        ensure_weight_tying=bool(l.get("modules_to_save")),
        use_gradient_checkpointing=t.get("gradient_checkpointing", "unsloth"),
        random_state=d.get("seed", 42),
    )

    datasets = load_train(train_path, tokenizer, prompt, d)
    print(f"Train: {len(datasets['train'])}  Eval: {len(datasets['eval'])}")

    out = Path(t["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    metrics_cb = _MetricsCallback(out / "training_log.jsonl")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        args=SFTConfig(
            output_dir=t["output_dir"],
            per_device_train_batch_size=t["per_device_train_batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            warmup_steps=t["warmup_steps"],
            max_steps=t.get("max_steps", -1),
            num_train_epochs=t.get("num_train_epochs", 1),
            learning_rate=t["learning_rate"],
            weight_decay=t["weight_decay"],
            lr_scheduler_type=t["lr_scheduler_type"],
            optim=t["optim"],
            bf16=t.get("bf16", True),
            fp16=t.get("fp16", False),
            logging_steps=t["logging_steps"],
            save_steps=t["save_steps"],
            eval_steps=t["eval_steps"],
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            dataset_text_field="text",
            max_seq_length=m["max_seq_length"],
            dataset_num_proc=d.get("num_proc", 2),
            packing=False,
            report_to="none",
            disable_tqdm=False,
        ),
        callbacks=[metrics_cb],
    )

    stats = trainer.train()
    print(f"\nDone - final loss: {stats.training_loss:.4f}")
    train_metrics = stats.metrics

    from lfm_train.plot import plot_run
    plot_path = out / "training_metrics.png"
    plot_run(out / "training_log.jsonl", plot_path,
             title=f"{m['name'].split('/')[-1]} - training metrics")
    print(f"Training plot  -> {plot_path}")

    lora_path = out / "lora_adapter"
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))
    save_prompt_config(lora_path, prompt.get("system_prompt", ""), prompt.get("input_label", "Query"))
    print(f"LoRA adapter   -> {lora_path}")

    merged = out / "merged_16bit"
    model.save_pretrained_merged(str(merged), tokenizer, save_method="merged_16bit")
    save_prompt_config(merged, prompt.get("system_prompt", ""), prompt.get("input_label", "Query"))
    print(f"Merged model   -> {merged}")

    gguf = out / "gguf"
    model.save_pretrained_gguf(str(gguf), tokenizer, quantization_method="q4_k_m")
    print(f"GGUF Q4_K_M    -> {gguf}")

    return train_metrics


def train_dpo(cfg: dict, train_path: str, prompt: dict) -> None:
    raise NotImplementedError("wire in trl.DPOTrainer with a preference dataset")


def train_grpo(cfg: dict, train_path: str, prompt: dict) -> None:
    raise NotImplementedError("wire in trl.GRPOTrainer with a reward function")


METHODS = {"sft": train_sft, "dpo": train_dpo, "grpo": train_grpo}


def main() -> None:
    from lfm_train.dataset import resolve_data

    pa = argparse.ArgumentParser()
    pa.add_argument("--config", default="configs/lfm2.5_350m.yaml",
                    help="model/training config (domain-agnostic)")
    pa.add_argument("--data", default="data/promql",
                    help="data prefix; loads <prefix>_train.jsonl + <prefix>_prompt.json")
    pa.add_argument("--method", choices=list(METHODS), default=None)
    args = pa.parse_args()

    cfg = _load_config(args.config)
    train_path, _eval_path, prompt = resolve_data(args.data)
    method = args.method or cfg["training"]["method"]
    print(f"method={method}  config={args.config}  data={args.data}")
    METHODS[method](cfg, str(train_path), prompt)


if __name__ == "__main__":
    main()
