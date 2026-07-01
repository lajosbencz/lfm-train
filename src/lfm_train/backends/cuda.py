"""CUDA backend: Unsloth + TRL + PEFT, NVIDIA GPUs.

Lift-and-shift of the original trainer.py / evaluate.py model-touching code
behind the Backend interface. Heavy imports (torch, unsloth, trl) stay
function-local so importing this module never pulls them in.
"""
from __future__ import annotations

import json
from pathlib import Path


class CudaBackend:
    name = "cuda"

    def train_sft(self, cfg: dict, train_path: str, prompt: dict) -> dict:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.recompile_limit = 64

        from transformers import TrainerCallback
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig

        from lfm_train.dataset import (align_eos_token, ensure_chat_template,
                                       load_train, save_prompt_config)

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

    def load_inference(self, path: str, max_seq_length: int = 512):
        from unsloth import FastLanguageModel
        from lfm_train.dataset import align_eos_token, ensure_chat_template

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=path, max_seq_length=max_seq_length, load_in_4bit=True, dtype=None,
            use_exact_model_name=True,
        )
        ensure_chat_template(tokenizer)
        align_eos_token(model, tokenizer)
        FastLanguageModel.for_inference(model)
        return model, tokenizer

    def generate(self, model, tokenizer, messages: list[dict],
                 max_new_tokens: int = 128) -> str:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)  # idempotent, keep for parity
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
            enable_thinking=False,
        ).to(model.device)
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    def reset_peak_memory(self) -> None:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def peak_memory_mb(self) -> float | None:
        import torch
        return torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else None
