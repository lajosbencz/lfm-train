"""Pure translation from this project's model-only config dict to mlx-lm's
LoRA/training knobs. No mlx import here on purpose: this stays unit-testable on
any platform.

The project config (configs/*.yaml, merged over base.yaml) is the single source
of truth for both backends. CUDA-only fields (``training.optim: adamw_8bit``,
``training.gradient_checkpointing: unsloth``, ``training.bf16/fp16``,
``model.dtype``) are simply not consumed here.
"""
from __future__ import annotations


def to_mlx_lora(cfg: dict) -> dict:
    """Translate a merged project config into the inputs the MLX backend needs.

    ``num_layers`` is left as None when the config doesn't pin it, meaning "all
    transformer layers" - the MLX backend resolves None to the loaded model's
    layer count (our CUDA path LoRA-wraps every layer, so we match that).

    ``iters`` is intentionally NOT computed here: it depends on the training set
    size, which only the backend knows after building the dataset. The raw
    schedule knobs (max_steps / num_train_epochs / batch_size / grad-accum) are
    passed through for the backend to resolve.
    """
    m = cfg.get("model", {})
    l = cfg.get("lora", {})
    t = cfg.get("training", {})

    r = int(l.get("r", 16))
    alpha = float(l.get("alpha", r))
    # PEFT (CUDA path) applies an effective LoRA scaling of alpha/r. mlx-lm's
    # LoRALinear multiplies the low-rank product by ``scale`` directly, so to
    # reproduce PEFT semantics we pass scale = alpha / r (== 1.0 for our configs,
    # where alpha == r). See open item: confirm against mlx_lm LoRALinear.
    scale = alpha / r if r else alpha

    lora = {
        "rank": r,
        "scale": scale,
        "dropout": float(l.get("dropout", 0.0)),
    }
    keys = l.get("target_modules")
    if keys:
        lora["keys"] = list(keys)

    return {
        "num_layers": l.get("num_layers"),  # None => all layers (resolved by backend)
        "lora": lora,
        "learning_rate": float(t.get("learning_rate", 2e-4)),
        "batch_size": int(t.get("per_device_train_batch_size", 4)),
        "grad_accumulation_steps": int(t.get("gradient_accumulation_steps", 1)),
        "max_steps": int(t.get("max_steps", -1)),
        "num_train_epochs": float(t.get("num_train_epochs", 1)),
        "warmup_steps": int(t.get("warmup_steps", 0)),
        "lr_scheduler_type": t.get("lr_scheduler_type", "cosine"),
        "max_seq_length": int(m.get("max_seq_length", 2048)),
        # CUDA uses unsloth's gradient checkpointing; map the truthy/"unsloth"
        # string onto MLX's boolean grad_checkpoint.
        "grad_checkpoint": bool(t.get("gradient_checkpointing")),
        "load_in_4bit": bool(m.get("load_in_4bit", True)),
    }
