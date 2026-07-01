"""Backend interface.

A backend owns every model-touching operation: loading + LoRA-wrapping a base
checkpoint, running the training loop, exporting artifacts, and generating text
at inference. Two implementations exist - ``cuda`` (Unsloth/TRL, NVIDIA) and
``mlx`` (Apple Silicon/Metal). Everything else in the codebase (dataset engine,
prompt rendering, config merge, eval scoring, publish) is backend-agnostic and
talks to a backend only through this surface.

This module stays import-light on purpose: importing it must not pull in torch,
unsloth, or mlx, so ``import lfm_train`` works on either OS regardless of which
heavy stack is installed.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    name: str  # "cuda" | "mlx"

    def train_sft(self, cfg: dict, train_path: str, prompt: dict) -> dict:
        """Train a LoRA adapter and write artifacts under ``cfg.training.output_dir``.

        Writes ``lora_adapter/`` (+ ``prompt_config.json``) and the backend's
        native merged/export artifacts. Returns a metrics dict shaped like TRL's
        TrainOutput.metrics (``train_runtime``, ``train_steps_per_second``, ...).
        """
        ...

    def load_inference(self, path: str, max_seq_length: int = 512):
        """Load a saved adapter dir for inference. Returns opaque (model, tokenizer)."""
        ...

    def generate(self, model, tokenizer, messages: list[dict],
                 max_new_tokens: int = 128) -> str:
        """Render ``messages`` with the tokenizer's chat template and greedily
        generate the assistant turn. The backend owns chat-templating, tokenize,
        generate, and decode. Returns the decoded completion (special tokens stripped)."""
        ...

    def reset_peak_memory(self) -> None:
        """Reset the device peak-memory counter (no-op when unsupported)."""
        ...

    def peak_memory_mb(self) -> float | None:
        """Peak device memory in MB since the last reset, or None when unsupported."""
        ...
