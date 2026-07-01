#!/usr/bin/env python3
"""
Entry point: uv run train --config configs/lfm2.5_1.2b.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


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


def train_sft(cfg: dict, train_path: str, prompt: dict, backend: str | None = None) -> dict:
    """Train a LoRA adapter via the selected backend (cuda/mlx). The actual
    model-touching code lives in ``lfm_train.backends.<backend>``; this is a
    thin dispatch so the CLI and config surface stay backend-agnostic."""
    from lfm_train.backends import get_backend
    return get_backend(backend).train_sft(cfg, train_path, prompt)


def train_dpo(cfg: dict, train_path: str, prompt: dict, backend: str | None = None) -> None:
    raise NotImplementedError("wire in trl.DPOTrainer with a preference dataset")


def train_grpo(cfg: dict, train_path: str, prompt: dict, backend: str | None = None) -> None:
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
    pa.add_argument("--backend", choices=["auto", "cuda", "mlx"], default="auto",
                    help="compute backend; 'auto' = LFM_TRAIN_BACKEND env or platform default")
    args = pa.parse_args()

    cfg = _load_config(args.config)
    train_path, _eval_path, prompt = resolve_data(args.data)
    method = args.method or cfg["training"]["method"]
    print(f"method={method}  config={args.config}  data={args.data}  backend={args.backend}")
    METHODS[method](cfg, str(train_path), prompt, backend=args.backend)


if __name__ == "__main__":
    main()
