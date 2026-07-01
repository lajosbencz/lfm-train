#!/usr/bin/env python3
"""
Standalone evaluation of a fine-tuned adapter against an eval set.

  uv run evaluate --finetuned outputs/lfm2.5-350m/lora_adapter --data data/promql

Reports per-category exact-match accuracy and saves a full results JSONL. The
prompt framing is read from the adapter's saved prompt_config.json, so this
stays domain-agnostic. For the full train+eval sweep across models, use
`uv run benchmark` instead.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from lfm_train.dataset import build_user, load_prompt_config


def _normalize(s: str) -> str:
    return " ".join(s.strip().split())


def _infer(backend, model, tokenizer, instruction: str, input_q: str,
           system_prompt: str, input_label: str = "Query",
           max_new_tokens: int = 128) -> str:
    """Build the chat messages and hand off generation to the backend. Backends
    own chat-templating + tokenize + generate + decode (see backends/*.generate)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": build_user(instruction, input_q, input_label)},
    ]
    return backend.generate(model, tokenizer, messages, max_new_tokens=max_new_tokens)


def _load_model(backend, name: str):
    return backend.load_inference(name, max_seq_length=512)


def _eval_model(backend, model, tokenizer, examples: list[dict], label: str,
                system_prompt: str, input_label: str) -> list[dict]:
    results = []
    cats: dict[str, list[bool]] = defaultdict(list)
    print(f"\n=== {label} ===")
    for ex in examples:
        pred = _infer(backend, model, tokenizer, ex["instruction"], ex.get("input", ""),
                      system_prompt, input_label)
        match = _normalize(pred) == _normalize(ex["output"])
        cats[ex.get("category", "unknown")].append(match)
        results.append({**ex, f"pred_{label}": pred, f"match_{label}": match})
        status = "OK" if match else " x"
        print(f"  {status} [{ex.get('category','?'):24s}] {ex['instruction'][:55]}")
        if not match:
            print(f"       expected: {ex['output']}")
            print(f"       got:      {pred}")

    print(f"\n  per-category accuracy ({label}):")
    total_ok = total = 0
    for cat, hits in sorted(cats.items()):
        ok = sum(hits)
        total_ok += ok
        total    += len(hits)
        print(f"    {cat:28s} {ok}/{len(hits)} ({100*ok//len(hits)}%)")
    print(f"  overall: {total_ok}/{total} ({100*total_ok//total}%)")
    return results


def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("--finetuned", required=True, help="path to a saved lora_adapter")
    pa.add_argument("--base",      default=None, help="optional base model to compare against")
    pa.add_argument("--data",      default="data/promql",
                    help="data prefix; loads <prefix>_eval.jsonl")
    pa.add_argument("--output",    default="outputs/eval_results.jsonl")
    pa.add_argument("--backend",   choices=["auto", "cuda", "mlx"], default="auto",
                    help="compute backend; 'auto' = LFM_TRAIN_BACKEND env or platform default")
    args = pa.parse_args()

    from lfm_train.backends import get_backend
    backend = get_backend(args.backend)

    eval_path = Path(f"{args.data}_eval.jsonl")
    examples = [json.loads(l) for l in eval_path.read_text().splitlines() if l.strip()]

    # prompt framing comes from the trained adapter (self-describing)
    prompt = load_prompt_config(args.finetuned)
    system_prompt, input_label = prompt["system_prompt"], prompt["input_label"]

    results: list[dict] = []
    if args.base:
        base_model, base_tok = _load_model(backend, args.base)
        results = _eval_model(backend, base_model, base_tok, examples, "base", system_prompt, input_label)
        del base_model

    ft_model, ft_tok = _load_model(backend, args.finetuned)
    ft_results = _eval_model(backend, ft_model, ft_tok, examples, "finetuned", system_prompt, input_label)
    if results:
        for i, r in enumerate(ft_results):
            results[i].update({k: v for k, v in r.items() if k.startswith(("pred_", "match_"))})
    else:
        results = ft_results

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in results) + "\n")
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    main()
