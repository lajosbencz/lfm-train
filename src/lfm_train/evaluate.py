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


def _infer(model, tokenizer, instruction: str, input_q: str,
           system_prompt: str, input_label: str = "Query",
           max_new_tokens: int = 128) -> str:
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": build_user(instruction, input_q, input_label)},
    ]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def _load_model(name: str):
    from unsloth import FastLanguageModel
    from lfm_train.dataset import align_eos_token, ensure_chat_template
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=name, max_seq_length=512, load_in_4bit=True, dtype=None,
        use_exact_model_name=True,
    )
    ensure_chat_template(tokenizer)
    align_eos_token(model, tokenizer)
    return model, tokenizer


def _eval_model(model, tokenizer, examples: list[dict], label: str,
                system_prompt: str, input_label: str) -> list[dict]:
    results = []
    cats: dict[str, list[bool]] = defaultdict(list)
    print(f"\n=== {label} ===")
    for ex in examples:
        pred = _infer(model, tokenizer, ex["instruction"], ex.get("input", ""),
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
    args = pa.parse_args()

    eval_path = Path(f"{args.data}_eval.jsonl")
    examples = [json.loads(l) for l in eval_path.read_text().splitlines() if l.strip()]

    # prompt framing comes from the trained adapter (self-describing)
    prompt = load_prompt_config(args.finetuned)
    system_prompt, input_label = prompt["system_prompt"], prompt["input_label"]

    results: list[dict] = []
    if args.base:
        base_model, base_tok = _load_model(args.base)
        results = _eval_model(base_model, base_tok, examples, "base", system_prompt, input_label)
        del base_model

    ft_model, ft_tok = _load_model(args.finetuned)
    ft_results = _eval_model(ft_model, ft_tok, examples, "finetuned", system_prompt, input_label)
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
