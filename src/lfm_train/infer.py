#!/usr/bin/env python3
"""
Inference for a fine-tuned adapter -- batch or interactive REPL.

The prompt framing (system prompt, input label) is read from the adapter's
saved prompt_config.json so this stays domain-agnostic and matches training.

Batch mode -- run a JSON list of prompts, print each result:

  uv run infer outputs/lfm2.5-350m/lora_adapter --prompts prompts.json

  prompts.json: [{"label": "...", "instruction": "...", "query": "..."}]

Interactive mode -- no --prompts, type instructions in a REPL:

  uv run infer outputs/lfm2.5-350m/lora_adapter

  Multi-line input, blank line submits, Ctrl-C/EOF quits. Include a line
  starting with "Query:" (case-insensitive) to split instruction vs query.
"""
from __future__ import annotations

import argparse
import json

from lfm_train.dataset import load_prompt_config
from lfm_train.evaluate import _infer


def load(adapter: str, max_seq_length: int = 512, backend: str | None = None):
    """Load an adapter for inference via the selected backend. Returns
    (backend, model, tokenizer)."""
    from lfm_train.backends import get_backend
    be = get_backend(backend)
    model, tokenizer = be.load_inference(adapter, max_seq_length=max_seq_length)
    return be, model, tokenizer


def _split_query(text: str) -> tuple[str, str]:
    """Split on a line starting with 'Query:' (case-insensitive)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lower().startswith("query:"):
            instruction = "\n".join(lines[:i]).strip()
            rest = line.split(":", 1)[1]
            input_q = "\n".join([rest] + lines[i + 1:]).strip()
            return instruction, input_q
    return text.strip(), ""


def _run_batch(backend, model, tokenizer, prompts: list[dict], system_prompt: str,
               input_label: str, max_new_tokens: int) -> None:
    for p in prompts:
        result = _infer(backend, model, tokenizer, p["instruction"], p.get("query", ""),
                        system_prompt, input_label, max_new_tokens=max_new_tokens)
        print(f"[{p.get('label', '')}]")
        print(f"  {result}\n")


def _run_repl(backend, model, tokenizer, system_prompt: str, input_label: str,
              max_new_tokens: int) -> None:
    print("Ready. Enter instruction (blank line submits, Ctrl-C/EOF quits).")
    print("Add a 'Query:' line to pass an input.\n")
    while True:
        try:
            lines: list[str] = []
            while True:
                line = input("> " if not lines else "  ")
                if line == "" and lines:
                    break
                lines.append(line)
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            break

        text = "\n".join(lines).strip()
        if not text:
            continue
        instruction, input_q = _split_query(text)
        result = _infer(backend, model, tokenizer, instruction, input_q,
                        system_prompt, input_label, max_new_tokens=max_new_tokens)
        print(f"\n{result}\n")


def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("adapter", help="path to a saved lora_adapter")
    pa.add_argument("--prompts", default=None, help="JSON list of prompts (batch mode)")
    pa.add_argument("--system", default=None, help="override the saved system prompt")
    pa.add_argument("--max-new-tokens", type=int, default=128)
    pa.add_argument("--backend", choices=["auto", "cuda", "mlx"], default="auto",
                    help="compute backend; 'auto' = LFM_TRAIN_BACKEND env or platform default")
    args = pa.parse_args()

    cfg = load_prompt_config(args.adapter)
    system_prompt = args.system if args.system is not None else cfg["system_prompt"]
    input_label = cfg["input_label"]

    print(f"Loading {args.adapter} ...", flush=True)
    backend, model, tokenizer = load(args.adapter, backend=args.backend)
    print("Ready.\n", flush=True)

    if args.prompts:
        prompts = json.loads(open(args.prompts).read())
        _run_batch(backend, model, tokenizer, prompts, system_prompt, input_label, args.max_new_tokens)
    else:
        _run_repl(backend, model, tokenizer, system_prompt, input_label, args.max_new_tokens)


if __name__ == "__main__":
    main()
