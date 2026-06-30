#!/usr/bin/env python3
"""
Dataset generator CLI -- thin wrapper over the domain-agnostic engine.

  uv run gen-dataset                      # default domain: promql
  uv run gen-dataset --domain promql
  uv run gen-dataset --spec path/to/spec.yaml --prefix mydomain

The heavy lifting lives in engine.py; domains are declarative YAML specs in
domains/. Add a new knowledge domain by writing a new spec, no code changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from lfm_train.data import engine

# Domain specs are declarative inputs (like configs/), kept at the repo root.
_DOMAINS_DIR = Path(__file__).parents[3] / "domains"


def _report(train: list[dict], eval_rows: list[dict]) -> None:
    print(f"train: {len(train)} examples")
    print(f"eval:  {len(eval_rows)} examples\n")

    cats = Counter(r["category"] for r in train)
    print("train category breakdown:")
    for cat, n in sorted(cats.items()):
        print(f"  {cat:24s} {n}")

    buckets = Counter(r["bucket"] for r in eval_rows)
    print("\neval buckets:")
    for b, n in sorted(buckets.items()):
        print(f"  {b:18s} {n}")

    # disjointness audit: no held-out eval input should appear in train
    train_inputs = {r["input"] for r in train}
    unseen = [r for r in eval_rows if r["bucket"] != "single_seen"]
    leaked = [r for r in unseen if r["input"] in train_inputs]
    print(f"\nheld-out eval inputs leaking into train: {len(leaked)}")
    if leaked:
        for r in leaked[:5]:
            print(f"  LEAK [{r['pattern']}] {r['input']}")


def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("--domain", default="promql", help="spec name under domains/")
    pa.add_argument("--spec",   default=None, help="explicit spec path (overrides --domain)")
    pa.add_argument("--prefix", default=None, help="output file prefix (default: domain name)")
    pa.add_argument("--out",    default=None, help="output dir (default: <repo>/data)")
    args = pa.parse_args()

    spec_path = Path(args.spec) if args.spec else _DOMAINS_DIR / f"{args.domain}.yaml"
    prefix = args.prefix or args.domain
    out_dir = Path(args.out) if args.out else Path(__file__).parents[3] / "data"

    train, eval_rows, prompt = engine.build(spec_path)
    engine.write(train, eval_rows, prompt, out_dir, prefix)
    _report(train, eval_rows)
    print(f"\nwrote {out_dir}/{prefix}_{{train,eval}}.jsonl and {out_dir}/{prefix}_prompt.json")


if __name__ == "__main__":
    main()
