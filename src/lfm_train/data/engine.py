#!/usr/bin/env python3
"""
Domain-agnostic instruction-tuning dataset engine.

A domain is a declarative YAML spec (see domains/*.yaml). The engine knows
nothing about any specific domain -- it samples vocabulary, renders Jinja2
templates, splits train/eval by held-out vocabulary, and writes JSONL.

Spec shape:

  config:
    seed: 42
    eval_seen_per_pattern: 1        # eval examples drawn from TRAIN vocab (delta probe)

  prompt:                           # task framing, copied next to the data
    input_label: "Query"           # label for the input block in the user turn
    system_prompt: "..."           # system message used at train + inference

  vocab:
    <dim>:                          # a vocabulary dimension
      train: [...]                  # values only the train set may sample
      eval:  [...]                  # held-out values only the eval set may sample
      shared: [...]                 # structural values both sets sample (windows, etc.)
    # values are scalars OR maps; maps are referenced field-wise: {{ dim.field }}

  patterns:
    - name: <id>
      category: <label>
      composite: false              # true => eval-only (tests compositional generalization)
      instructions: [ "...", ... ]  # Jinja2 paraphrases, rotated across samples
      input:  "<jinja2 template>"
      output: "<jinja2 template>"
      train: 12                     # how many train examples (ignored if composite)
      eval:  2                      # how many held-out-vocab eval examples

Templates are Jinja2: {{ dim }} for scalars, {{ dim.field }} for maps. PromQL's
own single-brace selectors ({job="x"}) pass through as literal text. Each dim is
sampled once per example, so {{ m.name }} and {{ m.rr }} refer to the same entry.

Eval buckets:
  single_seen      -- non-composite pattern, TRAIN vocab   (upper-bound probe)
  single_unseen    -- non-composite pattern, held-out vocab
  composed_trained -- composite pattern WITH train: N, evaluated on held-out vocab
                      (composition TYPE was trained -> tests vocab transfer)
  composed_heldout -- composite pattern with no train: N, held-out vocab
                      (composition TYPE never trained -> tests true compositional
                      generalization)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined, meta

_ENV = Environment(undefined=StrictUndefined, autoescape=False)


def _template_dims(text: str) -> set[str]:
    """Top-level variable names a template references (e.g. {{ m.rr }} -> {'m'})."""
    return meta.find_undeclared_variables(_ENV.parse(text))


def _pools(vocab: dict, split: str) -> dict[str, list]:
    """Per-dim sampling pool for 'train' or 'eval': shared values plus that split's."""
    return {
        dim: list(spec.get("shared", [])) + list(spec.get(split, []))
        for dim, spec in vocab.items()
    }


def _gen(pattern: dict, pools: dict, rng: random.Random, n: int, bucket: str) -> list[dict]:
    instrs = pattern["instructions"]
    in_tpl = _ENV.from_string(pattern["input"])
    out_tpl = _ENV.from_string(pattern["output"])
    instr_tpls = [_ENV.from_string(s) for s in instrs]

    dims: set[str] = set()
    for text in (pattern["input"], pattern["output"], *instrs):
        dims |= _template_dims(text)
    missing = [d for d in dims if not pools.get(d)]
    if missing:
        raise ValueError(f"pattern {pattern['name']!r}: no values for dim(s) {missing} in this split")
    dims = sorted(dims)

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    attempts, idx = 0, 0
    while len(rows) < n and attempts < max(n * 40, 200):
        attempts += 1
        binding = {d: rng.choice(pools[d]) for d in dims}
        inp = in_tpl.render(**binding)
        outp = out_tpl.render(**binding)
        if inp == outp:
            continue
        key = (inp, outp)
        if key in seen:
            continue
        seen.add(key)
        instr = instr_tpls[idx % len(instr_tpls)].render(**binding)
        idx += 1
        for ch in instr:
            assert ord(ch) < 128, f"non-ASCII {ch!r} in instruction: {instr!r}"
        rows.append({
            "category":    pattern["category"],
            "pattern":     pattern["name"],
            "bucket":      bucket,
            "instruction": instr,
            "input":       inp,
            "output":      outp,
        })
    return rows


def build(spec_path: str | Path) -> tuple[list[dict], list[dict], dict]:
    """Generate (train, eval, prompt) from a domain spec.

    `prompt` is the spec's task framing ({system_prompt, input_label}), passed
    through so the caller can persist it next to the data.
    """
    spec = yaml.safe_load(Path(spec_path).read_text())
    cfg = spec.get("config", {})
    rng = random.Random(cfg.get("seed", 42))
    seen_per = cfg.get("eval_seen_per_pattern", 1)

    vocab = spec["vocab"]
    pools_train = _pools(vocab, "train")
    pools_eval = _pools(vocab, "eval")

    train: list[dict] = []
    eval_rows: list[dict] = []

    for pat in spec["patterns"]:
        if pat.get("composite", False):
            n_train = pat.get("train", 0)
            if n_train:  # composition TYPE is trained -> eval bucket tests vocab transfer
                train += _gen(pat, pools_train, rng, n_train, "train")
                eval_rows += _gen(pat, pools_eval, rng, pat.get("eval", 4), "composed_trained")
            else:        # composition TYPE held out -> tests true compositional generalization
                eval_rows += _gen(pat, pools_eval, rng, pat.get("eval", 4), "composed_heldout")
            continue
        train += _gen(pat, pools_train, rng, pat.get("train", 12), "train")
        eval_rows += _gen(pat, pools_eval, rng, pat.get("eval", 2), "single_unseen")
        if seen_per:
            eval_rows += _gen(pat, pools_train, rng, seen_per, "single_seen")

    # dedup train; strip any held-out (unseen) eval pairs that leaked into train
    unseen_keys = {(r["input"], r["output"]) for r in eval_rows if r["bucket"] != "single_seen"}
    seen_keys: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for r in train:
        key = (r["input"], r["output"])
        if key in seen_keys or key in unseen_keys:
            continue
        seen_keys.add(key)
        deduped.append(r)

    rng.shuffle(deduped)
    rng.shuffle(eval_rows)
    return deduped, eval_rows, spec.get("prompt", {})


def write(train: list[dict], eval_rows: list[dict], prompt: dict,
          out_dir: str | Path, prefix: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{prefix}_prompt.json").write_text(json.dumps(prompt, indent=2) + "\n")
    (out / f"{prefix}_train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n")
    (out / f"{prefix}_eval.jsonl").write_text("\n".join(json.dumps(r) for r in eval_rows) + "\n")
