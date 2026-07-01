#!/usr/bin/env python3
"""
Entry point: uv run benchmark --configs configs/lfm2.5_1.2b.yaml configs/qwen3_0.6b.yaml configs/danube3_500m.yaml configs/smollm2_360m.yaml configs/lfm2.5_350m.yaml configs/lfm2.5_230m.yaml

Each model config is trained and evaluated in an isolated subprocess so torch.compile
state (fullgraph=True kernel cache) never bleeds between model sizes.

Worker mode (internal): called by the orchestrator with --_worker and --result-file.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

def _categories(rows: list[dict]) -> list[str]:
    """Category keys present across all results (domain-agnostic), sorted."""
    seen: set[str] = set()
    for r in rows:
        seen |= {k for k in r["accuracy"] if k != "__overall__"}
    return sorted(seen)


def _abbrev(name: str, width: int = 11) -> str:
    return name if len(name) <= width else name[:width - 1] + "."


# -- worker (isolated subprocess per model) ------------------------------------

def _worker(config_path: str, data_prefix: str, result_file: str,
            skip_train: bool = False, backend_name: str = "auto") -> None:
    """Train (unless skip_train) one config, evaluate, write JSON result to result_file."""
    from collections import defaultdict

    from lfm_train.backends import get_backend, resolve_name
    from lfm_train.trainer import _load_config, train_sft
    from lfm_train.evaluate import _infer, _normalize
    from lfm_train.dataset import resolve_data

    # The benchmark harness (subprocess isolation, GPU-memory telemetry, torch
    # param introspection) is CUDA-only by design; the MLX backend implements
    # train + inference but not this orchestrator.
    if resolve_name(backend_name) != "cuda":
        raise NotImplementedError(
            "benchmark is CUDA-only; the mlx backend supports train + infer "
            "(use `train` / `infer` / `evaluate` with --backend mlx)")
    backend = get_backend(backend_name)

    cfg = _load_config(config_path)
    output_dir = Path(cfg["training"]["output_dir"])
    train_path, eval_path, prompt = resolve_data(data_prefix)
    timing: dict = {}

    if skip_train:
        # train_sft's own TrainOutput.metrics isn't persisted anywhere on
        # disk (HF doesn't write it to trainer_state.json), so an already-
        # trained checkpoint has no train timing to recover here.
        pass
    else:
        backend.reset_peak_memory()
        train_metrics = train_sft(cfg, str(train_path), prompt, backend=backend_name)
        timing["train_runtime_sec"]        = train_metrics.get("train_runtime")
        timing["train_samples_per_second"] = train_metrics.get("train_samples_per_second")
        timing["train_steps_per_second"]   = train_metrics.get("train_steps_per_second")
        peak = backend.peak_memory_mb()
        if peak is not None:
            timing["train_gpu_mem_peak_mb"] = peak

    backend.reset_peak_memory()

    adapter_path = str(output_dir / "lora_adapter")
    print(f"\nLoading adapter for eval: {adapter_path}", flush=True)
    model, tokenizer = backend.load_inference(
        adapter_path, max_seq_length=cfg["model"]["max_seq_length"])

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    examples = [
        json.loads(line)
        for line in Path(eval_path).read_text().splitlines()
        if line.strip()
    ]

    system_prompt = prompt.get("system_prompt")
    input_label = prompt.get("input_label", "Query")
    cats: dict[str, list[bool]] = defaultdict(list)
    bkts: dict[str, list[bool]] = defaultdict(list)
    preds: list[dict] = []
    eval_tokens = 0
    eval_start = time.perf_counter()
    for ex in examples:
        pred  = _infer(backend, model, tokenizer, ex["instruction"], ex.get("input", ""),
                       system_prompt, input_label)
        eval_tokens += len(tokenizer(pred, add_special_tokens=False).input_ids)
        match = _normalize(pred) == _normalize(ex["output"])
        cats[ex.get("category", "unknown")].append(match)
        bkts[ex.get("bucket", "unknown")].append(match)
        preds.append({**ex, "pred": pred, "match": match})
    eval_runtime = time.perf_counter() - eval_start

    timing["eval_runtime_sec"]      = eval_runtime
    timing["eval_examples"]         = len(examples)
    timing["eval_examples_per_sec"] = len(examples) / eval_runtime if eval_runtime else 0
    timing["eval_tokens_generated"] = eval_tokens
    timing["eval_tokens_per_sec"]   = eval_tokens / eval_runtime if eval_runtime else 0
    eval_peak = backend.peak_memory_mb()
    if eval_peak is not None:
        timing["eval_gpu_mem_peak_mb"] = eval_peak

    total_ok = sum(sum(v) for v in cats.values())
    total    = sum(len(v) for v in cats.values())

    def _summarize(groups: dict[str, list[bool]]) -> dict:
        return {
            k: {"ok": sum(v), "total": len(v), "pct": 100 * sum(v) // len(v)}
            for k, v in groups.items()
        }

    acc = _summarize(cats)
    acc["__overall__"] = {
        "ok": total_ok, "total": total,
        "pct": 100 * total_ok // total if total else 0,
    }

    result = {
        "config":            config_path,
        "model":             cfg["model"]["name"],
        "accuracy":          acc,
        "buckets":           _summarize(bkts),
        "timing":            timing,
        "params":            {"total": total_params, "trainable": trainable_params},
        "lora":              {"r": cfg["lora"]["r"], "alpha": cfg["lora"]["alpha"],
                               "modules_to_save": cfg["lora"].get("modules_to_save")},
        "examples": preds,
    }
    Path(result_file).write_text(json.dumps(result, indent=2))
    print(f"\nResult written -> {result_file}", flush=True)


# -- table printing -------------------------------------------------------------

def _pct(acc: dict, key: str) -> str:
    d = acc.get(key, {})
    return f"{d['pct']}%" if d else "n/a"


def _print_table(rows: list[dict]) -> None:
    cats      = _categories(rows)
    col_keys  = ["__overall__"] + cats
    col_names = ["overall"] + [_abbrev(c) for c in cats]
    widths    = [len(n) for n in col_names]
    for r in rows:
        for i, k in enumerate(col_keys):
            widths[i] = max(widths[i], len(_pct(r["accuracy"], k)))
    model_w = max(len(r["model"].split("/")[-1]) for r in rows)
    model_w = max(model_w, len("model"))

    header = f"{'model':<{model_w}}"
    for n, w in zip(col_names, widths):
        header += f"  {n:>{w}}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        line = f"{r['model'].split('/')[-1]:<{model_w}}"
        for k, w in zip(col_keys, widths):
            line += f"  {_pct(r['accuracy'], k):>{w}}"
        print(line)
    print()


def _print_buckets(rows: list[dict]) -> None:
    """Generalization view: vocab transfer (single + trained compositions) vs
    true compositional generalization (held-out composition types)."""
    order = ["single_seen", "single_unseen", "composed_trained", "composed_heldout"]
    names = {"single_seen": "seen-vocab", "single_unseen": "unseen-vocab",
             "composed_trained": "comp-trained", "composed_heldout": "comp-heldout"}
    model_w = max(max(len(r["model"].split("/")[-1]) for r in rows), len("model"))

    header = f"{'model':<{model_w}}"
    for k in order:
        header += f"  {names[k]:>13}"
    print("\ngeneralization buckets:")
    print("  seen/unseen-vocab : single transforms  |  comp-trained : type trained, new vocab")
    print("  comp-heldout      : composition type NEVER trained (true compositional generalization)")
    print(header)
    print("-" * len(header))
    for r in rows:
        b = r.get("buckets", {})
        line = f"{r['model'].split('/')[-1]:<{model_w}}"
        for k in order:
            d = b.get(k)
            line += f"  {(str(d['pct'])+'%') if d else 'n/a':>13}"
        print(line)
    print()


def _print_timing(rows: list[dict]) -> None:
    cols = ["train_sec", "train_it/s", "eval_sec", "tok/sec", "train_gpu_mb", "eval_gpu_mb"]
    model_w = max(max(len(r["model"].split("/")[-1]) for r in rows), len("model"))
    widths  = [max(len(c), 8) for c in cols]

    header = f"{'model':<{model_w}}"
    for c, w in zip(cols, widths):
        header += f"  {c:>{w}}"
    print("\ntiming:")
    print(header)
    print("-" * len(header))
    for r in rows:
        t = r.get("timing", {})
        vals = [
            f"{t['train_runtime_sec']:.0f}" if t.get("train_runtime_sec") is not None else "n/a",
            f"{t['train_steps_per_second']:.2f}" if t.get("train_steps_per_second") is not None else "n/a",
            f"{t['eval_runtime_sec']:.1f}" if t.get("eval_runtime_sec") is not None else "n/a",
            f"{t['eval_tokens_per_sec']:.1f}" if t.get("eval_tokens_per_sec") is not None else "n/a",
            f"{t['train_gpu_mem_peak_mb']:.0f}" if t.get("train_gpu_mem_peak_mb") is not None else "n/a",
            f"{t['eval_gpu_mem_peak_mb']:.0f}" if t.get("eval_gpu_mem_peak_mb") is not None else "n/a",
        ]
        line = f"{r['model'].split('/')[-1]:<{model_w}}"
        for v, w in zip(vals, widths):
            line += f"  {v:>{w}}"
        print(line)
    print()


# -- orchestrator --------------------------------------------------------------

def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("--configs",     nargs="*", default=[])
    pa.add_argument("--data",        default="data/promql",
                     help="data prefix; loads <prefix>_eval.jsonl + <prefix>_prompt.json")
    pa.add_argument("--output",      default="outputs/benchmark.json")
    pa.add_argument("--skip-train",  action="store_true",
                     help="reuse each config's existing lora_adapter instead of retraining "
                          "(no train timing is recorded in this mode)")
    pa.add_argument("--backend",     choices=["auto", "cuda", "mlx"], default="auto",
                     help="compute backend (CUDA-only; mlx is unsupported for benchmark)")
    pa.add_argument("--_worker",     metavar="CONFIG", default=None, help=argparse.SUPPRESS)
    pa.add_argument("--result-file", default=None,                   help=argparse.SUPPRESS)
    args = pa.parse_args()

    if args._worker:
        _worker(args._worker, args.data, args.result_file,
                skip_train=args.skip_train, backend_name=args.backend)
        return

    if not args.configs:
        pa.error("--configs is required")

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for config_path in args.configs:
        cfg_name    = Path(config_path).stem
        result_file = str(out_dir / f".bench_{cfg_name}.json")

        print(f"\n{'='*60}")
        print(f"Benchmarking {cfg_name}  (config: {config_path})")
        print(f"{'='*60}\n", flush=True)

        worker_cmd = [sys.executable, "-m", "lfm_train.benchmark",
                      "--_worker",     config_path,
                      "--data",        args.data,
                      "--result-file", result_file,
                      "--backend",     args.backend]
        if args.skip_train:
            worker_cmd.append("--skip-train")
        ret = subprocess.run(worker_cmd)

        if ret.returncode != 0:
            print(f"ERROR: worker for {cfg_name} exited {ret.returncode}", flush=True)
            continue

        result = json.loads(Path(result_file).read_text())
        all_results.append(result)

        ok  = result["accuracy"]["__overall__"]["ok"]
        tot = result["accuracy"]["__overall__"]["total"]
        print(f"\n  {cfg_name}: {ok}/{tot} ({100*ok//tot}%)", flush=True)

    if all_results:
        _print_table(all_results)
        _print_buckets(all_results)
        _print_timing(all_results)
        Path(args.output).write_text(json.dumps(all_results, indent=2) + "\n")
        print(f"Full results -> {args.output}")

        from lfm_train.plot import plot_benchmark
        chart = Path(args.output).with_name("benchmark_chart.png")
        plot_benchmark(args.output, chart)


if __name__ == "__main__":
    main()
