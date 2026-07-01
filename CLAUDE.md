# lfm-train

Benchmarking harness for fine-tuning small (<=1.2B) base language models on a
domain-specific instruction -> rewrite task, with QLoRA SFT. Reference domain:
PromQL query optimization. `uv` for env.

## Backends (NVIDIA vs Apple Silicon)

Every model-touching operation goes through a `Backend` (`src/lfm_train/backends/`):
- `cuda` - Unsloth + TRL + PEFT + bitsandbytes on NVIDIA/Linux (RTX 3060 Ti 8GB
  reference). Full feature set: train, eval, benchmark, infer, GGUF publish.
- `mlx` - `mlx_lm` on Apple Silicon/Metal. Implements **train + inference** only;
  `benchmark` and `publish` raise `NotImplementedError` (the subprocess/CUDA-
  telemetry orchestrator and GGUF export don't port). *Untested on Metal* - no
  Apple hardware; delivered with structural checks on Linux.

Selection (`backends.get_backend`): explicit `--backend {auto,cuda,mlx}` flag ->
`LFM_TRAIN_BACKEND` env -> platform default (`mlx` on macOS, else `cuda`). The
install-time uv extra and this runtime default both key off platform, so they
align by default.

Install-time branching is via uv markers + conflicting extras (one `uv.lock`):
`uv sync --extra cuda` (Linux) / `uv sync --extra mlx` (macOS-arm64). `mlx-lm` is
marker-gated to darwin-arm64 and `torch/unsloth/bitsandbytes` to linux, so neither
host resolves the other's stack. `transformers>=5` is shared by both backends.

MLX specifics: QLoRA = point training at a 4-bit-converted base (`mlx_lm.convert
-q`); no GGUF (mlx-lm only exports GGUF for Llama/Mistral/Mixtral); serve the MLX
dir via `mlx_lm.server` or LM Studio. The dataset/prompt layer is shared verbatim
- `dataset._format` already renders `text` with `enable_thinking=False`, which is
the MLX-safe representation, so both backends train on identical strings.

The seam is the only place backends differ: `backends/base.py` (interface),
`cuda.py`, `mlx.py`, and the pure `config_map.py` (project config -> mlx LoRA
knobs, unit-testable with no mlx installed). The dataset engine, prompt rendering,
config merge, plotting, eval scoring, and `publish` (GGUF) are backend-agnostic.

## Separation of concerns (important)

- `configs/*.yaml` are MODEL-ONLY and domain-agnostic: model name, LoRA, training
  hyperparams. They contain no dataset path and no prompt. `base.yaml` holds
  shared defaults; per-model files override.
- The DOMAIN lives entirely in `domains/<domain>.yaml` (repo root, parallel to
  configs/): vocabulary, transformation patterns, AND the task framing
  (`prompt.system_prompt`, `prompt.input_label`).
- `gen-dataset` renders the domain spec into `data/<domain>_{train,eval}.jsonl`
  plus `data/<domain>_prompt.json` (the prompt sidecar). Training/eval/inference
  select a domain at the CLI via `--data data/<domain>`; the prompt travels with
  the data and is also saved next to each adapter (`prompt_config.json`) so
  inference reproduces the exact training prompt.

Adding a domain = one new `domains/<name>.yaml`, then `gen-dataset --domain <name>`.
No Python changes.

## Dataset generation is declarative

- `data/engine.py` - domain-agnostic: samples vocab, renders Jinja2 templates,
  splits train/eval by held-out vocab, dedups, audits leakage. `build()` returns
  `(train, eval, prompt)`.
- `domains/<domain>.yaml` - the whole domain as data (repo root).
- `data/generate.py` - thin CLI. Instruction strings must be plain ASCII.

## Generalization is measured, not assumed

Train and eval vocab are DISJOINT, so eval accuracy reflects learned transforms,
not memorized tokens. Eval buckets: `single_seen` (train vocab), `single_unseen`
(held-out vocab), `composed_trained` / `composed_heldout` (compositions whose
TYPE was/wasn't trained). Watch the seen-vs-unseen delta and the held-out
buckets - chasing 100% on overlapping vocab is overfitting.

## Key empirical finding

Single-transform generalization to unseen vocab is excellent and scale- AND
architecture-independent. Composition of two transforms does NOT emerge for
unseen combinations - not via coverage, instruction tricks, scale, or a different
model family. The only reliable lever is putting the composed form in training.
Full evidence: @docs/compositional-generalization.md

## Commands

- `uv run gen-dataset [--domain promql]` - regenerate data + prompt sidecar
- `uv run train --config configs/<model>.yaml --data data/promql`
- `uv run benchmark --configs <models...> --data data/promql --output outputs/benchmark.json`
  - trains + evals each model in an isolated subprocess (avoids torch.compile
  state bleed across model sizes), prints accuracy/bucket/timing tables, writes a chart.
- `uv run evaluate --finetuned outputs/<model>/lora_adapter --data data/promql`
- `uv run infer outputs/<model>/lora_adapter` (REPL; `--prompts file.json` for batch)
- `uv run publish outputs/<model>/merged_16bit <user>/<repo> --gguf outputs/<model>/gguf_gguf`
  (needs `HF_TOKEN` in env)

Or the Makefile: `make dataset|train|benchmark|eval|infer|publish`.

## Prompting format (read before touching inference)

The chat format is ChatML, applied identically at train and inference time:
- system = the domain `system_prompt`; user = `dataset.build_user()` (the
  instruction, optionally followed by `\n\n{input_label}:\n{input}`); assistant
  = the target.
- Rendered via `tokenizer.apply_chat_template(..., enable_thinking=False)` in
  BOTH `dataset._format` (train) and `evaluate._infer` (inference). The exact
  framing (system_prompt + input_label) is saved next to each adapter as
  `prompt_config.json`, so serving reproduces training.
- `enable_thinking=False` is a no-op for plain templates (LFM2.5) but for Qwen3
  it STILL injects an empty `<think>\n\n</think>\n\n` block into the assistant
  turn. The model is trained with that block, so inference must reproduce it.

Serving the GGUF: ALWAYS go through the model's embedded chat template -
`llama-server` (default), `llama-cli/llama-completion --jinja`, or Ollama. These
tokenize the special tokens (`<|im_start|>`, `<think>`) correctly. Do NOT
hand-build a raw ChatML string and feed it via `llama-completion -f`: that path
does not parse `<think>` as a special token, so Qwen3 (whose assistant turn
contains `<think></think>`) gets malformed input and rambles (it echoes an
instruction + "Query:" instead of answering). LFM2.5 has no think block, so a
raw prompt happens to work for it - which masks the bug. (A probe using raw `-f`
prompts once made the Qwen3 GGUF look broken when it was fine; verify GGUFs via
`--jinja` only.)

## Working notes

- Run benchmarks/training in the background and poll the log; don't pull whole
  logs into context.
- Subprocess isolation per model in `benchmark.py` is load-bearing - per-size
  kernel shapes conflict under `torch.compile` in one process.
- `FastLanguageModel.from_pretrained` always passes `use_exact_model_name=True`
  (else Unsloth silently swaps in a prequantized mirror, sometimes missing a
  chat template) and the result goes through `dataset.ensure_chat_template` +
  `dataset.align_eos_token`. Base checkpoints with tied embeddings whose chat
  special tokens were never pretrained (e.g. Qwen3's <|im_end|>) also need
  `lora.modules_to_save: [lm_head, embed_tokens]` to learn to stop.
- The real Q4_K_M GGUF is written to `outputs/<model>/gguf_gguf/`; `gguf/` holds
  an intermediate safetensors copy (Unsloth quirk).
- Apple Silicon port path, if needed: MLX / mlx-lm (not PyTorch MPS).

## Models benchmarked

LFM2.5 230M / 350M / 1.2B-Base (LFM Open License v1.0: free under $10M revenue),
Qwen3-0.6B-Base, h2o-danube3-500m-base, SmolLM2-360M (all Apache-2.0). One config
per model in `configs/`. See @BENCHMARK.md for results.
