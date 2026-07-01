.PHONY: install install-mlx dataset train eval benchmark infer publish clean

# Defaults (override on the command line, e.g. `make train CONFIG=configs/qwen3_0.6b.yaml`)
CONFIG  ?= configs/lfm2.5_350m.yaml
DATA    ?= data/promql
ADAPTER ?= outputs/lfm2.5-350m/lora_adapter

# All model configs, largest first.
CONFIGS = configs/lfm2.5_1.2b.yaml configs/qwen3_0.6b.yaml configs/danube3_500m.yaml \
          configs/smollm2_360m.yaml configs/lfm2.5_350m.yaml configs/lfm2.5_230m.yaml

# NVIDIA / Linux backend (default).
install:
	uv sync --extra cuda

# Apple Silicon / Metal backend.
install-mlx:
	uv sync --extra mlx

# Regenerate the dataset from the domain spec (src/lfm_train/data/domains/).
dataset:
	uv run gen-dataset

# Train one model. CONFIG selects the model; DATA selects the domain data.
train: dataset
	uv run train --config $(CONFIG) --data $(DATA)

# Evaluate one trained adapter against the eval split.
eval:
	uv run evaluate --finetuned $(ADAPTER) --data $(DATA) --output outputs/eval_results.jsonl

# Train + evaluate every model in an isolated subprocess; writes outputs/benchmark.json + chart.
benchmark: dataset
	uv run benchmark --configs $(CONFIGS) --data $(DATA) --output outputs/benchmark.json

# Interactive REPL (or batch with PROMPTS=file.json) against a trained adapter.
infer:
	uv run infer $(ADAPTER) $(if $(PROMPTS),--prompts $(PROMPTS),)

# Publish a merged model to the Hugging Face Hub (needs HF_TOKEN in the env).
# Note: the Q4_K_M GGUF lives in gguf_gguf/ (unsloth writes a safetensors copy to gguf/).
#   make publish MODEL_DIR=outputs/lfm2.5-350m/merged_16bit REPO=user/lfm2.5-350m-promql GGUF=outputs/lfm2.5-350m/gguf_gguf
publish:
	uv run publish $(MODEL_DIR) $(REPO) $(if $(GGUF),--gguf $(GGUF),)

clean:
	rm -rf outputs/ data/ .venv
