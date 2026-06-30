#!/usr/bin/env python3
"""Prompt the Q4_K_M GGUF from Python via llama-cpp-python.

    pip install llama-cpp-python        # builds llama.cpp
    python examples/prompt_gguf.py

Use create_chat_completion(messages=...) so the model's embedded chat template
is applied and special tokens (<|im_start|>, and Qwen3's <think>) tokenize
correctly. Do NOT build a raw prompt string and call the model directly
(create_completion / __call__): that path won't parse <think>, so Qwen3 rambles.
LFM2.5 has no <think> block, so a raw prompt happens to work for it - which hides
the bug. Always prompt GGUFs through the chat template.
"""
from llama_cpp import Llama

# Pull the GGUF straight from the Hub (or pass model_path="path/to/local.gguf").
llm = Llama.from_pretrained(
    repo_id="lazos/lfm2.5-350m-promql",   # or "lazos/qwen3-0.6b-promql"
    filename="*Q4_K_M.gguf",
    n_ctx=512,
    verbose=False,
)

# The system prompt the models were trained with (also in the repo's
# prompt_config.json); the input is passed under a "Query:" label.
SYSTEM = (
    "You are a PromQL optimization expert. Given a PromQL expression and an "
    "instruction, rewrite the expression according to the instruction. Output "
    "only the improved PromQL expression, no explanation."
)
instruction = 'add a job="payment-svc" filter before the rate so fewer series are scanned'
query = "rate(redis_commands_processed_total[5m])"

resp = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f"{instruction}\n\nQuery:\n{query}"},
    ],
    max_tokens=128,
    temperature=0.0,
)
print(resp["choices"][0]["message"]["content"].strip())
# expected: rate(redis_commands_processed_total{job="payment-svc"}[5m])
