#!/usr/bin/env python3
"""Prompt a published lfm-train model with transformers (merged fp16 weights).

    uv run python examples/prompt_transformers.py

The only thing that matters for correctness is using the tokenizer's chat
template (apply_chat_template) with the SAME system prompt the model was trained
with - never hand-concatenate the turns yourself.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "lazos/lfm2.5-350m-promql"   # or "lazos/qwen3-0.6b-promql"

# The system prompt the models were trained with (also saved in the repo as
# prompt_config.json). The input is passed under a "Query:" label.
SYSTEM = (
    "You are a PromQL optimization expert. Given a PromQL expression and an "
    "instruction, rewrite the expression according to the instruction. Output "
    "only the improved PromQL expression, no explanation."
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto")

instruction = 'add a job="payment-svc" filter before the rate so fewer series are scanned'
query = "rate(redis_commands_processed_total[5m])"

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user",   "content": f"{instruction}\n\nQuery:\n{query}"},
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt",
).to(model.device)
out = model.generate(inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip())
# expected: rate(redis_commands_processed_total{job="payment-svc"}[5m])
