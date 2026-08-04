import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

SYSTEM_PROMPT = """You are an expert data-quality assessor for pretraining corpora.

Score the document on a 1-10 integer scale for each dimension:
- coherence
- information_density
- lexical_diversity
- boilerplate
- overall_pretraining_utility

Return ONLY a JSON object with exactly these keys:
{
  "coherence": <int>,
  "information_density": <int>,
  "lexical_diversity": <int>,
  "boilerplate": <int>,
  "overall_pretraining_utility": <int>,
  "justification": "<one sentence>"
}
"""

MODEL = "Qwen/Qwen2.5-14B-Instruct"
TEST_FILE = "data/midcorpus.jsonl"      # <-- change if needed
N_DOCS = 50

print("Loading model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.eval()

print("Loading documents...")

docs = []
with open(TEST_FILE, encoding="utf-8") as f:
    for line in f:
        docs.append(json.loads(line))
        if len(docs) == N_DOCS:
            break


def build_prompt(doc):
    text = (doc.get("text") or "")[:6000]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content":
                f"Category: {doc.get('category','unknown')}\n"
                f"Source: {doc.get('source','unknown')}\n\n"
                f"Document:\n{text}",
        },
    ]


# -------------------------
# Warmup (not timed)
# -------------------------
print("Warmup...")

messages = build_prompt(docs[0])
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    model.generate(
        input_ids,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

torch.cuda.synchronize()

# -------------------------
# Timed benchmark
# -------------------------
latencies = []

print(f"Benchmarking {N_DOCS-1} documents...\n")

for i, doc in enumerate(docs[1:], start=1):

    messages = build_prompt(doc)

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        model.generate(
            input_ids,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    end = time.perf_counter()

    latencies.append(end - start)

    print(f"{i:2d}/{N_DOCS-1}: {latencies[-1]:.2f}s")

print("\n==============================")
print(f"Mean latency   : {np.mean(latencies):.3f} s/doc")
print(f"Median latency : {np.median(latencies):.3f} s/doc")
print(f"Min latency    : {np.min(latencies):.3f} s/doc")
print(f"Max latency    : {np.max(latencies):.3f} s/doc")
print("==============================")