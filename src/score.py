"""
score.py — Score curated corpus with Qwen2.5-32B-Instruct as LLM judge

Two backends, selected via --backend:

  local  (default) — loads Qwen 32B in 4-bit via bitsandbytes on GPU.
                     Intended for Kaggle 2xT4 or any CUDA machine with 30GB+ VRAM.

  api              — hits an OpenAI-compatible endpoint (Together AI, Fireworks, etc.)
                     Set SCORE_API_KEY and SCORE_BASE_URL env vars, or pass via args.
                     Good for local smoke testing without a GPU.

Output appends a "scores" field to each document:
  {
    "coherence": 8,
    "information_density": 7,
    "lexical_diversity": 6,
    "boilerplate": 2,
    "overall_pretraining_utility": 7,
    "justification": "..."
  }

Usage:
  # Kaggle / local GPU (4-bit)
  python score.py --backend local --input data/corpus.jsonl --output data/scored.jsonl

  # API (Kaggle-friendly fallback when no GPU is available)
  export SCORE_API_KEY=<your_key>
  export SCORE_BASE_URL=https://api.together.xyz/v1
  python score.py --backend api --input data/corpus.jsonl --output data/scored.jsonl

  # Smoke test on 20 docs
  python score.py --backend api --limit 20

Dependencies:
  pip install transformers bitsandbytes accelerate requests tqdm
  (bitsandbytes only needed for --backend local)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert data-quality assessor for pretraining corpora.

Score the document on a 1-10 integer scale for each dimension:
- coherence: how well the writing flows and is internally consistent
- information_density: how much useful, non-redundant information is packed in
- lexical_diversity: how varied the vocabulary and sentence structures are
- boilerplate: how much filler, repetition, or templated fluff is present
  (10 = extremely boilerplate, 1 = almost none)
- overall_pretraining_utility: how useful this document would be as LLM pretraining material

Return ONLY a JSON object with exactly these keys, no markdown fences, no extra text:
{
  "coherence": <int 1-10>,
  "information_density": <int 1-10>,
  "lexical_diversity": <int 1-10>,
  "boilerplate": <int 1-10>,
  "overall_pretraining_utility": <int 1-10>,
  "justification": "<one sentence>"
}"""


def build_user_message(doc: dict) -> str:
    text = (doc.get("text") or "").strip()
    if len(text) > 6000:
        text = text[:6000] + "\n[truncated]"
    return (
        f"Category: {doc.get('category', 'unknown')}\n"
        f"Source: {doc.get('source', 'unknown')}\n\n"
        f"Document:\n{text}"
    )


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "coherence", "information_density", "lexical_diversity",
    "boilerplate", "overall_pretraining_utility", "justification",
}


def parse_scores(raw: str) -> dict:
    text = raw.strip()
    text = re.sub(r"```json|```", "", text).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError(f"No JSON object found in model output: {text[:200]}")
        parsed = json.loads(match.group(0))

    missing = REQUIRED_KEYS - set(parsed.keys())
    if missing:
        raise ValueError(f"Model output missing keys: {missing}")

    return {
        "coherence":                  int(parsed["coherence"]),
        "information_density":        int(parsed["information_density"]),
        "lexical_diversity":          int(parsed["lexical_diversity"]),
        "boilerplate":                int(parsed["boilerplate"]),
        "overall_pretraining_utility": int(parsed["overall_pretraining_utility"]),
        "justification":              str(parsed["justification"])[:600],
    }


# ---------------------------------------------------------------------------
# Local backend (4-bit Qwen on GPU)
# ---------------------------------------------------------------------------

def load_local_model(model_id: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError:
        print("ERROR: transformers, torch, and bitsandbytes are required for --backend local", file=sys.stderr)
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found. Use --backend api for CPU/MPS inference.", file=sys.stderr)
        sys.exit(1)

    torch.cuda.empty_cache()

    print(f"Loading {model_id} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.\n")
    return tokenizer, model


def score_local(tokenizer, model, doc: dict, max_new_tokens: int = 256) -> dict:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_message(doc)},
    ]
    # Use the model's chat template
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # greedy — deterministic scores
            temperature=1.0,        # ignored when do_sample=False, silences warning
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return parse_scores(raw)


# ---------------------------------------------------------------------------
# API backend (OpenAI-compatible)
# ---------------------------------------------------------------------------

def score_api(doc: dict, model: str, api_key: str, base_url: str,
              retries: int = 3, backoff: float = 2.0) -> dict:
    """Call a Qwen-compatible API endpoint when a local GPU is unavailable."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_message(doc)},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    url = f"{base_url.rstrip('/')}/chat/completions"

    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"] or "{}"
            return parse_scores(content)
        except (requests.RequestException, ValueError, KeyError) as e:
            if attempt == retries - 1:
                raise
            wait = backoff * (2 ** attempt)
            print(f"  Attempt {attempt+1} failed ({e}), retrying in {wait:.0f}s...", file=sys.stderr)
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score corpus with Qwen 32B judge")
    parser.add_argument("--input",     default="data/corpus.jsonl")
    parser.add_argument("--output",    default="data/scored.jsonl")
    parser.add_argument("--backend",   default="local", choices=["local", "api"],
                        help="'local' = 4-bit GPU inference; 'api' = OpenAI-compatible endpoint")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct",
                        help="Model ID (HF hub name for local; provider model name for api)")
    parser.add_argument("--api-key",   default=os.getenv("SCORE_API_KEY"))
    parser.add_argument("--base-url",  default=os.getenv("SCORE_BASE_URL", "https://api.together.xyz/v1"))
    parser.add_argument("--limit",     type=int, default=None,
                        help="Score only the first N docs (smoke test)")
    parser.add_argument("--resume",    action="store_true",
                        help="Skip docs already present in --output (safe to re-run after interruption)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load corpus
    with input_path.open(encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        docs = docs[:args.limit]

    # Resume: skip already-scored doc IDs
    already_scored: set = set()
    if args.resume and output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    already_scored.add(json.loads(line)["id"])
        print(f"Resuming — {len(already_scored)} docs already scored, skipping.")

    docs_to_score = [d for d in docs if d["id"] not in already_scored]
    print(f"Scoring {len(docs_to_score)} documents with backend={args.backend}\n")

    # Load model if local
    tokenizer = model = None
    if args.backend == "local":
        tokenizer, model = load_local_model(args.model)
    else:
        if not args.api_key:
            print("ERROR: --backend api requires SCORE_API_KEY env var or --api-key", file=sys.stderr)
            sys.exit(1)

    errors = 0
    # Append mode so --resume works safely
    with output_path.open("a", encoding="utf-8") as out:
        for doc in tqdm(docs_to_score, unit="doc"):
            try:
                if args.backend == "local":
                    scores = score_local(tokenizer, model, doc)
                else:
                    scores = score_api(
                        doc, args.model, args.api_key, args.base_url
                    )
            except Exception as e:
                print(f"\n  ERROR scoring {doc['id']}: {e}", file=sys.stderr)
                errors += 1
                continue

            out.write(json.dumps({**doc, "scores": scores}, ensure_ascii=False) + "\n")

    total = len(docs_to_score) - errors
    print(f"\nDone. {total} scored, {errors} errors. Output: {output_path}")


if __name__ == "__main__":
    main()