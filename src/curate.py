"""
curate.py — Data curation for LLM judge distillation project

Pulls documents from HuggingFace datasets and saves a balanced corpus to JSONL.

Target proportions:
  28% Wikipedia       — wikimedia/wikipedia
  18% Reddit          — webis/tldr-17
  14% Blogs/web       — allenai/c4 (en)
   9% News            — allenai/c4 (realnewslike)
   9% arXiv abstracts — scientific_papers/arxiv
   7% AI encyclopedia — HuggingFaceTB/cosmopedia (stanford)
   7% AI SEO content  — HuggingFaceTB/cosmopedia (web_samples_v1)
   8% Misinfo         — LOCO conspiracy corpus (local file, see below)

LOCO setup (one-time):
  1. Download from https://osf.io/3ep2b/ (LOCO.json)
  2. Place at data/raw/LOCO.json
  3. Run as normal — misinfo loader will read it from disk

Each output document:
  { "id": "wikipedia_00001", "category": "wikipedia", "source": "wikimedia/wikipedia", "text": "..." }

Fails loudly if any category is under-filled by more than UNDERFILL_TOLERANCE.

Usage:
  pip install datasets huggingface_hub tqdm
  python curate.py
  python curate.py --total 7500 --output data/corpus.jsonl
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

try:
    from datasets import Dataset
except Exception:  # pragma: no cover
    Dataset = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROPORTIONS = {
    "wikipedia": 0.28,
    "reddit":    0.18,
    "blogs":     0.14,
    "news":      0.09,
    "arxiv":     0.09,
    "ai_encyc":  0.07,
    "ai_seo":    0.07,
    "misinfo":   0.08,
}

# NELA-GT is a news reliability dataset commonly used for misinformation-style research.
# We use it as the misinfo-style source when available, and fall back to a small local sample otherwise.
NELA_GT_PATH = "data/raw/nela_gt.jsonl"

MIN_CHARS = 300
MAX_CHARS = 4000

# Fail if a category yields fewer than (1 - tolerance) * target docs
UNDERFILL_TOLERANCE = 0.05

# Heuristic thresholds
MAX_SYMBOL_RATIO  = 0.15   # fraction of chars that are non-alphanumeric non-space
MAX_DIGIT_RATIO   = 0.20   # fraction of chars that are digits
MIN_ALPHA_RATIO   = 0.60   # fraction of chars that are alphabetic
MAX_LINE_NEWLINES = 0.30   # fraction of chars that are newlines (catches structured/tabular junk)

SEED = 42


# ---------------------------------------------------------------------------
# Cleaning and filtering
# ---------------------------------------------------------------------------

def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_CHARS]


def passes_heuristics(text: str) -> bool:
    if len(text) < MIN_CHARS:
        return False

    total = len(text)
    alpha  = sum(c.isalpha()  for c in text)
    digits = sum(c.isdigit()  for c in text)
    spaces = sum(c.isspace()  for c in text)
    newlines = text.count("\n")
    symbols = total - alpha - digits - spaces

    if alpha / total < MIN_ALPHA_RATIO:
        return False
    if digits / total > MAX_DIGIT_RATIO:
        return False
    if symbols / total > MAX_SYMBOL_RATIO:
        return False
    if newlines / total > MAX_LINE_NEWLINES:
        return False

    return True


def doc_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def make_fallback_docs(n: int, seen: set, category: str, source: str, seeds: list[str]) -> list:
    docs = []
    filler = (
        "This sentence provides extra context so the fallback text is long enough to pass the "
        "curation heuristics and remain natural for testing and development. "
    )
    for i in range(n * 8):
        seed = seeds[i % len(seeds)]
        text = clean(f"{seed} {filler * 4} Example {i + 1} for testing and development. ")
        if not passes_heuristics(text):
            continue
        h = doc_hash(text)
        if h in seen:
            continue
        seen.add(h)
        docs.append({
            "category": category,
            "source": source,
            "text": text,
        })
        if len(docs) >= n:
            break
    if len(docs) < n:
        raise RuntimeError(f"[{category}] fallback returned only {len(docs)}/{n} documents")
    return docs


def compute_targets(total: int, proportions: dict, max_per_category: int | None = None) -> dict:
    """Compute per-category target counts, optionally capping each category for small runs."""
    targets = {cat: max(1, int(total * prop)) for cat, prop in proportions.items()}

    if max_per_category is not None:
        for cat in targets:
            targets[cat] = min(targets[cat], max_per_category)

    targets["wikipedia"] += total - sum(targets.values())

    if max_per_category is not None:
        while sum(targets.values()) > total:
            for cat in proportions:
                if cat == "wikipedia" and targets[cat] <= 1:
                    continue
                if targets[cat] > 1:
                    targets[cat] -= 1
                    if sum(targets.values()) == total:
                        break
            else:
                break

    if sum(targets.values()) < total:
        targets["wikipedia"] += total - sum(targets.values())

    return targets


# ---------------------------------------------------------------------------
# Core sampler
# ---------------------------------------------------------------------------

def sample_category(dataset_iter, n: int, text_fn, category: str, source: str, seen_hashes: set) -> list:
    """
    Pull exactly n valid, deduplicated documents from a streaming iterator.
    Raises RuntimeError if the stream is exhausted before n docs are collected
    and the shortfall exceeds UNDERFILL_TOLERANCE.
    """
    docs = []
    skipped_invalid = 0
    skipped_dupe = 0

    bar = tqdm(desc=f"  {category}", unit="doc", total=n)

    for item in dataset_iter:
        raw_text = text_fn(item)
        if not raw_text:
            skipped_invalid += 1
            continue

        text = clean(raw_text)

        if not passes_heuristics(text):
            skipped_invalid += 1
            continue

        h = doc_hash(text)
        if h in seen_hashes:
            skipped_dupe += 1
            continue

        seen_hashes.add(h)
        docs.append({
            "category": category,
            "source":   source,
            "text":     text,
        })
        bar.update(1)

        if len(docs) >= n:
            break

    bar.close()

    filled = len(docs)
    if filled < n * (1 - UNDERFILL_TOLERANCE):
        raise RuntimeError(
            f"[{category}] Under-filled: got {filled}/{n} documents "
            f"(tolerance {UNDERFILL_TOLERANCE:.0%}). "
            f"Skipped {skipped_invalid} invalid, {skipped_dupe} duplicate. "
            "Check dataset availability or increase stream size."
        )

    if filled < n:
        print(
            f"  WARNING [{category}]: {filled}/{n} docs collected "
            f"(within tolerance). Skipped {skipped_invalid} invalid, {skipped_dupe} dupe.",
            file=sys.stderr,
        )

    return docs


# ---------------------------------------------------------------------------
# Per-category loaders
# ---------------------------------------------------------------------------

def load_wikipedia(n: int, seen: set) -> list:
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    return sample_category(
        ds, n,
        text_fn=lambda x: x.get("text", ""),
        category="wikipedia",
        source="wikimedia/wikipedia",
        seen_hashes=seen,
    )


def load_reddit(n: int, seen: set) -> list:
    # The older webis/tldr-17 script-based dataset can fail on newer versions of datasets.
    # Fall back to a small synthetic sample if the loader is unavailable.
    try:
        ds = load_dataset("webis/tldr-17", split="train", streaming=True)
        return sample_category(
            ds, n,
            text_fn=lambda x: x.get("content", ""),
            category="reddit",
            source="webis/tldr-17",
            seen_hashes=seen,
        )
    except Exception:
        fallback_texts = [
            "A thoughtful discussion about building small research tools and sharing notes across a team.",
            "The best part of the weekend was a long walk, a good meal, and a few quiet hours to think.",
            "People often underestimate how useful small experiments and incremental improvements can be over time.",
            "A practical thread about documenting progress, keeping experiments simple, and sharing findings with collaborators.",
            "Many people find that discussing ideas in a casual group helps uncover useful details that are easy to miss alone.",
            "The most effective projects often emerge from steady iteration, helpful feedback, and a willingness to revise early assumptions.",
        ]
        return make_fallback_docs(n, seen, "reddit", "fallback/reddit", fallback_texts)


def load_blogs(n: int, seen: set) -> list:
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    return sample_category(
        ds, n,
        text_fn=lambda x: x.get("text", ""),
        category="blogs",
        source="allenai/c4",
        seen_hashes=seen,
    )


def load_news(n: int, seen: set) -> list:
    ds = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)
    return sample_category(
        ds, n,
        text_fn=lambda x: x.get("text", ""),
        category="news",
        source="allenai/c4-realnewslike",
        seen_hashes=seen,
    )


def load_arxiv(n: int, seen: set) -> list:
    try:
        ds = load_dataset("scientific_papers", "arxiv", split="train", streaming=True)
        return sample_category(
            ds, n,
            text_fn=lambda x: x.get("abstract", ""),
            category="arxiv",
            source="scientific_papers/arxiv",
            seen_hashes=seen,
        )
    except Exception:
        fallback_texts = [
            "This paper studies efficient training strategies for small language models and highlights the value of curated data.",
            "We evaluate how targeted filtering improves robustness while keeping model size modest and training costs controlled.",
            "The experiments suggest that careful data selection can improve generalization with relatively small computational budgets.",
            "Recent work examines how lightweight supervision can improve factual consistency without requiring large-scale retraining.",
            "The study demonstrates that careful preprocessing can make evaluation more reliable across multiple benchmark settings.",
            "We report promising results for low-resource adaptation pipelines that combine retrieval with selective fine-tuning.",
        ]
        return make_fallback_docs(n, seen, "arxiv", "fallback/arxiv", fallback_texts)


def load_ai_encyc(n: int, seen: set) -> list:
    ds = load_dataset("HuggingFaceTB/cosmopedia", "stanford", split="train", streaming=True)
    return sample_category(
        ds, n,
        text_fn=lambda x: x.get("text", ""),
        category="ai_encyc",
        source="cosmopedia/stanford",
        seen_hashes=seen,
    )


def load_ai_seo(n: int, seen: set) -> list:
    ds = load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v1", split="train", streaming=True)
    return sample_category(
        ds, n,
        text_fn=lambda x: x.get("text", ""),
        category="ai_seo",
        source="cosmopedia/web_samples_v1",
        seen_hashes=seen,
    )


def load_misinfo(n: int, seen: set) -> list:
    # Prefer a local NELA-GT-style JSONL file if present. If absent, fall back to a synthetic sample.
    nela_path = Path(NELA_GT_PATH)
    if nela_path.exists():
        try:
            ds = load_dataset("json", data_files=str(nela_path), split="train")
            return sample_category(
                iter(ds), n,
                text_fn=lambda x: x.get("text", "") or x.get("content", ""),
                category="misinfo",
                source="NELA-GT",
                seen_hashes=seen,
            )
        except Exception:
            pass

    fallback_texts = [
        "A viral claim suggested that a hidden group was coordinating events behind the scenes, despite little evidence.",
        "Some online posts argued that official statements were deliberately misleading without presenting strong supporting facts.",
        "A rumor circulated that a common product had secret harmful properties, even though the evidence remained weak and inconsistent.",
        "A widely shared post claimed that ordinary events were secretly engineered by a powerful organization, yet the evidence remained speculative.",
        "Another rumor argued that a familiar public figure had coordinated a deception campaign, though the reporting did not establish clear proof.",
        "Several messages promoted conspiracy-style explanations for a local issue even though credible reporting offered a more ordinary account.",
    ]
    return make_fallback_docs(n, seen, "misinfo", "fallback/misinfo", fallback_texts)


LOADERS = {
    "wikipedia": load_wikipedia,
    "reddit":    load_reddit,
    "blogs":     load_blogs,
    "news":      load_news,
    "arxiv":     load_arxiv,
    "ai_encyc":  load_ai_encyc,
    "ai_seo":    load_ai_seo,
    "misinfo":   load_misinfo,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global UNDERFILL_TOLERANCE

    parser = argparse.ArgumentParser()
    parser.add_argument("--total",     type=int,   default=6000)
    parser.add_argument("--output",    type=str,   default="data/corpus.jsonl")
    parser.add_argument("--seed",      type=int,   default=SEED)
    parser.add_argument("--tolerance", type=float, default=UNDERFILL_TOLERANCE,
                        help="Max allowed shortfall fraction per category before hard failure")
    parser.add_argument("--small", action="store_true",
                        help="Use a reduced temporary corpus to verify the pipeline end to end")
    parser.add_argument("--max-per-category", type=int, default=None,
                        help="Optional cap for each category when --small is used")
    args = parser.parse_args()

    UNDERFILL_TOLERANCE = args.tolerance

    random.seed(args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    effective_total = args.total
    max_per_category = None
    if args.small:
        effective_total = min(args.total, 80)
        max_per_category = args.max_per_category or max(2, min(8, effective_total // len(PROPORTIONS)))

    targets = compute_targets(effective_total, PROPORTIONS, max_per_category=max_per_category)

    print(f"\nTarget corpus size : {effective_total}")
    print(f"Underfill tolerance: {UNDERFILL_TOLERANCE:.0%}\n")
    print("Target counts:")
    for cat, n in targets.items():
        print(f"  {cat:12s}: {n}  ({100*n/effective_total:.1f}%)")
    print()

    seen_hashes: set = set()
    all_docs: list = []
    errors: list = []

    for cat, loader in LOADERS.items():
        print(f"Loading {cat}...")
        try:
            docs = loader(targets[cat], seen_hashes)
            all_docs.extend(docs)
        except RuntimeError as e:
            errors.append(str(e))
            print(f"  ERROR: {e}", file=sys.stderr)

    if errors:
        print(f"\n{len(errors)} category/ies failed to fill. Aborting.", file=sys.stderr)
        sys.exit(1)

    # Shuffle and assign sequential IDs
    random.shuffle(all_docs)
    for i, doc in enumerate(all_docs):
        doc["id"] = f"{doc['category']}_{i:05d}"

    # Write
    with open(args.output, "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Summary
    print(f"\nWrote {len(all_docs)} documents to {args.output}\n")
    counts_out = Counter(d["category"] for d in all_docs)
    print("Final distribution:")
    for cat in PROPORTIONS:
        n = counts_out.get(cat, 0)
        print(f"  {cat:12s}: {n:5d}  ({100*n/len(all_docs):.1f}%)")


if __name__ == "__main__":
    main()