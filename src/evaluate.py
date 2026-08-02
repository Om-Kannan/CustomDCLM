"""
evaluate.py — Evaluate finetuned student models against the Qwen judge

Computes per-dimension per-category Spearman correlation and MSE between
student model predictions and judge scores on the held-out test set.

Produces:
  - results/spearman_heatmap_{model}.png     — category × dimension Spearman matrix
  - results/mse_heatmap_{model}.png          — category × dimension MSE matrix
  - results/score_distributions_{model}.png  — judge vs student score histograms
  - results/dimension_vs_size.png            — Spearman per dimension across model sizes
  - results/latency_vs_correlation.png       — accuracy/latency tradeoff scatter
  - results/interdim_correlation_{model}.png — inter-dimension correlation matrix
  - results/qualitative_examples.jsonl       — highest-MSE cases per category
  - results/summary.json                     — all metrics in one file

Usage:
  # Evaluate one model
  python evaluate.py \\
    --model     Qwen/Qwen2.5-0.5B-Instruct \\
    --adapter   checkpoints/Qwen2.5-0.5B-Instruct/final \\
    --test      data/test.jsonl

  # After running all three, generate cross-model plots
  python evaluate.py --plot-only --summary results/summary.json

Dependencies:
  pip install transformers peft bitsandbytes accelerate scipy scikit-learn
              matplotlib seaborn tqdm
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # no display needed on Kaggle
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIMENSIONS = [
    "coherence",
    "information_density",
    "lexical_diversity",
    "boilerplate",
    "overall_pretraining_utility",
]

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


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model(model_id: str, adapter_path: str):
    print(f"Loading {model_id} with adapter from {adapter_path}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    print("Model loaded.\n")
    return tokenizer, model


def build_prompt(doc: dict) -> list:
    text = (doc.get("text") or "").strip()
    if len(text) > 6000:
        text = text[:6000] + "\n[truncated]"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"Category: {doc.get('category', 'unknown')}\n"
            f"Source: {doc.get('source', 'unknown')}\n\n"
            f"Document:\n{text}"
        )},
    ]


def parse_scores(raw: str) -> dict | None:
    text = re.sub(r"```json|```", "", raw).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if not all(k in parsed for k in DIMENSIONS):
        return None

    return {dim: int(parsed[dim]) for dim in DIMENSIONS}


def infer_one(tokenizer, model, doc: dict, max_new_tokens: int = 128) -> tuple[dict | None, float]:
    messages = build_prompt(doc)
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - t0

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    scores = parse_scores(raw)
    return scores, latency


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(judge_scores: list, student_scores: list, latencies: list) -> dict:
    """
    Given parallel lists of judge and student score dicts and per-doc latencies,
    return per-dimension Spearman, MSE, and latency stats.
    """
    results = {}
    for dim in DIMENSIONS:
        j = [s[dim] for s in judge_scores]
        s = [s[dim] for s in student_scores]
        corr, pval = spearmanr(j, s)
        mse = mean_squared_error(j, s)
        results[dim] = {
            "spearman": round(float(corr), 4),
            "spearman_pval": round(float(pval), 4),
            "mse": round(float(mse), 4),
        }

    results["latency"] = {
        "mean_seconds": round(float(np.mean(latencies)), 3),
        "median_seconds": round(float(np.median(latencies)), 3),
        "total_seconds": round(float(np.sum(latencies)), 1),
    }
    return results


def compute_interdim_correlation(student_scores: list) -> np.ndarray:
    """Correlation matrix of student scores across dimensions."""
    matrix = np.array([[s[dim] for dim in DIMENSIONS] for s in student_scores])
    corr = np.corrcoef(matrix.T)
    return corr


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmap(matrix: np.ndarray, row_labels: list, col_labels: list,
                 title: str, path: Path, vmin: float = 0, vmax: float = 1,
                 fmt: str = ".2f", cmap: str = "YlGnBu") -> None:
    fig, ax = plt.subplots(figsize=(len(col_labels) * 1.4, len(row_labels) * 0.8 + 1))
    sns.heatmap(
        matrix, annot=True, fmt=fmt, cmap=cmap,
        xticklabels=col_labels, yticklabels=row_labels,
        vmin=vmin, vmax=vmax, ax=ax, linewidths=0.5,
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Category")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_score_distributions(judge_scores: list, student_scores: list,
                             model_slug: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(DIMENSIONS), figsize=(4 * len(DIMENSIONS), 4))
    for ax, dim in zip(axes, DIMENSIONS):
        j = [s[dim] for s in judge_scores]
        s = [s[dim] for s in student_scores]
        bins = range(1, 12)
        ax.hist(j, bins=bins, alpha=0.6, label="Judge", color="steelblue", align="left")
        ax.hist(s, bins=bins, alpha=0.6, label="Student", color="coral",   align="left")
        ax.set_title(dim.replace("_", "\n"), fontsize=9)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    fig.suptitle(f"Score distributions — {model_slug}", y=1.02)
    plt.tight_layout()
    path = out_dir / f"score_distributions_{model_slug}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_interdim_correlation(corr_matrix: np.ndarray, model_slug: str, out_dir: Path) -> None:
    path = out_dir / f"interdim_correlation_{model_slug}.png"
    plot_heatmap(
        corr_matrix,
        row_labels=[d.replace("_", "\n") for d in DIMENSIONS],
        col_labels=[d.replace("_", "\n") for d in DIMENSIONS],
        title=f"Inter-dimension correlation — {model_slug}",
        path=path,
        vmin=-1, vmax=1, cmap="coolwarm",
    )


def plot_cross_model(summary: dict, out_dir: Path) -> None:
    """
    Two cross-model plots:
    1. Spearman per dimension across model sizes (line plot)
    2. Latency vs aggregate Spearman (scatter, model size as legend)
    """
    model_slugs = list(summary.keys())
    if len(model_slugs) < 2:
        print("  Need at least 2 models for cross-model plots, skipping.")
        return

    # --- 1. Dimension vs model size ---
    # X: model index (proxy for size), Y: Spearman, one line per dimension
    fig, ax = plt.subplots(figsize=(8, 5))
    for dim in DIMENSIONS:
        y = [summary[slug]["overall"][dim]["spearman"] for slug in model_slugs]
        ax.plot(model_slugs, y, marker="o", label=dim.replace("_", " "))
    ax.set_xlabel("Model")
    ax.set_ylabel("Spearman correlation")
    ax.set_title("Spearman correlation per dimension across model sizes")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = out_dir / "dimension_vs_size.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

    # --- 2. Latency vs aggregate Spearman ---
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = ["o", "s", "^", "D", "v"]
    for i, slug in enumerate(model_slugs):
        agg_spearman = np.mean([
            summary[slug]["overall"][dim]["spearman"] for dim in DIMENSIONS
        ])
        latency = summary[slug]["overall"]["latency"]["mean_seconds"]
        ax.scatter(latency, agg_spearman, s=120, marker=markers[i % len(markers)], label=slug, zorder=5)
        ax.annotate(slug, (latency, agg_spearman), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Mean inference latency (seconds / doc)")
    ax.set_ylabel("Aggregate Spearman correlation")
    ax.set_title("Accuracy–latency tradeoff")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = out_dir / "latency_vs_correlation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Qualitative examples
# ---------------------------------------------------------------------------

def find_worst_cases(docs: list, judge_scores: list, student_scores: list,
                     n: int = 5) -> list:
    """Return the n docs with the highest average absolute score error."""
    cases = []
    for doc, j, s in zip(docs, judge_scores, student_scores):
        avg_err = np.mean([abs(j[dim] - s[dim]) for dim in DIMENSIONS])
        cases.append({
            "id":            doc["id"],
            "category":      doc.get("category", ""),
            "text_preview":  doc.get("text", "")[:300],
            "judge_scores":  j,
            "student_scores": s,
            "avg_abs_error": round(float(avg_err), 3),
        })
    cases.sort(key=lambda x: x["avg_abs_error"], reverse=True)
    return cases[:n]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_model(args, out_dir: Path) -> dict:
    # Load test set
    test_path = Path(args.test)
    with test_path.open(encoding="utf-8") as f:
        test_docs = [json.loads(l) for l in f if l.strip()]
    print(f"Test set: {len(test_docs)} documents\n")

    tokenizer, model = load_model(args.model, args.adapter)

    judge_all, student_all, latencies_all = [], [], []
    by_cat_judge   = defaultdict(list)
    by_cat_student = defaultdict(list)
    errors = 0

    for doc in tqdm(test_docs, desc="Inferring", unit="doc"):
        judge = doc.get("scores")
        if not judge or not all(k in judge for k in DIMENSIONS):
            errors += 1
            continue

        student, latency = infer_one(tokenizer, model, doc)
        if student is None:
            errors += 1
            continue

        judge_all.append(judge)
        student_all.append(student)
        latencies_all.append(latency)
        by_cat_judge[doc["category"]].append(judge)
        by_cat_student[doc["category"]].append(student)

    print(f"\nScored {len(judge_all)} docs, {errors} errors\n")

    model_slug = args.model.split("/")[-1]

    # Overall metrics
    overall = compute_metrics(judge_all, student_all, latencies_all)

    # Per-category metrics
    per_cat = {}
    categories = sorted(by_cat_judge.keys())
    for cat in categories:
        per_cat[cat] = compute_metrics(
            by_cat_judge[cat], by_cat_student[cat],
            [0.0] * len(by_cat_judge[cat]),   # latency not meaningful per-cat
        )

    # Heatmaps
    # Spearman
    spearman_matrix = np.array([
        [per_cat[cat][dim]["spearman"] for dim in DIMENSIONS]
        for cat in categories
    ])
    plot_heatmap(
        spearman_matrix, categories,
        [d.replace("_", "\n") for d in DIMENSIONS],
        f"Spearman correlation — {model_slug}",
        out_dir / f"spearman_heatmap_{model_slug}.png",
        vmin=0, vmax=1,
    )

    # MSE
    mse_matrix = np.array([
        [per_cat[cat][dim]["mse"] for dim in DIMENSIONS]
        for cat in categories
    ])
    plot_heatmap(
        mse_matrix, categories,
        [d.replace("_", "\n") for d in DIMENSIONS],
        f"MSE — {model_slug}",
        out_dir / f"mse_heatmap_{model_slug}.png",
        vmin=0, vmax=9, fmt=".2f", cmap="YlOrRd",
    )

    # Score distributions
    plot_score_distributions(judge_all, student_all, model_slug, out_dir)

    # Inter-dimension correlation
    interdim = compute_interdim_correlation(student_all)
    plot_interdim_correlation(interdim, model_slug, out_dir)

    # Qualitative worst cases
    worst = find_worst_cases(test_docs[:len(judge_all)], judge_all, student_all)
    worst_path = out_dir / f"qualitative_worst_{model_slug}.jsonl"
    with worst_path.open("w") as f:
        for case in worst:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"  Saved {worst_path}")

    return {
        "model": args.model,
        "n_docs": len(judge_all),
        "errors": errors,
        "overall": overall,
        "per_category": per_cat,
        "interdim_correlation": interdim.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter",   default="checkpoints/Qwen2.5-0.5B-Instruct/final")
    parser.add_argument("--test",      default="data/test.jsonl")
    parser.add_argument("--out-dir",   default="results")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip inference, just regenerate cross-model plots from summary.json")
    parser.add_argument("--summary",   default="results/summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.summary)

    if args.plot_only:
        if not summary_path.exists():
            print(f"ERROR: {summary_path} not found", file=sys.stderr)
            sys.exit(1)
        with summary_path.open() as f:
            summary = json.load(f)
        print("Generating cross-model plots...")
        plot_cross_model(summary, out_dir)
        return

    # Run evaluation
    result = evaluate_model(args, out_dir)
    model_slug = args.model.split("/")[-1]

    # Load or create summary
    summary = {}
    if summary_path.exists():
        with summary_path.open() as f:
            summary = json.load(f)

    summary[model_slug] = result

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary to {summary_path}")

    # Regenerate cross-model plots if we have multiple models
    if len(summary) > 1:
        print("\nGenerating cross-model plots...")
        plot_cross_model(summary, out_dir)


if __name__ == "__main__":
    main()