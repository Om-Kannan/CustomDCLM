"""
evaluate.py — Evaluate finetuned student models against the Qwen judge

Computes per-dimension per-category Spearman correlation and MSE between
student model predictions and judge scores on the held-out test set.

Produces:
  - results/spearman_heatmap_{slug}.png
  - results/mse_heatmap_{slug}.png
  - results/score_distributions_{slug}.png
  - results/interdim_correlation_{slug}.png
  - results/qualitative_worst_{slug}.jsonl
  - results/summary.json                     — all metrics, accumulates across runs
  - results/baseline_vs_finetuned.txt        — 3x2 table: base vs finetuned per model

  Cross-model plots (auto when 2+ models in summary):
  - results/dimension_vs_size.png
  - results/latency_vs_correlation.png

Usage:
  # Baseline (no adapter)
  python evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --test data/test.jsonl

  # Finetuned
  python evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter checkpoints/Qwen2.5-0.5B-Instruct/final --test data/test.jsonl

  # Regenerate plots only
  python evaluate.py --plot-only

Dependencies:
  pip install transformers peft bitsandbytes accelerate scipy scikit-learn matplotlib seaborn tqdm
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import torch
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

def load_model(model_id: str, adapter_path: str = None):
    tag = f"with adapter {adapter_path}" if adapter_path else "base (no adapter)"
    print(f"Loading {model_id} — {tag}...")

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
        device_map={"": torch.cuda.current_device()},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    print("Model loaded.\n")
    return tokenizer, model


def build_prompt(doc: dict) -> list:
    text = (doc.get("text") or "").strip()
    if len(text) > 6000:
        text = text[:6000] + "\n[truncated]"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
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

    # Guard against model returning non-dict JSON
    if not isinstance(parsed, dict):
        return None

    if not all(k in parsed for k in DIMENSIONS):
        return None

    return {dim: int(parsed[dim]) for dim in DIMENSIONS}


def infer_one(tokenizer, model, doc: dict, max_new_tokens: int = 128) -> tuple[dict | None, float]:
    messages = build_prompt(doc)
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
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
    return parse_scores(raw), latency


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(judge_scores: list, student_scores: list, latencies: list) -> dict:
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
    matrix = np.array([[s[dim] for dim in DIMENSIONS] for s in student_scores])
    return np.corrcoef(matrix.T)


# ---------------------------------------------------------------------------
# Baseline vs finetuned table
# ---------------------------------------------------------------------------

def write_baseline_finetuned_table(summary: dict, path: Path) -> None:
    """
    3 x 2 table: rows = model sizes, cols = base / finetuned
    Cell value = aggregate Spearman (mean across all dimensions)
    Also includes per-dimension breakdown.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("Baseline vs Finetuned — Aggregate Spearman (mean across dimensions)")
    lines.append("=" * 70)

    # Find base/finetuned pairs
    base_keys      = [k for k in summary if k.endswith("-base")]
    finetuned_keys = [k for k in summary if k.endswith("-finetuned")]

    # Header
    lines.append(f"\n{'Model':<30} {'Base':>10} {'Finetuned':>12} {'Delta':>8}")
    lines.append("-" * 65)

    model_names = sorted({k.replace("-base", "").replace("-finetuned", "") for k in summary})
    for name in model_names:
        base_key = f"{name}-base"
        ft_key   = f"{name}-finetuned"
        base_sp  = np.mean([summary[base_key]["overall"][d]["spearman"] for d in DIMENSIONS]) if base_key in summary else None
        ft_sp    = np.mean([summary[ft_key]["overall"][d]["spearman"]   for d in DIMENSIONS]) if ft_key   in summary else None

        base_str = f"{base_sp:.4f}" if base_sp is not None else "—"
        ft_str   = f"{ft_sp:.4f}"   if ft_sp   is not None else "—"
        delta_str = f"{ft_sp - base_sp:+.4f}" if (base_sp is not None and ft_sp is not None) else "—"
        lines.append(f"{name:<30} {base_str:>10} {ft_str:>12} {delta_str:>8}")

    # Per-dimension breakdown
    lines.append("\n" + "=" * 70)
    lines.append("Per-dimension Spearman — Base vs Finetuned")
    lines.append("=" * 70)

    for name in model_names:
        base_key = f"{name}-base"
        ft_key   = f"{name}-finetuned"
        lines.append(f"\n{name}")
        lines.append(f"  {'Dimension':<32} {'Base':>8} {'Finetuned':>12} {'Delta':>8}")
        lines.append("  " + "-" * 60)
        for dim in DIMENSIONS:
            b = summary[base_key]["overall"][dim]["spearman"] if base_key in summary else None
            f = summary[ft_key]["overall"][dim]["spearman"]   if ft_key   in summary else None
            b_str = f"{b:.4f}" if b is not None else "—"
            f_str = f"{f:.4f}" if f is not None else "—"
            d_str = f"{f - b:+.4f}" if (b is not None and f is not None) else "—"
            lines.append(f"  {dim:<32} {b_str:>8} {f_str:>12} {d_str:>8}")

    text = "\n".join(lines) + "\n"
    path.write_text(text)
    print(f"  Saved {path}")
    print("\n" + text)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmap(matrix, row_labels, col_labels, title, path,
                 vmin=0, vmax=1, fmt=".2f", cmap="YlGnBu"):
    fig, ax = plt.subplots(figsize=(len(col_labels) * 1.4, len(row_labels) * 0.8 + 1))
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=col_labels, yticklabels=row_labels,
                vmin=vmin, vmax=vmax, ax=ax, linewidths=0.5)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Category")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_score_distributions(judge_scores, student_scores, model_slug, out_dir):
    fig, axes = plt.subplots(1, len(DIMENSIONS), figsize=(4 * len(DIMENSIONS), 4))
    for ax, dim in zip(axes, DIMENSIONS):
        j = [s[dim] for s in judge_scores]
        s = [s[dim] for s in student_scores]
        bins = range(1, 12)
        ax.hist(j, bins=bins, alpha=0.6, label="Judge",   color="steelblue", align="left")
        ax.hist(s, bins=bins, alpha=0.6, label="Student", color="coral",     align="left")
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


def plot_interdim_correlation(corr_matrix, model_slug, out_dir):
    plot_heatmap(
        corr_matrix,
        row_labels=[d.replace("_", "\n") for d in DIMENSIONS],
        col_labels=[d.replace("_", "\n") for d in DIMENSIONS],
        title=f"Inter-dimension correlation — {model_slug}",
        path=out_dir / f"interdim_correlation_{model_slug}.png",
        vmin=-1, vmax=1, cmap="coolwarm",
    )


def plot_cross_model(summary: dict, out_dir: Path) -> None:
    # Only use finetuned entries for cross-model plots
    finetuned = {k: v for k, v in summary.items() if k.endswith("-finetuned")}
    if len(finetuned) < 2:
        print("  Need at least 2 finetuned models for cross-model plots, skipping.")
        return

    slugs = list(finetuned.keys())

    # Spearman per dimension across model sizes
    fig, ax = plt.subplots(figsize=(8, 5))
    for dim in DIMENSIONS:
        y = [finetuned[slug]["overall"][dim]["spearman"] for slug in slugs]
        ax.plot(slugs, y, marker="o", label=dim.replace("_", " "))
    ax.set_xlabel("Model")
    ax.set_ylabel("Spearman correlation")
    ax.set_title("Spearman per dimension across model sizes (finetuned)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = out_dir / "dimension_vs_size.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

    # Latency vs aggregate Spearman
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = ["o", "s", "^", "D", "v"]
    for i, slug in enumerate(slugs):
        agg = np.mean([finetuned[slug]["overall"][d]["spearman"] for d in DIMENSIONS])
        lat = finetuned[slug]["overall"]["latency"]["mean_seconds"]
        ax.scatter(lat, agg, s=120, marker=markers[i % len(markers)], label=slug, zorder=5)
        ax.annotate(slug.replace("-finetuned", ""), (lat, agg),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Mean inference latency (seconds / doc)")
    ax.set_ylabel("Aggregate Spearman correlation")
    ax.set_title("Accuracy–latency tradeoff (finetuned models)")
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

def find_worst_cases(docs, judge_scores, student_scores, n=5):
    cases = []
    for doc, j, s in zip(docs, judge_scores, student_scores):
        avg_err = np.mean([abs(j[dim] - s[dim]) for dim in DIMENSIONS])
        cases.append({
            "id":             doc["id"],
            "category":       doc.get("category", ""),
            "text_preview":   doc.get("text", "")[:300],
            "judge_scores":   j,
            "student_scores": s,
            "avg_abs_error":  round(float(avg_err), 3),
        })
    cases.sort(key=lambda x: x["avg_abs_error"], reverse=True)
    return cases[:n]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_model(args, out_dir: Path) -> dict:
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

    # Slug: ModelName-base or ModelName-finetuned
    model_slug = args.model.split("/")[-1]
    run_slug   = f"{model_slug}-{'finetuned' if args.adapter else 'base'}"

    overall  = compute_metrics(judge_all, student_all, latencies_all)
    per_cat  = {}
    categories = sorted(by_cat_judge.keys())
    for cat in categories:
        per_cat[cat] = compute_metrics(
            by_cat_judge[cat], by_cat_student[cat],
            [0.0] * len(by_cat_judge[cat]),
        )

    # Heatmaps
    spearman_matrix = np.array([[per_cat[c][d]["spearman"] for d in DIMENSIONS] for c in categories])
    plot_heatmap(spearman_matrix, categories,
                 [d.replace("_", "\n") for d in DIMENSIONS],
                 f"Spearman — {run_slug}",
                 out_dir / f"spearman_heatmap_{run_slug}.png")

    mse_matrix = np.array([[per_cat[c][d]["mse"] for d in DIMENSIONS] for c in categories])
    plot_heatmap(mse_matrix, categories,
                 [d.replace("_", "\n") for d in DIMENSIONS],
                 f"MSE — {run_slug}",
                 out_dir / f"mse_heatmap_{run_slug}.png",
                 vmin=0, vmax=9, fmt=".2f", cmap="YlOrRd")

    plot_score_distributions(judge_all, student_all, run_slug, out_dir)

    interdim = compute_interdim_correlation(student_all)
    plot_interdim_correlation(interdim, run_slug, out_dir)

    worst = find_worst_cases(test_docs[:len(judge_all)], judge_all, student_all)
    worst_path = out_dir / f"qualitative_worst_{run_slug}.jsonl"
    with worst_path.open("w") as f:
        for case in worst:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"  Saved {worst_path}")

    return {
        "model":               args.model,
        "adapter":             args.adapter,
        "n_docs":              len(judge_all),
        "errors":              errors,
        "overall":             overall,
        "per_category":        per_cat,
        "interdim_correlation": interdim.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter",   default=None,
                        help="Path to LoRA adapter. Omit for baseline run.")
    parser.add_argument("--test",      default="data/test.jsonl")
    parser.add_argument("--out-dir",   default="results")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--summary",   default="results/summary.json")
    return parser.parse_args()


def main():
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
        plot_cross_model(summary, out_dir)
        write_baseline_finetuned_table(summary, out_dir / "baseline_vs_finetuned.txt")
        return

    result = evaluate_model(args, out_dir)

    model_slug = args.model.split("/")[-1]
    run_slug   = f"{model_slug}-{'finetuned' if args.adapter else 'base'}"

    summary = {}
    if summary_path.exists():
        with summary_path.open() as f:
            summary = json.load(f)
    summary[run_slug] = result
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary to {summary_path}")

    # Write table and cross-model plots whenever we have enough data
    write_baseline_finetuned_table(summary, out_dir / "baseline_vs_finetuned.txt")
    if len([k for k in summary if k.endswith("-finetuned")]) > 1:
        plot_cross_model(summary, out_dir)


if __name__ == "__main__":
    main()