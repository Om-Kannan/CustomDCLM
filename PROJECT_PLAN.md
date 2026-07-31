# LLM Judge Distillation — Project Plan

## Research Question

How well does model distillation preserve LLM-like document quality scoring, and where does it break down — by model size and by document category?

## Hypothesis

Distillation quality degrades faster on out-of-distribution document types (arXiv, AI-generated) than on common web text (Wikipedia, Reddit), and this degradation is disproportionate relative to model size reduction.

---

## Pipeline Overview

```
curate.py → score.py → finetune.py → evaluate.py
```

---

## Step 1 — Data Curation (`curate.py`)

Assemble a corpus of 6,000–8,000 documents from the following sources:

| Category       | Proportion | Source                          |
|----------------|------------|---------------------------------|
| Wikipedia      | 30%        | wikimedia/wikipedia             |
| Reddit         | 20%        | sentence-transformers/reddit    |
| Blogs/web      | 15%        | allenai/c4                      |
| News           | 10%        | allenai/c4 (realnewslike)       |
| arXiv          | 10%        | scientific_papers/arxiv         |
| AI encyclopedia| 7.5%       | HuggingFaceTB/cosmopedia        |
| AI SEO content | 7.5%       | HuggingFaceTB/cosmopedia        |

Output: `data/corpus.jsonl` — one document per line with fields `id`, `category`, `source`, `text`.

---

## Step 2 — LLM Scoring (`score.py`)

Score every document with **Qwen 32B** (open source, Apache 2.0 — license-safe for training data).

Each document receives a structured JSON score:

```json
{
  "coherence": 8,
  "information_density": 7,
  "lexical_diversity": 6,
  "boilerplate": 2,
  "overall_pretraining_utility": 7,
  "justification": "..."
}
```

Output: `data/scored.jsonl` — corpus.jsonl with a `scores` field appended to each document.

80/20 train/test split after scoring, stratified by category.

---

## Step 3 — Finetuning (`finetune.py`)

Finetune three student models on the judge's scored outputs using **QLoRA**:

| Model       | Size  |
|-------------|-------|
| Qwen3-0.6B  | 0.6B  |
| Qwen3-1.7B  | 1.7B  |
| Qwen3-4B    | 4B    |

All from the same family as the judge (Qwen) — controls for tokenizer and pretraining distribution differences.

Task: given a document, predict the same structured JSON rubric scores.

---

## Step 4 — Evaluation (`evaluate.py`)

Primary metric: **Spearman correlation per rubric dimension per category** between student model predictions and judge scores on the held-out test set.

Key analyses:
- Accuracy vs model size curves (one curve per rubric dimension)
- Per-category breakdown — where does each student model diverge most from the judge
- Latency and inference cost vs accuracy tradeoff across model sizes

---

## Infrastructure

- **Compute**: Kaggle free tier (T4 GPUs, 30hr/week)
- **Repo structure**: Python scripts in GitHub, thin Kaggle notebook as launcher (`!git pull && !python src/score.py`)
- **Checkpointing**: save to Kaggle persistent storage or Google Drive after each script

---

## Deliverable

Accuracy vs model size curves broken down by document category, with a clear story about where and why distillation degrades — targeting an arXiv preprint or workshop paper at EMNLP/ACL.
