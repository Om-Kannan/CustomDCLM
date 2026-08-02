"""
finetune.py — QLoRA finetuning of student models on LLM judge scores

Trains a small Qwen2.5 model to replicate the scoring behavior of the
Qwen2.5-14B judge. Run once per student model.

Usage:
  # Finetune each student
  python finetune.py --model Qwen/Qwen2.5-0.5B-Instruct --input data/scored_mid.jsonl
  python finetune.py --model Qwen/Qwen2.5-1.5B-Instruct --input data/scored_mid.jsonl
  python finetune.py --model Qwen/Qwen2.5-3B-Instruct   --input data/scored_mid.jsonl

  # Output checkpoints saved to:
  #   checkpoints/Qwen2.5-0.5B-Instruct/
  #   checkpoints/Qwen2.5-1.5B-Instruct/
  #   checkpoints/Qwen2.5-3B-Instruct/

Dependencies:
  pip install transformers peft bitsandbytes accelerate datasets tqdm
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
import torch


# ---------------------------------------------------------------------------
# Prompt formatting
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


def format_target(scores: dict) -> str:
    """Serialize scores dict to the exact JSON string the model should output."""
    return json.dumps({
        "coherence":                   scores["coherence"],
        "information_density":         scores["information_density"],
        "lexical_diversity":           scores["lexical_diversity"],
        "boilerplate":                 scores["boilerplate"],
        "overall_pretraining_utility": scores["overall_pretraining_utility"],
        "justification":               scores["justification"],
    }, ensure_ascii=False)


def build_messages(doc: dict) -> tuple[list, str]:
    """Return (messages_for_chat_template, target_string)."""
    text = (doc.get("text") or "").strip()
    if len(text) > 6000:
        text = text[:6000] + "\n[truncated]"

    user_content = (
        f"Category: {doc.get('category', 'unknown')}\n"
        f"Source: {doc.get('source', 'unknown')}\n\n"
        f"Document:\n{text}"
    )

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
    ]
    target = format_target(doc["scores"])
    return messages, target


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

def load_scored(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stratified_split(docs: list[dict], test_size: float, seed: int) -> tuple[list, list]:
    """80/20 split stratified by category."""
    random.seed(seed)
    by_cat = defaultdict(list)
    for doc in docs:
        by_cat[doc.get("category", "unknown")].append(doc)

    train, test = [], []
    for cat, items in by_cat.items():
        random.shuffle(items)
        n_test = max(1, int(len(items) * test_size))
        test.extend(items[:n_test])
        train.extend(items[n_test:])

    random.shuffle(train)
    random.shuffle(test)
    return train, test


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_example(example: dict, tokenizer, max_length: int = 1024) -> dict:
    """
    Tokenize one (prompt, completion) pair.
    Labels are -100 for the prompt tokens (not trained on) and the
    actual token ids for the completion tokens.
    """
    messages = example["messages"]
    target   = example["target"]

    # Encode prompt using chat template (no generation prompt yet)
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors=None,
    )

    # Encode target (the JSON scores)
    target_ids = tokenizer.encode(
        target + tokenizer.eos_token,
        add_special_tokens=False,
    )

    input_ids = prompt_ids + target_ids
    labels    = [-100] * len(prompt_ids) + target_ids

    # Truncate if over max_length
    input_ids = input_ids[:max_length]
    labels    = labels[:max_length]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_student_model(model_id: str):
    print(f"Loading {model_id} in 4-bit for QLoRA...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": torch.cuda.current_device()},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        # Target the attention and MLP projection layers
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return tokenizer, model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA finetune a student model on judge scores")
    parser.add_argument("--model",      default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--input",      default="data/scored_mid.jsonl")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--test-size",  type=float, default=0.2)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--max-length", type=int,   default=1024)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--save-train-test", action="store_true",
                        help="Save train/test split to data/ for reuse across models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: scored corpus not found at {input_path}", file=sys.stderr)
        sys.exit(1)

    # Slug the model name for output directory
    model_slug = args.model.split("/")[-1]
    output_dir = Path(args.output_dir) / model_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Load and split data
    # ---------------------------------------------------------------------------
    print(f"Loading scored corpus from {input_path}...")
    docs = load_scored(input_path)
    print(f"  {len(docs)} documents loaded")

    # Reuse existing split if available so all models train/eval on same data
    train_path = Path("data/train.jsonl")
    test_path  = Path("data/test.jsonl")

    if train_path.exists() and test_path.exists():
        print("  Reusing existing train/test split from data/")
        with train_path.open() as f:
            train_docs = [json.loads(l) for l in f if l.strip()]
        with test_path.open() as f:
            test_docs = [json.loads(l) for l in f if l.strip()]
    else:
        print(f"  Splitting {len(docs)} docs ({1-args.test_size:.0%} train / {args.test_size:.0%} test)...")
        train_docs, test_docs = stratified_split(docs, args.test_size, args.seed)
        if args.save_train_test:
            Path("data").mkdir(exist_ok=True)
            with train_path.open("w") as f:
                for d in train_docs: f.write(json.dumps(d) + "\n")
            with test_path.open("w") as f:
                for d in test_docs: f.write(json.dumps(d) + "\n")
            print(f"  Saved split to data/train.jsonl and data/test.jsonl")

    print(f"  Train: {len(train_docs)} | Test: {len(test_docs)}")

    # ---------------------------------------------------------------------------
    # Load model
    # ---------------------------------------------------------------------------
    tokenizer, model = load_student_model(args.model)

    # ---------------------------------------------------------------------------
    # Tokenize
    # ---------------------------------------------------------------------------
    print("Tokenizing...")

    def make_hf_dataset(doc_list: list[dict]) -> Dataset:
        raw = []
        for doc in tqdm(doc_list, desc="  formatting"):
            messages, target = build_messages(doc)
            raw.append({"messages": messages, "target": target, "category": doc.get("category", "")})

        dataset = Dataset.from_list(raw)
        dataset = dataset.map(
            lambda ex: tokenize_example(ex, tokenizer, args.max_length),
            remove_columns=["messages", "target", "category"],
            desc="  tokenizing",
        )
        return dataset

    train_dataset = make_hf_dataset(train_docs)
    eval_dataset  = make_hf_dataset(test_docs)

    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    print(f"\nFinetuning {args.model}...")
    trainer.train()

    # Save final adapter weights
    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nSaved to {final_path}")


if __name__ == "__main__":
    main()