import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split scored corpus into train/test by category")
    parser.add_argument("--input", type=str, default="data/scored.jsonl", help="Scored JSONL input")
    parser.add_argument("--train-output", type=str, default="data/train.jsonl", help="Train split output")
    parser.add_argument("--test-output", type=str, default="data/test.jsonl", help="Test split output")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction to reserve for test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_by_category(records: List[Dict[str, Any]], test_size: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    random.seed(seed)
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[record.get("category", "unknown")].append(record)

    train: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    for category, items in by_category.items():
        random.shuffle(items)
        cutoff = max(1, int(len(items) * test_size))
        test.extend(items[:cutoff])
        train.extend(items[cutoff:])

    random.shuffle(train)
    random.shuffle(test)
    return train, test


def write_records(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Scored corpus not found: {input_path}")

    records = load_records(input_path)
    train, test = split_by_category(records, args.test_size, args.seed)
    write_records(Path(args.train_output), train)
    write_records(Path(args.test_output), test)
    print(f"Wrote {len(train)} train and {len(test)} test records")


if __name__ == "__main__":
    main()
