"""Download, split, validate, and convert GSM8K for verl.

Run on the rental server, with its data-disk paths, for example::

    PYTHONPATH=src python -m math_grpo.data.gsm8k \
        --revision <dataset-commit> \
        --cache-dir <data-disk>/huggingface \
        --output-dir <data-disk>/math-verl-grpo/data/processed/gsm8k
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from math_grpo.answer_parser import normalize_number

DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
EXPECTED_TRAIN_SIZE = 7_473
EXPECTED_TEST_SIZE = 1_319
DEV_SIZE = 512
SPLIT_SEED = 20_260_910

PROMPT_SUFFIX = (
    "Solve the problem step by step. End your response with the final numeric "
    'answer in the form "#### <number>".'
)

_FINAL_ANSWER = re.compile(
    r"####\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*$"
)


def extract_ground_truth(answer: str) -> str:
    """Extract the terminal ``#### <number>`` answer from a GSM8K solution."""

    match = _FINAL_ANSWER.search(answer)
    if match is None:
        raise ValueError("GSM8K answer does not end with '#### <number>'")
    return normalize_number(match.group(1))


def normalize_question(question: str) -> str:
    """Normalize a question only for exact duplicate detection."""

    text = unicodedata.normalize("NFKC", question)
    return " ".join(text.casefold().split())


def choose_dev_indices(
    train_size: int,
    dev_size: int = DEV_SIZE,
    seed: int = SPLIT_SEED,
) -> set[int]:
    """Choose a deterministic dev subset from stable source indices."""

    if not 0 < dev_size < train_size:
        raise ValueError("dev_size must be between zero and train_size")

    def rank(index: int) -> bytes:
        return hashlib.sha256(f"{seed}:{index}".encode()).digest()

    ranked_indices = sorted(range(train_size), key=rank)
    return set(ranked_indices[:dev_size])


def make_row(
    example: dict[str, str],
    *,
    split: str,
    source_split: str,
    source_index: int,
    revision: str,
) -> dict[str, Any]:
    """Convert one source example to the small schema expected by verl."""

    question = example["question"]
    answer = example["answer"]
    return {
        "data_source": DATASET_ID,
        "prompt": [
            {
                "role": "user",
                "content": f"{question}\n\n{PROMPT_SUFFIX}",
            }
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": extract_ground_truth(answer),
        },
        "extra_info": {
            "split": split,
            "source_split": source_split,
            "source_index": source_index,
            "dataset_revision": revision,
            "question": question,
            "answer": answer,
        },
    }


def assert_no_question_overlap(rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    """Fail if any normalized question appears in more than one split."""

    questions: dict[str, set[str]] = {}
    for split, rows in rows_by_split.items():
        questions[split] = {
            normalize_question(row["extra_info"]["question"]) for row in rows
        }

    names = list(questions)
    for position, left in enumerate(names):
        for right in names[position + 1 :]:
            overlap = questions[left] & questions[right]
            if overlap:
                raise ValueError(
                    f"Question leakage between {left} and {right}: "
                    f"{len(overlap)} normalized duplicate(s)"
                )


def build_rows(dataset: Any, revision: str) -> dict[str, list[dict[str, Any]]]:
    """Create the fixed train/dev/test rows and run integrity checks."""

    if set(dataset) != {"train", "test"}:
        raise ValueError(f"Expected train/test splits, found: {sorted(dataset)}")
    if len(dataset["train"]) != EXPECTED_TRAIN_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_SIZE} train rows, found {len(dataset['train'])}"
        )
    if len(dataset["test"]) != EXPECTED_TEST_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SIZE} test rows, found {len(dataset['test'])}"
        )

    dev_indices = choose_dev_indices(len(dataset["train"]))
    rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "dev": [],
        "test": [],
    }

    for index, example in enumerate(dataset["train"]):
        split = "dev" if index in dev_indices else "train"
        rows[split].append(
            make_row(
                example,
                split=split,
                source_split="train",
                source_index=index,
                revision=revision,
            )
        )

    for index, example in enumerate(dataset["test"]):
        rows["test"].append(
            make_row(
                example,
                split="test",
                source_split="test",
                source_index=index,
                revision=revision,
            )
        )

    expected_counts = {
        "train": EXPECTED_TRAIN_SIZE - DEV_SIZE,
        "dev": DEV_SIZE,
        "test": EXPECTED_TEST_SIZE,
    }
    actual_counts = {name: len(split_rows) for name, split_rows in rows.items()}
    if actual_counts != expected_counts:
        raise ValueError(f"Unexpected split sizes: {actual_counts}")

    assert_no_question_overlap(rows)
    return rows


def write_parquet(rows_by_split: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    """Write one verl-compatible Parquet file per split."""

    from datasets import Dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        Dataset.from_list(rows).to_parquet(output_dir / f"{split}.parquet")


def write_metadata(
    output_dir: Path,
    revision: str,
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> None:
    """Save the few values needed to reproduce the generated files."""

    metadata = {
        "dataset": DATASET_ID,
        "config": DATASET_CONFIG,
        "revision": revision,
        "split_seed": SPLIT_SEED,
        "dev_size": DEV_SIZE,
        "counts": {name: len(rows) for name, rows in rows_by_split.items()},
    }
    path = output_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision",
        required=True,
        help="Exact Hugging Face dataset commit to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory on the server data disk.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Hugging Face cache directory on the server data disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        revision=args.revision,
        cache_dir=str(args.cache_dir),
    )
    rows = build_rows(dataset, args.revision)
    write_parquet(rows, args.output_dir)
    write_metadata(args.output_dir, args.revision, rows)

    counts = {name: len(split_rows) for name, split_rows in rows.items()}
    print(f"Prepared GSM8K at {args.output_dir}: {counts}")


if __name__ == "__main__":
    main()
