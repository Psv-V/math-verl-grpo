"""Independent vLLM evaluation for GSM8K base and trained models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from math_grpo.answer_parser import normalize_number, parse_flexible, parse_strict


def evaluate_response(
    response: str,
    ground_truth: str,
    *,
    response_tokens: int = 0,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """Evaluate one response without calling the training reward function."""

    expected = normalize_number(ground_truth)
    strict = parse_strict(response)
    flexible = parse_flexible(response)

    return {
        "ground_truth": expected,
        "response": response,
        "strict_answer": strict.value,
        "flexible_answer": flexible.value,
        "strict_correct": strict.value == expected,
        "flexible_correct": flexible.value == expected,
        "format_valid": strict.success,
        "parse_failed": not flexible.success,
        "parse_source": flexible.source,
        "parse_error": flexible.error,
        "response_tokens": response_tokens,
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, float | int]:
    """Aggregate per-example records into the frozen evaluation metrics."""

    if not records:
        raise ValueError("Cannot summarize an empty evaluation")

    count = len(records)

    def rate(field: str) -> float:
        return sum(bool(record[field]) for record in records) / count

    return {
        "num_examples": count,
        "strict_accuracy": rate("strict_correct"),
        "flexible_accuracy": rate("flexible_correct"),
        "format_compliance": rate("format_valid"),
        "parse_failure_rate": rate("parse_failed"),
        "truncation_rate": rate("truncated"),
        "mean_response_tokens": sum(
            int(record["response_tokens"]) for record in records
        )
        / count,
    }


def load_rows(input_file: Path, limit: int | None) -> list[dict[str, Any]]:
    """Load one prepared Parquet split."""

    from datasets import load_dataset

    dataset = load_dataset(
        "parquet",
        data_files={"evaluation": str(input_file)},
        split="evaluation",
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        dataset = dataset.select(range(min(limit, len(dataset))))
    return list(dataset)


def get_dataset_revision(rows: list[dict[str, Any]]) -> str:
    """Require one recorded dataset revision across the evaluation split."""

    revisions = {
        row["extra_info"].get("dataset_revision")
        for row in rows
        if row["extra_info"].get("dataset_revision")
    }
    if len(revisions) != 1:
        raise ValueError(f"Expected exactly one dataset revision, found: {revisions}")
    return revisions.pop()


def run_vllm(
    rows: list[dict[str, Any]],
    *,
    model: str,
    revision: str,
    download_dir: Path,
    max_tokens: int,
    max_model_len: int,
    seed: int,
    gpu_memory_utilization: float,
) -> list[dict[str, Any]]:
    """Generate one deterministic response per row with offline vLLM."""

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        revision=revision,
        tokenizer_revision=revision,
        download_dir=str(download_dir),
        tensor_parallel_size=1,
        dtype="bfloat16",
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=False,
    )
    sampling = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
    )
    conversations = [row["prompt"] for row in rows]
    outputs = llm.chat(conversations, sampling_params=sampling, use_tqdm=True)

    if len(outputs) != len(rows):
        raise RuntimeError("vLLM returned a different number of outputs than inputs")

    records: list[dict[str, Any]] = []
    for row, output in zip(rows, outputs):
        if len(output.outputs) != 1:
            raise RuntimeError("Expected exactly one completion per example")

        completion = output.outputs[0]
        finish_reason = (
            str(completion.finish_reason)
            if completion.finish_reason is not None
            else None
        )
        record = evaluate_response(
            completion.text,
            row["reward_model"]["ground_truth"],
            response_tokens=len(completion.token_ids),
            finish_reason=finish_reason,
        )
        record.update(
            {
                "source_split": row["extra_info"]["source_split"],
                "source_index": row["extra_info"]["source_index"],
                "question": row["extra_info"]["question"],
            }
        )
        records.append(record)

    return records


def write_results(
    output_dir: Path,
    records: list[dict[str, Any]],
    run_config: dict[str, Any],
) -> None:
    """Write predictions and a self-contained summary to a new directory."""

    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    result = {"metrics": summarize(records), "generation": run_config}
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--download-dir", required=True, type=Path)
    parser.add_argument("--max-tokens", required=True, type=int)
    parser.add_argument("--max-model-len", required=True, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu-memory-utilization", default=0.85, type=float)
    parser.add_argument(
        "--limit",
        type=int,
        help="Use only the first N rows for a dev smoke run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_file, args.limit)
    records = run_vllm(
        rows,
        model=args.model,
        revision=args.revision,
        download_dir=args.download_dir,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    run_config = {
        "model": args.model,
        "revision": args.revision,
        "input_file": str(args.input_file),
        "dataset_revision": get_dataset_revision(rows),
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "temperature": 0.0,
        "top_p": 1.0,
        "num_completions": 1,
        "seed": args.seed,
        "limit": args.limit,
    }
    write_results(args.output_dir, records, run_config)
    print(json.dumps(summarize(records), indent=2))


if __name__ == "__main__":
    main()
