import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from math_grpo.evaluate import (  # noqa: E402
    evaluate_response,
    get_dataset_revision,
    summarize,
    write_results,
)


def test_evaluate_response_separates_correctness_and_format():
    strict = evaluate_response("Work.\n#### 42", "42", response_tokens=8)
    flexible = evaluate_response("The final answer is 42.", "42")

    assert strict["strict_correct"]
    assert strict["flexible_correct"]
    assert strict["format_valid"]
    assert strict["response_tokens"] == 8

    assert not flexible["strict_correct"]
    assert flexible["flexible_correct"]
    assert not flexible["format_valid"]


def test_evaluate_response_marks_length_finish_as_truncated():
    record = evaluate_response("unfinished 40 + 2", "42", finish_reason="length")

    assert record["truncated"]
    assert record["parse_failed"]


def test_summarize_computes_expected_rates():
    records = [
        evaluate_response("#### 42", "42", response_tokens=4),
        evaluate_response("The final answer is 42.", "42", response_tokens=6),
        evaluate_response("#### 40", "42", response_tokens=4),
        evaluate_response("No answer.", "42", response_tokens=2),
    ]

    metrics = summarize(records)

    assert metrics["num_examples"] == 4
    assert metrics["strict_accuracy"] == 0.25
    assert metrics["flexible_accuracy"] == 0.5
    assert metrics["format_compliance"] == 0.5
    assert metrics["parse_failure_rate"] == 0.25
    assert metrics["mean_response_tokens"] == 4.0


def test_summarize_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        summarize([])


def test_dataset_revision_must_be_unique():
    rows = [
        {"extra_info": {"dataset_revision": "abc"}},
        {"extra_info": {"dataset_revision": "abc"}},
    ]
    assert get_dataset_revision(rows) == "abc"

    rows[1]["extra_info"]["dataset_revision"] = "def"
    with pytest.raises(ValueError, match="exactly one"):
        get_dataset_revision(rows)


def test_write_results_creates_predictions_and_summary(tmp_path):
    records = [evaluate_response("#### 42", "42", response_tokens=4)]
    output_dir = tmp_path / "base-dev"

    write_results(output_dir, records, {"model": "test-model"})

    prediction = json.loads(
        (output_dir / "predictions.jsonl").read_text().strip()
    )
    summary = json.loads((output_dir / "summary.json").read_text())
    assert prediction["strict_correct"]
    assert summary["metrics"]["strict_accuracy"] == 1.0
    assert summary["generation"]["model"] == "test-model"
