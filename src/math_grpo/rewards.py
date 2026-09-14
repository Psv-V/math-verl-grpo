"""Rule-based GSM8K rewards and verl-compatible entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from math_grpo.answer_parser import (
    ParseResult,
    normalize_number,
    parse_flexible,
    parse_strict,
)

DATA_SOURCE = "openai/gsm8k"
FORMAT_WEIGHT = 0.1


@dataclass(frozen=True)
class RewardResult:
    """Complete score details for one generated response."""

    score: float
    correctness: float
    format: float
    parsed_answer: str | None
    parse_source: str | None
    parse_error: str | None

    def to_verl_dict(self) -> dict[str, float]:
        """Return numeric fields that verl can aggregate and log."""

        return {
            "score": self.score,
            "correctness": self.correctness,
            "format": self.format,
            "parse_success": float(self.parsed_answer is not None),
        }


def score_response(
    solution_str: str,
    ground_truth: str,
    *,
    format_weight: float = FORMAT_WEIGHT,
) -> RewardResult:
    """Score one response without depending on verl internals."""

    if format_weight < 0:
        raise ValueError("format_weight must not be negative")

    expected = normalize_number(ground_truth)
    strict = parse_strict(solution_str)
    flexible = parse_flexible(solution_str)

    correctness = float(flexible.value == expected)
    format_score = float(strict.success)
    total = correctness + format_weight * format_score

    return RewardResult(
        score=total,
        correctness=correctness,
        format=format_score,
        parsed_answer=flexible.value,
        parse_source=flexible.source,
        parse_error=flexible.error,
    )


def _check_data_source(data_source: str) -> None:
    if data_source != DATA_SOURCE:
        raise ValueError(f"Unsupported data_source: {data_source!r}")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    """verl entry point for correctness plus 0.1 format reward."""

    del extra_info
    _check_data_source(data_source)
    return score_response(solution_str, ground_truth).to_verl_dict()


def compute_score_correctness_only(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    """verl entry point for the correctness-only ablation."""

    del extra_info
    _check_data_source(data_source)
    return score_response(
        solution_str,
        ground_truth,
        format_weight=0.0,
    ).to_verl_dict()

