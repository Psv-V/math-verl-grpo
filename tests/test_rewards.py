import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from math_grpo.rewards import (  # noqa: E402
    DATA_SOURCE,
    compute_score,
    compute_score_correctness_only,
    score_response,
)


@pytest.mark.parametrize(
    ("response", "score", "correctness", "format_score"),
    [
        ("Work.\n#### 42", 1.1, 1.0, 1.0),
        ("The final answer is 42.", 1.0, 1.0, 0.0),
        ("Work.\n#### 40", 0.1, 0.0, 1.0),
        ("I do not know.", 0.0, 0.0, 0.0),
    ],
)
def test_reward_combinations(response, score, correctness, format_score):
    result = score_response(response, "42")

    assert result.score == pytest.approx(score)
    assert result.correctness == correctness
    assert result.format == format_score


def test_reward_normalizes_model_and_ground_truth():
    result = score_response("#### 1,200.50", "1200.5")

    assert result.score == pytest.approx(1.1)
    assert result.parsed_answer == "1200.5"


def test_ambiguous_answer_is_not_guessed():
    result = score_response("It could be 40 or 42.", "42")

    assert result.score == 0.0
    assert result.parse_error == "ambiguous_answers"


def test_verl_entry_point_returns_numeric_metrics():
    result = compute_score(DATA_SOURCE, "#### 42", "42", extra_info={})

    assert result == {
        "score": 1.1,
        "correctness": 1.0,
        "format": 1.0,
        "parse_success": 1.0,
    }


def test_correctness_only_never_adds_format_reward():
    correct = compute_score_correctness_only(DATA_SOURCE, "#### 42", "42")
    incorrect = compute_score_correctness_only(DATA_SOURCE, "#### 40", "42")

    assert correct["score"] == 1.0
    assert incorrect["score"] == 0.0
    assert incorrect["format"] == 1.0


def test_unknown_data_source_fails_loudly():
    with pytest.raises(ValueError, match="Unsupported data_source"):
        compute_score("another/dataset", "#### 42", "42")

