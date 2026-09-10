import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from math_grpo.answer_parser import (  # noqa: E402
    normalize_number,
    parse_flexible,
    parse_strict,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1,234", "1234"), ("-2.500", "-2.5"), ("-0.00", "0")],
)
def test_normalize_number(raw, expected):
    assert normalize_number(raw) == expected


def test_strict_parser_accepts_one_terminal_answer():
    result = parse_strict("Compute 40 + 2.\n#### 42")

    assert result.success
    assert result.value == "42"
    assert result.source == "strict"


@pytest.mark.parametrize(
    ("response", "error"),
    [
        ("", "empty_response"),
        ("The answer is 42.", "missing_final_marker"),
        ("##### 42", "missing_final_marker"),
        ("#### 40\n#### 42", "multiple_final_markers"),
        ("#### 42\nMore text", "malformed_or_nonterminal_answer"),
    ],
)
def test_strict_parser_rejects_invalid_format(response, error):
    result = parse_strict(response)

    assert not result.success
    assert result.error == error


def test_flexible_parser_accepts_named_answer():
    result = parse_flexible("After checking the work, the final answer is $2.50.")

    assert result.value == "2.5"
    assert result.source == "named"


def test_flexible_parser_accepts_only_number():
    result = parse_flexible("42")

    assert result.value == "42"
    assert result.source == "single_number"


def test_flexible_parser_does_not_guess_between_numbers():
    result = parse_flexible("It could be 40 or 42.")

    assert not result.success
    assert result.error == "ambiguous_answers"


def test_flexible_parser_reports_missing_number():
    result = parse_flexible("I cannot solve this problem.")

    assert not result.success
    assert result.error == "no_numeric_answer"
