import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from math_grpo.data.gsm8k import (  # noqa: E402
    assert_no_question_overlap,
    choose_dev_indices,
    extract_ground_truth,
)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("reasoning\n#### 1,234", "1234"),
        ("reasoning\n#### -2.50", "-2.5"),
        ("reasoning\n#### 0.00", "0"),
    ],
)
def test_extract_ground_truth(answer, expected):
    assert extract_ground_truth(answer) == expected


def test_extract_ground_truth_requires_terminal_marker():
    with pytest.raises(ValueError):
        extract_ground_truth("#### 42\nextra text")


def test_dev_split_is_fixed_and_has_requested_size():
    first = choose_dev_indices(100, dev_size=12, seed=7)
    second = choose_dev_indices(100, dev_size=12, seed=7)

    assert first == second
    assert len(first) == 12


def test_overlap_check_normalizes_case_and_whitespace():
    rows = {
        "train": [{"extra_info": {"question": "How  many?"}}],
        "dev": [{"extra_info": {"question": "how many?"}}],
        "test": [{"extra_info": {"question": "A different question"}}],
    }

    with pytest.raises(ValueError, match="leakage"):
        assert_no_question_overlap(rows)

