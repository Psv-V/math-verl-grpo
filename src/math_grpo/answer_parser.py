"""Conservative parsing of final numeric answers from model responses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

NUMBER_PATTERN = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

_FINAL_MARKER = re.compile(r"(?<!#)####(?!#)")
_STRICT_ANSWER = re.compile(rf"(?<!#)####(?!#)\s*({NUMBER_PATTERN})\s*$")
_MARKED_ANSWER = re.compile(rf"(?<!#)####(?!#)\s*\$?\s*({NUMBER_PATTERN})")
_NAMED_ANSWER = re.compile(
    rf"(?:final\s+answer|answer)\s*(?:is|=|:)\s*\$?\s*({NUMBER_PATTERN})",
    re.IGNORECASE,
)
_ANY_NUMBER = re.compile(rf"(?<![\w.])({NUMBER_PATTERN})(?![\w.])")


@dataclass(frozen=True)
class ParseResult:
    """The parsed answer, its source, or a stable failure reason."""

    value: str | None
    source: str | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.value is not None


def normalize_number(value: str) -> str:
    """Return one stable string representation of a finite decimal number."""

    cleaned = value.replace(",", "").strip()
    try:
        number = Decimal(cleaned)
    except InvalidOperation as error:
        raise ValueError(f"Invalid numeric answer: {value!r}") from error

    if not number.is_finite():
        raise ValueError(f"Answer must be finite: {value!r}")
    if number == 0:
        return "0"

    normalized = format(number, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def parse_strict(response: str) -> ParseResult:
    """Accept exactly one terminal ``#### <number>`` answer."""

    if not response.strip():
        return ParseResult(value=None, error="empty_response")

    marker_count = len(_FINAL_MARKER.findall(response))
    if marker_count == 0:
        return ParseResult(value=None, error="missing_final_marker")
    if marker_count > 1:
        return ParseResult(value=None, error="multiple_final_markers")

    match = _STRICT_ANSWER.search(response)
    if match is None:
        return ParseResult(value=None, error="malformed_or_nonterminal_answer")

    return ParseResult(value=normalize_number(match.group(1)), source="strict")


def _unique_match(pattern: re.Pattern[str], response: str) -> ParseResult | None:
    matches = pattern.findall(response)
    if len(matches) == 1:
        return ParseResult(value=normalize_number(matches[0]))
    if len(matches) > 1:
        return ParseResult(value=None, error="ambiguous_answers")
    return None


def parse_flexible(response: str) -> ParseResult:
    """Extract only an unambiguous numeric answer for diagnostic evaluation.

    Preference order is strict format, one ``####`` answer, one explicit
    ``final answer`` phrase, and finally a response containing only one number.
    Multiple plausible candidates are rejected instead of guessing.
    """

    strict = parse_strict(response)
    if strict.success:
        return strict
    if not response.strip():
        return strict

    marked = _unique_match(_MARKED_ANSWER, response)
    if marked is not None:
        if marked.success:
            return ParseResult(value=marked.value, source="marked")
        return marked

    named = _unique_match(_NAMED_ANSWER, response)
    if named is not None:
        if named.success:
            return ParseResult(value=named.value, source="named")
        return named

    numbers = _unique_match(_ANY_NUMBER, response)
    if numbers is not None:
        if numbers.success:
            return ParseResult(value=numbers.value, source="single_number")
        return numbers

    return ParseResult(value=None, error="no_numeric_answer")
