"""Tests for char-based token estimates and deterministic padding."""

from __future__ import annotations

import pytest

from benchmark_suite.workloads.tokens import (
    CHARS_PER_TOKEN,
    approx_tokens,
    content_tokens,
    pad_last_user_to_total,
    pad_to_tokens,
)


def test_approx_tokens_empty_is_zero() -> None:
    assert approx_tokens("") == 0


def test_approx_tokens_is_len_over_four() -> None:
    assert approx_tokens("abcd") == 1
    assert approx_tokens("a" * 40) == 10
    assert CHARS_PER_TOKEN == 4


def test_pad_to_tokens_hits_target_exactly() -> None:
    padded = pad_to_tokens("hello", 20, seed="unit")
    assert approx_tokens(padded) == 20
    assert padded.startswith("hello")


def test_pad_to_tokens_does_not_truncate_oversize_source() -> None:
    source = "x" * 80
    assert approx_tokens(source) == 20
    assert pad_to_tokens(source, 10, seed="unit") == source


def test_pad_to_tokens_is_deterministic_and_seed_sensitive() -> None:
    a = pad_to_tokens("task", 50, seed="alpha")
    b = pad_to_tokens("task", 50, seed="alpha")
    c = pad_to_tokens("task", 50, seed="beta")
    assert a == b
    assert a != c


def test_pad_last_user_to_total_pads_only_user() -> None:
    messages = [
        {"role": "system", "content": "sys " * 8},
        {"role": "user", "content": "do the thing"},
    ]
    padded = pad_last_user_to_total(messages, 40, seed="sess-1")
    assert padded[0]["content"] == messages[0]["content"]
    assert padded[1]["content"].startswith("do the thing")
    assert content_tokens(padded) == 40


def test_pad_last_user_to_total_requires_last_user() -> None:
    messages = [{"role": "system", "content": "only system"}]
    with pytest.raises(ValueError, match="role=user"):
        pad_last_user_to_total(messages, 40, seed="x")


def test_content_tokens_sums_messages() -> None:
    messages = [
        {"role": "system", "content": "abcd"},
        {"role": "user", "content": "efghijkl"},
    ]
    assert content_tokens(messages) == 3
