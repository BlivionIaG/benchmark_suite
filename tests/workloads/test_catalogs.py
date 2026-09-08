"""Invariants for the shipped sauce chat and session prompt catalogs."""

from __future__ import annotations

from collections import Counter
from itertools import pairwise

from benchmark_suite.workloads.chat_catalog import (
    CHAT_LADDER,
    CHAT_PROMPTS,
    assemble_chat_messages,
    prompts_for_concurrency,
    validate_chat_catalog,
)
from benchmark_suite.workloads.session_catalog import (
    SESSION_SPECS,
    SESSION_SYSTEM_PROMPT,
    plan_turn_input_targets,
    session_followups,
)
from benchmark_suite.workloads.tokens import approx_tokens, content_tokens


def test_chat_catalog_ladder_and_unique_prompts() -> None:
    validate_chat_catalog()
    assert CHAT_LADDER == (16, 8, 4, 2, 1)
    counts = Counter(p.concurrency for p in CHAT_PROMPTS)
    assert dict(counts) == {16: 16, 8: 8, 4: 4, 2: 2, 1: 1}
    assert len(CHAT_PROMPTS) == 31
    assert len({p.prompt_id for p in CHAT_PROMPTS}) == 31
    assert len({p.subject for p in CHAT_PROMPTS}) == 31
    assert len({p.system for p in CHAT_PROMPTS}) == 31
    assert len({p.task for p in CHAT_PROMPTS}) == 31


def test_prompts_for_concurrency_returns_matching_bucket() -> None:
    for conc in CHAT_LADDER:
        bucket = prompts_for_concurrency(conc)
        assert len(bucket) == conc
        assert all(p.concurrency == conc for p in bucket)


def test_assemble_chat_messages_hits_input_budget() -> None:
    spec = prompts_for_concurrency(1)[0]
    messages = assemble_chat_messages(spec, input_tokens=512)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[0]["content"] == spec.system
    assert spec.task in messages[1]["content"]
    assert content_tokens(messages) == 512


def test_session_catalog_has_sixteen_distinct_coding_tasks() -> None:
    assert len(SESSION_SPECS) == 16
    assert len({s.session_id for s in SESSION_SPECS}) == 16
    assert len({s.initial for s in SESSION_SPECS}) == 16
    assert len({s.title for s in SESSION_SPECS}) == 16
    assert "coding agent" in SESSION_SYSTEM_PROMPT.lower()
    for spec in SESSION_SPECS:
        followups = session_followups(spec)
        assert len(followups) == 12
        assert len(set(followups)) == 12
        assert spec.feature in followups[0]


def test_plan_turn_input_targets_grows_and_clips_for_small_context() -> None:
    large = plan_turn_input_targets(budget=197952)
    small = plan_turn_input_targets(budget=6000)
    assert large[-1] == 197952
    assert small[-1] == 6000
    assert len(large) > len(small)
    assert all(a < b for a, b in pairwise(large))
    assert all(a < b for a, b in pairwise(small))
    assert large[0] >= 2048 or large[0] == large[-1]


def test_session_system_prompt_is_substantial() -> None:
    assert approx_tokens(SESSION_SYSTEM_PROMPT) >= 400
