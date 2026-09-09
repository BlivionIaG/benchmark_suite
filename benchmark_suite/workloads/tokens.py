"""Char-based token estimates and deterministic padding.

Scorers must not depend on torch or a model tokenizer. We treat 4 characters as
one token (the same heuristic llama.cpp uses for naive estimates). That is close
enough to pin prefill/decode *load*, which is what these workloads measure.
"""

from __future__ import annotations

import hashlib

CHARS_PER_TOKEN = 4


def approx_tokens(text: str) -> int:
    """Estimate tokens as ``len(text) // 4``. Empty string is 0."""
    if not text:
        return 0
    return len(text) // CHARS_PER_TOKEN


def content_tokens(messages: list[dict[str, str]]) -> int:
    """Sum of ``approx_tokens`` over each message's content."""
    return sum(approx_tokens(m["content"]) for m in messages)


def pad_to_tokens(text: str, target: int, *, seed: str) -> str:
    """Append deterministic background text so ``approx_tokens`` equals ``target``.

    If ``text`` is already at or above ``target``, it is returned unchanged (never
    truncated — the unique task always survives).
    """
    if target <= 0:
        return text
    target_chars = target * CHARS_PER_TOKEN
    if len(text) >= target_chars:
        return text
    parts: list[str] = [text, "\n\n## Background material\n\n"]
    n = 0
    body = "".join(parts)
    while len(body) < target_chars:
        n += 1
        body += _filler_paragraph(seed, n)
        body += "\n"
    return body[:target_chars]


def pad_last_user_to_total(
    messages: list[dict[str, str]], target: int, *, seed: str
) -> list[dict[str, str]]:
    """Pad the last user message so conversation content is ``target`` tokens.

    If the conversation already meets or exceeds ``target``, messages are
    returned unchanged.
    """
    if not messages:
        return messages
    current = content_tokens(messages)
    if current >= target:
        return [dict(m) for m in messages]
    last = messages[-1]
    if last.get("role") != "user":
        raise ValueError("pad_last_user_to_total requires the last message to be role=user")
    others = content_tokens(messages[:-1])
    user_target = max(approx_tokens(last["content"]), target - others)
    padded = pad_to_tokens(last["content"], user_target, seed=seed)
    out = [dict(m) for m in messages]
    out[-1] = {"role": "user", "content": padded}
    return out


def _filler_paragraph(seed: str, n: int) -> str:
    digest = hashlib.sha256(f"{seed}:{n}".encode()).hexdigest()
    return (
        f"Section {n} ({digest[:12]}). This attached source continues the "
        f"working context for '{seed}'. It exists so the prompt reaches a "
        f"fixed token budget without a repeating 16-byte loop. Notes for "
        f"section {n}: figures, asides, and references the assistant may ignore "
        f"unless the user question cites them. Cross-ref {digest[12:20]}. "
    )
