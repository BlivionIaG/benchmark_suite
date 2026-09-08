"""Pi-style coding-agent system prompt plus 16 multi-turn coding sessions.

Follow-up user turns are session-specific (add a feature, fix CI, refactor,
etc.). Context growth is applied by the session scorer via token padding of
the latest user message so the conversation can climb toward 200k tokens.
Smaller ``max_model_len`` values simply plan fewer turns.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmark_suite.workloads.tokens import pad_to_tokens

SESSION_SYSTEM_PROMPT = """\
You are a coding agent in a continuing software session. You work from the
workspace context the user attaches (files, test logs, diffs, notes). Treat
attached material as ground truth. Do not invent APIs, files, or test
output that were not provided.

How you work:
- Read the latest user request and the attached workspace before answering.
- Propose concrete file-level changes: path, then a unified diff or a full
  file when a diff would be unclear.
- When you would run a command (tests, linters, formatters), write it as a
  fenced shell block the user could paste. Do not claim you already ran it
  unless a log is attached.
- Prefer small, reviewable patches over rewrites. Match the style of attached
  code (naming, types, error handling).
- If a constraint is missing (language version, license, target OS), state
  the assumption in one line and proceed.
- Never dump secrets, API keys, or credentials. If a test fixture needs a key,
  use a placeholder.

Session memory:
- The user will come back with follow-ups: extra features, failing tests,
  refactors, performance, docs. Keep earlier decisions unless they explicitly
  change them.
- Workspace attachments grow over the session. Later files supersede earlier
  ones when paths collide.

Output shape for each turn:
1. Short plan (3-8 bullets).
2. Patches / files.
3. Commands to run.
4. Risks or tests still missing.

You are not a general chatbot. Stay inside the repository task. If the user
asks for something outside the attached project, say so and offer the
smallest related change that still helps.

When the attached workspace is long, do not summarize the whole tree. Work
from the files the latest user message highlights. Quote paths when you
change them so a later turn can attach a patched snapshot.
"""


@dataclass(frozen=True)
class SessionSpec:
    """One coding session: unique initial task plus interpolations for follow-ups."""

    session_id: str
    title: str
    language: str
    initial: str
    feature: str
    test_failure: str
    bug: str
    refactor_target: str
    extra_module: str


SESSION_SPECS: tuple[SessionSpec, ...] = (
    SessionSpec(
        session_id="library-rest-api",
        title="library REST API",
        language="Python",
        initial=(
            "Start a small library catalog API: books with isbn, title, "
            "author, and copies. In-memory store is fine. Need list, get by "
            "isbn, create, and a borrow endpoint that decrements copies and "
            "rejects when none remain. Include a request/response sketch and "
            "the first module layout."
        ),
        feature="POST /books/{isbn}/return that increments copies and records a timestamp",
        test_failure=(
            "FAILED tests/test_borrow.py::test_borrow_last_copy - "
            "AssertionError: expected 409, got 200"
        ),
        bug="Borrowing the same isbn twice in one request double-decrements copies",
        refactor_target="the in-memory dict store into a BookRepository protocol",
        extra_module="src/library/auth.py — optional API key on mutating routes",
    ),
    SessionSpec(
        session_id="file-organizer-cli",
        title="file organizer CLI",
        language="Python",
        initial=(
            "Write a CLI that sorts a download folder into subdirs by "
            "extension (pdf, images, archives, other) and prints a dry-run "
            "table. Must refuse to overwrite, and must keep a manifest.json "
            "of moves so they can be undone."
        ),
        feature="--by-date mode that nests files under YYYY/MM/ in addition to type",
        test_failure="FAILED tests/test_undo.py::test_undo_restores_name_collision",
        bug="Symlinks are followed and the target is moved, leaving a broken link",
        refactor_target="path classification into a pure function with tests",
        extra_module="src/organize/watch.py — optional watchdog loop",
    ),
    SessionSpec(
        session_id="metrics-dashboard",
        title="metrics dashboard",
        language="HTML/JS",
        initial=(
            "Build a single-page dashboard that fetches /metrics.json every "
            "5s and draws a sparkline for request_rate and p95_latency. No "
            "framework: one HTML file, one JS module, CSS variables for a "
            "dark theme. Handle fetch failure without blanking the last "
            "good chart."
        ),
        feature="a toggle to pause polling and a CSV export of the in-memory buffer",
        test_failure="jsdom: sparkline throws when the metrics array is empty",
        bug="Overlapping fetches can apply an older payload after a newer one",
        refactor_target="chart drawing into a function that takes a canvas and samples",
        extra_module="static/thresholds.js — color the p95 line when it exceeds a limit",
    ),
    SessionSpec(
        session_id="sqlite-migrations",
        title="SQLite migration kit",
        language="Python",
        initial=(
            "Design a tiny forward-only migration runner for SQLite: a "
            "schema_migrations table, numbered SQL files, and a CLI `migrate "
            "up` / `status`. Fail if checksum of an applied file changed. No "
            "down migrations in v1."
        ),
        feature="`migrate stamp <version>` for existing databases created by hand",
        test_failure="FAILED tests/test_checksum.py::test_detects_edited_applied_file",
        bug="Two processes running migrate up both pass the lock and apply twice",
        refactor_target="SQL file discovery so versions can be 001_ and 20260908_ mixed",
        extra_module="src/migrate/sqlite_lock.py — BEGIN IMMEDIATE wrapper",
    ),
    SessionSpec(
        session_id="config-parser",
        title="config language parser",
        language="Python",
        initial=(
            "Parse a minimal config language: keys, dotted sections, "
            "strings, ints, bools, and comments with #. No nested arrays "
            "yet. Return a nested dict. On error, include line/column. "
            "Provide a handful of example files as the spec."
        ),
        feature="unquoted identifiers and a include = \"other.cfg\" directive",
        test_failure="FAILED tests/test_span.py::test_error_column_on_unterminated_string",
        bug="A comment after a value eats the next key on the following line",
        refactor_target="lexer vs parser so the lexer is unit-testable",
        extra_module="src/cfg/pretty.py — emit canonical form for diffs",
    ),
    SessionSpec(
        session_id="token-bucket",
        title="token-bucket rate limiter",
        language="Go",
        initial=(
            "Implement an in-process token bucket for HTTP middleware: "
            "capacity, refill per second, Allow(n) bool. Needs a test that "
            "time is injected (no sleeps). Document behavior when n > "
            "capacity."
        ),
        feature="a per-key map of buckets with idle eviction after TTL",
        test_failure="TestAllowBurst: got true after exhausting the bucket at t=0",
        bug="Refill uses wall time that jumps backward and grants a huge burst",
        refactor_target="the map + mutex into a type with a small interface",
        extra_module="ratelimit/http.go — middleware that keys on X-API-Key hash",
    ),
    SessionSpec(
        session_id="markdown-ssg",
        title="markdown static site generator",
        language="Python",
        initial=(
            "Write a static site generator: read content/*.md with YAML "
            "front matter, apply templates/*.html with {{title}} and "
            "{{body}}, emit public/. Support a posts list page. Keep it "
            "dependency-light; a stdlib markdown-ish converter may be a "
            "stub that wraps a function we can swap."
        ),
        feature="draft: true in front matter skips emit unless --include-drafts",
        test_failure="FAILED tests/test_index.py::test_posts_sorted_by_date_desc",
        bug="Relative links in markdown are not rewritten when a post is in a subfolder",
        refactor_target="template replacement so it does not use str.replace for braces in code",
        extra_module="ssg/sitemap.py — emit sitemap.xml from the public tree",
    ),
    SessionSpec(
        session_id="websocket-chat",
        title="WebSocket chat room",
        language="TypeScript",
        initial=(
            "Sketch a small WebSocket chat server: rooms, join, broadcast "
            "text, and a last-50-messages replay on join. In-memory. Define "
            "the JSON message schema and the server loop. No auth yet."
        ),
        feature="typing notifications that expire after 3s of silence",
        test_failure="join replay sends messages from other rooms",
        bug="A disconnected socket stays in the room set and broadcast throws",
        refactor_target="room state into a class with join/leave/broadcast",
        extra_module="src/rateLimit.ts — drop clients who send > 20 messages/s",
    ),
    SessionSpec(
        session_id="csv-cleaner",
        title="CSV cleaner",
        language="Python",
        initial=(
            "Build a CLI that reads a CSV, trims cells, drops empty rows, "
            "normalizes header names to snake_case, and writes UTF-8. "
            "Must work streaming for a 500 MB file (no full DataFrame). "
            "Report counts of dropped rows."
        ),
        feature="--dedupe-on col1,col2 keeping the last row",
        test_failure="FAILED tests/test_stream.py::test_does_not_slurp_whole_file",
        bug="Quoted commas inside fields split columns after trim",
        refactor_target="header normalization into a tested function",
        extra_module="csvclean/types.py — optional --infer-ints that fails closed",
    ),
    SessionSpec(
        session_id="toy-vcs",
        title="toy version-control snapshotter",
        language="Python",
        initial=(
            "Implement a toy VCS: `init`, `commit <msg>` that stores a "
            "content-addressed snapshot of a directory (sha256 of each file), "
            "and `log`. Ignore a .toyignore with glob-ish lines. Do not try "
            "to be git-compatible."
        ),
        feature="`diff <commit> <commit>` showing added/removed/changed paths",
        test_failure="FAILED tests/test_ignore.py::test_nested_glob_star",
        bug="Commit hashes change because file iteration order is not sorted",
        refactor_target="tree walking so ignore rules are applied in one place",
        extra_module="toyvcs/checkout.py — restore a snapshot to a target dir",
    ),
    SessionSpec(
        session_id="mini-regex",
        title="mini regex engine",
        language="Python",
        initial=(
            "Implement a matcher for a tiny regex subset: literals, `.`, "
            "`*`, concatenation, and grouping `(...)`. Recursive backtracking "
            "is fine. Provide match(pattern, text) -> bool. Document what "
            "you do not support (lookaround, backrefs)."
        ),
        feature="`+` and `?` and a `^`/`$` anchor",
        test_failure="FAILED tests/test_star.py::test_star_is_greedy_enough_for_aaa",
        bug="`a*` on a long unmatched suffix blows the recursion limit",
        refactor_target="NFA compilation if you are still on naive recursion",
        extra_module="regex/compile.py — emit an explicit instruction list",
    ),
    SessionSpec(
        session_id="job-queue",
        title="in-process job queue",
        language="Python",
        initial=(
            "Write a FIFO job queue with at-least-once workers: enqueue a "
            "callable name + JSON payload, workers heartbeating, and a "
            "visibility timeout that re-queues stale inflight jobs. "
            "SQLite-backed. Include a tiny worker loop."
        ),
        feature="named queues and a delay/run_at timestamp",
        test_failure="FAILED tests/test_lease.py::test_expired_lease_is_taken_by_second_worker",
        bug="Successful jobs are not deleted, so they retry forever",
        refactor_target="SQL statements into a store class",
        extra_module="jobs/dead.py — move jobs that fail N times to a dead table",
    ),
    SessionSpec(
        session_id="tui-todo",
        title="terminal todo TUI",
        language="Python",
        initial=(
            "Design a terminal todo app: list, add, toggle done, persist "
            "to ~/.todo.json. Keyboard: j/k, space, a, q. Describe the "
            "draw loop and a curses-or-not choice. Keep the model separate "
            "from rendering so tests do not need a tty."
        ),
        feature="priority flag and a filter to hide completed items",
        test_failure="FAILED tests/test_store.py::test_corrupt_json_does_not_crash",
        bug="Toggle uses list index after filter so the wrong item is completed",
        refactor_target="pure model (TodoList) vs renderer",
        extra_module="todo/search.py — incremental filter by substring",
    ),
    SessionSpec(
        session_id="http-mock",
        title="HTTP mock server",
        language="Python",
        initial=(
            "Write a stdlib http.server mock that loads fixtures from "
            "yaml/json: method, path, status, body. Support path params "
            "like /users/{id} and a default 404. Need a way to assert "
            "observed requests after a test."
        ),
        feature="request body JSON matchers and a sequence of responses for one path",
        test_failure="FAILED tests/test_params.py::test_path_param_does_not_match_extra_segment",
        bug="GET / is registered but /?x=1 404s because query is in the path key",
        refactor_target="matcher objects instead of a nested dict",
        extra_module="httpmock/https.py — optional TLS via a generated test cert",
    ),
    SessionSpec(
        session_id="thumbnail-pipeline",
        title="image thumbnail pipeline",
        language="Python",
        initial=(
            "Design a pipeline that reads a directory of images, writes "
            "WxH thumbnails (default 256x256, contain-fit, JPEG q=85), and "
            "a manifest of src hash → dest path. You may assume Pillow. "
            "Must skip unchanged hashes. No GPU."
        ),
        feature="`--sizes 256,1024` emitting several widths per source",
        test_failure="FAILED tests/test_skip.py::test_unchanged_hash_skips_encode",
        bug="Orientation EXIF is ignored so some thumbs are rotated",
        refactor_target="hashing vs encode so tests can stub encode",
        extra_module="thumbs/webp.py — optional webp output when requested",
    ),
    SessionSpec(
        session_id="typed-config",
        title="typed config loader",
        language="Python",
        initial=(
            "Load YAML/JSON config into a dataclass tree with required "
            "fields, defaults, and env-var overrides like APP_DB__HOST. "
            "Unknown keys are errors. Give a small example Config and the "
            "loader API. No pydantic required — stdlib + typing is the point."
        ),
        feature="a --print-effective command that shows env overrides in comments",
        test_failure="FAILED tests/test_env.py::test_nested_double_underscore_override",
        bug="Empty string env vars are treated as missing instead of empty",
        refactor_target="walk of nested dataclasses so lists of structs work",
        extra_module="config/dotenv.py — optional .env file, never committed examples",
    ),
)


_FOLLOWUP_COUNT = 12


def session_followups(spec: SessionSpec) -> tuple[str, ...]:
    """Twelve follow-up user turns: features, tests, fixes, polish."""
    return (
        (
            f"Add this feature to {spec.title}: {spec.feature}. Show the files "
            f"you would change and keep the existing public API working."
        ),
        (
            f"Tests failed on {spec.title}:\n\n```\n{spec.test_failure}\n```\n"
            f"Fix the code, not the assertion, unless the assertion is wrong."
        ),
        (
            f"We hit a bug in production-shaped use: {spec.bug}. Reproduce "
            f"with a test name, then patch."
        ),
        (
            f"Refactor {spec.refactor_target}. Behavior must stay the same; "
            f"add a characterization test if none exists."
        ),
        (
            f"Add structured logging around the hot path of {spec.title}. "
            f"No secrets in log lines. Include a sample log line."
        ),
        (
            f"Performance: {spec.title} feels slow on a 10x larger fixture. "
            f"Identify the likely bottleneck from the attached files and "
            f"propose one change with a before/after complexity note."
        ),
        (
            f"Harden error handling in {spec.title}: invalid input, missing "
            f"files, and a mid-run failure must not corrupt the store. "
            f"Return actionable errors."
        ),
        (
            f"Write a README section for {spec.title}: install, one happy-path "
            f"command, and a troubleshooting bullet for the bug we already saw."
        ),
        (
            f"UX/CLI: add flags or help text so a new user can discover "
            f"{spec.feature} without reading source. Keep POSIX-ish flags."
        ),
        (
            f"Persistence: make sure {spec.title} survives process restart. "
            f"If it already does, add a test that proves it with a temp dir."
        ),
        (
            f"Introduce {spec.extra_module}. Wire it in behind a default-off "
            f"flag or optional dependency."
        ),
        (
            f"Prepare a v0.1 release checklist for {spec.title}: version "
            f"bump, changelog, and what you would still not ship. {spec.language} "
            f"packaging notes only — no secrets."
        ),
    )


def plan_turn_input_targets(*, budget: int, max_turns: int = 13) -> list[int]:
    """Monotone input-token targets from a first-turn floor up to ``budget``.

    Smaller budgets drop turns so the step stays meaningful (>= 512 tokens
    when possible). ``budget`` should already be ``min(max_context, model_len)
    - output_reserve``. ``max_turns`` caps how many points are produced.
    """
    if budget < 1:
        raise ValueError("budget must be >= 1")
    if max_turns < 1:
        raise ValueError("max_turns must be >= 1")
    first = min(2048, max(256, budget // 4))
    if first >= budget or max_turns == 1:
        return [budget]
    min_step = 512 if budget >= 2048 else max(64, budget // 16)
    n = max_turns
    while n > 1 and (budget - first) / (n - 1) < min_step:
        n -= 1
    if n == 1:
        return [budget]
    out = [int(first + i * (budget - first) / (n - 1)) for i in range(n)]
    out[-1] = budget
    compacted: list[int] = [out[0]]
    for target in out[1:]:
        if target > compacted[-1]:
            compacted.append(target)
    if compacted[-1] != budget:
        compacted[-1] = budget
    return compacted


def workspace_blob(session_id: str, turn: int, extra_tokens: int) -> str:
    """Fake attached source files used to grow agentic context."""
    header = (
        f"## Attached workspace (turn {turn}, session {session_id})\n\n"
        f"Paths below are the current tree. Later turns replace files with "
        f"the same path.\n"
    )
    if extra_tokens <= 0:
        return header
    return pad_to_tokens(header, extra_tokens, seed=f"{session_id}:ws:{turn}")
