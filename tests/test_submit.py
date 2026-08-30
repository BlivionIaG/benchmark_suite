"""tests/test_submit.py — bs submit HTTP POST flow.

Tests use a MockHttpx context manager that swaps benchmark_suite.submission's
httpx.Client for one backed by httpx.MockTransport. The tarball contents are
checked by decoding what the mock received, not by re-implementing the bundle
logic (which is covered by test_export.py).
"""
from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from benchmark_suite.cli import app
from benchmark_suite.submission import (
    DEFAULT_LEADERBOARD_URL,
    SubmitResult,
    resolve_leaderboard_url,
    submit_submission,
)

runner = CliRunner()


# ----- fixtures -----


RecipeDict = dict[str, Any]


@pytest.fixture
def sample_result_dir(tmp_path: Path) -> Path:
    """Minimal result dir with summary.json/csv/README.md + a recipe-shaped blob."""
    rd = tmp_path / "cell"
    rd.mkdir()
    recipe: RecipeDict = {
        "meta": {"name": "test-recipe", "description": "x", "version": "1.0.0", "tags": []},
        "backend": {"type": "external", "model_path": "/models/x"},
        "endpoint": {"url": "http://127.0.0.1:8000"},
        "resources": {"tensor_parallel_size": 1, "dtype": "float16"},
        "bench": {
            "load": {"concurrencies": [1], "num_prompts": 4, "input_len": 16, "output_len": 8},
            "scoring": [{"kind": "throughput", "tool": "vllm-bench"}],
        },
    }
    (rd / "summary.json").write_text(json.dumps({"recipe": recipe, "scores": []}))
    (rd / "summary.csv").write_text("metric,value\noutput_tok_s,42.5\nttft_mean_ms,120\n")
    (rd / "README.md").write_text("# test\n\nsample run\n")
    return rd


class MockHttpx:
    """Context manager that swaps `benchmark_suite.submission.httpx.Client`
    for one backed by a MockTransport.

    Usage:
        with MockHttpx(handler) as captured:
            submit_submission(...)
        # captured dict is populated by the handler with whatever it wants
    """

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self._handler = handler
        self._transport = httpx.MockTransport(handler)
        self._orig_client: type[httpx.Client] | None = None
        self.captured: dict[str, Any] = {}

    def __enter__(self) -> dict[str, Any]:
        import benchmark_suite.submission as sub

        self._orig_client = sub.httpx.Client
        sub.httpx.Client = self._factory
        return self.captured

    def __exit__(self, *exc: object) -> None:
        import benchmark_suite.submission as sub

        if self._orig_client is not None:
            sub.httpx.Client = self._orig_client

    def _factory(self, *args: object, **kwargs: object) -> httpx.Client:
        orig = self._orig_client
        if orig is None:
            raise RuntimeError("MockHttpx used outside `with` block")
        merged: dict[str, object] = {"transport": self._transport, **kwargs}
        return orig(*args, **merged)


def _assert_tarball_contents(tar_bytes: bytes) -> set[str]:
    """Return the set of filenames inside the tarball."""
    names: set[str] = set()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        for member in tf.getmembers():
            names.add(member.name)
    return names


# ----- unit tests -----


def test_submit_success_201(sample_result_dir: Path) -> None:
    """Happy path: server returns 201 + submission_id, we return the public URL."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            201,
            json={
                "submission_id": "abc-123",
                "public_url": "https://bench.uncool.red/runs/abc-123",
            },
        )

    with MockHttpx(handler):
        result = submit_submission(
            sample_result_dir,
            handle="testuser",
            endpoint="https://bench.uncool.red",
        )

    assert result.get("submission_id") == "abc-123"
    assert result.get("public_url") == "https://bench.uncool.red/runs/abc-123"
    assert result.get("http_status") == 201
    assert result.get("endpoint") == "https://bench.uncool.red"


def test_submit_post_body_has_required_form_fields(sample_result_dir: Path) -> None:
    """The HTTP request must include file/handle/date/metadata fields."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        ctype = request.headers.get("content-type", "")
        body: bytes = request.read()
        captured["content_type"] = ctype
        captured["body"] = body
        return httpx.Response(
            201,
            json={"submission_id": "ok", "public_url": "https://bench.uncool.red/runs/ok"},
        )

    with MockHttpx(handler):
        submit_submission(
            sample_result_dir,
            handle="testuser",
            endpoint="https://bench.uncool.red",
        )

    body: bytes = captured["body"]
    assert b'name="handle"' in body
    assert b"testuser" in body
    assert b'name="date"' in body
    assert b'name="metadata"' in body
    assert b'name="file"' in body
    assert b"submission.tar.gz" in body


def test_submit_tarball_contains_required_files(sample_result_dir: Path) -> None:
    """Tarball in POST body must contain recipe.yaml + summary.* + README.md + metadata.json."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        ctype = request.headers.get("content-type", "")
        boundary = ctype.split("boundary=", 1)[1].split(";", 1)[0]
        boundary_b = b"--" + boundary.encode()
        chunks = body.split(boundary_b)
        for chunk in chunks:
            if b'name="file"' in chunk:
                start = chunk.find(b"\r\n\r\n") + 4
                end = chunk.rfind(b"\r\n")
                captured["file_bytes"] = chunk[start:end]
                break
        return httpx.Response(
            201,
            json={"submission_id": "ok", "public_url": "https://bench.uncool.red/runs/ok"},
        )

    with MockHttpx(handler):
        submit_submission(
            sample_result_dir,
            handle="testuser",
            endpoint="https://bench.uncool.red",
        )

    file_bytes: bytes = captured["file_bytes"]
    names = _assert_tarball_contents(file_bytes)
    assert "recipe.yaml" in names
    assert "summary.csv" in names
    assert "summary.json" in names
    assert "README.md" in names
    assert "metadata.json" in names


def test_submit_server_400_returns_structured_error(sample_result_dir: Path) -> None:
    """Server-side validation failure surfaces error + details from the response."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": "handle_invalid", "details": "must match ^[a-z0-9]..."},
        )

    with MockHttpx(handler):
        result = submit_submission(
            sample_result_dir, handle="testuser", endpoint="https://bench.uncool.red"
        )

    typed: SubmitResult = result
    assert typed.get("error") == "handle_invalid"
    assert "must match" in typed.get("details", "")
    assert typed.get("http_status") == 400
    assert "submission_id" not in typed


def test_submit_offline_fallback_writes_to_cache(
    sample_result_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Network failure -> tarball lands in ~/.cache/bs/submissions/."""
    monkeypatch.setenv("BS_CACHE_DIR", str(sample_result_dir.parent / "cache"))

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with MockHttpx(handler):
        result = submit_submission(
            sample_result_dir, handle="testuser", endpoint="https://bench.uncool.red"
        )

    typed: SubmitResult = result
    assert typed.get("error") == "offline"
    local_path = typed.get("local_path")
    assert local_path
    local = Path(local_path)
    assert local.exists()
    assert local.stat().st_size > 0
    assert local.suffix == ".gz"
    assert "testuser" in local.name


def test_submit_offline_no_fallback_returns_error(sample_result_dir: Path) -> None:
    """With allow_offline_fallback=False, a connection error surfaces in the result."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with MockHttpx(handler):
        result = submit_submission(
            sample_result_dir,
            handle="testuser",
            endpoint="https://bench.uncool.red",
            allow_offline_fallback=False,
        )

    typed: SubmitResult = result
    assert typed.get("error") == "offline"
    assert "local_path" not in typed


def test_submit_leaderboard_url_env_var(
    sample_result_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LEADERBOARD_URL env var is respected when --endpoint is not given."""
    seen_url: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_url["url"] = str(request.url)
        return httpx.Response(
            201, json={"submission_id": "ok", "public_url": "https://example.test/runs/ok"}
        )

    monkeypatch.setenv("LEADERBOARD_URL", "https://example.test")
    with MockHttpx(handler):
        result = submit_submission(sample_result_dir, handle="testuser")

    typed: SubmitResult = result
    assert seen_url["url"] == "https://example.test/api/submissions"
    assert typed.get("endpoint") == "https://example.test"


def test_submit_endpoint_flag_overrides_env(
    sample_result_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--endpoint (passed as positional arg) takes precedence over $LEADERBOARD_URL."""
    seen_url: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_url["url"] = str(request.url)
        return httpx.Response(
            201, json={"submission_id": "ok", "public_url": "https://override.test/runs/ok"}
        )

    monkeypatch.setenv("LEADERBOARD_URL", "https://env.example.com")
    with MockHttpx(handler):
        submit_submission(
            sample_result_dir, handle="testuser", endpoint="https://override.test"
        )

    assert seen_url["url"] == "https://override.test/api/submissions"


def test_submit_default_endpoint_is_bench_uncool_red(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no env var and no --endpoint, default is https://bench.uncool.red."""
    monkeypatch.delenv("LEADERBOARD_URL", raising=False)
    assert resolve_leaderboard_url(None) == DEFAULT_LEADERBOARD_URL
    assert resolve_leaderboard_url(None) == "https://bench.uncool.red"


def test_submit_empty_handle_raises(sample_result_dir: Path) -> None:
    with pytest.raises(ValueError, match="handle"):
        submit_submission(sample_result_dir, handle="", endpoint="https://x.test")


def test_submit_missing_summary_json_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=re_escape("summary.json")):
        submit_submission(
            tmp_path / "no-such-dir",
            handle="testuser",
            endpoint="https://x.test",
        )


# ----- CLI tests -----


def test_cli_submit_help() -> None:
    """The submit command is registered and its help renders."""
    result = runner.invoke(app, ["submit", "--help"])
    assert result.exit_code == 0
    assert "--handle" in result.output
    assert "--endpoint" in result.output
    assert "bench.uncool.red" in result.output


def test_cli_submit_handle_required(sample_result_dir: Path) -> None:
    """Without --handle, the CLI exits 2 with a clear error."""
    result = runner.invoke(app, ["submit", str(sample_result_dir)])
    assert result.exit_code == 2
    assert "--handle" in result.output


def test_cli_submit_missing_dir() -> None:
    """A non-existent result dir exits with an error."""
    result = runner.invoke(
        app, ["submit", "/no/such/dir", "--handle", "testuser"]
    )
    assert result.exit_code != 0


def test_cli_submit_calls_endpoint_and_prints_url(
    sample_result_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end CLI invocation: --endpoint captures the request, output shows URL."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            201,
            json={
                "submission_id": "cli-ok",
                "public_url": "https://bench.uncool.red/runs/cli-ok",
            },
        )

    with MockHttpx(handler):
        result = runner.invoke(
            app,
            [
                "submit",
                str(sample_result_dir),
                "--handle",
                "testuser",
                "--endpoint",
                "https://bench.uncool.red",
                "--hardware-gpu",
                "Radeon PRO V620",
                "--hardware-gpu-count",
                "4",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "cli-ok" in result.output
    assert "bench.uncool.red/runs/cli-ok" in result.output


def test_cli_submit_offline_uses_env_var_endpoint(
    sample_result_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LEADERBOARD_URL env var is picked up by the CLI as the endpoint."""
    monkeypatch.setenv("BS_CACHE_DIR", str(sample_result_dir.parent / "cli-cache"))

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setenv("LEADERBOARD_URL", "https://offline.test")
    with MockHttpx(handler):
        result = runner.invoke(
            app,
            [
                "submit",
                str(sample_result_dir),
                "--handle",
                "testuser",
            ],
        )

    assert result.exit_code == 1
    assert "offline" in result.output.lower()
    assert "cached" in result.output.lower()


def test_cli_submit_no_offline_fallback_exits_clean(
    sample_result_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With --no-offline-fallback, the CLI prints the offline error without writing cache."""
    cache_dir = sample_result_dir.parent / "no-fallback-cache"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setenv("BS_CACHE_DIR", str(cache_dir))
    with MockHttpx(handler):
        result = runner.invoke(
            app,
            [
                "submit",
                str(sample_result_dir),
                "--handle",
                "testuser",
                "--endpoint",
                "https://x.test",
                "--no-offline-fallback",
            ],
        )

    assert result.exit_code == 1
    assert not cache_dir.exists() or not any(cache_dir.glob("*.tar.gz"))


def re_escape(s: str) -> str:
    """Escape regex special characters for use in pytest.raises(match=...)."""
    import re

    return re.escape(s)