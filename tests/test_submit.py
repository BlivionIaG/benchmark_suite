"""tests/test_submit.py — bs submit subprocess shell-out to localmaxxing's `lmx`.

`bs submit` builds a JSON payload, writes it to a temp file, and shells
out to `lmx speed-test submit` (or `dry-run`). Tests mock the lmx
binary with a small bash script that records invocations and prints
synthetic success output.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from benchmark_suite.cli import app
from benchmark_suite.recipe import (
    BackendSection,
    EndpointSection,
    HardwareSection,
    MetaSection,
    Recipe,
)
from benchmark_suite.scoring.base import ScoreRecord
from benchmark_suite.submission import (
    find_lmx,
    submit_submission,
)

runner = CliRunner()


# ----- fixtures -----


@pytest.fixture
def fake_lmx(tmp_path: Path) -> Path:
    script = tmp_path / "fake-lmx"
    invocations = tmp_path / "invocations"
    invocations.mkdir()
    invocations_env = invocations.resolve().as_posix()
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"INV_DIR={invocations_env}\n"
        "mkdir -p \"$INV_DIR\"\n"
        "rec=\"$INV_DIR/$(basename \"$0\")-$RANDOM-$$.jsonl\"\n"
        "python3 - \"$rec\" \"$@\" <<'PYEOF'\n"
        "import json, sys\n"
        "rec = sys.argv[1]\n"
        "with open(rec, 'a') as f:\n"
        "    f.write(json.dumps({'argv': sys.argv[2:]}) + '\\n')\n"
        "PYEOF\n"
        "subcmd=\"${2:-submit}\"\n"
        "if [ \"${LMX_FAIL:-0}\" = \"1\" ]; then\n"
        "  echo \"lmx rejected payload: hfId required\" >&2\n"
        "  exit 1\n"
        "fi\n"
        "if [ \"$subcmd\" = \"dry-run\" ]; then\n"
        "  echo \"valid: true\"\n"
        "  exit 0\n"
        "fi\n"
        "echo \"submitted: abc-123\"\n"
        "echo \"view at: https://www.localmaxxing.com/speed-tests/abc-123\"\n"
        "exit 0\n"
    )
    script.chmod(0o755)
    return script


def _seed_recipe() -> Recipe:
    return Recipe(
        meta=MetaSection(name="test", description="x"),
        backend=BackendSection(type="vllm", model_path="Qwen/Qwen3-8B"),
        endpoint=EndpointSection(url="http://127.0.0.1:8000"),
        hardware=HardwareSection(
            vendor="amd",
            model="Radeon PRO V620",
            count=4,
            vram_gb=32,
        ),
    )


def _seed_score() -> ScoreRecord:
    return ScoreRecord(
        kind="throughput",
        cell_id="dense_triton_triton_cg0_mtp0",
        status="success",
        metrics={"output_tok_s": 87.4, "ttft_mean_ms": 142.5, "input_tok_s": 1210.5},
    )


@pytest.fixture
def sample_result_dir(tmp_path: Path) -> Path:
    rd = tmp_path / "cell"
    rd.mkdir()
    recipe = _seed_recipe()
    score = _seed_score()
    (rd / "summary.json").write_text(
        json.dumps({"recipe": recipe.model_dump(mode="json"), "scores": [score.to_dict()]})
    )
    (rd / "summary.csv").write_text("metric,value\noutput_tok_s,87.4\n")
    (rd / "README.md").write_text("# test\n\nsample run\n")
    return rd


def _latest_invocation(fake_lmx: Path) -> dict[str, Any]:
    invocations_dir = fake_lmx.parent / "invocations"
    files = sorted(invocations_dir.glob("*.jsonl"))
    assert files, "fake lmx was not invoked"
    last = files[-1].read_text().strip().splitlines()[-1]
    return json.loads(last)


# ----- find_lmx -----


def test_find_lmx_explicit_path(fake_lmx: Path) -> None:
    assert find_lmx(str(fake_lmx)) == str(fake_lmx)


def test_find_lmx_explicit_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        find_lmx(str(tmp_path / "does-not-exist"))


def test_find_lmx_not_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty_path = tmp_path / "empty-bin"
    empty_path.mkdir()
    monkeypatch.setenv("PATH", str(empty_path))
    with pytest.raises(FileNotFoundError, match="binary not found"):
        find_lmx(None)


def test_find_lmx_on_path(
    fake_lmx: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_which(name: str) -> str | None:
        return str(fake_lmx) if name == "lmx" else None

    monkeypatch.setattr("shutil.which", cast(Callable[[str], str | None], fake_which))
    assert find_lmx(None) == str(fake_lmx)


# ----- submit_submission (unit) -----


def test_submit_success_calls_lmx_submit(fake_lmx: Path, sample_result_dir: Path) -> None:
    result = submit_submission(
        sample_result_dir,
        lmx_bin=str(fake_lmx),
        endpoint="https://www.localmaxxing.com",
    )

    assert result.get("submission_id") == "abc-123"
    assert result.get("public_url") == "https://www.localmaxxing.com/speed-tests/abc-123"
    assert result.get("lmx_exit_code") == 0

    inv = _latest_invocation(fake_lmx)
    argv = inv["argv"]
    assert argv[0] == "speed-test"
    assert argv[1] == "submit"
    assert "--api-url" in argv
    assert "https://www.localmaxxing.com" in argv


def test_submit_dry_run_calls_lmx_dry_run(
    fake_lmx: Path, sample_result_dir: Path
) -> None:
    result = submit_submission(
        sample_result_dir,
        lmx_bin=str(fake_lmx),
        dry_run=True,
    )

    assert result.get("dry_run_valid") is True
    assert "valid: true" in result.get("dry_run_stdout", "")

    inv = _latest_invocation(fake_lmx)
    argv = inv["argv"]
    assert argv[1] == "dry-run"


def test_submit_no_endpoint_omits_api_url_flag(
    fake_lmx: Path, sample_result_dir: Path
) -> None:
    result = submit_submission(sample_result_dir, lmx_bin=str(fake_lmx))

    assert result.get("submission_id") == "abc-123"
    inv = _latest_invocation(fake_lmx)
    assert "--api-url" not in inv["argv"]


def test_submit_payload_temp_file_deleted_after_run(
    fake_lmx: Path, sample_result_dir: Path
) -> None:
    submit_submission(sample_result_dir, lmx_bin=str(fake_lmx))

    inv = _latest_invocation(fake_lmx)
    payload_path = Path(inv["argv"][2])
    assert not payload_path.exists(), (
        "temp payload file must be cleaned up after lmx runs"
    )


def test_submit_lmx_failure_returns_structured_error(
    tmp_path: Path, sample_result_dir: Path
) -> None:
    fake = tmp_path / "fail-lmx"
    fake.write_text("#!/usr/bin/env bash\necho 'lmx rejected payload' >&2\nexit 1\n")
    fake.chmod(0o755)

    result = submit_submission(sample_result_dir, lmx_bin=str(fake))

    assert result.get("error") == "lmx_failed"
    assert result.get("lmx_exit_code") == 1
    assert "lmx rejected payload" in result.get("lmx_stderr", "")
    assert result.get("submission_id", "") == ""


def test_submit_lmx_not_found(
    tmp_path: Path, sample_result_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_path = tmp_path / "empty-bin"
    empty_path.mkdir()
    monkeypatch.setenv("PATH", str(empty_path))

    result = submit_submission(sample_result_dir, lmx_bin=None)

    assert result.get("lmx_not_found") is True
    assert result.get("error") == "lmx_not_found"
    assert "lmx" in result.get("details", "").lower()


def test_submit_missing_summary_json_raises(tmp_path: Path, fake_lmx: Path) -> None:
    with pytest.raises(FileNotFoundError, match=re.escape("summary.json")):
        submit_submission(
            tmp_path / "no-such-dir",
            lmx_bin=str(fake_lmx),
        )


def test_submit_no_output_tok_s_raises(
    tmp_path: Path, fake_lmx: Path
) -> None:
    rd = tmp_path / "cell"
    rd.mkdir()
    recipe = Recipe(
        meta=MetaSection(name="t", description="x"),
        backend=BackendSection(type="external"),
    )
    (rd / "summary.json").write_text(
        json.dumps({"recipe": recipe.model_dump(mode="json"), "scores": []})
    )
    with pytest.raises(ValueError, match="output_tok_s"):
        submit_submission(rd, lmx_bin=str(fake_lmx))


# ----- CLI -----


def test_cli_submit_help() -> None:
    result = runner.invoke(app, ["submit", "--help"])
    assert result.exit_code == 0
    assert "--lmx-bin" in result.output
    assert "--endpoint" in result.output
    assert "--dry-run" in result.output
    assert "--api-key" not in result.output, (
        "lmx handles auth; --api-key was removed in the shell-out rewrite"
    )
    assert "lmx" in result.output


def test_cli_submit_missing_dir(fake_lmx: Path) -> None:
    result = runner.invoke(
        app, ["submit", "/no/such/dir", "--lmx-bin", str(fake_lmx)]
    )
    assert result.exit_code != 0


def test_cli_submit_happy_path_prints_url(
    fake_lmx: Path, sample_result_dir: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "submit",
            str(sample_result_dir),
            "--lmx-bin",
            str(fake_lmx),
            "--endpoint",
            "https://www.localmaxxing.com",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "abc-123" in result.output
    assert "localmaxxing.com/speed-tests/abc-123" in result.output


def test_cli_submit_dry_run_prints_valid(
    fake_lmx: Path, sample_result_dir: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "submit",
            str(sample_result_dir),
            "--lmx-bin",
            str(fake_lmx),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "payload valid" in result.output


def test_cli_submit_lmx_not_found(
    tmp_path: Path, sample_result_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_path = tmp_path / "empty-bin"
    empty_path.mkdir()
    monkeypatch.setenv("PATH", str(empty_path))

    result = runner.invoke(app, ["submit", str(sample_result_dir)])

    assert result.exit_code == 1
    assert "lmx" in result.output.lower()


def test_cli_submit_lmx_failure(
    tmp_path: Path, sample_result_dir: Path
) -> None:
    fake = tmp_path / "fail-lmx"
    fake.write_text("#!/usr/bin/env bash\necho 'rejected' >&2\nexit 1\n")
    fake.chmod(0o755)

    result = runner.invoke(
        app,
        ["submit", str(sample_result_dir), "--lmx-bin", str(fake)],
    )

    assert result.exit_code == 1
    assert "rejected" in result.output
