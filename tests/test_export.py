"""Tests for `bs export` — metadata collection + submission tarball bundling."""
from __future__ import annotations

import json
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchmark_suite import cli
from benchmark_suite.recipe import BackendSection, EndpointSection, MetaSection, Recipe
from benchmark_suite.scoring.base import ScoreRecord
from benchmark_suite.scoring.metadata_collector import (
    build_metadata,
    collect_hardware,
    collect_software,
)

runner = CliRunner()

T0 = datetime(2026, 8, 30, 12, 0, 0, tzinfo=UTC)
T1 = datetime(2026, 8, 30, 12, 5, 0, tzinfo=UTC)


def _fail_proc() -> subprocess.CompletedProcess[str]:
    """A subprocess.run stand-in that always reports failure."""
    return subprocess.CompletedProcess(["rocm-smi"], 1, "", "")


# ----- helpers -----


def make_recipe() -> Recipe:
    return Recipe(
        meta=MetaSection(name="test", description="a test recipe"),
        backend=BackendSection(type="external"),
        endpoint=EndpointSection(url="http://x"),
    )


def seed_result_dir(result_dir: Path) -> None:
    """Write a minimal summary.json + summary.csv + README.md into result_dir."""
    recipe = make_recipe()
    score = ScoreRecord(
        kind="throughput",
        cell_id="dense_triton_triton_cg0_mtp0",
        status="success",
        started_at=T0,
        finished_at=T1,
        metrics={"output_tok_s": 50.0},
    )
    summary = {
        "recipe": recipe.model_dump(mode="json"),
        "scores": [score.to_dict()],
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (result_dir / "summary.csv").write_text(
        "cell_id,scorer_kind\ndense_triton_triton_cg0_mtp0,throughput\n"
    )
    (result_dir / "README.md").write_text("# Test\n")


def list_tar_members(path: Path) -> list[str]:
    with tarfile.open(path, "r:gz") as tf:
        return [m.name for m in tf.getmembers()]


# ----- collect_hardware -----


def test_collect_hardware_with_torch_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch.cuda.get_device_properties supplies gpu name + vram when rocm-smi absent."""
    import sys

    class FakeProps:
        name = "Radeon PRO V620"
        total_memory = 32_000_000_000

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_properties(idx: int) -> FakeProps:
            assert idx == 0
            return FakeProps()

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(
        "benchmark_suite.scoring.metadata_collector.subprocess.run", _run_fails
    )
    hw = collect_hardware()
    assert hw["gpu"] == "Radeon PRO V620"
    assert hw["vram_gb"] == 32


def _run_fails(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
    """subprocess.run stand-in: always fails, ignoring args."""
    del args, kwargs
    return _fail_proc()


def _path_missing(self: object) -> bool:
    """Path.exists stand-in: always False."""
    del self
    return False


def test_collect_hardware_no_torch_no_rocm_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both rocm-smi and torch unavailable → no GPU keys, no raise."""
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(
        "benchmark_suite.scoring.metadata_collector.subprocess.run", _run_fails
    )
    hw = collect_hardware()
    assert "gpu" not in hw
    assert "gpu_count" not in hw
    assert "vram_gb" not in hw


# ----- collect_software -----


def test_collect_software_finds_vllm_via_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    """pip show vllm Version field populates the 'vllm' key."""

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        if cmd[:2] == ["pip", "show"]:
            return subprocess.CompletedProcess(
                cmd, 0, "Name: vllm\nVersion: 0.20.1\n", ""
            )
        return _fail_proc()

    monkeypatch.setattr(
        "benchmark_suite.scoring.metadata_collector.subprocess.run", fake_run
    )
    monkeypatch.setattr(
        "benchmark_suite.scoring.metadata_collector.Path.exists", _path_missing
    )
    sw = collect_software()
    assert sw["vllm"] == "0.20.1"


def test_collect_software_no_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    """/opt/rocm/.info/version missing → 'rocm': 'n/a'."""
    monkeypatch.setattr(
        "benchmark_suite.scoring.metadata_collector.subprocess.run", _run_fails
    )
    monkeypatch.setattr(
        "benchmark_suite.scoring.metadata_collector.Path.exists", _path_missing
    )
    sw = collect_software()
    assert sw["rocm"] == "n/a"


# ----- build_metadata -----


def test_build_metadata_validates_required_fields() -> None:
    """Empty submitter raises ValueError."""
    with pytest.raises(ValueError):
        build_metadata(submitter="", date_str="2026-08-30")


def test_build_metadata_merges_overrides() -> None:
    """Explicit override wins over auto-detected value."""
    md = build_metadata(
        submitter="alice",
        date_str="2026-08-30",
        hardware={"gpu": "auto-gpu", "gpu_count": 4},
        software={"rocm": "7.2.0"},
        model={"hf_repo": "Qwen/Qwen2.5-7B"},
    )
    assert md["hardware"]["gpu"] == "auto-gpu"
    assert md["hardware"]["gpu_count"] == 4
    assert md["software"]["rocm"] == "7.2.0"
    assert md["model"]["hf_repo"] == "Qwen/Qwen2.5-7B"
    assert md["submitter"] == "alice"
    assert md["date"] == "2026-08-30"


# ----- export_submission -----


def test_export_submission_bundles_required_files(tmp_path: Path) -> None:
    """recipe.yaml, summary.csv/json, README.md, metadata.json all present."""
    result_dir = tmp_path / "rd"
    result_dir.mkdir()
    seed_result_dir(result_dir)
    out = cli.export_submission(
        result_dir,
        submitter="alice",
        output=tmp_path / "sub.tar.gz",
        metadata_overrides={"hardware": {"gpu": "Radeon PRO V620"}},
    )
    assert out.name.endswith(".tar.gz")
    members = list_tar_members(out)
    for expected in ("recipe.yaml", "summary.csv", "summary.json", "README.md", "metadata.json"):
        assert expected in members, f"missing {expected} in {members}"


def test_export_submission_metadata_overrides_win(tmp_path: Path) -> None:
    """Explicit hardware_gpu override lands in metadata.json."""
    result_dir = tmp_path / "rd"
    result_dir.mkdir()
    seed_result_dir(result_dir)
    out = cli.export_submission(
        result_dir,
        submitter="alice",
        output=tmp_path / "sub.tar.gz",
        metadata_overrides={"hardware": {"gpu": "custom"}},
    )
    with tarfile.open(out, "r:gz") as tf:
        member = tf.extractfile("metadata.json")
        assert member is not None
        metadata = json.loads(member.read())
    assert metadata["hardware"]["gpu"] == "custom"


def test_export_submission_writes_tar_zst(tmp_path: Path) -> None:
    """Output is a valid gzipped tar archive."""
    result_dir = tmp_path / "rd"
    result_dir.mkdir()
    seed_result_dir(result_dir)
    out = cli.export_submission(
        result_dir, submitter="alice", output=tmp_path / "sub.tar.gz"
    )
    assert out.suffix == ".gz"
    assert tarfile.is_tarfile(out)


def test_export_submission_includes_artifacts_if_small(tmp_path: Path) -> None:
    """Small artifacts/ files are bundled; the artifacts/ dir is preserved."""
    result_dir = tmp_path / "rd"
    result_dir.mkdir()
    seed_result_dir(result_dir)
    artifacts = result_dir / "artifacts"
    artifacts.mkdir()
    (artifacts / "foo.json").write_text("{}")
    out = cli.export_submission(
        result_dir, submitter="alice", output=tmp_path / "sub.tar.gz"
    )
    members = list_tar_members(out)
    assert "artifacts/foo.json" in members


def test_export_submission_missing_summary_json_raises(tmp_path: Path) -> None:
    """Empty result_dir → FileNotFoundError."""
    result_dir = tmp_path / "rd"
    result_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        cli.export_submission(
            result_dir, submitter="alice", output=tmp_path / "sub.tar.gz"
        )


# ----- CLI -----


def test_cli_export_command(tmp_path: Path) -> None:
    """`bs export` exits 0 and writes the tarball."""
    result_dir = tmp_path / "rd"
    result_dir.mkdir()
    seed_result_dir(result_dir)
    out = tmp_path / "sub.tar.gz"
    result = runner.invoke(
        cli.app,
        ["export", str(result_dir), "--submitter", "alice", "-o", str(out)],
    )
    assert result.exit_code == 0
    assert out.exists()
    assert f"wrote {out}" in result.output


def test_cli_export_command_missing_result_dir(tmp_path: Path) -> None:
    """Non-existent result dir → typer rejects with a non-zero exit."""
    result = runner.invoke(
        cli.app,
        ["export", str(tmp_path / "nope"), "--submitter", "alice"],
    )
    assert result.exit_code != 0