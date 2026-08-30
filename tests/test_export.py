"""Tests for `bs export` — metadata collection + localmaxxing payload building."""
from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchmark_suite import cli
from benchmark_suite.recipe import (
    BackendSection,
    EndpointSection,
    HardwareSection,
    MetaSection,
    Recipe,
)
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


# ----- build_lmx_payload + export_payload -----


def test_build_lmx_payload_minimal(tmp_path: Path) -> None:
    result_dir = tmp_path / "rd"
    result_dir.mkdir()
    recipe = Recipe(
        meta=MetaSection(name="test", description="x"),
        backend=BackendSection(type="vllm", model_path="Qwen/Qwen3-8B"),
        hardware=HardwareSection(vendor="amd", model="Radeon PRO V620", count=4, vram_gb=32),
    )
    score = ScoreRecord(
        kind="throughput",
        cell_id="dense_triton_triton_cg0_mtp0",
        status="success",
        started_at=T0,
        finished_at=T1,
        metrics={"output_tok_s": 87.4, "input_tok_s": 1210.5, "ttft_mean_ms": 142.5},
    )
    (result_dir / "summary.json").write_text(
        json.dumps({"recipe": recipe.model_dump(mode="json"), "scores": [score.to_dict()]})
    )

    from benchmark_suite.submission import build_lmx_payload

    payload = build_lmx_payload(result_dir=result_dir)
    assert payload["engineName"] == "vllm"
    assert payload["quantization"] == "FP16"
    assert payload["tokSOut"] == 87.4
    assert payload["tokSPrefill"] == 1210.5
    assert payload["ttftMs"] == 142.5
    assert payload["hardware"]["hwClass"] == "DISCRETE_GPU"
    assert payload["hardware"]["gpuName"] == "AMD Radeon PRO V620"
    assert payload["hardware"]["gpuCount"] == 4
    assert payload["hardware"]["vramGb"] == 32
    assert payload["engineFlags"]["cellId"] == "dense_triton_triton_cg0_mtp0"


def test_build_lmx_payload_hf_id_from_model_path(tmp_path: Path) -> None:
    result_dir = tmp_path / "rd"
    result_dir.mkdir()
    recipe_local = Recipe(
        meta=MetaSection(name="test", description="x"),
        backend=BackendSection(type="vllm", model_path="/models/Qwen3-8B"),
        hardware=HardwareSection(vendor="amd", model="V620", count=1, vram_gb=32),
    )
    score = ScoreRecord(
        kind="throughput",
        cell_id="x",
        status="success",
        started_at=T0,
        finished_at=T1,
        metrics={"output_tok_s": 1.0},
    )
    (result_dir / "summary.json").write_text(
        json.dumps({"recipe": recipe_local.model_dump(mode="json"), "scores": [score.to_dict()]})
    )
    from benchmark_suite.submission import build_lmx_payload

    assert build_lmx_payload(result_dir=result_dir)["hfId"] == "Qwen3-8B"

    recipe_hf = Recipe(
        meta=MetaSection(name="test", description="x"),
        backend=BackendSection(type="vllm", model_path="Qwen/Qwen3-8B"),
        hardware=HardwareSection(vendor="amd", model="V620", count=1, vram_gb=32),
    )
    (result_dir / "summary.json").write_text(
        json.dumps({"recipe": recipe_hf.model_dump(mode="json"), "scores": [score.to_dict()]})
    )
    assert build_lmx_payload(result_dir=result_dir)["hfId"] == "Qwen/Qwen3-8B"


def test_build_lmx_payload_omits_optional_fields_when_absent(tmp_path: Path) -> None:
    result_dir = tmp_path / "rd"
    result_dir.mkdir()
    recipe = Recipe(
        meta=MetaSection(name="test", description="x"),
        backend=BackendSection(type="external"),
    )
    score = ScoreRecord(
        kind="throughput",
        cell_id="x",
        status="success",
        started_at=T0,
        finished_at=T1,
        metrics={"output_tok_s": 42.0},
    )
    (result_dir / "summary.json").write_text(
        json.dumps({"recipe": recipe.model_dump(mode="json"), "scores": [score.to_dict()]})
    )
    from benchmark_suite.submission import build_lmx_payload

    payload = build_lmx_payload(result_dir=result_dir)
    assert "tokSPrefill" not in payload
    assert "ttftMs" not in payload
    assert "peakVramGb" not in payload
    assert "hardware" not in payload
    assert payload["tokSOut"] == 42.0


def test_build_lmx_payload_raises_without_output_tok_s(tmp_path: Path) -> None:
    result_dir = tmp_path / "rd"
    result_dir.mkdir()
    recipe = Recipe(
        meta=MetaSection(name="test", description="x"),
        backend=BackendSection(type="external"),
    )
    (result_dir / "summary.json").write_text(
        json.dumps({"recipe": recipe.model_dump(mode="json"), "scores": []})
    )
    from benchmark_suite.submission import build_lmx_payload

    with pytest.raises(ValueError, match="output_tok_s"):
        build_lmx_payload(result_dir=result_dir)


def test_build_lmx_payload_raises_on_missing_summary(tmp_path: Path) -> None:
    result_dir = tmp_path / "rd"
    result_dir.mkdir()
    from benchmark_suite.submission import build_lmx_payload

    with pytest.raises(FileNotFoundError):
        build_lmx_payload(result_dir=result_dir)


def test_build_lmx_payload_notes_truncated_to_2000(tmp_path: Path) -> None:
    result_dir = tmp_path / "rd"
    result_dir.mkdir()
    recipe = Recipe(
        meta=MetaSection(name="test", description="x"),
        backend=BackendSection(type="external"),
    )
    score = ScoreRecord(
        kind="throughput",
        cell_id="x",
        status="success",
        started_at=T0,
        finished_at=T1,
        metrics={"output_tok_s": 1.0},
    )
    (result_dir / "summary.json").write_text(
        json.dumps({"recipe": recipe.model_dump(mode="json"), "scores": [score.to_dict()]})
    )
    from benchmark_suite.submission import build_lmx_payload

    long_notes = "x" * 3000
    payload = build_lmx_payload(result_dir=result_dir, notes=long_notes)
    assert len(payload["notes"]) == 2000


def test_export_payload_writes_json_file(tmp_path: Path) -> None:
    result_dir = tmp_path / "rd"
    result_dir.mkdir()
    recipe = Recipe(
        meta=MetaSection(name="test", description="x"),
        backend=BackendSection(type="vllm", model_path="Qwen/Qwen3-8B"),
        hardware=HardwareSection(vendor="amd", model="Radeon PRO V620", count=1, vram_gb=32),
    )
    score = ScoreRecord(
        kind="throughput",
        cell_id="x",
        status="success",
        started_at=T0,
        finished_at=T1,
        metrics={"output_tok_s": 87.4},
    )
    (result_dir / "summary.json").write_text(
        json.dumps({"recipe": recipe.model_dump(mode="json"), "scores": [score.to_dict()]})
    )
    out = tmp_path / "payload.json"
    from benchmark_suite.submission import export_payload

    written = export_payload(result_dir=result_dir, output=out)
    assert written == out
    body = json.loads(out.read_text())
    assert body["tokSOut"] == 87.4
    assert body["engineName"] == "vllm"


# ----- CLI -----


def test_cli_export_command(tmp_path: Path) -> None:
    result_dir = tmp_path / "rd"
    result_dir.mkdir()
    seed_result_dir(result_dir)
    out = tmp_path / "payload.json"
    result = runner.invoke(
        cli.app,
        ["export", str(result_dir), "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert f"wrote {out}" in result.output


def test_cli_export_command_missing_result_dir(tmp_path: Path) -> None:
    """Non-existent result dir → typer rejects with a non-zero exit."""
    result = runner.invoke(
        cli.app,
        ["export", str(tmp_path / "nope"), "-o", str(tmp_path / "p.json")],
    )
    assert result.exit_code != 0