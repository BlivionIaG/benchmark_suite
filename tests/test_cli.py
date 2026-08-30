"""Tests for benchmark_suite.cli — the `bs` entry point (CliRunner, mocked machinery)."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
import yaml
from typer.testing import CliRunner

from benchmark_suite import __version__, cli
from benchmark_suite._matrix import MatrixAxis
from benchmark_suite.recipe import BenchSection, Recipe, load_recipe
from benchmark_suite.report import write_summary
from benchmark_suite.scoring.base import Scorer, ScoreRecord, ScorerRegistry, ScoreStatus
from benchmark_suite.tools.convert_kl_logits import KLCacheManifest

BASE_URL = "http://127.0.0.1:8000"
T0 = datetime(2026, 8, 30, 12, 0, 0, tzinfo=UTC)
T1 = datetime(2026, 8, 30, 12, 5, 0, tzinfo=UTC)

runner = CliRunner()


# ----- shared helpers -----


class StubScorer(Scorer):
    """Deterministic scorer stub returning a canned ScoreRecord."""

    kind = "stub"

    def __init__(self, status: str) -> None:
        super().__init__(config=None)
        self._status = status

    def score(
        self,
        recipe: Recipe,
        *,
        result_dir: Path,
        endpoint_url: str | None = None,
    ) -> ScoreRecord:
        del result_dir, endpoint_url
        return ScoreRecord(
            kind=self.kind,
            cell_id=recipe.cell.render(),
            status=self._status,
            started_at=T0,
            finished_at=T1,
            metrics={"output_tok_s": 42.0},
        )


def install_stub_scorer(monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    """Replace ScorerRegistry.build_scorers with one StubScorer of the given status."""
    stub = StubScorer(status)

    def build(cls: type[ScorerRegistry], bench: BenchSection) -> list[Scorer]:
        del cls, bench
        return [stub]

    monkeypatch.setattr(ScorerRegistry, "build_scorers", classmethod(build))


def write_recipe(path: Path, data: dict[str, Any]) -> Path:
    """Write a recipe dict to a YAML file."""
    path.write_text(yaml.safe_dump(data))
    return path


def seed_result_dir(tmp_path: Path, name: str, output_tok_s: float) -> Path:
    """Create a result dir with one SUCCESS throughput score via write_summary."""
    recipe = Recipe.model_validate({"meta": {"name": name}})
    score = ScoreRecord(
        kind="throughput",
        cell_id=recipe.cell.render(),
        status=ScoreStatus.SUCCESS,
        started_at=T0,
        finished_at=T1,
        metrics={"output_tok_s": output_tok_s, "total_tok_s": output_tok_s * 2},
    )
    rd = tmp_path / name
    write_summary(recipe, [score], rd)
    return rd


# ----- fixtures -----


@pytest.fixture
def fake_binaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make all binary checks fast + deterministic (no real subprocess)."""

    def fake_check(name: str) -> cli.BinaryCheck:
        del name
        return {"installed": True, "version": "1.0.0"}

    monkeypatch.setattr(cli, "_check_binary", fake_check)


@pytest.fixture
def mock_endpoint_down(respx_mock: respx.MockRouter) -> respx.MockRouter:
    """GET /health -> 502 (unreachable)."""
    respx_mock.get(f"{BASE_URL}/health").mock(return_value=httpx.Response(502))
    return respx_mock


@pytest.fixture
def mock_endpoint_up(respx_mock: respx.MockRouter) -> respx.MockRouter:
    """Health + models + logprobs all OK; serves 'test-model'."""
    respx_mock.get(f"{BASE_URL}/health").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    respx_mock.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "test-model"}]})
    )
    respx_mock.post(f"{BASE_URL}/v1/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"text": "t", "logprobs": {"tokens": ["t"], "token_logprobs": [-0.1]}}
                ]
            },
        )
    )
    return respx_mock


@pytest.fixture
def doctor_recipe_path(tmp_path: Path) -> Path:
    """Valid recipe against the mocked endpoint with a throughput (llm-perf) scorer."""
    return write_recipe(
        tmp_path / "doctor.yaml",
        {
            "meta": {"name": "doctor-test"},
            "endpoint": {"url": BASE_URL, "model_name": "test-model"},
            "bench": {"scoring": [{"kind": "throughput", "tool": "llm-perf"}]},
        },
    )


@pytest.fixture
def run_recipe_path(tmp_path: Path) -> Path:
    """Minimal valid recipe (external backend, everything else defaults)."""
    return write_recipe(tmp_path / "run.yaml", {"meta": {"name": "run-test"}})


# ----- version / help -----


def test_version_command() -> None:
    result = runner.invoke(cli.app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_version_flag() -> None:
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_args_prints_help() -> None:
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0
    assert "Usage" in result.output


# ----- doctor -----


def test_doctor_dry_run_exits_0_even_if_endpoint_down(
    doctor_recipe_path: Path,
    mock_endpoint_down: respx.MockRouter,
    fake_binaries: None,
) -> None:
    result = runner.invoke(cli.app, ["doctor", str(doctor_recipe_path), "--dry-run"])
    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["endpoint"]["reachable"] is False
    assert report["ok"] is False


def test_doctor_no_dry_run_exits_nonzero_if_endpoint_down(
    doctor_recipe_path: Path,
    mock_endpoint_down: respx.MockRouter,
    fake_binaries: None,
) -> None:
    result = runner.invoke(cli.app, ["doctor", str(doctor_recipe_path)])
    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report["endpoint"]["reachable"] is False


def test_doctor_recipe_load_failure(tmp_path: Path, fake_binaries: None) -> None:
    bad = write_recipe(tmp_path / "bad.yaml", {"meta": {"name": "Bad Name!"}})
    result = runner.invoke(cli.app, ["doctor", str(bad)])
    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report["recipe"]["valid"] is False
    assert report["recipe"]["errors"] != []
    assert report["ok"] is False


def test_doctor_reports_missing_binaries(
    doctor_recipe_path: Path,
    mock_endpoint_up: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_check(name: str) -> cli.BinaryCheck:
        if name == "llm-perf":
            return {"installed": False, "version": ""}
        return {"installed": True, "version": "1.0.0"}

    monkeypatch.setattr(cli, "_check_binary", fake_check)
    result = runner.invoke(cli.app, ["doctor", str(doctor_recipe_path)])
    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report["binaries"]["llm-perf"]["installed"] is False
    assert report["scorers"]["throughput"]["missing"] == ["llm-perf"]
    assert report["ok"] is False


def test_doctor_ok_when_everything_present(
    doctor_recipe_path: Path,
    mock_endpoint_up: respx.MockRouter,
    fake_binaries: None,
) -> None:
    result = runner.invoke(cli.app, ["doctor", str(doctor_recipe_path)])
    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["ok"] is True
    assert report["recipe"]["valid"] is True
    assert report["endpoint"]["reachable"] is True
    assert report["endpoint"]["requested_model_served"] is True
    assert report["endpoint"]["logprobs_supported"] is True
    assert report["scorers"]["throughput"]["available"] is True


# ----- run -----


def test_run_recipe_loads_and_executes_scorers(
    run_recipe_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_stub_scorer(monkeypatch, ScoreStatus.SUCCESS)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["run", str(run_recipe_path)])
    assert result.exit_code == 0
    summaries = list((tmp_path / "results").glob("*/*/summary.csv"))
    assert len(summaries) == 1
    rd = summaries[0].parent
    assert (rd / "summary.json").exists()
    assert (rd / "README.md").exists()
    assert (rd / "logs").is_dir()
    assert (rd / "artifacts").is_dir()
    data = json.loads((rd / "summary.json").read_text())
    assert data["scores"][0]["kind"] == "stub"
    assert data["scores"][0]["status"] == "success"


def test_run_recipe_failure_exits_1(
    run_recipe_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_stub_scorer(monkeypatch, ScoreStatus.FAILURE)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["run", str(run_recipe_path)])
    assert result.exit_code == 1
    summaries = list((tmp_path / "results").glob("*/*/summary.csv"))
    assert len(summaries) == 1  # result dir still produced on failure


def test_run_recipe_invalid_recipe_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = write_recipe(tmp_path / "bad.yaml", {"meta": {"name": "Bad Name!"}})
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["run", str(bad)])
    assert result.exit_code == 1


# ----- run-all -----


def test_run_all_iterates_glob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_stub_scorer(monkeypatch, ScoreStatus.SUCCESS)
    for name in ("alpha", "beta"):
        write_recipe(tmp_path / f"{name}.yaml", {"meta": {"name": name}})
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["run-all", "*.yaml"])
    assert result.exit_code == 0
    top = tmp_path / "results"
    run_names = sorted(d.name for d in top.iterdir())
    assert any(n.startswith("alpha_") for n in run_names)
    assert any(n.startswith("beta_") for n in run_names)


def test_run_all_no_match_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["run-all", "nope-*.yaml"])
    assert result.exit_code == 1


# ----- matrix -----


def test_matrix_axis_parsing() -> None:
    axis = cli.parse_axis("cell.attn=triton,fardna2")
    assert axis == MatrixAxis(path=("cell", "attn"), values=["triton", "fardna2"])


def test_matrix_axis_parsing_coerces_int_and_bool() -> None:
    assert cli.parse_axis("cell.cg=0,1").values == [0, 1]
    assert cli.parse_axis("runtime.gpu_lock=true,false").values == [True, False]


def test_matrix_expands_axes_and_runs_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_stub_scorer(monkeypatch, ScoreStatus.SUCCESS)
    recipe = write_recipe(tmp_path / "matrix.yaml", {"meta": {"name": "matrix-test"}})
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        cli.app, ["matrix", str(recipe), "--axis", "cell.attn=triton,fardna2"]
    )
    assert result.exit_code == 0
    summaries = list((tmp_path / "results").glob("*/*/summary.csv"))
    assert len(summaries) == 2
    cell_names = {s.parent.name for s in summaries}
    assert cell_names == {"dense_triton_triton_cg0_mtp0", "dense_fardna2_triton_cg0_mtp0"}


def test_matrix_bad_axis_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = write_recipe(tmp_path / "matrix.yaml", {"meta": {"name": "matrix-test"}})
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["matrix", str(recipe), "--axis", "no-equals-sign"])
    assert result.exit_code != 0


# ----- compare -----


def test_compare_creates_delta_artifacts(tmp_path: Path) -> None:
    a = seed_result_dir(tmp_path, "a", output_tok_s=10.0)
    b = seed_result_dir(tmp_path, "b", output_tok_s=12.0)
    result = runner.invoke(cli.app, ["compare", str(a), str(b)])
    assert result.exit_code == 0
    assert (b / "delta.csv").exists()
    assert (b / "delta.md").exists()
    delta_csv = (b / "delta.csv").read_text()
    assert "output_tok_s" in delta_csv


# ----- report -----


def test_report_regenerates_from_existing_artifacts(tmp_path: Path) -> None:
    recipe = Recipe.model_validate({"meta": {"name": "report-test"}})
    score = ScoreRecord(
        kind="throughput",
        cell_id="dense_triton_triton_cg0_mtp0",
        status=ScoreStatus.SUCCESS,
        started_at=T0,
        finished_at=T1,
        metrics={"output_tok_s": 5.0},
    )
    rd = tmp_path / "rd"
    write_summary(recipe, [score], rd)
    original_csv = (rd / "summary.csv").read_text()
    (rd / "summary.csv").unlink()
    (rd / "README.md").unlink()

    result = runner.invoke(cli.app, ["report", str(rd)])
    assert result.exit_code == 0
    assert (rd / "summary.csv").read_text() == original_csv
    assert (rd / "README.md").exists()


def test_report_missing_summary_json_exits_1(tmp_path: Path) -> None:
    rd = tmp_path / "empty"
    rd.mkdir()
    result = runner.invoke(cli.app, ["report", str(rd)])
    assert result.exit_code == 1


# ----- init -----


def test_init_scaffolds_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["init", "foo"])
    assert result.exit_code == 0
    target = tmp_path / "recipes" / "foo.yaml"
    assert target.exists()
    recipe = load_recipe(target)
    assert recipe.meta.name == "foo"
    assert recipe.backend.type == "vllm"
    assert recipe.bench.scoring[0].kind == "throughput"


def test_init_refuses_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(cli.app, ["init", "foo"]).exit_code == 0
    result = runner.invoke(cli.app, ["init", "foo"])
    assert result.exit_code == 1


def test_init_rejects_non_slug_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["init", "Bad Name"])
    assert result.exit_code == 1
    assert not (tmp_path / "recipes" / "Bad Name.yaml").exists()


# ----- convert-logits -----


def _fake_manifest(tokenizer: str, model_name: str, prompt_count: int) -> KLCacheManifest:
    return KLCacheManifest(
        model_name=model_name,
        tokenizer_id=tokenizer,
        vocab_size=100,
        prompt_count=prompt_count,
        max_prompt_tokens=4,
        shape_per_prompt=[(4, 100)],
        dtype="float16",
        created_at="2026-08-30T00:00:00+00:00",
        source_files=["f_0.fp16"],
    )


def test_convert_logits_calls_convert_logit_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "logits"
    input_dir.mkdir()
    (input_dir / "f_0.fp16").write_bytes(b"fake")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        '{"prompt": "What is the capital of France?"}\n{"prompt": "2+2?"}\n'
    )
    out = tmp_path / "out-cache"
    captured: dict[str, object] = {}

    def fake_convert(
        input_dir: Path,
        output_dir: Path,
        *,
        tokenizer: str,
        model_name: str,
        prompts: list[str],
        max_prompts: int | None,
    ) -> KLCacheManifest:
        captured["input_dir"] = input_dir
        captured["output_dir"] = output_dir
        captured["tokenizer"] = tokenizer
        captured["model_name"] = model_name
        captured["prompts"] = prompts
        captured["max_prompts"] = max_prompts
        return _fake_manifest(tokenizer, model_name, len(prompts))

    monkeypatch.setattr(cli, "convert_logit_cache", fake_convert)
    result = runner.invoke(
        cli.app,
        [
            "convert-logits",
            str(input_dir),
            "--tokenizer",
            "Qwen/Qwen3-0.6B",
            "--model-name",
            "fp16-ref",
            "--prompts",
            str(prompts),
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0
    assert captured["input_dir"] == input_dir
    assert captured["output_dir"] == out
    assert captured["tokenizer"] == "Qwen/Qwen3-0.6B"
    assert captured["model_name"] == "fp16-ref"
    assert captured["prompts"] == ["What is the capital of France?", "2+2?"]
    assert captured["max_prompts"] is None
    assert "2 prompt shards" in result.output


def test_convert_logits_malformed_jsonl_exits_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "logits"
    input_dir.mkdir()
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "ok"}\nnot-json\n')
    out = tmp_path / "out-cache"
    called = False

    def fake_convert(
        input_dir: Path,
        output_dir: Path,
        *,
        tokenizer: str,
        model_name: str,
        prompts: list[str],
        max_prompts: int | None,
    ) -> KLCacheManifest:
        del input_dir, output_dir, tokenizer, model_name, prompts, max_prompts
        nonlocal called
        called = True
        return _fake_manifest("t", "m", 1)

    monkeypatch.setattr(cli, "convert_logit_cache", fake_convert)
    result = runner.invoke(
        cli.app,
        [
            "convert-logits",
            str(input_dir),
            "--tokenizer",
            "t",
            "--model-name",
            "m",
            "--prompts",
            str(prompts),
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 1
    assert called is False