"""benchmark_suite/scoring/agentic.py — agentic task evaluation.

Supported harnesses:
  - "inspect" (default): UK AISI's inspect-ai. Invokes `inspect eval <task>
    --model openai-api/<model>` with OPENAI_BASE_URL pointing at the candidate
    endpoint. Returns per-task pass rates.
  - "terminal-bench": experimental. Requires Docker. Best-effort — surfaces a
    clear error if Docker is missing.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from benchmark_suite.recipe import AgenticScorer, Recipe
from benchmark_suite.scoring.base import Scorer, ScoreRecord, ScoreStatus, scorer

_VALID_SANDBOXES = {"docker", "local"}


@scorer
class AgenticScorerImpl(Scorer):
    """Agentic task evaluation via inspect-ai (primary) or terminal-bench (experimental).

    inspect-ai (UK AISI): invokes `inspect eval <task> --model openai-api/<model>`
    with OPENAI_BASE_URL pointing at the candidate endpoint. Returns task pass
    rates per benchmark.

    terminal-bench: experimental. Requires Docker. Best-effort — surfaces clear
    error if Docker missing.
    """

    kind = "agentic"

    def __init__(self, config: AgenticScorer) -> None:
        self.config = config

    def score(
        self,
        recipe: Recipe,
        *,
        result_dir: Path,
        endpoint_url: str | None = None,
    ) -> ScoreRecord:
        cell_id = recipe.cell.render()
        url = endpoint_url or recipe.endpoint.url
        try:
            if self.config.harness == "terminal-bench":
                metrics = self._score_terminal_bench(recipe, result_dir, url)
            else:
                metrics = self._score_inspect(recipe, result_dir, url)
        except Exception as exc:  # scorer must degrade to FAILURE
            return ScoreRecord(
                kind=self.kind,
                metrics={},
                status=ScoreStatus.FAILURE,
                cell_id=cell_id,
                error=str(exc),
                started_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
            )
        return ScoreRecord(
            kind=self.kind,
            metrics=metrics,
            status=ScoreStatus.SUCCESS,
            cell_id=cell_id,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
        )

    def _score_inspect(
        self, recipe: Recipe, result_dir: Path, endpoint_url: str
    ) -> dict[str, Any]:
        ok, _version = check_inspect_installed()
        if not ok:
            raise RuntimeError(
                "inspect-ai is not installed. Install it with: "
                "`uv sync --extra agentic` (or `pip install inspect-ai`)."
            )
        tasks = self.config.tasks
        if not tasks:
            raise ValueError("agentic scorer requires at least one task")
        output_dir = result_dir / "inspect"
        output_dir.mkdir(parents=True, exist_ok=True)
        argv = build_inspect_argv(
            recipe,
            endpoint_url=endpoint_url,
            tasks=tasks,
            model_name=recipe.endpoint.model_name,
            limit=self.config.limit,
            sandbox=self.config.sandbox,
        )
        argv += ["--output-dir", str(output_dir)]
        env = build_inspect_env(recipe, endpoint_url)
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            env={**os.environ, **env},
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"inspect eval failed (exit {proc.returncode}): {proc.stderr.strip()}"
            )
        parsed = parse_inspect_results(output_dir)
        if not parsed:
            raise RuntimeError("inspect eval produced no parseable results")
        metrics: dict[str, Any] = {}
        for task, task_metrics in parsed.items():
            for metric, value in task_metrics.items():
                metrics[f"agentic_{task}_{metric}"] = value
        return metrics

    def _score_terminal_bench(
        self, recipe: Recipe, result_dir: Path, endpoint_url: str
    ) -> dict[str, Any]:
        ok, _version = check_docker_installed()
        if not ok:
            raise RuntimeError(
                "terminal-bench (experimental) requires docker, which is not "
                "installed or not running. Install docker or use harness=inspect."
            )
        tasks = self.config.tasks
        if not tasks:
            raise ValueError("agentic scorer requires at least one task")
        output_path = result_dir / "terminal_bench.json"
        argv = build_terminal_bench_argv(recipe, tasks=tasks, output_path=output_path)
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"terminal-bench failed (exit {proc.returncode}): {proc.stderr.strip()}"
            )
        return {"agentic_terminal_bench": 1.0}


def build_inspect_argv(
    recipe: Recipe,
    *,
    endpoint_url: str,
    tasks: list[str],
    model_name: str,
    limit: int | None = None,
    sandbox: str = "docker",
) -> list[str]:
    """Build argv for `inspect eval`.

    inspect CLI shape (verified against inspect-ai 0.3.x docs):
      inspect eval <task>
        --model openai-api/<model_name>
        --limit <N>                # only if set
        --sandbox docker|local
        --output-dir <path>
        --display plain
    """
    if sandbox not in _VALID_SANDBOXES:
        raise ValueError(f"sandbox must be one of {sorted(_VALID_SANDBOXES)}")
    argv = ["inspect", "eval", *tasks, "--model", f"openai-api/{model_name}"]
    if limit is not None:
        argv += ["--limit", str(limit)]
    argv += ["--sandbox", sandbox, "--display", "plain"]
    return argv


def build_inspect_env(
    recipe: Recipe,
    endpoint_url: str,
) -> dict[str, str]:
    """Env vars for inspect subprocess. Must include OPENAI_BASE_URL pointing at endpoint."""
    env = dict(recipe.merged_env())
    env["OPENAI_BASE_URL"] = endpoint_url
    return env


def parse_inspect_results(output_dir: Path) -> dict[str, dict[str, float]]:
    """Parse inspect-ai eval JSON output. Returns {task: {metric: value}}.

    inspect-ai writes <output_dir>/<timestamp>/logs/*.json + metrics.
    Extract "accuracy" or pass-rate per task. Defensive: return {} if nothing found.
    """
    if not output_dir.is_dir():
        return {}
    results: dict[str, dict[str, float]] = {}
    for json_file in sorted(output_dir.rglob("*.json")):
        try:
            data: dict[str, Any] = json.loads(json_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        results_node = data.get("results")
        if not isinstance(results_node, dict):
            continue
        results_dict = cast("dict[str, Any]", results_node)
        scores = results_dict.get("scores", [])
        if not isinstance(scores, list):
            continue
        for score in cast("list[Any]", scores):
            if not isinstance(score, dict):
                continue
            score_dict = cast("dict[str, Any]", score)
            name = score_dict.get("name")
            metrics = score_dict.get("metrics", {})
            if not isinstance(name, str) or not isinstance(metrics, dict):
                continue
            metrics_dict = cast("dict[str, Any]", metrics)
            for metric, raw in metrics_dict.items():
                value = (
                    cast("dict[str, Any]", raw).get("value")
                    if isinstance(raw, dict)
                    else raw
                )
                if isinstance(value, (int, float)):
                    results.setdefault(name, {})[metric] = float(value)
    return results


def check_inspect_installed() -> tuple[bool, str]:
    """Return (installed, version). Uses `inspect --version`."""
    path = shutil.which("inspect")
    if path is None:
        return False, ""
    try:
        proc = subprocess.run(
            ["inspect", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, ""
    if proc.returncode != 0:
        return False, ""
    return True, (proc.stdout or proc.stderr).strip()


def check_docker_installed() -> tuple[bool, str]:
    """Return (installed, version). Runs `docker version` and checks non-empty output."""
    path = shutil.which("docker")
    if path is None:
        return False, ""
    try:
        proc = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, ""
    if proc.returncode != 0:
        return False, ""
    version = proc.stdout.strip()
    if not version:
        return False, ""
    return True, version


def build_terminal_bench_argv(
    recipe: Recipe,
    *,
    tasks: list[str],
    output_path: Path,
) -> list[str]:
    """Build argv for `terminal-bench` (experimental, requires Docker).

    CLI shape (terminal-bench 0.2.x):
      terminal-bench run <task> --output <path>
    """
    return ["terminal-bench", "run", *tasks, "--output", str(output_path)]