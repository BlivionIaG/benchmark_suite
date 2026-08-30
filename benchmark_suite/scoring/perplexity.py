"""benchmark_suite/scoring/perplexity.py — perplexity scorer via lm-eval-harness.

Runs EleutherAI's lm-eval-harness as a subprocess with ``--model
local-completions`` (NOT ``local-chat-completions`` — the chat-completions
endpoint does not expose prompt logprobs in most API implementations, while
``/v1/completions`` does, and vLLM serves both). Captures per-task perplexity
from the ``results.json`` lm_eval writes into the output directory.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, cast

from benchmark_suite.recipe import PerplexityScorer, Recipe
from benchmark_suite.scoring.base import Scorer, ScoreRecord, ScoreStatus, scorer

_LM_EVAL_INSTALL_HINT = (
    "lm_eval not installed. Install with: uv add --optional perplexity lm-eval==0.4.12"
)
_DEFAULT_TIMEOUT_S = 1800.0


def _ensure_v1(url: str) -> str:
    """Append ``/v1`` to an endpoint URL if not already present."""
    u = url.rstrip("/")
    return u if u.endswith("/v1") else u + "/v1"


def build_lm_eval_argv(
    recipe: Recipe,
    *,
    endpoint_url: str,
    output_path: Path,
    tasks: list[str],
    num_fewshot: int = 0,
    limit: int | None = None,
    extra_args: list[str] | None = None,
    lm_eval_path: str = "lm_eval",
) -> list[str]:
    """Construct the lm_eval subprocess argv.

    Standard incantation::

        lm_eval
          --model local-completions
          --model_args base_url=<endpoint>/v1,model=<recipe.endpoint.model_name>
          --tasks <task1,task2,...>
          --num_fewshot <N>
          --output_path <output_path>
          [--limit <N>]    # only if set
          [<extra_args>]

    We use ``--model local-completions`` (NOT ``local-chat-completions``)
    because the chat-completions endpoint does not expose prompt logprobs in
    most API implementations; ``/v1/completions`` does, and vLLM supports both.
    """
    base_url = _ensure_v1(endpoint_url)
    model_args = f"base_url={base_url},model={recipe.endpoint.model_name}"
    argv = [
        lm_eval_path,
        "--model",
        "local-completions",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--num_fewshot",
        str(num_fewshot),
        "--output_path",
        str(output_path),
    ]
    if limit is not None:
        argv.extend(["--limit", str(limit)])
    if extra_args:
        argv.extend(extra_args)
    return argv


def parse_lm_eval_results(output_path: Path) -> dict[str, dict[str, float]]:
    """Parse lm_eval results JSON. Returns ``{task_name: {metric_name: value}}``.

    lm_eval output format: ``results`` is a dict keyed by task name; each value
    has subkeys like ``ppl``, ``ppl_stderr``, ``acc,none``, etc. We extract the
    ``results`` dict verbatim (defensive: empty dict if the file is absent or
    malformed).
    """
    results_file = output_path / "results.json"
    if not results_file.exists():
        return {}
    try:
        raw: Any = json.loads(results_file.read_text())
    except (ValueError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    raw_dict = cast(dict[str, Any], raw)
    results = raw_dict.get("results")
    if not isinstance(results, dict):
        return {}
    typed_results = cast(dict[str, dict[str, Any]], results)
    out: dict[str, dict[str, float]] = {}
    for task, metrics in typed_results.items():
        out[task] = {
            str(k): float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
    return out


def check_lm_eval_installed() -> tuple[bool, str]:
    """Returns ``(True, version_string)`` if lm_eval is on PATH, else ``(False, '')``."""
    try:
        proc = subprocess.run(
            ["lm_eval", "--version"],
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, ""
    if proc.returncode != 0:
        return False, ""
    return True, (proc.stdout or "").strip()


@scorer
class PerplexityScorerImpl(Scorer):
    """Runs lm-eval-harness via subprocess, model=local-completions, captures per-task ppl."""

    kind = "perplexity"

    def __init__(self, config: PerplexityScorer) -> None:
        self.config = config

    def score(
        self,
        recipe: Recipe,
        *,
        result_dir: Path,
        endpoint_url: str | None = None,
    ) -> ScoreRecord:
        cell_id = recipe.cell.render()

        ok, _version = check_lm_eval_installed()
        if not ok:
            return ScoreRecord(
                kind=self.kind,
                cell_id=cell_id,
                status=ScoreStatus.FAILURE,
                error=_LM_EVAL_INSTALL_HINT,
            )

        base_url = _ensure_v1(endpoint_url or recipe.endpoint.base_url_v1)
        output_path = result_dir / "lm_eval"
        argv = build_lm_eval_argv(
            recipe,
            endpoint_url=base_url,
            output_path=output_path,
            tasks=self.config.tasks,
            num_fewshot=self.config.num_fewshot,
            limit=self.config.limit,
            extra_args=self.config.lm_eval_extra_args or None,
        )

        env = {**os.environ, **recipe.merged_env()}
        try:
            subprocess.run(
                argv,
                check=True,
                env=env,
                timeout=_DEFAULT_TIMEOUT_S,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            return ScoreRecord(
                kind=self.kind,
                cell_id=cell_id,
                status=ScoreStatus.FAILURE,
                error=f"lm_eval failed: {exc}",
            )

        results = parse_lm_eval_results(output_path)
        metrics: dict[str, float | int | str] = {}
        for task, task_metrics in results.items():
            if "ppl" in task_metrics:
                metrics[f"perplexity_{task}"] = task_metrics["ppl"]

        artifacts: dict[str, str] = {}
        if (output_path / "results.json").exists():
            artifacts["results.json"] = "lm_eval/results.json"

        return ScoreRecord(
            kind=self.kind,
            cell_id=cell_id,
            status=ScoreStatus.SUCCESS,
            metrics=metrics,
            artifacts=artifacts,
        )