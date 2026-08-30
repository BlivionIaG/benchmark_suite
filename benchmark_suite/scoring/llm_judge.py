"""benchmark_suite/scoring/llm_judge.py — LLM-judge scorer (native + promptfoo)."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, cast

import httpx
import yaml

from benchmark_suite.recipe import LLMJudgeScorer, Recipe
from benchmark_suite.scoring.base import Scorer, ScoreRecord, ScoreStatus, scorer

_SCORE_RE = re.compile(r'score["\s:]+(\d+)')
_RATIONALE_RE = re.compile(r'rationale["\s:]+([^"\\}]+)')


@scorer
class LLMJudgeScorerImpl(Scorer):
    """Uses a judge LLM (typically local vLLM endpoint) to score candidate answers.

    Two drivers:
      - "native" (default): httpx POST to judge_url + judge_model, parses a
        rubric response, returns 0-10 score + rationale. Zero Node dependency.
      - "promptfoo": generates a promptfooconfig.yaml + invokes `npx promptfoo@<pin>`.
        Optional path; requires Node.

    The judge endpoint may be the same as the candidate endpoint (self-judge)
    or different (recommended). Schema allows any judge_url.
    """

    kind = "llm_judge"

    def __init__(self, config: LLMJudgeScorer) -> None:
        self.config = config

    def score(
        self,
        recipe: Recipe,
        *,
        result_dir: Path,
        endpoint_url: str | None = None,
    ) -> ScoreRecord:
        cell_id = recipe.cell.render()
        try:
            if self.config.driver == "promptfoo":
                judge_score = self._score_promptfoo(recipe, result_dir)
            else:
                judge_score = self._score_native(recipe, result_dir)
        except Exception as exc:  # scorer must degrade to FAILURE, never raise
            return ScoreRecord(
                kind=self.kind,
                metrics={},
                status=ScoreStatus.FAILURE,
                cell_id=cell_id,
                error=str(exc),
            )
        return ScoreRecord(
            kind=self.kind,
            metrics={"judge_score": judge_score},
            status=ScoreStatus.SUCCESS,
            cell_id=cell_id,
        )

    def _score_native(self, recipe: Recipe, result_dir: Path) -> float:
        prompts = self._load_prompts()
        api_key = os.environ.get(self.config.judge_api_key_env, "")
        scores: list[int] = []
        for item in prompts:
            result = native_judge(
                self.config.judge_url,
                self.config.judge_model,
                api_key,
                prompt=item["prompt"],
                candidate_answer=item["candidate_answer"],
                rubric=self.config.rubric,
            )
            scores.append(int(result["score"]))
        if not scores:
            raise ValueError("no prompts to judge")
        return sum(scores) / len(scores)

    def _score_promptfoo(self, recipe: Recipe, result_dir: Path) -> float:
        prompts = self._load_prompts()
        config_path = build_promptfoo_config(
            recipe, self.config, prompts, result_dir
        )
        output = run_promptfoo(
            config_path, promptfoo_version=self.config.promptfoo_version
        )
        results = output.get("results", {}).get("results", [])
        if not results:
            raise ValueError("promptfoo returned no results")
        scores = [float(r.get("score", 0.0)) for r in results]
        return sum(scores) / len(scores)

    def _load_prompts(self) -> list[dict[str, str]]:
        if self.config.prompts_file is None:
            raise ValueError("llm_judge requires prompts_file")
        path = self.config.prompts_file
        prompts: list[dict[str, str]] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(
                {
                    "prompt": str(obj["prompt"]),
                    "candidate_answer": str(obj.get("candidate_answer", "")),
                }
            )
        return prompts


def native_judge(
    judge_url: str,
    judge_model: str,
    judge_api_key: str,
    *,
    prompt: str,
    candidate_answer: str,
    rubric: str,
    timeout_s: float = 120.0,
    max_retries: int = 3,
) -> dict[str, Any]:
    """POST to {judge_url}/chat/completions with a rubric prompt; parse response.

    Returns: {"score": int, "rationale": str, "raw": str} on success; raises on failure.
    """
    judge_prompt = (
        f"{rubric}\n\n"
        f"Candidate answer:\n{candidate_answer}\n\n"
        f"Original prompt (for context):\n{prompt}\n\n"
        'Respond with JSON: {"score": <int 0-10>, "rationale": "<short explanation>"}.'
    )
    headers = {"Authorization": f"Bearer {judge_api_key}"} if judge_api_key else {}
    url = judge_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "user", "content": judge_prompt},
        ],
        "temperature": 0.0,
    }

    last_error: Exception | None = None
    with httpx.Client(timeout=timeout_s) as client:
        for _ in range(max_retries):
            try:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                parsed = parse_judge_response(content)
                return {
                    "score": parsed["score"],
                    "rationale": parsed["rationale"],
                    "raw": content,
                }
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                last_error = exc
    raise RuntimeError(f"judge request failed after {max_retries} attempts: {last_error}")


def parse_judge_response(response_text: str) -> dict[str, Any]:
    """Parse the judge's JSON response. Tolerate surrounding text/markdown fences.

    Returns: {"score": int, "rationale": str}; raises ValueError if score not found.
    """
    text = response_text.strip()

    # 1. Direct JSON parse.
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "score" in data:
            obj = cast(dict[str, Any], data)
            return {
                "score": int(obj["score"]),
                "rationale": str(obj.get("rationale", "")),
            }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 2. Find the first {...} block (strip markdown fences first).
    fenced = re.sub(r"```(?:json)?\s*", "", text)
    for match in re.finditer(r"\{.*?\}", fenced, flags=re.DOTALL):
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict) and "score" in data:
                obj = cast(dict[str, Any], data)
                return {
                    "score": int(obj["score"]),
                    "rationale": str(obj.get("rationale", "")),
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    # 3. Regex fallback.
    score_match = _SCORE_RE.search(text)
    if score_match is None:
        raise ValueError("could not extract score from judge response")
    score = int(score_match.group(1))
    rationale = ""
    rationale_match = _RATIONALE_RE.search(text)
    if rationale_match is not None:
        rationale = rationale_match.group(1).strip()
    return {"score": score, "rationale": rationale}


def build_promptfoo_config(
    recipe: Recipe,
    judge_config: LLMJudgeScorer,
    prompts: list[dict[str, str]],
    output_path: Path,
) -> Path:
    """Generate a promptfooconfig.yaml from the recipe + judge config + prompts."""
    output_path.mkdir(parents=True, exist_ok=True)
    config_path = output_path / "promptfooconfig.yaml"

    provider: dict[str, Any] = {
        "id": f"openai:chat:{judge_config.judge_model}",
        "config": {
            "apiBaseUrl": judge_config.judge_url,
        },
    }
    if judge_config.judge_api_key_env:
        provider["config"]["apiKeyEnvar"] = judge_config.judge_api_key_env

    tests: list[dict[str, Any]] = []
    for item in prompts:
        tests.append(
            {
                "vars": {
                    "prompt": item["prompt"],
                    "candidate_answer": item["candidate_answer"],
                },
                "assert": [
                    {
                        "type": "llm-rubric",
                        "value": judge_config.rubric,
                    }
                ],
            }
        )

    config: dict[str, Any] = {
        "providers": [provider],
        "prompts": ["{{candidate_answer}}"],
        "tests": tests,
        "defaultTest": {
            "options": {
                "provider": provider["id"],
            }
        },
    }

    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def run_promptfoo(
    config_path: Path,
    *,
    promptfoo_version: str = "0.118.5",
    timeout_s: float = 1800.0,
) -> dict[str, Any]:
    """Invoke `npx promptfoo@<pin> eval -c <config> -o json`. Returns parsed JSON output."""
    cmd = [
        "npx",
        f"promptfoo@{promptfoo_version}",
        "eval",
        "-c",
        str(config_path),
        "-o",
        "json",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"promptfoo eval failed (exit {proc.returncode}): {proc.stderr.strip()}"
        )
    return json.loads(proc.stdout)