"""benchmark_suite/cli.py — the `bs` entry point: run, run-all, matrix, compare,
report, init, doctor, convert-logits.

This module is the composition root: importing it registers every scorer in
ScorerRegistry so `bs run` can build whichever scorers a recipe lists.

# allow: SIZE_OK — single-file CLI composition root mandated by PLAN.md §1
# ("cli.py — typer app → `bs` entry point (all subcommands)"); the bulk is flat
# command wiring, TypedDict report shapes, and the init template constant.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, TypedDict, assert_never, cast

import typer
import yaml
from pydantic import ValidationError

from benchmark_suite import __version__
from benchmark_suite._matrix import AxisValue, MatrixAxis, expand_matrix
from benchmark_suite.compare import compare_results
from benchmark_suite.paths import result_dir as make_result_dir
from benchmark_suite.recipe import (
    SLUG_RE,
    AgenticScorer,
    BenchSection,
    KLDScorer,
    LLMJudgeScorer,
    PerplexityScorer,
    Recipe,
    ThroughputScorer,
    load_recipe,
)
from benchmark_suite.report import write_summary
from benchmark_suite.runner.endpoint import probe_endpoint
from benchmark_suite.runner.serve import managed_server

# Importing the scorer modules registers them in ScorerRegistry (side effect).
from benchmark_suite.scoring import (  # noqa: F401
    agentic,  # pyright: ignore[reportUnusedImport]
    kl_divergence,  # pyright: ignore[reportUnusedImport]
    llm_judge,  # pyright: ignore[reportUnusedImport]
    perplexity,  # pyright: ignore[reportUnusedImport]
    throughput,  # pyright: ignore[reportUnusedImport]
)
from benchmark_suite.scoring.base import ScoreRecord, ScorerRegistry, ScoreStatus
from benchmark_suite.submission import export_payload, submit_submission
from benchmark_suite.tools.convert_kl_logits import convert_logit_cache

app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    pretty_exceptions_enable=False,
    help="Declarative benchmarking for OpenAI-compatible LLM endpoints.",
)

_BINARIES: tuple[str, ...] = ("llm-perf", "vllm", "lm_eval", "npx", "inspect", "docker")
_DOCTOR_PROBE_TIMEOUT_S: float = 10.0

_RECIPE_TEMPLATE: str = """\
meta:
  name: {name}
  description: ""
  version: "1.0.0"
  tags: []
backend:
  type: vllm                    # change to llamacpp / tgi / external as needed
  model_path: /models/{name}    # fill in
endpoint:
  url: http://127.0.0.1:8000
resources:
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.85
  max_model_len: 4096
  dtype: float16
runtime:
  startup_wait_s: 600
bench:
  load:
    concurrencies: [1, 4, 8]
    num_prompts: 32
    input_len: 256
    output_len: 64
  scoring:
    - kind: throughput
      tool: llm-perf
cell:
  family: dense
  attn: triton
  linear: triton
  cg: 0
  mtp: 0
"""


# ----- doctor report types -----


class BinaryCheck(TypedDict):
    installed: bool
    version: str


class RecipeCheck(TypedDict):
    path: str
    valid: bool
    errors: list[str]


class EndpointCheck(TypedDict):
    url: str
    reachable: bool
    logprobs_supported: bool
    served_models: list[str]
    requested_model_served: bool
    errors: list[str]


class ScorerCheck(TypedDict):
    available: bool
    missing: list[str]


class DoctorReport(TypedDict):
    ok: bool
    recipe: RecipeCheck
    endpoint: EndpointCheck
    binaries: dict[str, BinaryCheck]
    scorers: dict[str, ScorerCheck]
    errors: list[str]


# ----- root callback + version -----


@app.callback()
def main(
    ctx: typer.Context,
    version: Annotated[
        bool, typer.Option("--version", "-V", help="Print version and exit.")
    ] = False,
) -> None:
    """bs — declarative benchmarking for OpenAI-compatible LLM endpoints."""
    if version:
        typer.echo(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command("version")
def version_cmd() -> None:
    """Print version and exit."""
    typer.echo(__version__)


# ----- doctor -----


def _check_binary(name: str) -> BinaryCheck:
    """Probe one binary: on PATH (shutil.which) + `--version` capture."""
    if shutil.which(name) is None:
        return {"installed": False, "version": ""}
    try:
        proc = subprocess.run(
            [name, "--version"],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"installed": False, "version": ""}
    if proc.returncode != 0:
        return {"installed": False, "version": ""}
    combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    version = combined.splitlines()[0] if combined else ""
    return {"installed": True, "version": version}


def _scorer_requirements(bench: BenchSection) -> dict[str, list[str]]:
    """Map each scorer kind in the bench to the binaries it requires."""
    requirements: dict[str, list[str]] = {}
    for cfg in bench.scoring:
        required: list[str]
        match cfg:
            case ThroughputScorer(tool="llm-perf"):
                required = ["llm-perf"]
            case ThroughputScorer():
                required = ["vllm"]
            case PerplexityScorer():
                required = ["lm_eval"]
            case KLDScorer(source="llm-perf"):
                required = ["llm-perf"]
            case KLDScorer():
                required = []
            case LLMJudgeScorer(driver="promptfoo"):
                required = ["npx"]
            case LLMJudgeScorer():
                required = []
            case AgenticScorer(harness="inspect"):
                required = ["inspect"]
            case AgenticScorer():
                required = ["docker"]
            case _:
                assert_never(cfg)
        kind_requirements = requirements.setdefault(cfg.kind, [])
        for name in required:
            if name not in kind_requirements:
                kind_requirements.append(name)
    return requirements


def _doctor_report(recipe_path: Path) -> DoctorReport:
    """Build the doctor report: recipe validity, endpoint probe, binary checks."""
    recipe: Recipe | None = None
    recipe_errors: list[str] = []
    try:
        recipe = load_recipe(recipe_path)
    except yaml.YAMLError as exc:
        recipe_errors.append(f"YAML parse error: {exc}")
    except ValidationError as exc:
        recipe_errors.extend(f"{err['loc']}: {err['msg']}" for err in exc.errors())
    except OSError as exc:
        recipe_errors.append(f"cannot read recipe: {exc}")

    binaries = {name: _check_binary(name) for name in _BINARIES}

    endpoint_check: EndpointCheck = {
        "url": "",
        "reachable": False,
        "logprobs_supported": False,
        "served_models": [],
        "requested_model_served": False,
        "errors": [],
    }
    scorers: dict[str, ScorerCheck] = {}

    if recipe is not None:
        caps = probe_endpoint(
            recipe.endpoint.url,
            api_key_env=recipe.endpoint.api_key_env,
            requested_model=recipe.endpoint.model_name,
            timeout_s=_DOCTOR_PROBE_TIMEOUT_S,
            max_retries=0,
        )
        endpoint_check["url"] = recipe.endpoint.url
        endpoint_check["reachable"] = caps.reachable
        endpoint_check["logprobs_supported"] = caps.logprobs_supported
        endpoint_check["served_models"] = list(caps.served_models)
        endpoint_check["requested_model_served"] = caps.requested_model_served
        if not caps.reachable:
            endpoint_check["errors"].append(
                f"endpoint {recipe.endpoint.url} is not reachable (health check failed)"
            )
        elif recipe.endpoint.model_name and not caps.requested_model_served:
            endpoint_check["errors"].append(
                f"requested model {recipe.endpoint.model_name!r} is not served; "
                f"served models: {list(caps.served_models)}"
            )
        needs_logprobs = any(
            isinstance(cfg, PerplexityScorer | KLDScorer) for cfg in recipe.bench.scoring
        )
        if caps.reachable and needs_logprobs and not caps.logprobs_supported:
            endpoint_check["errors"].append(
                "endpoint does not support logprobs; perplexity/kld scorers require them"
            )

        for kind, required in _scorer_requirements(recipe.bench).items():
            missing = [name for name in required if not binaries[name]["installed"]]
            scorers[kind] = {"available": not missing, "missing": missing}

    errors: list[str] = [*recipe_errors, *endpoint_check["errors"]]
    for kind, check in scorers.items():
        for name in check["missing"]:
            errors.append(f"scorer {kind!r} requires binary {name!r} which is not installed")

    ok = (
        recipe is not None
        and endpoint_check["reachable"]
        and all(check["available"] for check in scorers.values())
    )
    return DoctorReport(
        ok=ok,
        recipe={"path": str(recipe_path), "valid": recipe is not None, "errors": recipe_errors},
        endpoint=endpoint_check,
        binaries=binaries,
        scorers=scorers,
        errors=errors,
    )


@app.command()
def doctor(
    recipe: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True)
    ],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print report and exit 0 regardless.")
    ] = False,
) -> None:
    """Validate recipe + probe endpoint + check binaries."""
    report = _doctor_report(recipe)
    typer.echo(json.dumps(report, indent=2))
    if not dry_run and not report["ok"]:
        raise typer.Exit(code=1)


# ----- run / run-all -----


def _unique_result_dir(recipe: Recipe) -> Path:
    """Allocate a fresh results/<name>_<ts>/<cell>/ dir, bumping the ts on collision."""
    base = datetime.now(UTC)
    offset = 0
    while True:
        rd = make_result_dir(recipe.meta.name, base + timedelta(seconds=offset), recipe.cell)
        if not rd.exists():
            return rd
        offset += 1


def _run_recipe_object(recipe: Recipe) -> int:
    """Execute one in-memory Recipe; 0 = no scorer failed, 1 = any FAILURE."""
    rd = _unique_result_dir(recipe)
    (rd / "logs").mkdir(parents=True, exist_ok=True)
    (rd / "artifacts").mkdir(parents=True, exist_ok=True)

    scores: list[ScoreRecord] = []
    # managed_server yields None for external backends (no lifecycle) and runs
    # the full spawn/wait/teardown cycle (under the GPU file lock) otherwise.
    with managed_server(recipe, log_dir=rd / "logs", gpu_lock=recipe.runtime.gpu_lock):
        for scorer in ScorerRegistry.build_scorers(recipe.bench):
            record = scorer.score(recipe, result_dir=rd, endpoint_url=recipe.endpoint.url)
            scores.append(record)
            typer.echo(f"  [{record.kind}] {record.status}: {len(record.metrics)} metrics")

    write_summary(recipe, scores, rd)
    typer.echo(f"results: {rd}")
    return 1 if any(s.status == ScoreStatus.FAILURE for s in scores) else 0


def _run_recipe(recipe_path: Path) -> int:
    """Load + execute one recipe file; 0 = no scorer failed, 1 = any FAILURE."""
    try:
        recipe = load_recipe(recipe_path)
    except Exception as exc:
        typer.echo(f"error: cannot load recipe {recipe_path}: {exc}", err=True)
        return 1
    return _run_recipe_object(recipe)


@app.command()
def run(
    recipe: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True)
    ],
) -> None:
    """Execute one recipe end-to-end."""
    raise typer.Exit(code=_run_recipe(recipe))


@app.command("run-all")
def run_all(
    glob_pattern: Annotated[
        str, typer.Argument(help="Glob of recipe YAML files, relative to cwd.")
    ],
) -> None:
    """Execute all recipes matching the glob serially."""
    paths = sorted(p for p in Path().glob(glob_pattern) if p.is_file())
    if not paths:
        typer.echo(f"no recipes match glob {glob_pattern!r}", err=True)
        raise typer.Exit(code=1)
    failed: list[str] = []
    for path in paths:
        typer.echo(f"=== {path} ===")
        if _run_recipe(path) != 0:
            failed.append(str(path))
    if failed:
        typer.echo(f"failed recipes: {', '.join(failed)}", err=True)
        raise typer.Exit(code=1)


# ----- matrix -----


def parse_axis_value(raw: str) -> AxisValue:
    """Coerce one axis value: bool literal, then int, else str."""
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        return raw


def parse_axis(spec: str) -> MatrixAxis:
    """Parse ``dotted.path=v1,v2`` into a MatrixAxis."""
    path_str, sep, values_str = spec.partition("=")
    if not sep or not path_str.strip() or not values_str.strip():
        raise typer.BadParameter(f"expected dotted-path=v1,v2 form, got {spec!r}")
    path = tuple(seg.strip() for seg in path_str.strip().split("."))
    values = [
        parse_axis_value(v.strip())
        for v in values_str.split(",")
        if v.strip()
    ]
    if not values:
        raise typer.BadParameter(f"axis {spec!r} has no values")
    return MatrixAxis(path=path, values=values)


def _load_recipe_or_exit(recipe_path: Path) -> Recipe:
    try:
        return load_recipe(recipe_path)
    except Exception as exc:
        typer.echo(f"error: cannot load recipe {recipe_path}: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def matrix(
    recipe: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True)
    ],
    axis: Annotated[
        list[str],
        typer.Option("--axis", help="Axis in dotted-path=v1,v2 form (repeatable)."),
    ],
) -> None:
    """Expand axes and run each variant as a cell."""
    try:
        axes = [parse_axis(spec) for spec in axis]
    except typer.BadParameter as exc:
        typer.echo(str(exc.message), err=True)
        raise typer.Exit(code=1) from exc
    base = _load_recipe_or_exit(recipe)
    variants = expand_matrix(base, axes)
    failed = False
    for idx, variant in enumerate(variants, start=1):
        typer.echo(f"=== cell {idx}/{len(variants)}: {variant.cell.render()} ===")
        if _run_recipe_object(variant) != 0:
            failed = True
    raise typer.Exit(code=1 if failed else 0)


# ----- compare -----


@app.command()
def compare(
    dir_a: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    dir_b: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write delta files here instead of dir_b."),
    ] = None,
) -> None:
    """Compare two result dirs; emit delta.csv + delta.md."""
    paths = compare_results(dir_a, dir_b, output_dir=output)
    typer.echo(f"delta.csv: {paths['csv']}")
    typer.echo(f"delta.md: {paths['md']}")


# ----- report -----


def _record_from_dict(entry: dict[str, Any]) -> ScoreRecord:
    """Rebuild a ScoreRecord from its summary.json to_dict() form."""
    return ScoreRecord(
        kind=str(entry["kind"]),
        cell_id=str(entry["cell_id"]),
        status=str(entry["status"]),
        started_at=datetime.fromisoformat(str(entry["started_at"])),
        finished_at=datetime.fromisoformat(str(entry["finished_at"])),
        metrics={str(k): v for k, v in entry["metrics"].items()},
        artifacts={str(k): str(v) for k, v in entry["artifacts"].items()},
        error=entry["error"],
        notes=dict(entry["notes"]),
    )


@app.command()
def report(
    result_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Regenerate summary.csv + summary.json + README.md from existing artifacts."""
    summary_json = result_dir / "summary.json"
    if not summary_json.exists():
        typer.echo(f"error: no summary.json in {result_dir}; nothing to re-emit", err=True)
        raise typer.Exit(code=1)
    data: Any = json.loads(summary_json.read_text())
    recipe = Recipe.model_validate(data["recipe"])
    scores = [_record_from_dict(entry) for entry in data["scores"]]
    write_summary(recipe, scores, result_dir)
    typer.echo(f"re-emitted summary.csv, summary.json, README.md in {result_dir}")


# ----- init -----


@app.command()
def init(
    name: Annotated[str, typer.Argument(help="Recipe slug; writes recipes/<name>.yaml.")],
) -> None:
    """Scaffold a new recipe at recipes/<name>.yaml."""
    if not SLUG_RE.match(name):
        typer.echo(f"error: name must be a slug [a-z0-9-_], got {name!r}", err=True)
        raise typer.Exit(code=1)
    target = Path("recipes") / f"{name}.yaml"
    if target.exists():
        typer.echo(f"error: already exists: {target}", err=True)
        raise typer.Exit(code=1)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_RECIPE_TEMPLATE.format(name=name))
    typer.echo(f"wrote {target}")


# ----- convert-logits -----


def _load_prompts(path: Path) -> list[str]:
    """Read a JSONL file of {"prompt": ...} objects into a prompt list."""
    prompts: list[str] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: malformed JSON line: {exc}") from exc
        if not isinstance(entry, dict):
            raise ValueError(f"{path}:{lineno}: expected a JSON object with a 'prompt' key")
        prompt = cast("dict[str, object]", entry).get("prompt")
        if not isinstance(prompt, str):
            raise ValueError(f"{path}:{lineno}: expected a JSON object with a 'prompt' key")
        prompts.append(prompt)
    return prompts


@app.command("convert-logits")
def convert_logits_cmd(
    input_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="Directory of f_*.fp16 files.")
    ],
    tokenizer: Annotated[str, typer.Option("--tokenizer", help="HF tokenizer id or path.")],
    model_name: Annotated[
        str, typer.Option("--model-name", help="Name of the model that produced the logits.")
    ],
    prompts: Annotated[
        Path,
        typer.Option(
            "--prompts", exists=True, dir_okay=False, help='JSONL: one {"prompt": ...} per line.'
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output", "-o", file_okay=False, help="Output cache dir (manifest.json lands here)."
        ),
    ],
    max_prompts: Annotated[
        int | None, typer.Option("--max-prompts", help="Convert only the first N prompts.")
    ] = None,
) -> None:
    """Convert f_*.fp16 → safetensors cache + manifest.json."""
    try:
        prompt_list = _load_prompts(prompts)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    manifest = convert_logit_cache(
        input_dir,
        output,
        tokenizer=tokenizer,
        model_name=model_name,
        prompts=prompt_list,
        max_prompts=max_prompts,
    )
    typer.echo(f"wrote {manifest.prompt_count} prompt shards to {output}")
    typer.echo(f"manifest: {output}/manifest.json")


# ----- export -----


@app.command()
def export(
    result_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="Result dir produced by `bs run`.")
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Path to write the localmaxxing JSON payload.")
    ],
    notes: Annotated[str, typer.Option("--notes")] = "",
) -> None:
    """Write the localmaxxing JSON payload that `bs submit` hands to `lmx`.

    Useful for inspecting or diffing payloads without invoking the
    network. Pass the same file to `lmx speed-test submit <file>`
    directly if you prefer to bypass the shell-out wrapper.
    """
    out = export_payload(result_dir=result_dir, output=output, notes=notes)
    typer.echo(f"wrote {out}")


# ----- submit -----


@app.command()
def submit(
    result_dir: Annotated[
        Path,
        typer.Argument(
            exists=True, file_okay=False, help="Result dir produced by `bs run`."
        ),
    ],
    lmx_bin: Annotated[
        str | None,
        typer.Option(
            "--lmx-bin",
            help="Path to the `lmx` binary. Defaults to $PATH lookup.",
        ),
    ] = None,
    endpoint: Annotated[
        str | None,
        typer.Option(
            "--endpoint",
            "-e",
            help="Override the localmaxxing base URL (passed to lmx as --api-url).",
        ),
    ] = None,
    notes: Annotated[
        str, typer.Option("--notes", help="Free-form notes (max 2000 chars).")
    ] = "",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Validate the payload via `lmx speed-test dry-run` "
                "without submitting. Does not consume the 1/min rate limit."
            ),
        ),
    ] = False,
) -> None:
    """Submit a result dir to localmaxxing.com by shelling out to `lmx`.

    Reads `summary.json` from the result dir, maps it to the
    localmaxxing schema, writes it to a temp file, and invokes
    `lmx speed-test submit <file>` (or `lmx speed-test dry-run <file>`
    with `--dry-run`). Authentication is handled by `lmx` itself
    via `$LMX_API_KEY`, `lmx auth --key`, or `~/.config/localmaxxing/`.

    Install `lmx` from
    https://github.com/LottoLottoLotto/localmaxxing-cli/releases/latest
    or `go install github.com/LottoLottoLotto/localmaxxing-cli/cmd/lmx@latest`.
    """
    try:
        result = submit_submission(
            result_dir=result_dir,
            lmx_bin=lmx_bin,
            endpoint=endpoint,
            notes=notes,
            dry_run=dry_run,
        )
    except FileNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if result.get("lmx_not_found"):
        typer.echo(f"error: {result.get('details', 'lmx not found')}", err=True)
        raise typer.Exit(code=1)

    if dry_run:
        if result.get("dry_run_valid"):
            typer.echo("dry-run: payload valid (lmx accepted)")
            typer.echo(result.get("dry_run_stdout", ""))
            return
        typer.echo("dry-run: lmx rejected the payload", err=True)
        typer.echo(result.get("details", ""), err=True)
        raise typer.Exit(code=1)

    if "submission_id" in result:
        sub_id = result.get("submission_id", "")
        public = result.get("public_url", "")
        typer.echo(f"submitted: {sub_id}")
        typer.echo(f"view at:    {public}")
        return

    typer.echo(
        f"error: {result.get('error', 'lmx_failed')} "
        f"(lmx exit {result.get('lmx_exit_code', '?')})",
        err=True,
    )
    typer.echo(f"details: {result.get('details', '')}", err=True)
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()