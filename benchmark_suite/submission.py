"""benchmark_suite/submission.py — build a localmaxxing.com speed-test JSON
payload from a result dir and submit it via the official `lmx` CLI.

`bs submit` builds the JSON payload that `lmx speed-test submit` consumes,
writes it to a temp file, and shells out to `lmx`. The localmaxxing
HTTP API is not called directly; `lmx` is the canonical client and is
maintained by the localmaxxing team.

The recipe's `hardware:` block maps to the `hardware` field of the
localmaxxing payload (`hwClass: DISCRETE_GPU`). The recipe's
`quantization:` field maps directly. The recipe's `backend.type`
maps to the canonical `engineName` (`vllm` / `llama.cpp` / `tgi`).
A free-form `notes` string can be passed via `--notes` for context.

Authentication is handled entirely by `lmx` (`$LMX_API_KEY` env var,
or `lmx auth --key ...`, or `~/.config/localmaxxing/config.json`). We
do not accept or pass an API key directly.

Install `lmx`: see https://github.com/LottoLottoLotto/localmaxxing-cli
(release binaries, or `go install github.com/LottoLottoLotto/localmaxxing-cli/cmd/lmx@latest`).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

if TYPE_CHECKING:
    from benchmark_suite.recipe import Recipe


class SubmitResult(TypedDict, total=False):
    """Result of `bs submit`. Exactly one of `submission_id`, `error`,
    or `lmx_not_found` is set on the happy / error path.

    - submission_id + public_url + lmx_stdout: lmx submitted successfully.
    - dry_run_valid + dry_run_stdout: lmx dry-run validated the payload.
    - error + details + lmx_stderr: lmx exited non-zero; payload was rejected.
    - lmx_not_found: `lmx` binary not on $PATH (and --lmx-bin not set).
    """

    submission_id: str
    public_url: str
    endpoint: str
    error: str
    details: str
    lmx_stdout: str
    lmx_stderr: str
    lmx_exit_code: int
    dry_run_valid: bool
    dry_run_stdout: str
    lmx_not_found: bool


def build_lmx_payload(
    *,
    result_dir: Path,
    recipe: Recipe | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Build the JSON payload for `lmx speed-test submit` from a result dir.

    Reads `summary.json` from the result dir and maps it to the
    localmaxxing schema (hardware + metrics + engine + quantization +
    notes). The only required field is `tokSOut`; everything else is
    best-effort and omitted when absent.

    Args:
        result_dir: the result dir produced by `bs run`; must contain
            `summary.json`.
        recipe: optional pre-loaded `Recipe` instance. When None, the recipe
            is re-loaded from `summary.json["recipe"]`.
        notes: free-form notes (max 2000 chars per localmaxxing schema).

    Raises:
        FileNotFoundError: if `summary.json` is missing.
        ValueError: if no throughput score with `output_tok_s` is present.
    """
    summary_json = result_dir / "summary.json"
    if not summary_json.exists():
        raise FileNotFoundError(f"no summary.json in {result_dir}")

    data: dict[str, Any] = json.loads(summary_json.read_text())
    recipe_dict: dict[str, Any] = data.get("recipe", {})

    if recipe is None:
        from benchmark_suite.recipe import Recipe

        recipe = Recipe.model_validate(recipe_dict)

    scores_list: list[dict[str, Any]] = data.get("scores", [])
    throughput_score: dict[str, Any] = next(
        (s for s in scores_list if s.get("kind") == "throughput"),
        cast("dict[str, Any]", {}),
    )
    metrics: dict[str, Any] = throughput_score.get("metrics", {})

    tok_s_out = metrics.get("output_tok_s") or metrics.get("tok_s_out")
    if tok_s_out is None:
        raise ValueError(
            "no output_tok_s in summary.json — run `bs run` first to populate it"
        )

    backend_type = recipe_dict.get("backend", {}).get("type", "external")
    payload: dict[str, Any] = {
        "hfId": _derive_hf_id(recipe_dict),
        "engineName": _engine_name(backend_type),
        "quantization": recipe.quantization or "FP16",
        "tokSOut": float(tok_s_out),
    }

    hardware_obj = _build_hardware_payload(recipe)
    if hardware_obj is not None:
        payload["hardware"] = hardware_obj

    if (prefill := metrics.get("input_tok_s") or metrics.get("tok_s_prefill")) is not None:
        payload["tokSPrefill"] = float(prefill)
    if (ttft := metrics.get("ttft_mean_ms") or metrics.get("ttft_ms")) is not None:
        payload["ttftMs"] = float(ttft)
    if (peak_vram := metrics.get("peak_vram_gb")) is not None:
        payload["peakVramGb"] = float(peak_vram)
    if (ctx := metrics.get("context_length")) is not None:
        payload["contextLength"] = int(ctx)
    if (out_tokens := metrics.get("output_tokens")) is not None:
        payload["outputTokens"] = int(out_tokens)

    engine_flags: dict[str, Any] = {
        "cellId": recipe.cell.render(),
    }
    if (tp := recipe.resources.tensor_parallel_size) and tp > 1:
        engine_flags["tensorParallel"] = int(tp)
    if (max_model_len := recipe.resources.max_model_len) and max_model_len > 0:
        engine_flags["maxModelLen"] = int(max_model_len)
    payload["engineFlags"] = engine_flags

    if notes:
        payload["notes"] = notes[:2000]

    return payload


def _derive_hf_id(recipe_dict: dict[str, Any]) -> str:
    """Derive the `hfId` from the recipe's backend.model_path.

    localmaxxing validates the model id server-side; absolute local
    paths fall back to the last path segment.
    """
    path = recipe_dict.get("backend", {}).get("model_path", "")
    if "/" in path and not path.startswith("/"):
        return path
    return Path(path).name if path else "unknown/unknown"


def _engine_name(backend_type: str) -> str:
    """Map benchmark_suite backend.type → localmaxxing engineName."""
    return {
        "vllm": "vllm",
        "llamacpp": "llama.cpp",
        "tgi": "tgi",
        "external": "external",
    }.get(backend_type, backend_type or "external")


def _gpu_name(vendor: str, model: str) -> str:
    """Compose the localmaxxing `gpuName` field from vendor + model."""
    if not model:
        return ""
    vendor_prefix = {
        "amd": "AMD",
        "nvidia": "NVIDIA",
        "intel": "Intel",
        "apple": "Apple",
        "tenstorrent": "Tenstorrent",
        "other": "",
    }.get(vendor, vendor.title())
    return f"{vendor_prefix} {model}".strip() if vendor_prefix else model


def _build_hardware_payload(recipe: Recipe) -> dict[str, Any] | None:
    """Build the `hardware` field for the localmaxxing payload."""
    hw = recipe.hardware
    if not hw.is_complete():
        return None

    obj: dict[str, Any] = {
        "hwClass": "DISCRETE_GPU",
        "gpuName": _gpu_name(hw.vendor, hw.model),
        "gpuCount": int(hw.count),
        "vramGb": int(hw.vram_gb),
    }
    if hw.cpu:
        obj["cpu"] = hw.cpu
    if hw.ram_gb:
        obj["ramGb"] = int(hw.ram_gb)
    if hw.os:
        obj["os"] = hw.os
    if hw.power_watts:
        obj["powerWatts"] = int(hw.power_watts)
    return obj


def export_payload(
    result_dir: Path,
    output: Path,
    *,
    notes: str = "",
) -> Path:
    """Write the localmaxxing payload as a JSON file to `output`.

    Useful for inspecting what `bs submit` would hand to `lmx`, or for
    running `lmx speed-test submit <file>` directly.
    """
    payload = build_lmx_payload(result_dir=result_dir, notes=notes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return output


def find_lmx(explicit: str | None) -> str:
    """Resolve the path to the `lmx` binary.

    Priority: explicit flag > $PATH lookup. Returns the absolute path
    if found, or raises FileNotFoundError with install instructions.
    """
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"--lmx-bin={explicit} does not exist")
        return str(path)
    found = shutil.which("lmx")
    if found:
        return found
    raise FileNotFoundError(
        "`lmx` binary not found on $PATH. Install it:\n"
        "  • Download a release: "
        "https://github.com/LottoLottoLotto/localmaxxing-cli/releases/latest\n"
        "  • Or build from source: "
        "`go install github.com/LottoLottoLotto/localmaxxing-cli/cmd/lmx@latest`\n"
        "Then either add it to $PATH or pass `--lmx-bin /path/to/lmx`."
    )


def submit_submission(
    result_dir: Path,
    *,
    lmx_bin: str | None = None,
    endpoint: str | None = None,
    notes: str = "",
    dry_run: bool = False,
) -> SubmitResult:
    """Build the payload and shell out to `lmx` to submit it.

    Args:
        result_dir: produced by `bs run`; must contain `summary.json`.
        lmx_bin: path to the `lmx` binary; defaults to $PATH lookup.
        endpoint: override the localmaxxing base URL (passed as
            `--api-url` to lmx). Defaults to lmx's built-in default
            (`https://www.localmaxxing.com`).
        notes: free-form notes (max 2000 chars).
        dry_run: when True, invokes `lmx speed-test dry-run` instead
            of `lmx speed-test submit`. No rate-limit consumption,
            no write.

    Returns:
        SubmitResult with one of:
        - submission_id + public_url + lmx_stdout (live submit success)
        - dry_run_valid + dry_run_stdout (dry-run success)
        - error + details + lmx_stderr (lmx rejected payload)
        - lmx_not_found (binary missing on $PATH)

    Raises:
        ValueError: when summary.json has no output_tok_s.
        FileNotFoundError: when summary.json is missing OR when
            `lmx` is not found on $PATH.
    """
    try:
        lmx_path = find_lmx(lmx_bin)
    except FileNotFoundError as exc:
        return SubmitResult(lmx_not_found=True, error="lmx_not_found", details=str(exc))

    payload = build_lmx_payload(result_dir=result_dir, notes=notes)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="bs-submit-", delete=False
    ) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)

    try:
        cmd = [lmx_path, "speed-test", "dry-run" if dry_run else "submit", str(tmp_path)]
        if endpoint:
            cmd.extend(["--api-url", endpoint])

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    finally:
        tmp_path.unlink(missing_ok=True)

    stdout = proc.stdout
    stderr = proc.stderr

    if proc.returncode == 0:
        sub_id, public_url = _parse_submission_id(stdout)
        if dry_run:
            return SubmitResult(
                endpoint=endpoint or "https://www.localmaxxing.com",
                dry_run_valid=True,
                dry_run_stdout=stdout,
            )
        return SubmitResult(
            submission_id=sub_id,
            public_url=public_url,
            endpoint=endpoint or "https://www.localmaxxing.com",
            lmx_stdout=stdout,
            lmx_exit_code=0,
        )

    return SubmitResult(
        endpoint=endpoint or "https://www.localmaxxing.com",
        error="lmx_failed",
        details=_truncate(stderr or stdout, 500),
        lmx_stdout=stdout,
        lmx_stderr=stderr,
        lmx_exit_code=proc.returncode,
    )


def _parse_submission_id(stdout: str) -> tuple[str, str]:
    """Best-effort extraction of (submission_id, public_url) from lmx stdout.

    `lmx` prints human-readable text by default. We look for:
      - A line containing "id:" or "submitted:"
      - A URL matching `https://www.localmaxxing.com/speed-tests/<id>`
    If neither matches, returns ("", "").
    """
    import re

    url_match = re.search(
        r"https?://[^\s]*?/speed-tests/([A-Za-z0-9_-]+)", stdout
    )
    sub_id = url_match.group(1) if url_match else ""
    public_url = url_match.group(0) if url_match else ""
    if not sub_id:
        id_match = re.search(
            r"(?:id|submission[_-]?id|run[_-]?id)[:\s]+([A-Za-z0-9_-]+)",
            stdout,
            re.IGNORECASE,
        )
        if id_match:
            sub_id = id_match.group(1)
    return sub_id, public_url


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "..."
