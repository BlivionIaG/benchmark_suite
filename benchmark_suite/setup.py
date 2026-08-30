"""benchmark_suite/setup.py — onboarding wizard for localmaxxing.com + lmx.

`bs setup` runs four checks in sequence, each independent:

  1. Detect GPU → hardware.json
  2. Verify lmx is installed
  3. Verify auth (LMX_API_KEY env or ~/.config/localmaxxing/config.json)
  4. Print summary card

Steps that fail print a clear message and exit non-zero; steps that
succeed print a ✓. The wizard never hangs (browser opens are
detached via subprocess.Popen if a TTY is present).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

LMX_INSTALL_HINT = (
    "Install `lmx` from "
    "https://github.com/LottoLottoLotto/localmaxxing-cli/releases/latest\n"
    "    Linux (amd64):  curl -fsSLO <base>/lmx-linux-amd64.tar.gz && "
    "tar -xzf lmx-linux-amd64.tar.gz && sudo mv lmx /usr/local/bin/\n"
    "    macOS (arm64):  curl -fsSLO <base>/lmx-darwin-arm64.tar.gz && "
    "tar -xzf lmx-darwin-arm64.tar.gz && sudo mv lmx /usr/local/bin/\n"
    "    Or:  go install github.com/LottoLottoLotto/localmaxxing-cli/cmd/lmx@latest"
)

LMX_AUTH_HINT = (
    "Authenticate with one of:\n"
    "    export LMX_API_KEY=bhk_...                                 # env var\n"
    "    lmx auth login                                             # browser device-flow\n"
    "    printf '%s\\n' \"$LMX_API_KEY\" | lmx auth --key-stdin       # safe (no shell history)"
)

LMX_CONFIG_PATH = Path.home() / ".config" / "localmaxxing" / "config.json"


@dataclass
class SetupStep:
    name: str
    ok: bool
    summary: str


@dataclass
class SetupResult:
    steps: list[SetupStep]
    hardware: dict[str, object] | None
    lmx_path: str | None
    lmx_version: str | None
    api_key_prefix: str | None


def detect_hardware(out_path: Path | None = None) -> dict[str, object] | None:
    """Write the localmaxxing hardware JSON to `out_path` and return it.

    Tries `lmx hardware --out <path>` first (canonical); falls back to
    `torch.cuda.get_device_properties(0)` if lmx is unavailable and
    torch is importable; otherwise returns None.

    Returns None on every detection path failing — the caller should
    treat this as "skip hardware step" not as an error.
    """
    out_path = out_path or (Path.cwd() / "hardware.json")

    lmx_path = shutil.which("lmx")
    if lmx_path:
        try:
            proc = subprocess.run(
                [lmx_path, "hardware", "--out", str(out_path)],
                capture_output=True, text=True, check=False,
            )
            if proc.returncode == 0 and out_path.exists():
                payload = json.loads(out_path.read_text())
                if isinstance(payload, dict):
                    return cast("dict[str, object]", payload)
        except (OSError, json.JSONDecodeError):
            pass

    torch_payload = _detect_hardware_via_torch()
    if torch_payload:
        out_path.write_text(json.dumps(torch_payload, indent=2) + "\n")
        return torch_payload

    return None


def _detect_hardware_via_torch() -> dict[str, object] | None:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return None
    torch_mod = cast("Any", torch)
    try:
        if not torch_mod.cuda.is_available():
            return None
        props = torch_mod.cuda.get_device_properties(0)
    except Exception:
        return None
    name = getattr(props, "name", "unknown GPU")
    total_mem = getattr(props, "total_memory", 0)
    vram_gb = round(total_mem / (1024**3)) if total_mem else 0
    return {
        "hwClass": "DISCRETE_GPU",
        "gpuName": name,
        "gpuCount": torch_mod.cuda.device_count(),
        "vramGb": vram_gb,
    }


def lmx_version(lmx_path: str) -> str | None:
    """Return the version string from `lmx --version`, or None on failure."""
    try:
        proc = subprocess.run(
            [lmx_path, "--version"], capture_output=True, text=True, check=False,
        )
    except OSError:
        return None
    out = (proc.stdout or "") + (proc.stderr or "")
    match = re.search(r"(\d+\.\d+\.\d+)", out)
    return match.group(1) if match else (out.strip().splitlines()[-1] if out.strip() else None)


def find_api_key() -> str | None:
    """Return the API key from $LMX_API_KEY env or lmx config file, or None."""
    env_key = os.environ.get("LMX_API_KEY", "").strip()
    if env_key:
        return env_key
    if LMX_CONFIG_PATH.exists():
        try:
            cfg = json.loads(LMX_CONFIG_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        key = cfg.get("apiKey")
        return str(key) if isinstance(key, str) and key else None
    return None


def redact_key(key: str) -> str:
    """Return the first 8 chars of a key with the rest masked, e.g. `bhk_1a2b****`."""
    if not key:
        return ""
    if len(key) <= 8:
        return key[:4] + "****"
    return key[:8] + "****"


def is_tty() -> bool:
    return bool(os.isatty(0))


def open_browser_for_auth(lmx_path: str) -> bool:
    """Run `lmx auth login` (which opens a browser) and return True on success.

    Only meaningful in a TTY — silently returns False if stdin isn't a tty.
    """
    if not is_tty():
        return False
    try:
        proc = subprocess.run([lmx_path, "auth", "login"], check=False)
    except OSError:
        return False
    return proc.returncode == 0


def run_setup(
    *,
    lmx_bin: str | None = None,
    hardware_out: Path | None = None,
    skip_auth: bool = False,
    run_auth_if_missing: bool = True,
) -> SetupResult:
    """Run the full onboarding wizard.

    Args:
        lmx_bin: explicit path to the lmx binary; defaults to $PATH lookup.
        hardware_out: where to write hardware.json; defaults to cwd.
        skip_auth: don't check / prompt for the API key (for CI smoke tests).
        run_auth_if_missing: if no key is found AND stdin is a TTY, spawn
            `lmx auth login` (which opens a browser). Set False for
            non-interactive contexts.

    Returns a SetupResult with the per-step outcomes; never raises for
    expected errors (lmx missing, no key, no GPU). The CLI decides
    exit codes based on which steps succeeded.
    """
    steps: list[SetupStep] = []

    hardware = None
    try:
        hardware = detect_hardware(hardware_out)
        if hardware:
            gpu = hardware.get("gpuName", "unknown")
            count = hardware.get("gpuCount", "?")
            vram = hardware.get("vramGb", "?")
            steps.append(
                SetupStep("GPU detected", True, f"{gpu} x{count} ({vram} GB each)")
            )
        else:
            steps.append(
                SetupStep("GPU detected", False, "neither lmx hardware nor torch found a GPU")
            )
    except OSError as exc:
        steps.append(SetupStep("GPU detected", False, f"lmx hardware failed: {exc}"))

    lmx_path: str | None = None
    lmx_ver: str | None = None
    found = shutil.which("lmx") if lmx_bin is None else (
        str(Path(lmx_bin)) if Path(lmx_bin).is_file() else None
    )
    if found:
        lmx_path = found
        lmx_ver = lmx_version(found)
        steps.append(
            SetupStep("lmx installed", True, f"{found}" + (f" (v{lmx_ver})" if lmx_ver else ""))
        )
    else:
        steps.append(SetupStep("lmx installed", False, "lmx not found on $PATH"))

    api_key: str | None = None
    if skip_auth:
        steps.append(SetupStep("API key", False, "skipped (--skip-auth)"))
    else:
        api_key = find_api_key()
        if api_key:
            source = "LMX_API_KEY" if os.environ.get("LMX_API_KEY") else LMX_CONFIG_PATH.as_posix()
            steps.append(
                SetupStep(
                    "API key", True, f"{redact_key(api_key)} (via {source})"
                )
            )
        elif run_auth_if_missing and lmx_path and is_tty():
            typer_echo("No API key found. Launching `lmx auth login` (opens browser)...")
            if open_browser_for_auth(lmx_path):
                api_key = find_api_key()
                if api_key:
                    steps.append(
                        SetupStep(
                            "API key",
                            True,
                            f"{redact_key(api_key)} (after `lmx auth login`)",
                        )
                    )
                else:
                    steps.append(
                        SetupStep(
                            "API key",
                            False,
                            "`lmx auth login` ran but no key was saved",
                        )
                    )
            else:
                steps.append(
                    SetupStep(
                        "API key",
                        False,
                        "`lmx auth login` failed (browser not opened?)",
                    )
                )
        else:
            steps.append(SetupStep("API key", False, "no key in env or config"))

    return SetupResult(
        steps=steps,
        hardware=hardware,
        lmx_path=lmx_path,
        lmx_version=lmx_ver,
        api_key_prefix=redact_key(api_key) if api_key else None,
    )


def typer_echo(msg: str) -> None:
    """Best-effort import-and-use of typer.echo; falls back to print."""
    try:
        import typer

        typer.echo(msg)
    except ImportError:
        print(msg)