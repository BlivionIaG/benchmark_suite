"""benchmark_suite/tools/convert_kl_logits.py — one-time f_*.fp16 → safetensors cache.

This is the ONLY module in the suite that imports torch. It converts a directory
of legacy ``f_*.fp16`` files (``torch.save`` zips of a tensor list) into a
memmap-friendly safetensors cache plus a ``manifest.json`` recording provenance.

The conversion is deliberately metadata-requiring: it never guesses which
model/tokenizer/prompts produced the logits. The caller must pass ``--tokenizer``,
``--model-name``, and ``--prompts`` explicitly; the resulting manifest records
them so the KLD scorer can refuse cross-tokenizer comparisons later.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

MANIFEST_FILENAME = "manifest.json"
CACHE_FORMAT = "safetensors+manifest"
SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class KLCacheManifest:
    """Provenance metadata for a converted logit cache."""

    model_name: str
    tokenizer_id: str
    vocab_size: int
    prompt_count: int
    max_prompt_tokens: int
    shape_per_prompt: list[tuple[int, int]]  # (prompt_len, vocab) per prompt
    dtype: str  # "float16" | "float32"
    created_at: str  # ISO 8601
    source_files: list[str]
    cache_format: str = CACHE_FORMAT
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # JSON-safe: tuples become lists so a round-trip through json is stable.
        data["shape_per_prompt"] = [list(pair) for pair in self.shape_per_prompt]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KLCacheManifest:
        shapes = [
            (int(pair[0]), int(pair[1])) for pair in data["shape_per_prompt"]
        ]
        return cls(
            model_name=str(data["model_name"]),
            tokenizer_id=str(data["tokenizer_id"]),
            vocab_size=int(data["vocab_size"]),
            prompt_count=int(data["prompt_count"]),
            max_prompt_tokens=int(data["max_prompt_tokens"]),
            shape_per_prompt=shapes,
            dtype=str(data["dtype"]),
            created_at=str(data["created_at"]),
            source_files=[str(f) for f in data["source_files"]],
            cache_format=str(data.get("cache_format", CACHE_FORMAT)),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
        )


def _load_tokenizer_vocab_size(tokenizer: str) -> int:
    """Resolve a tokenizer id/path to its vocab size (lazy transformers import)."""
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    tok = cast(Any, AutoTokenizer).from_pretrained(tokenizer)
    return len(tok)


def _load_fp16_tensor(path: Path) -> np.ndarray:
    """Load one f_*.fp16 file into a numpy array.

    Legacy files are ``torch.save`` zips of a tensor list. We try
    ``weights_only=True`` first (safe), then fall back to ``weights_only=False``
    for older zips that embed non-tensor metadata.
    """
    import torch  # type: ignore[import-not-found]

    torch_mod = cast(Any, torch)
    try:
        obj: Any = torch_mod.load(path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch_mod.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, (list, tuple)):
        items = cast(list[Any], obj)
        if len(items) != 1:
            raise ValueError(
                f"{path.name}: expected a single tensor, got {len(items)} items"
            )
        obj = items[0]
    if not isinstance(obj, torch_mod.Tensor):
        raise TypeError(f"{path.name}: expected a torch.Tensor, got {type(obj).__name__}")
    return cast(np.ndarray, obj.detach().cpu().numpy())


def convert_logit_cache(
    input_dir: Path,
    output_dir: Path,
    *,
    tokenizer: str,
    model_name: str,
    prompts: list[str],
    max_prompts: int | None = None,
) -> KLCacheManifest:
    """Read f_*.fp16 (torch.save zip of tensor list), re-save as safetensors.

    Writes ``output_dir/manifest.json`` and one ``prompt_{idx:06d}.safetensors``
    per prompt. Requires explicit ``tokenizer`` + ``model_name`` + ``prompts`` —
    never guesses provenance.
    """
    from safetensors.numpy import save_file  # type: ignore[reportUnknownVariableType]

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(p.name for p in input_dir.glob("f_*.fp16"))
    if not source_files:
        raise FileNotFoundError(f"no f_*.fp16 files found in {input_dir}")

    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    if len(prompts) != len(source_files):
        raise ValueError(
            f"prompt count ({len(prompts)}) does not match f_*.fp16 file count "
            f"({len(source_files)}); pass --prompts with exactly one entry per file"
        )

    vocab_size = _load_tokenizer_vocab_size(tokenizer)

    shapes: list[tuple[int, int]] = []
    max_prompt_tokens = 0
    dtype = "float32"
    for idx, (prompt, filename) in enumerate(zip(prompts, source_files, strict=True)):
        del prompt  # prompt text is provenance, not used for shape
        arr = _load_fp16_tensor(input_dir / filename)
        if arr.ndim != 2:
            raise ValueError(
                f"{filename}: expected 2D (prompt_len, vocab), got shape {arr.shape}"
            )
        prompt_len, vocab = arr.shape
        if vocab != vocab_size:
            raise ValueError(
                f"{filename}: vocab dim {vocab} != tokenizer vocab_size {vocab_size}"
            )
        shapes.append((prompt_len, vocab))
        max_prompt_tokens = max(max_prompt_tokens, prompt_len)
        dtype = "float16" if arr.dtype == np.float16 else "float32"
        save_file({"logits": arr}, output_dir / f"prompt_{idx:06d}.safetensors")

    manifest = KLCacheManifest(
        model_name=model_name,
        tokenizer_id=tokenizer,
        vocab_size=vocab_size,
        prompt_count=len(prompts),
        max_prompt_tokens=max_prompt_tokens,
        shape_per_prompt=shapes,
        dtype=dtype,
        created_at=datetime.now(UTC).isoformat(),
        source_files=source_files,
    )
    (output_dir / MANIFEST_FILENAME).write_text(
        _json_dumps(manifest.to_dict())
    )
    return manifest


def _json_dumps(data: dict[str, Any]) -> str:
    import json

    return json.dumps(data, indent=2, sort_keys=True)