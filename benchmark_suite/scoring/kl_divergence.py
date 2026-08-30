"""benchmark_suite/scoring/kl_divergence.py — per-token KL divergence scorer.

Two sources (per ``KLDScorer.source``):

- ``"logits_dir"``: a converted safetensors cache (one shard per prompt) plus a
  ``manifest.json`` recording provenance + vocab. The scorer queries the
  candidate endpoint's ``top_logprobs`` for the manifest's prompts and computes
  top-k KL with float32 logsumexp renormalization against the reference logits.
- ``"llm-perf"``: invokes ``llm-perf logprobs`` on both endpoints, then
  ``llm-perf kl-divergence`` to get a native KL metric.

Metadata-tolerant by design: if ``manifest.json`` is missing or the vocab does
not match the endpoint tokenizer, the scorer REFUSES with a clear error pointing
to ``bs convert-logits``. No baked-in assumptions about pre-existing caches.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from safetensors.numpy import (
    load_file as np_load_safetensors,  # type: ignore[reportUnknownVariableType]
)

from benchmark_suite.recipe import KLDScorer, Recipe
from benchmark_suite.runner.llm_perf import (
    run_llm_perf_kl_divergence,
    run_llm_perf_logprobs,
)
from benchmark_suite.scoring.base import Scorer, ScoreRecord, ScoreStatus, scorer

MANIFEST_FILENAME = "manifest.json"


class _Tokenizer(Protocol):
    """Minimal tokenizer interface: vocab size + token→id mapping."""

    vocab_size: int

    def convert_tokens_to_ids(self, token: str) -> int | None: ...


def _logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """Numerically stable log-sum-exp in float32."""
    x = x.astype(np.float32)
    m = np.max(x, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=keepdims))


def _top_k_kl(
    reference_logits: np.ndarray,
    candidate_logits: np.ndarray,
    top_k: int = 128,
) -> dict[str, Any]:
    """Per-token KL(ref || cand) over the reference's top-k vocab.

    Both inputs are ``(prompt_len, vocab)`` float arrays of logits. The top-k
    restriction uses the REFERENCE distribution's top-k indices only (not the
    union) — the standard KL-on-top-k that avoids ``log(0)`` for long-tail tokens.

    Returns ``{"mean_kl", "max_kl", "p99_kl", "kl_per_token"}``.
    """
    ref = np.asarray(reference_logits, dtype=np.float32)
    cand = np.asarray(candidate_logits, dtype=np.float32)
    if ref.shape != cand.shape:
        raise ValueError(f"shape mismatch: reference {ref.shape} vs candidate {cand.shape}")
    if ref.ndim != 2:
        raise ValueError(f"expected 2D (prompt_len, vocab), got {ref.shape}")

    vocab = ref.shape[-1]
    k = min(top_k, vocab)

    log_p = ref - _logsumexp(ref, axis=-1, keepdims=True)
    log_q = cand - _logsumexp(cand, axis=-1, keepdims=True)

    topk_idx = np.argpartition(ref, -k, axis=-1)[..., -k:]

    log_p_k = np.take_along_axis(log_p, topk_idx, axis=-1)
    log_q_k = np.take_along_axis(log_q, topk_idx, axis=-1)

    log_p_k = log_p_k - _logsumexp(log_p_k, axis=-1, keepdims=True)
    log_q_k = log_q_k - _logsumexp(log_q_k, axis=-1, keepdims=True)

    p_k = np.exp(log_p_k)
    kl_per_token = np.sum(p_k * (log_p_k - log_q_k), axis=-1)

    return {
        "mean_kl": float(np.mean(kl_per_token)),
        "max_kl": float(np.max(kl_per_token)),
        "p99_kl": float(np.percentile(kl_per_token, 99)),
        "kl_per_token": kl_per_token.tolist(),
    }


def _load_manifest(logits_dir: Path) -> dict[str, Any]:
    """Load manifest.json from a converted logit cache; refuse if missing."""
    manifest_path = Path(logits_dir) / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest.json found at {manifest_path}.\n"
            "Run `bs convert-logits <input_dir> --tokenizer <hf-id> "
            "--model-name <name> --prompts <file>` first."
        )
    return json.loads(manifest_path.read_text())


def _vocab_check(manifest: dict[str, Any], endpoint_tokenizer: _Tokenizer | int) -> bool:
    """Compare manifest vocab_size to the endpoint tokenizer's vocab size.

    ``endpoint_tokenizer`` may be an int (vocab size) or a tokenizer object with
    a ``vocab_size`` attribute. Returns True on match; raises ValueError on
    mismatch (mentioning both values).
    """
    manifest_vocab = int(manifest["vocab_size"])
    if isinstance(endpoint_tokenizer, int):
        tokenizer_vocab = endpoint_tokenizer
    else:
        tokenizer_vocab = endpoint_tokenizer.vocab_size
    tokenizer_vocab = int(tokenizer_vocab)
    if manifest_vocab != tokenizer_vocab:
        raise ValueError(
            f"vocab mismatch: manifest vocab_size={manifest_vocab} but endpoint "
            f"tokenizer vocab_size={tokenizer_vocab}. Re-run `bs convert-logits` "
            "with the correct --tokenizer."
        )
    return True


def _load_tokenizer(tokenizer_id: str) -> _Tokenizer:
    """Lazily load a tokenizer (transformers/torch only touched here)."""
    from transformers import AutoTokenizer  # type: ignore[reportMissingImports]

    return cast(_Tokenizer, AutoTokenizer.from_pretrained(tokenizer_id))  # type: ignore[reportUnknownMemberType]


def _candidate_logits_from_top_logprobs(
    top_logprobs: list[dict[str, Any]],
    vocab_size: int,
    token_to_id: Callable[[str], int | None],
) -> np.ndarray:
    """Reconstruct a candidate logits vector from endpoint top_logprobs.

    Returns a ``(vocab,)`` float32 array of logits: ``logprob`` at each returned
    token's index, ``-inf`` elsewhere.
    """
    logits = np.full(vocab_size, -np.inf, dtype=np.float32)
    for entry in top_logprobs:
        token = entry.get("token")
        logprob = float(entry.get("logprob", -np.inf))
        if token is None:
            continue
        token_id = token_to_id(token)
        if token_id is None or not (0 <= token_id < vocab_size):
            continue
        logits[token_id] = logprob
    return logits


@scorer
class KLDScorerImpl(Scorer):
    """Per-token KL divergence between a reference distribution and a candidate."""

    kind = "kld"

    def __init__(self, config: KLDScorer) -> None:
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
            if self.config.source == "llm-perf":
                metrics = self._score_llm_perf(recipe, result_dir, endpoint_url)
            else:
                metrics = self._score_logits_dir(recipe, result_dir, endpoint_url)
        except Exception as exc:
            return ScoreRecord(
                kind=self.kind,
                status=ScoreStatus.FAILURE,
                cell_id=cell_id,
                error=str(exc),
            )
        return ScoreRecord(
            kind=self.kind,
            metrics=metrics,
            status=ScoreStatus.SUCCESS,
            cell_id=cell_id,
        )

    def _score_logits_dir(
        self,
        recipe: Recipe,
        result_dir: Path,
        endpoint_url: str | None,
    ) -> dict[str, Any]:
        del result_dir  # artifacts written by caller; scorer is pure
        logits_dir = self.config.reference_logits_dir
        manifest = _load_manifest(logits_dir)

        tokenizer = _load_tokenizer(manifest["tokenizer_id"])
        if self.config.vocab_check:
            _vocab_check(manifest, tokenizer)

        prompts = self._load_prompts()
        if len(prompts) != int(manifest["prompt_count"]):
            raise ValueError(
                f"prompt count ({len(prompts)}) != manifest prompt_count "
                f"({manifest['prompt_count']})"
            )

        vocab_size = int(manifest["vocab_size"])
        top_k = self.config.top_k
        base_url = (endpoint_url or recipe.endpoint.base_url_v1).rstrip("/")
        model_name = recipe.endpoint.model_name

        mean_kls: list[float] = []
        max_kls: list[float] = []
        p99_kls: list[float] = []
        for idx, prompt in enumerate(prompts):
            reference = np_load_safetensors(
                logits_dir / f"prompt_{idx:06d}.safetensors"
            )["logits"]
            top_logprobs = self._query_top_logprobs(base_url, model_name, prompt, top_k)
            candidate = _candidate_logits_from_top_logprobs(
                top_logprobs, vocab_size, tokenizer.convert_tokens_to_ids
            )
            candidate_b = np.broadcast_to(candidate, reference.shape)
            result = _top_k_kl(reference, candidate_b, top_k=top_k)
            mean_kls.append(float(result["mean_kl"]))
            max_kls.append(float(result["max_kl"]))
            p99_kls.append(float(result["p99_kl"]))

        return {
            "mean_kl": float(np.mean(mean_kls)) if mean_kls else 0.0,
            "max_kl": float(np.max(max_kls)) if max_kls else 0.0,
            "p99_kl": float(np.max(p99_kls)) if p99_kls else 0.0,
        }

    def _score_llm_perf(
        self,
        recipe: Recipe,
        result_dir: Path,
        endpoint_url: str | None,
    ) -> dict[str, Any]:
        del endpoint_url  # llm-perf reads the endpoint from the recipe
        baseline_jsonl = result_dir / "kld_baseline.jsonl"
        candidate_jsonl = result_dir / "kld_candidate.jsonl"
        output_json = result_dir / "kld_result.json"

        prompts = self._load_prompts() or None
        run_llm_perf_logprobs(
            recipe,
            output_jsonl_path=baseline_jsonl,
            max_tokens=self.config.max_tokens,
            prompts=prompts,
        )
        run_llm_perf_logprobs(
            recipe,
            output_jsonl_path=candidate_jsonl,
            max_tokens=self.config.max_tokens,
            prompts=prompts,
        )
        report = run_llm_perf_kl_divergence(
            baseline_jsonl, candidate_jsonl, output_json_path=output_json
        )
        return {"kl_divergence": float(report["kl_divergence"])}

    def _load_prompts(self) -> list[str]:
        if self.config.prompts_file is None:
            return []
        prompts: list[str] = []
        for line in self.config.prompts_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(str(obj["prompt"]))
        return prompts

    def _query_top_logprobs(
        self, base_url: str, model_name: str, prompt: str, top_k: int
    ) -> list[dict[str, Any]]:
        import httpx

        url = base_url + "/completions"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": self.config.max_tokens,
            "logprobs": top_k,
        }
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = cast(dict[str, Any], resp.json())
        choices = cast(list[dict[str, Any]], data.get("choices", []))
        if not choices:
            raise ValueError("endpoint returned no choices")
        logprobs = cast(dict[str, Any], choices[0].get("logprobs") or {})
        return cast(list[dict[str, Any]], logprobs.get("top_logprobs", []))