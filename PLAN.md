# benchmark_suite — Implementation Plan (PLAN.md)

**Status**: decision-complete, ready for execution
**Date**: 2026-08-30
**Scope**: v1 of a standalone repo at `/home/kletorch/Projects/infrastructure/gfx1030_optimized/benchmark_suite/` (own git remote, e.g. `BlivionIaG/benchmark_suite`). Declarative YAML recipes → run benchmarks + quality scorers against any OpenAI-compatible endpoint (vLLM, llama.cpp, TGI, OpenAI API) → comparable reports.

## 0. Context & Verified Ground Truth (execution-relevant findings)

| Finding | Consequence |
|---|---|
| `llm-perf` (iopsystems/llm-perf, Apache-2.0/MIT) is **TOML-config-driven**, subcommands `bench` (default) / `logprobs` / `kl-divergence baseline.jsonl candidate.jsonl` / `mmlu-pro`; JSON output via `[output] format="json" file=...`; TTFT/TPOT/ITL percentiles + context-bucketed metrics; **no crates.io, no prebuilt releases; tags up to `v0.1.16`; needs Rust ≥1.85** (2024 edition) | Runner generates a temp TOML per run; install via `scripts/install_llm_perf.sh v0.1.16` (cargo build --release); `bs doctor` checks binary; `vllm bench` stays fallback. Dev box has cargo; `.176` Rust version must be checked by doctor |
| PyPI verified: `lm-eval 0.4.12` (≥3.10), `inspect-ai 0.3.260` (≥3.10), `terminal-bench 0.2.18` (**≥3.12**), `promptfoo 0.1.4` (official **wrapper around the Node CLI**), `typer 0.27.2`, `pydantic 2.13.5`, `numpy 2.5.2` (**≥3.12**), `httpx 0.28.1`, `pyyaml 6.0.3`, `respx 0.23.1`, `ruff 0.16.5`, `basedpyright 1.39.10` | `requires-python = ">=3.12,<3.14"`, `.python-version = 3.13` (system python is 3.14.3 — too new for lm-eval's dep tree); core stays light, heavy tools in optional extras |
| We do **not** assume the existing 12 GB `models/kl_logits/f_*.fp16` cache has known origin. The tool must work for any user with any captured logits, with no baked-in knowledge of which model/tokenizer/prompts produced them. | KLD scorer is **metadata-tolerant**: `bs convert-logits` requires explicit `--tokenizer` + `--model-name` + `--prompts` flags and writes a `manifest.json` recording the provenance. If the cache already exists without metadata, the scorer refuses and points to `bs convert-logits` to recreate with explicit metadata. No silent assumptions. |
| Parent conventions verified in source: parser fields `output_tok_s, peak_output_tok_s, total_tok_s, ttft_mean_ms, ttft_median_ms, ttft_p99_ms, tpot_mean_ms, tpot_median_ms, duration_s, successful, failed` (11 regexes in `benchmark_results/parse_matrix_results.py`); cell-id `{family}_{attn}_{linear}_cg{cg}_mtp{mtp}`; result README sections Date/Goal/Configuration/Results/Interpretation/Verdict/Files; canonical gfx1030 env block | Ported verbatim as golden-tested code; `summary.csv` reuses the exact legacy column names; `report.py` emits the same README structure |
| `benchmark_suite/` does not exist yet; parent dir **is** a git repo | Nested-repo handling: P1 adds `benchmark_suite/` to parent `.gitignore` (with user confirmation) and `git init` inside the new dir |

---

## 1. Repository Layout

```
benchmark_suite/
├── PLAN.md                        # this document (commit C0)
├── README.md                      # install, quickstart, recipe anatomy, tool matrix
├── AGENTS.md                      # repo-local agent rules (no-secrets, atomic commits, TDD flow)
├── pyproject.toml                 # uv project; deps + ruff/basedpyright config (below)
├── .python-version                # 3.13
├── .gitignore                     # results/, .venv/, __pycache__/, dist/, *.egg-info, .triton/
├── .pre-commit-config.yaml        # ruff (lint+format), basedpyright, yaml check
├── uv.lock                        # locked reproducible env
├── scripts/
│   └── install_llm_perf.sh        # pin-tag cargo build of iopsystems/llm-perf (default v0.1.16)
├── tools/
│   └── convert_kl_logits.py       # one-time torch.load(f_*.fp16) → safetensors/npy + manifest.json
├── benchmark_suite/
│   ├── __init__.py                # __version__, public re-exports (Recipe, load_recipe)
│   ├── cli.py                     # typer app → `bs` entry point (all subcommands)
│   ├── recipe.py                  # Pydantic v2 schema + load_recipe() + env-merge + matrix expansion
│   ├── paths.py                   # result-dir layout, CellId render ({family}_{attn}_{linear}_cg{cg}_mtp{mtp})
│   ├── runner/
│   │   ├── __init__.py
│   │   ├── endpoint.py            # health probe, /v1/models, logprobs capability probe, retry/backoff
│   │   ├── serve.py               # server lifecycle: cmd synthesis (vllm/llamacpp), spawn, wait-health, teardown, GPU file-lock
│   │   ├── llm_perf.py            # TOML generation, `llm-perf bench|logprobs|kl-divergence`, JSON parse
│   │   └── vllm_bench.py          # `vllm bench throughput` fallback + golden-tested stdout regex parser
│   ├── scoring/
│   │   ├── __init__.py            # kind→scorer registry
│   │   ├── base.py                # Scorer protocol + ScoreRecord (kind, metrics, status, artifacts, cell_id)
│   │   ├── throughput.py          # ScoreRecord from llm-perf JSON / vllm-bench parse (legacy column names)
│   │   ├── perplexity.py          # lm-eval subprocess, model=local-completions, wikitext ppl
│   │   ├── kl_divergence.py       # dual source: llm-perf native | logits_dir (memmap + top-k KL)
│   │   ├── llm_judge.py           # driver=native (httpx rubric judge) | promptfoo (npx/PyPI wrapper)
│   │   └── agentic.py             # harness=inspect (inspect-ai subprocess) | terminal-bench (best-effort)
│   ├── report.py                  # summary.csv (legacy columns), summary.json, README.md (parent style)
│   └── compare.py                 # diff two result dirs → md + csv delta tables
├── recipes/
│   ├── qwen36-27b-gptq-tp4.yaml         # dense W4A16, V2+FPP, canonical gfx1030 env
│   ├── qwen36-35b-a3b-fp16-tp4.yaml     # MoE FP16 reference
│   ├── perplexity-compare.yaml          # wikitext ppl across two endpoints
│   └── kld-vs-fp16-reference.yaml       # logits_dir KLD vs models/kl_logits (quant drift)
├── tests/
│   ├── conftest.py                # fixtures: respx mock OpenAI server, fake llm-perf binary, tmp recipe
│   ├── test_recipe.py             # schema validation, union discrimination, env merge, cell render
│   ├── test_paths.py
│   ├── runner/
│   │   ├── test_endpoint.py       # respx-probed health/logprobs capability
│   │   ├── test_llm_perf.py       # TOML gen snapshot + JSON parse fixture
│   │   └── test_vllm_bench.py     # golden stdout fixture → exact field mapping
│   └── scoring/
│       ├── test_throughput.py
│       ├── test_perplexity.py     # subprocess argv + lm-eval JSON fixture parse
│       ├── test_kl_divergence.py  # tiny synthetic logits, hand-computed KL equality
│       ├── test_llm_judge.py      # native rubric flow via respx; promptfoo argv build
│       └── test_agentic.py        # inspect argv build; missing-binary errors
└── results/                       # gitignored
```

### `pyproject.toml` (canonical content — exact pins where stability matters)

```toml
[project]
name = "benchmark-suite"
version = "0.1.0"
description = "Declarative benchmarking and quality scoring for OpenAI-compatible LLM endpoints"
requires-python = ">=3.12,<3.14"
dependencies = [
  "pydantic==2.13.5",
  "PyYAML==6.0.3",
  "typer==0.27.2",
  "httpx==0.28.1",
  "numpy==2.5.2",
]

[project.optional-dependencies]
perplexity = ["lm-eval==0.4.12"]                      # pulls torch+datasets (heavy by design)
judge      = ["promptfoo==0.1.4"]                     # official wrapper; Node still required
agentic    = ["inspect-ai==0.3.260", "terminal-bench==0.2.18"]
kld-torch  = ["torch>=2.6"]                            # only for legacy f_*.fp16 conversion
dev        = ["pytest>=8.3", "pytest-cov>=5", "respx==0.23.1", "ruff==0.16.5", "basedpyright==1.39.10", "pre-commit>=4"]

[project.scripts]
bs = "benchmark_suite.cli:app"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "RUF", "ANN", "TID"]

[tool.basedpyright]
typeCheckingMode = "strict"
pythonVersion = "3.12"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

---

## 2. Recipe Schema (concrete Pydantic v2 — syntactically valid)

```python
"""benchmark_suite/recipe.py — declarative recipe schema (Pydantic v2)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")

# Canonical gfx1030 env block (parent AGENTS.md); runtime.env merges OVER these.
CANONICAL_ENV: dict[str, str] = {
    "VLLM_ROCM_USE_AITER": "0",
    "VLLM_ROCM_USE_AITER_MOE": "0",
    "FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE",
    "VLLM_RDNA_FORCE_FP16": "1",
    "TORCH_BLAS_PREFER_HIPBLASLT": "0",
    "PYTORCH_TUNABLEOP_ENABLED": "1",
    "PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED": "0",
    "VLLM_BATCH_INVARIANT": "0",
    "GPU_MAX_HW_QUEUES": "2",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
}


class MetaSection(BaseModel):
    name: str
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    tags: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not SLUG_RE.match(v):
            raise ValueError("meta.name must be a slug [a-z0-9-_]")
        return v


class CellId(BaseModel):
    """Renders the parent cell convention: dense_fardna2_rdna2_cg1_mtp2."""
    family: str = "dense"
    attn: str = "triton"
    linear: str = "triton"
    cg: int = 0
    mtp: int = 0
    extra: dict[str, str] = Field(default_factory=dict)

    def render(self) -> str:
        base = f"{self.family}_{self.attn}_{self.linear}_cg{self.cg}_mtp{self.mtp}"
        if self.extra:
            base += "_" + "_".join(f"{k}{v}" for k, v in sorted(self.extra.items()))
        return base


class BackendSection(BaseModel):
    type: Literal["vllm", "llamacpp", "tgi", "external"] = "external"
    model_path: str = ""                                # local path or HF id (not for external)
    served_model_name: str = ""                         # default: basename(model_path)
    vllm: dict[str, object] = Field(default_factory=dict)      # e.g. compilation-config, language-model-only
    llamacpp: dict[str, object] = Field(default_factory=dict)  # e.g. n-gpu-layers, flash-attn
    tgi: dict[str, object] = Field(default_factory=dict)


class EndpointSection(BaseModel):
    url: str = "http://127.0.0.1:8000"
    api_key_env: str = "OPENAI_API_KEY"
    model_name: str = ""                                # payload `model`; defaulted in Recipe validator
    timeout_s: float = 600.0
    max_retries: int = 3

    @property
    def base_url_v1(self) -> str:
        u = self.url.rstrip("/")
        return u if u.endswith("/v1") else u + "/v1"


class ResourcesSection(BaseModel):
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    max_num_seqs: int = 8
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = "float16"
    enforce_eager: bool = False
    devices: str = ""                                   # "0,1,2,3" → HIP_/CUDA_VISIBLE_DEVICES
    extra_args: dict[str, object] = Field(default_factory=dict)  # escape-hatch CLI flags


class RuntimeSection(BaseModel):
    env: dict[str, str] = Field(default_factory=dict)   # merged over CANONICAL_ENV
    server_cmd: str = ""                                # explicit override; "" → synthesize
    startup_wait_s: float = 900.0
    health_path: str = "/health"
    teardown_timeout_s: float = 30.0
    gpu_lock: bool = True                               # serialize GPU-owning cells (file lock)


class LoadSection(BaseModel):
    concurrencies: list[int] = Field(default_factory=lambda: [1, 4, 8])
    num_prompts: int = 32
    input_len: int = 256
    output_len: int = 64
    random_range_ratio: float = 0.0                     # 0 = fixed lengths
    dataset: Optional[Path] = None                      # JSONL {"prompt": ...}; None → synthetic
    warmup_requests: int = 4
    qps: Optional[float] = None                         # fixed-QPS mode when set
    duration_s: Optional[float] = None                  # soak mode when set


class StopConditions(BaseModel):
    max_duration_s: Optional[float] = None
    max_failures: int = 0
    min_output_tok_s: Optional[float] = None


class ThroughputScorer(BaseModel):
    kind: Literal["throughput"] = "throughput"
    tool: Literal["llm-perf", "vllm-bench"] = "llm-perf"
    extra_flags: list[str] = Field(default_factory=list)


class PerplexityScorer(BaseModel):
    kind: Literal["perplexity"] = "perplexity"
    tasks: list[str] = Field(default_factory=lambda: ["wikitext"])
    num_fewshot: int = 0
    limit: Optional[int] = None
    lm_eval_extra_args: list[str] = Field(default_factory=list)


class KLDScorer(BaseModel):
    kind: Literal["kld"] = "kld"
    source: Literal["logits_dir", "llm-perf"] = "logits_dir"
    reference_logits_dir: Path = Path("models/kl_logits")   # converted cache + manifest.json
    reference_endpoint: str = ""                        # source=llm-perf: baseline endpoint URL
    prompts_file: Optional[Path] = None
    top_k: int = 128
    max_tokens: int = 1
    vocab_check: bool = True                            # refuse cross-tokenizer comparison


class LLMJudgeScorer(BaseModel):
    kind: Literal["llm_judge"] = "llm_judge"
    driver: Literal["native", "promptfoo"] = "native"   # native = httpx rubric judge (no Node)
    judge_url: str = "http://127.0.0.1:8000"
    judge_model: str = ""
    judge_api_key_env: str = "OPENAI_API_KEY"
    prompts_file: Optional[Path] = None
    rubric: str = "Score the answer 0-10 for correctness and clarity."
    promptfoo_config: Optional[Path] = None
    promptfoo_version: str = "0.118.5"                  # npm pin when driver=promptfoo


class AgenticScorer(BaseModel):
    kind: Literal["agentic"] = "agentic"
    harness: Literal["inspect", "terminal-bench"] = "inspect"
    tasks: list[str] = Field(default_factory=list)
    limit: Optional[int] = None
    sandbox: str = "docker"


ScorerConfig = Annotated[
    Union[ThroughputScorer, PerplexityScorer, KLDScorer, LLMJudgeScorer, AgenticScorer],
    Field(discriminator="kind"),
]


class BenchSection(BaseModel):
    load: LoadSection = Field(default_factory=LoadSection)
    scoring: list[ScorerConfig] = Field(default_factory=list)
    stop_conditions: StopConditions = Field(default_factory=StopConditions)


class ReportSection(BaseModel):
    output_format: Literal["csv", "md", "json", "all"] = "all"
    output_dir: Path = Path("results")
    keep_artifacts: bool = True


class Recipe(BaseModel):
    meta: MetaSection
    backend: BackendSection = Field(default_factory=BackendSection)
    endpoint: EndpointSection = Field(default_factory=EndpointSection)
    resources: ResourcesSection = Field(default_factory=ResourcesSection)
    runtime: RuntimeSection = Field(default_factory=RuntimeSection)
    bench: BenchSection = Field(default_factory=BenchSection)
    report: ReportSection = Field(default_factory=ReportSection)
    cell: CellId = Field(default_factory=CellId)

    @model_validator(mode="after")
    def _coherence(self) -> "Recipe":
        if self.backend.type != "external" and not self.backend.model_path:
            raise ValueError("backend.model_path required when backend.type != 'external'")
        if not self.endpoint.model_name:
            self.endpoint.model_name = (
                self.backend.served_model_name or Path(self.backend.model_path).name
            )
        return self

    def merged_env(self) -> dict[str, str]:
        return {**CANONICAL_ENV, **self.runtime.env}


def load_recipe(path: Path) -> Recipe:
    return Recipe.model_validate(yaml.safe_load(Path(path).read_text()))
```

### Example recipe (`recipes/qwen36-27b-gptq-tp4.yaml`)

```yaml
meta:
  name: qwen36-27b-gptq-tp4
  description: Qwen3.6-27B-GPTQ-W4A16-G32, 4x V620, RDNA2 kernels
  tags: [gfx1030, gptq, w4a16, tp4]
backend:
  type: vllm
  model_path: /data/triton/models/Qwen3.6-27B-GPTQ-W4A16-G32
  vllm:
    compilation-config: '{"cudagraph_mode": "FULL_AND_PIECEWISE", "compile_ranges_endpoints": []}'
    language-model-only: true
    skip-mm-profiling: true
    trust-remote-code: true
endpoint:
  url: http://127.0.0.1:8000
resources:
  tensor_parallel_size: 4
  gpu_memory_utilization: 0.85
  max_model_len: 2048
  max_num_seqs: 8
  dtype: float16
  devices: "0,1,2,3"
runtime:
  env: {VLLM_USE_V2_MODEL_RUNNER: "1"}        # rest comes from CANONICAL_ENV
  startup_wait_s: 1200
bench:
  load:
    concurrencies: [1, 4, 8, 16]
    num_prompts: 32
    input_len: 256
    output_len: 64
  scoring:
    - kind: throughput
      tool: llm-perf
cell: {family: dense, attn: triton, linear: rdna2, cg: 1, mtp: 0}
```

---

## 3. CLI Commands

Entry point `bs` (typer). All commands exit non-zero on validation failure; `--json` on report/compare for machine consumption.

| Command | Purpose | Example |
|---|---|---|
| `bs run <recipe.yaml>` | Execute one recipe: (optional) server lifecycle → bench load → scorers → report dir | `bs run recipes/qwen36-27b-gptq-tp4.yaml` |
| `bs run-all <glob>` | Execute all matching recipes sequentially (GPU file-lock serializes cells) | `bs run-all 'recipes/*.yaml'` |
| `bs matrix <recipe.yaml> --axis k=v1,v2 ...` | Sweep axes: cartesian product over `cell` fields / env vars / resources; one result cell per combo | `bs matrix base.yaml --axis attn=triton,fardna2 linear=rdna2,exllama` |
| `bs compare <dirA> <dirB>` | Diff two result dirs → delta table (md + csv), per-metric % change, regression flags vs stop_conditions | `bs compare results/a_cell/ results/b_cell/` |
| `bs report <result_dir>` | Regenerate summary.csv + summary.json + README.md from raw artifacts | `bs report results/qwen36_2026-08-30/dense_triton_rdna2_cg1_mtp0/` |
| `bs init <name>` | Scaffold a new recipe from template (prompts backend type) | `bs init my-llamacpp-bench` |
| `bs doctor <recipe.yaml>` | Validate recipe; probe endpoint health + logprobs capability; check binaries (`llm-perf`, `lm_eval`, `npx`, `inspect`, docker); check Rust/cargo if install needed | `bs doctor recipes/kld-vs-fp16-reference.yaml` |
| `bs convert-logits <dir>` *(addition, required by KLD)* | One-time torch.load of `f_*.fp16` → safetensors + `manifest.json` (shape/dtype/vocab/prompt-order) | `bs convert-logits models/kl_logits --with-torch` |

---

## 4. Implementation Phases (TDD, 7 phases)

Every phase: **write failing tests first → implement → green → lint/type clean → one atomic commit**. Est. LOC = source + tests.

### Phase 1 — Repo init & scaffolding (~180 LOC, no deps)
- **Deliverable**: git repo, `pyproject.toml` + `uv.lock`, `.python-version=3.13`, ruff + basedpyright-strict config, pre-commit, `.gitignore` (incl. `results/`), README skeleton, `PLAN.md` committed, `benchmark_suite/__init__.py` + `cli.py` stub (`bs --help` works). Parent `.gitignore` gets `benchmark_suite/` (ask user first).
- **Validation**: `uv sync --extra dev` green; `ruff check .` + `basedpyright .` green; `bs --help` exits 0; `git log` shows C0/C1.

### Phase 2 — Recipe schema (~450 src + 300 test, deps: P1)
- **Deliverable**: `recipe.py` (schema above), `paths.py` (result-dir layout + `CellId.render()`), `load_recipe()`, `merged_env()`, matrix axis expansion (`axis attn=a,b linear=c,d` → recipe variants + cell overrides).
- **TDD**: minimal-valid recipe; each scorer discriminates correctly; unknown `kind` → ValidationError; env merge precedence (recipe > canonical); cell render equals legacy `dense_fardna2_rdna2_cg1_mtp2`; matrix 2×2 → 4 recipes; missing `model_path` for `backend.type=vllm` → error.
- **Validation**: `pytest tests/test_recipe.py tests/test_paths.py` green; basedpyright strict green on the two modules.

### Phase 3 — Runner layer (~700 src + 450 test, deps: P2)
- **Deliverable**: `runner/endpoint.py` (health, `/v1/models` model-name check, **logprobs capability probe** — `POST /v1/completions` with `logprobs=5, max_tokens=1`; retry w/ backoff), `runner/serve.py` (command synthesis for vllm/llamacpp from `resources`+`backend` knobs, spawn with merged env, wait-for-health up to `startup_wait_s`, teardown with timeout, GPU file-lock `~/.cache/benchmark_suite/gpu.lock`), `runner/llm_perf.py` (TOML `[endpoint]/[load]/[input]/[output]` generation, `bench`/`logprobs`/`kl-divergence` invocation, JSON output parsing), `runner/vllm_bench.py` (subprocess wrapper + **the 11 regexes ported verbatim** from `parse_matrix_results.py`).
- **TDD**: respx-mocked endpoint probe; fake `llm-perf` shell script emitting canned JSON; golden `vllm bench` stdout fixture → exact field equality; serve.py cmd-synthesis snapshots; lock serializes two fake servers.
- **Validation**: runner tests green; `bs doctor` against a live mock endpoint reports capabilities correctly.

### Phase 4 — Scoring A: throughput + KLD (~450 src + 300 test, deps: P3)
- **Deliverable**: `scoring/base.py` (`ScoreRecord`, `Scorer` protocol, registry), `scoring/throughput.py` (llm-perf JSON → legacy `summary.csv` column names exactly; per-concurrency rows), `scoring/kl_divergence.py` (`source=logits_dir`: memmap converted safetensors, query endpoint `top_logprobs` for the manifest's prompts, per-token KL over top-k with logsumexp renormalization in float32; `source=llm-perf`: `llm-perf logprobs` both endpoints → `llm-perf kl-divergence`), `tools/convert_kl_logits.py` + `manifest.json` writer (torch only here, `--with-torch` gate).
- **TDD**: synthetic logits fixture (vocab=8, 4 tokens) → hand-computed KL equality; memmap never loads full file (memory guard); vocab-mismatch → loud refusal; llm-perf JSON → legacy columns byte-exact vs fixture.
- **Validation**: KLD of identical distributions ≈ 0.0; conversion of one real `f_0.fp16` verified against `torch.load` reference on dev box.

### Phase 5 — Scoring B: perplexity + LLM-judge + agentic (~800 src + 400 test, deps: P2, P3 probe)
- **Deliverable**: `scoring/perplexity.py` (subprocess `lm_eval --model local-completions --model_args base_url=...,model=... --tasks wikitext --output_path ...`, result JSON parse; endpoint logprobs capability pre-check), `scoring/llm_judge.py` (`driver=native`: candidate answers via endpoint, judge via `judge_url` + rubric, httpx-only, scores 0-10 + rationale artifact; `driver=promptfoo`: generated `promptfooconfig.yaml` → `npx promptfoo@<pin> eval` → parse JSON), `scoring/agentic.py` (`harness=inspect`: `inspect eval <task> --model openai-api/bench/<model>` with `OPENAI_BASE_URL` env; `harness=terminal-bench`: subprocess wrapper, **best-effort — requires docker**).
- **TDD**: argv construction unit tests (mock subprocess); lm-eval/promptfoo/inspect JSON fixture parsing; missing binary → actionable error message; native judge flow fully respx-mocked.
- **Validation**: fixture-driven tests green; each scorer degrades gracefully with clear remediation text when its tool/binary is absent.

### Phase 6 — CLI + report + compare (~700 src + 350 test, deps: P4, P5)
- **Deliverable**: full `cli.py` (all 8 commands), `report.py` (`summary.csv` with legacy columns, `summary.json`, `README.md` in parent bench_results style: Date/Goal/Configuration/Results/Verdict/Files), `compare.py` (md + csv delta tables, %-change, regression highlighting).
- **TDD**: `typer.testing.CliRunner` for every command on mocked runners; matrix expansion → expected cell list; report from fixture artifacts → golden README/csv; compare of two fixture dirs → golden delta.
- **Validation**: `bs run` on a fully mocked end-to-end path produces the complete result dir layout.

### Phase 7 — Example recipes + documentation + CI scaffolding (~250 YAML/docs, deps: P6) — **GPU smoke DEFERRED**
- **Deliverable**: the 4 recipes, full README usage section, GitHub Actions CI workflow (pytest + ruff + basedpyright + `bs doctor --dry-run` on a mock server).
- **Out of scope for v1**: actual GPU run smoke (hardware busy with bug-investigation work — to be done in a separate session when GPUs are free). CI runs the mock path; real-hardware validation is on the user.
- **Validation**: all docs/recipes parse cleanly; CI green on a fresh clone; `bs --help` + `bs doctor recipes/qwen36-27b-gptq-tp4.yaml --dry-run` exit 0 against respx-mocked endpoint.

**Total estimate**: ~3.5k LOC source + ~1.6k LOC tests.

---

## 5. Risks & Open Questions

| # | Risk / Open question | Mitigation / when resolved |
|---|---|---|
| R1 | **llm-perf has no binaries/crate** — needs Rust ≥1.85 + `cargo build --release` (pinned `v0.1.16`). Dev box has cargo; `.176` toolchain unverified. TOML-config-only interface may lag our flag needs | `scripts/install_llm_perf.sh` pins the tag; `bs doctor` checks binary + rustc version; `vllm-bench` fallback scorer covers boxes without Rust |
| R2 | **PyPI `promptfoo==0.1.4` is a thin wrapper around the Node CLI** — Node/npx required regardless; wrapper behavior at 0.1.4 unverified against current npm promptfoo | Default `driver=native` (pure httpx rubric judge → zero Node dependency); promptfoo driver validated first thing in P5 task T8 before relying on the wrapper; npm version pinned in schema |
| R3 | **lm-eval 0.4.12 pulls torch + datasets** (heavy, ~GBs) and its dep tree may lag Python 3.14 | Optional extra `perplexity`; project venv pinned 3.13 (`<3.14`); scorer invokes lm_eval as **subprocess** (isolation); `bs doctor` reports extra-missing with exact install command |
| R4 | **KLD cache provenance is opaque by design** — we do not assume what model/tokenizer/prompts produced any existing logits file | `bs convert-logits` REQUIRES explicit `--tokenizer` + `--model-name` + `--prompts` flags; writes `manifest.json` with shape/dtype/vocab/prompt-order + provenance hash; scorer refuses if manifest missing or vocab mismatches endpoint tokenizer (`vocab_check: true`); memmaps the cache, never loads full file into RAM |
| R5 | **GPU contention / server ownership** — matrix cells and run-all serialize on shared GPUs; a crashed server must not leak VRAM into the next cell | `runtime.gpu_lock` file-lock around GPU-owning phases; teardown with timeout + `rocm-smi` VRAM check between cells; `backend.type=external` = probe-only, no lifecycle |
| Q1 | Agentic scope: `terminal-bench` needs docker-in-host and its own agent abstraction — **v1 keeps it best-effort**; confirm acceptable | Resolved at P5/T9: inspect-ai is the supported harness; terminal-bench ships behind a clear "experimental" warning |
| Q2 | Default judge model for `llm_judge` when judging on the same endpoint being benchmarked (self-judge bias) | Schema allows any `judge_url`; README recommends a fixed external judge endpoint for comparability |

---

## 6. Acceptance Criteria (v1 "done")

1. **Recipe → results**: I write a recipe YAML, run `bs run it.yaml`, and get `results/<name>_<ts>/<cell_id>/` containing `summary.csv` (legacy columns), `summary.json`, `README.md` (parent style), `logs/`, `artifacts/`.
2. **Comparison**: `bs compare results/a/ results/b/` emits a markdown + csv delta table with %-change per metric.
3. **KLD continuity**: the KLD scorer reads my existing `models/kl_logits/` (after one-time `bs convert-logits`) and scores a quantized endpoint against the fp16 reference; identical-distribution sanity check ≈ 0.
4. **Local judge**: `llm_judge` scores a candidate using my local vLLM endpoint as judge with zero Node dependency (`driver=native`).
5. **Doctor**: `bs doctor recipe.yaml` validates the recipe, probes endpoint health + logprobs capability, and reports missing tools with install commands — before any GPU work starts.
5. **Matrix**: `bs matrix base.yaml --axis attn=triton,fardna2 linear=rdna2,exllama` produces 4 correctly-named cells (`dense_triton_rdna2_cg1_mtp0`, …) serially under the GPU lock.
6. **Quality gates**: `pytest` green, `ruff check` clean, `basedpyright` strict clean; all commits atomic and pushed.

---

## 7. Task Dependency Graph

| Task | Depends On | Reason |
|---|---|---|
| T1 Repo init | None | Starting point |
| T2 Recipe schema + paths | T1 | Needs pyproject/venv/lint config |
| T3 Endpoint probe + serve lifecycle | T2 | Consumes Recipe/merged_env/resources |
| T4 llm-perf wrapper + vllm-bench fallback | T2 | Consumes Recipe load/bench sections; mocked independently of T3 |
| T5 Throughput scorer + base/registry | T4 | Maps llm-perf/vllm-bench outputs to ScoreRecord |
| T6 KLD scorer + convert-logits | T2 | Schema + filesystem only; endpoint probe soft-dep (mocked) |
| T7 Perplexity scorer | T3 | Needs live-endpoint capability probe |
| T8 LLM-judge scorer | T3 | Needs endpoint probe (candidate + judge endpoints) |
| T9 Agentic scorer | T2 | Schema + subprocess wrappers; endpoint env wiring from T3 (mocked) |
| T10 report.py + compare.py | T5 | summary.csv/json schema defined by ScoreRecord |
| T11 CLI (all 8 commands) | T3, T4, T5, T6, T7, T8, T9, T10 | Wires everything |
| T12 Example recipes + E2E smoke | T11 | Uses the real CLI on real hardware |

**Critical path**: T1 → T2 → T3 → T8 → T11 → T12 (longest scorer chain through judge; throughput path T4 → T5 → T10 is shorter but all feed T11).

## 8. Parallel Execution Graph

```
Wave 1 (no deps):           T1
Wave 2:                     T2
Wave 3 (after T2):          T3, T4                      ← parallel
Wave 4 (after T3/T4):       T5, T6, T7, T8, T9          ← 5 parallel scorer tasks
Wave 5 (after T5):          T10                         ← can start once T5 lands, overlaps Wave 4
Wave 6 (after all scorers): T11
Wave 7:                     T12 (real-hardware E2E)
```

Estimated speedup vs fully sequential: ~35–40% (five scorers + report parallelized).

## 9. Tasks & Delegation (category + skills per task)

### T1: Repo init & scaffolding
- **Category**: `quick` — mechanical scaffolding, single-purpose edits.
- **Skills**: `["git-master"]` — git init + first atomic commits.
- **Skills evaluation**: OMITTED `programming` (no substantive Python yet); OMITTED `tdd` (no logic to test); INCLUDED `git-master` (commit hygiene from commit zero).

### T2: Recipe schema + paths
- **Category**: `deep` — the schema is the contract everything else consumes; mistakes here multiply.
- **Skills**: `["programming", "tdd"]` — strict Pydantic v2 typing, basedpyright-strict; red-green for the validation matrix.
- **Skills evaluation**: INCLUDED `programming` (Python strict-types domain); INCLUDED `tdd` (schema is pure logic, ideal red-green target); OMITTED `refactor` (greenfield); OMITTED `data-scientist` (no data processing).

### T3: Endpoint probe + serve lifecycle
- **Category**: `deep` — subprocess lifecycle, GPU lock, health probing have real failure modes.
- **Skills**: `["programming", "tdd", "endpoint-testing"]` — httpx/respx probing discipline from endpoint-testing; TDD for lifecycle state machine.
- **Skills evaluation**: INCLUDED `endpoint-testing` (this task builds the probe that skill formalizes); OMITTED `debugging` (nothing broken yet).

### T4: llm-perf wrapper + vllm-bench fallback
- **Category**: `deep` — external-binary contract (TOML gen, JSON parse) + verbatim regex port with golden fixtures.
- **Skills**: `["programming", "tdd"]`.
- **Skills evaluation**: INCLUDED both; OMITTED `endpoint-testing` (wrapper doesn't probe, T3 did).

### T5: Throughput scorer + base/registry
- **Category**: `quick` — mapping parsed results → legacy CSV columns is small once T4 exists; but it defines ScoreRecord, so correctness matters more than size.
- **Skills**: `["programming", "tdd"]`.
- **Skills evaluation**: INCLUDED both; OMITTED `data-scientist` (fixed column mapping, no analysis).

### T6: KLD scorer + convert-logits
- **Category**: `deep` — 12 GB memmap handling, float32 logsumexp numerics, torch.save zip format, vocab alignment.
- **Skills**: `["programming", "tdd", "data-scientist"]` — numpy memmap/dtype verification is data-scientist territory; TDD with hand-computed KL fixture.
- **Skills evaluation**: INCLUDED `data-scientist` (binary format verification + conversion validation); OMITTED `debugging` (unless conversion mismatches reference — then escalate).

### T7: Perplexity scorer
- **Category**: `deep` — lm-eval subprocess contract, optional-extra gating, logprobs capability pre-check.
- **Skills**: `["programming", "tdd"]`.
- **Skills evaluation**: INCLUDED both; OMITTED `endpoint-testing` (probe already built in T3; scorer consumes it).

### T8: LLM-judge scorer
- **Category**: `deep` — two drivers (native httpx, promptfoo), judge-endpoint auth, promptfoo wrapper verification (R2).
- **Skills**: `["programming", "tdd", "endpoint-testing"]` — endpoint-testing for validating judge + candidate endpoints before scoring.
- **Skills evaluation**: INCLUDED `endpoint-testing` (judge-endpoint coherence probe); OMITTED `playwright` (no browser).

### T9: Agentic scorer
- **Category**: `deep` — inspect-ai subprocess + env wiring; terminal-bench docker gating.
- **Skills**: `["programming", "tdd"]`.
- **Skills evaluation**: INCLUDED both; OMITTED `endpoint-testing` (uses T3 probe).

### T10: report.py + compare.py
- **Category**: `unspecified-high` — multi-format golden-output work (csv/json/md) across all scorer kinds.
- **Skills**: `["programming", "tdd", "writing"]` — README.md generation must match the parent bench_results house style.
- **Skills evaluation**: INCLUDED `writing` (report prose/templates); OMITTED `visual-qa` (markdown tables, not UI).

### T11: CLI (8 commands)
- **Category**: `unspecified-high` — broad integration surface; typer + CliRunner tests.
- **Skills**: `["programming", "tdd"]`.
- **Skills evaluation**: INCLUDED both; OMITTED `frontend`/`playwright` (CLI, not web UI).

### T12: Example recipes + E2E smoke on real hardware
- **Category**: `deep` — real-endpoint validation on gfx1030/gfx1100; the only task that touches GPUs.
- **Skills**: `["endpoint-testing", "debugging"]` — probe output coherence before trusting throughput; debugging for the inevitable real-hardware surprises.
- **Skills evaluation**: INCLUDED `endpoint-testing` (mandatory pre-bench coherence check per repo skill); INCLUDED `debugging` (live-system validation); OMITTED `tdd` (E2E is a checklist run, not red-green).

## 10. Atomic Commit Strategy

One commit per red→green slice, tests + implementation together (TDD: commit at green). Conventional-commit style per parent repo rules (type + description + What/Why/Testing body). Push after each commit — never leave green work uncommitted.

| # | Commit message | Lands |
|---|---|---|
| C0 | `docs: add PLAN.md — decision-complete implementation plan` | PLAN.md |
| C1 | `chore: repo scaffold (pyproject, uv.lock, ruff/basedpyright, pre-commit, README skeleton)` | P1 |
| C2 | `feat(recipe): Pydantic v2 schema, YAML loader, cell-id render, matrix expansion` | P2 |
| C3 | `feat(runner): endpoint capability probe + server lifecycle with GPU lock` | P3a |
| C4 | `feat(runner): llm-perf TOML wrapper + vllm-bench fallback parser (golden fixtures)` | P3b |
| C5 | `feat(scoring): throughput scorer + ScoreRecord/registry (legacy summary.csv columns)` | P4a |
| C6 | `feat(scoring): KLD scorer (logits_dir memmap + llm-perf native) + convert-logits tool` | P4b |
| C7 | `feat(scoring): perplexity scorer via lm-eval local-completions (subprocess)` | P5a |
| C8 | `feat(scoring): LLM-judge scorer — native httpx rubric + promptfoo driver` | P5b |
| C9 | `feat(scoring): agentic scorer — inspect-ai (terminal-bench experimental)` | P5c |
| C10 | `feat(report): summary/README writer (parent bench_results style) + compare` | P6a |
| C11 | `feat(cli): bs run/run-all/matrix/compare/report/init/doctor/convert-logits` | P6b |
| C12 | `feat(recipes): 4 example recipes + README usage + CI workflow (GPU smoke deferred)` | P7 |

Remote: create `BlivionIaG/benchmark_suite` (PRIVATE) at C1; `git push -u origin master` then push each commit. User has accepted PRIVATE visibility for v1.

## TODO List (ADD THESE)

### Wave 1 (start immediately)

- [ ] **1. T1 Repo init & scaffolding**
  - What: mkdir benchmark_suite; write PLAN.md (this file verbatim); pyproject.toml + `.python-version=3.13`; `uv sync --extra dev`; ruff/basedpyright-strict/pre-commit configs; .gitignore (results/, .venv/); README skeleton; package + `cli.py` stub (`bs --help`); `git init`; add `benchmark_suite/` to parent .gitignore (ask user); C0+C1; create remote + push.
  - Depends: none. Blocks: T2.
  - Category: `quick`. Skills: [`git-master`]
  - QA: `uv run bs --help` exits 0; `ruff check . && basedpyright .` clean; `git log --oneline` = C0,C1; remote tracking set.

### Wave 2

- [ ] **2. T2 Recipe schema + paths**
  - What: implement `recipe.py` exactly per §2 schema + `paths.py` (CellId.render, result-dir layout) + matrix axis expansion; tests first (discrimination, defaults, env merge, cell render = `dense_fardna2_rdna2_cg1_mtp2`, 2×2 matrix → 4 variants).
  - Depends: 1. Blocks: T3, T4, T6, T9.
  - Category: `deep`. Skills: [`programming`, `tdd`]
  - QA: `pytest tests/test_recipe.py tests/test_paths.py` green; basedpyright strict clean on both modules; C2 pushed.

### Wave 3 (after T2 — run in parallel)

- [ ] **3. T3 Endpoint probe + serve lifecycle**
  - What: `runner/endpoint.py` (health, /v1/models, logprobs capability, retry) + `runner/serve.py` (cmd synthesis vllm/llamacpp, spawn, wait-health, teardown, GPU file-lock); respx-mocked tests + cmd-synthesis snapshots + lock serialization test.
  - Depends: 2. Blocks: T7, T8, T11.
  - Category: `deep`. Skills: [`programming`, `tdd`, `endpoint-testing`]
  - QA: runner tests green; probe against respx mock classifies health/logprobs correctly; C3 pushed.

- [ ] **4. T4 llm-perf wrapper + vllm-bench fallback**
  - What: `runner/llm_perf.py` (TOML gen `[endpoint]/[load]/[input]/[output]`, bench/logprobs/kl-divergence, JSON parse) + `runner/vllm_bench.py` (11 regexes verbatim from parent parser); fake-binary + golden-stdout fixtures.
  - Depends: 2. Blocks: T5.
  - Category: `deep`. Skills: [`programming`, `tdd`]
  - QA: TOML snapshot test matches llm-perf v0.1.16 config shape; golden vllm stdout → exact legacy field values; C4 pushed.

### Wave 4 (after T3/T4 — run in parallel)

- [ ] **5. T5 Throughput scorer + base/registry**
  - What: `scoring/base.py` (ScoreRecord, Scorer protocol, registry) + `scoring/throughput.py` → legacy summary.csv columns.
  - Depends: 4. Blocks: T10, T11.
  - Category: `quick`. Skills: [`programming`, `tdd`]
  - QA: llm-perf JSON fixture → summary.csv byte-exact columns; C5 pushed.

- [ ] **6. T6 KLD scorer + convert-logits**
  - What: `tools/convert_kl_logits.py` (torch.load → safetensors + manifest.json; REQUIRES explicit `--tokenizer` + `--model-name` + `--prompts` flags — refuses to guess); `scoring/kl_divergence.py` dual-source (memmap top-k KL float32 logsumexp; llm-perf native); synthetic-fixture KL equality; vocab-mismatch refusal; missing-manifest refusal (points user to `bs convert-logits`).
  - Depends: 2. Blocks: T11.
  - Category: `deep`. Skills: [`programming`, `tdd`, `data-scientist`]
  - QA: KL(identical)≈0.0 on fixture; conversion of synthetic fp16 logit blob matches torch.load reference; missing-manifest → clear "run `bs convert-logits`" error; never loads >1 GiB RSS (memmap); C6 pushed.

- [ ] **7. T7 Perplexity scorer**
  - What: `scoring/perplexity.py` lm_eval subprocess (`local-completions`, wikitext), JSON parse, capability pre-check, extra-missing remediation error.
  - Depends: 3. Blocks: T11.
  - Category: `deep`. Skills: [`programming`, `tdd`]
  - QA: argv construction test + lm-eval JSON fixture parse green; missing-extra error actionable; C7 pushed.

- [ ] **8. T8 LLM-judge scorer**
  - What: verify PyPI `promptfoo` wrapper behavior FIRST (R2); `scoring/llm_judge.py` driver=native (httpx rubric, respx-tested) + driver=promptfoo (config gen, npx pinned); judge endpoint probe reuse.
  - Depends: 3. Blocks: T11.
  - Category: `deep`. Skills: [`programming`, `tdd`, `endpoint-testing`]
  - QA: native judge flow fully mocked green; promptfoo argv/config golden; works with `judge_url=http://127.0.0.1:8000`; C8 pushed.

- [ ] **9. T9 Agentic scorer**
  - What: `scoring/agentic.py` inspect-ai subprocess (`--model openai-api/bench/<model>`, OPENAI_BASE_URL wiring); terminal-bench wrapper with docker check + experimental warning.
  - Depends: 2. Blocks: T11.
  - Category: `deep`. Skills: [`programming`, `tdd`]
  - QA: argv/env construction tests green; missing docker/inspect → clear remediation; C9 pushed.

### Wave 5 (overlaps Wave 4 after T5)

- [ ] **10. T10 report.py + compare.py**
  - What: summary.csv (legacy columns)/summary.json/README.md (parent style) writers; compare → md+csv delta with %-change + regression flags.
  - Depends: 5. Blocks: T11.
  - Category: `unspecified-high`. Skills: [`programming`, `tdd`, `writing`]
  - QA: golden README/csv from fixture artifacts; compare of two fixture dirs → golden delta; C10 pushed.

### Wave 6

- [ ] **11. T11 CLI — all 8 commands**
  - What: `cli.py` run/run-all/matrix/compare/report/init/doctor/convert-logits wiring T3–T10; CliRunner tests on mocked runners.
  - Depends: 3,4,5,6,7,8,9,10. Blocks: T12.
  - Category: `unspecified-high`. Skills: [`programming`, `tdd`]
  - QA: mocked end-to-end `bs run` produces full result-dir layout; `bs matrix` 2×2 → 4 correctly-named cells; `bs doctor` mock-endpoint report correct; C11 pushed.

### Wave 7

- [ ] **12. T12 Example recipes + README + CI (GPU smoke deferred)**
  - What: 4 recipes per §1; full README usage section; GitHub Actions CI workflow (pytest + ruff + basedpyright + `bs doctor recipes/*.yaml --dry-run` against respx mock); docs note for "real-hardware smoke" as a future manual step.
  - Depends: 11. Blocks: none (final).
  - Category: `unspecified-high`. Skills: [`programming`, `writing`]
  - QA: CI green on fresh clone; all 4 recipes parse + `bs doctor --dry-run` exit 0 against mock; README has install/quickstart/recipe-anatomy/scorer-reference/troubleshooting sections; C12 pushed; `git status` clean.

## Execution Instructions

1. **Wave 1**: `task(category="quick", load_skills=["git-master"], run_in_background=false, prompt="T1: ...")`
2. **Wave 2**: `task(category="deep", load_skills=["programming","tdd"], prompt="T2: ...")`
3. **Wave 3** (parallel): T3 + T4 as two background tasks.
4. **Wave 4** (parallel): T5, T6, T7, T8, T9 as background tasks; start T10 as soon as T5 lands.
5. **Wave 6**: T11 after all scorers + T10 green.
6. **Wave 7**: T12 on real hardware; verify every §6 box.
7. Final QA: `pytest && ruff check . && basedpyright .` clean; 13 commits (C0–C12) all pushed; `git status` clean.

## Success Criteria (final verification, GPU smoke deferred)

- §6 acceptance criteria 1–5 pass on a fresh clone via CI + mock endpoint (criterion 6 — real-hardware matrix — deferred until GPUs are free).
- `bs doctor` prevents at least one misconfiguration class (missing logprobs, missing binary, wrong model name, vocab mismatch) with an actionable message.
- Legacy continuity: summary.csv columns, cell-id format, README structure match parent `bench_results/` conventions byte-for-byte where applicable.
- 13 atomic commits on `master`, all pushed to PRIVATE `BlivionIaG/benchmark_suite`; repo clonable + `uv sync && uv run bs --help` works from a fresh clone.
