"""benchmark_suite/recipe.py — declarative recipe schema (Pydantic v2)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal

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
    # e.g. compilation-config, language-model-only
    vllm: dict[str, object] = Field(default_factory=dict)
    # e.g. n-gpu-layers, flash-attn
    llamacpp: dict[str, object] = Field(default_factory=dict)
    tgi: dict[str, object] = Field(default_factory=dict)


class EndpointSection(BaseModel):
    url: str = "http://127.0.0.1:8000"
    api_key_env: str = "OPENAI_API_KEY"
    # payload `model`; defaulted in Recipe validator
    model_name: str = ""
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


# Vendor identifiers for the localmaxxing `hardware.gpuName` mapping.
# localmaxxing accepts any string for `gpuName`; this enum keeps recipes
# consistent and exposes a typed surface for tooling.
GpuVendor = Literal["amd", "nvidia", "intel", "apple", "tenstorrent", "other"]


class HardwareSection(BaseModel):
    """Hardware identity for the run, surfaced to leaderboard submissions.

    Maps to the `hardware` field of the
    [localmaxxing.com](https://www.localmaxxing.com) `POST /api/speed-tests`
    schema (`hwClass: DISCRETE_GPU` for dGPU / multi-dGPU setups). The
    fields below are the subset the leaderboard API exposes for dGPU
    hardware; other `hwClass` values (`UNIFIED`, `CPU_ONLY`) require
    fields this schema does not currently model — those are out of scope
    for v1.

    All fields are optional at parse time. `bs submit` requires
    `vendor`, `model`, and at least one of `vram_gb` / `count` to
    construct a valid payload.
    """
    vendor: GpuVendor = "amd"
    model: str = ""
    count: int = 1
    vram_gb: int = 0
    cpu: str = ""
    ram_gb: int = 0
    os: str = ""
    power_watts: int = 0

    def is_complete(self) -> bool:
        return bool(self.model) and self.vram_gb > 0 and self.count > 0


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
    dataset: Path | None = None                      # JSONL {"prompt": ...}; None → synthetic
    warmup_requests: int = 4
    qps: float | None = None                         # fixed-QPS mode when set
    duration_s: float | None = None                  # soak mode when set


class StopConditions(BaseModel):
    max_duration_s: float | None = None
    max_failures: int = 0
    min_output_tok_s: float | None = None


class ThroughputScorer(BaseModel):
    kind: Literal["throughput"] = "throughput"
    tool: Literal["llm-perf", "vllm-bench"] = "llm-perf"
    extra_flags: list[str] = Field(default_factory=list)


class PerplexityScorer(BaseModel):
    kind: Literal["perplexity"] = "perplexity"
    tasks: list[str] = Field(default_factory=lambda: ["wikitext"])
    num_fewshot: int = 0
    limit: int | None = None
    lm_eval_extra_args: list[str] = Field(default_factory=list)


class KLDScorer(BaseModel):
    kind: Literal["kld"] = "kld"
    source: Literal["logits_dir", "llm-perf"] = "logits_dir"
    reference_logits_dir: Path = Path("models/kl_logits")   # converted cache + manifest.json
    reference_endpoint: str = ""                        # source=llm-perf: baseline endpoint URL
    prompts_file: Path | None = None
    top_k: int = 128
    max_tokens: int = 1
    vocab_check: bool = True                            # refuse cross-tokenizer comparison


class LLMJudgeScorer(BaseModel):
    kind: Literal["llm_judge"] = "llm_judge"
    driver: Literal["native", "promptfoo"] = "native"   # native = httpx rubric judge (no Node)
    judge_url: str = "http://127.0.0.1:8000"
    judge_model: str = ""
    judge_api_key_env: str = "OPENAI_API_KEY"
    prompts_file: Path | None = None
    rubric: str = "Score the answer 0-10 for correctness and clarity."
    promptfoo_config: Path | None = None
    promptfoo_version: str = "0.118.5"                  # npm pin when driver=promptfoo


class AgenticScorer(BaseModel):
    kind: Literal["agentic"] = "agentic"
    harness: Literal["inspect", "terminal-bench"] = "inspect"
    tasks: list[str] = Field(default_factory=list)
    limit: int | None = None
    sandbox: str = "docker"


ScorerConfig = Annotated[
    ThroughputScorer | PerplexityScorer | KLDScorer | LLMJudgeScorer | AgenticScorer,
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
    hardware: HardwareSection = Field(default_factory=HardwareSection)
    quantization: str = ""

    @model_validator(mode="after")
    def _coherence(self) -> Recipe:
        if self.backend.type != "external" and not self.backend.model_path:
            raise ValueError("backend.model_path required when backend.type != 'external'")
        if not self.endpoint.model_name:
            self.endpoint.model_name = (
                self.backend.served_model_name or Path(self.backend.model_path).name
            )
        if not self.quantization and self.resources.dtype == "float16":
            self.quantization = "FP16"
        return self

    def merged_env(self) -> dict[str, str]:
        return {**CANONICAL_ENV, **self.runtime.env}


def load_recipe(path: Path) -> Recipe:
    return Recipe.model_validate(yaml.safe_load(Path(path).read_text()))