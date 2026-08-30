# benchmark_suite

Declarative benchmarking and quality scoring for OpenAI-compatible LLM endpoints.

[![CI](https://github.com/BlivionIaG/benchmark_suite/actions/workflows/ci.yml/badge.svg)](https://github.com/BlivionIaG/benchmark_suite/actions)

## What it is

A CLI tool (`bs`) that runs reproducible benchmarks against any OpenAI-compatible HTTP endpoint — vLLM, llama.cpp server, TGI, OpenAI API — and produces comparable reports.

Five scorers shipped:

| Scorer | What it measures | Tool |
|---|---|---|
| `throughput` | output tokens/s, TTFT, TPOT, ITL | llm-perf (primary) / vllm-bench (fallback) |
| `perplexity` | standard lm-eval tasks (wikitext etc) | lm-eval-harness via /v1/completions |
| `kld` | per-token KL divergence vs reference distribution | custom top-k KL (memmap'd safetensors cache or llm-perf logprobs) |
| `llm_judge` | 0–10 quality score from a judge LLM | httpx (native, no Node) or promptfoo |
| `agentic` | task pass rate (inspect-ai) or terminal tasks (experimental) | inspect-ai / terminal-bench |

## Install

```bash
# Requires uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/BlivionIaG/benchmark_suite.git
cd benchmark_suite
uv sync --extra dev         # core only
uv sync --all-extras        # with lm-eval, promptfoo, inspect-ai, terminal-bench
```

## Quickstart

```bash
# 1. Validate a recipe and probe the endpoint
bs doctor recipes/qwen36-27b-gptq-tp4.yaml

# 2. Run a recipe end-to-end (spawns vllm if needed, runs scorers, writes summary)
bs run recipes/qwen36-27b-gptq-tp4.yaml

# 3. Sweep axes — one cell per combination
bs matrix recipes/qwen36-27b-gptq-tp4.yaml --axis cell.attn=triton,fardna2 --axis cell.linear=rdna2,exllama

# 4. Compare two result dirs
bs compare results/cell_a results/cell_b

# 5. Regenerate summary from existing artifacts
bs report results/qwen36_2026-08-30/dense_fardna2_rdna2_cg1_mtp0/

# 6. Scaffold a new recipe
bs init my-new-bench
```

## Recipe Anatomy

A recipe is a YAML file with these top-level sections:

| Section | Purpose |
|---|---|
| `meta` | Name, description, version, tags |
| `backend` | Backend type (`vllm`, `llamacpp`, `tgi`, `external`) + model path |
| `endpoint` | URL, API key env var, model name, timeouts |
| `resources` | TP/PP size, GPU mem utilization, max model len, dtype |
| `runtime` | Environment overrides, server startup config, GPU lock |
| `bench` | Load profile + scorers |
| `report` | Output format, output dir |
| `cell` | Cell ID fields (family, attn, linear, cg, mtp) |

Example recipe (`recipes/qwen36-27b-gptq-tp4.yaml`):

```yaml
meta:
  name: qwen36-27b-gptq-tp4
  description: Qwen3.6-27B-GPTQ-W4A16-G32 on 4x V620
  tags: [gfx1030, gptq, w4a16, tp4]

backend:
  type: vllm
  model_path: /models/Qwen3.6-27B-GPTQ-W4A16-G32

endpoint:
  url: http://127.0.0.1:8000

resources:
  tensor_parallel_size: 4
  gpu_memory_utilization: 0.85
  max_model_len: 2048
  dtype: float16

bench:
  load:
    concurrencies: [1, 4, 8, 16]
    num_prompts: 32
    input_len: 256
    output_len: 64
  scoring:
    - kind: throughput
      tool: llm-perf

cell:
  family: dense
  attn: fardna2
  linear: rdna2
  cg: 1
  mtp: 0
```

### Shipped recipes

| Recipe | Purpose |
|---|---|
| `recipes/qwen36-27b-gptq-tp4.yaml` | Dense W4A16 production cell on 4x V620 (gfx1030) — V2 runner, FPP cudagraphs, RDNA2 kernels |
| `recipes/qwen36-35b-a3b-fp16-tp4.yaml` | MoE FP16 reference — baseline for quantization comparison |
| `recipes/perplexity-compare.yaml` | Cross-platform wikitext perplexity against any external endpoint |
| `recipes/kld-vs-fp16-reference.yaml` | KLD divergence vs a converted fp16 reference cache |

`backend.type: external` recipes never spawn a server — `bs run` probes the endpoint and runs scorers only. Point `endpoint.url` at whatever is already running.

## Result Layout

Each `bs run` produces:
```
results/<name>_<ts>/<cell_id>/
├── summary.csv           # legacy column names (output_tok_s, ttft_mean_ms, …)
├── summary.json          # structured: recipe + scores
├── README.md             # human-readable report
├── logs/                 # server stdout/stderr (per cell)
└── artifacts/            # raw scorer outputs (llm-perf JSON, lm-eval results, etc.)
```

## Scorer Reference

### throughput

```yaml
scoring:
  - kind: throughput
    tool: llm-perf      # or "vllm-bench"
    extra_flags: []     # extra CLI flags
```

### perplexity

```yaml
scoring:
  - kind: perplexity
    tasks: [wikitext]   # any lm-eval task
    num_fewshot: 0
    limit: 100          # optional sample cap
```

**Note**: uses `/v1/completions` (not `/v1/chat/completions`) because OpenAI does not expose per-token logprobs on the chat endpoint. vLLM and most OpenAI-compatible servers support this.

### kld

```yaml
scoring:
  - kind: kld
    source: logits_dir              # or "llm-perf"
    reference_logits_dir: ./kl_ref  # must have manifest.json
    top_k: 128
    max_tokens: 1
    vocab_check: true               # refuse cross-tokenizer comparison
```

**Prerequisite**: run `bs convert-logits` to produce the cache:
```bash
bs convert-logits /path/to/fp16/logits \
  --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
  --model-name qwen2.5-0.5b-fp16 \
  --prompts prompts.jsonl \
  --output ./kl_ref
```

The KLD scorer **refuses to run** if `manifest.json` is missing or the endpoint's vocab size doesn't match the reference. This is by design — silently comparing different tokenizers produces meaningless numbers.

### llm_judge

```yaml
scoring:
  - kind: llm_judge
    driver: native                # or "promptfoo"
    judge_url: http://127.0.0.1:8001   # use a different endpoint for judging
    judge_model: ""
    rubric: "Score 0-10 for correctness and clarity."
```

**Note**: For comparability across benchmarks, use a fixed external judge endpoint rather than self-judging against the candidate endpoint (self-judge bias).

### agentic

```yaml
scoring:
  - kind: agentic
    harness: inspect               # or "terminal-bench" (experimental)
    tasks: [inspect-ai-task-name]
    sandbox: docker
```

`terminal-bench` requires Docker on the host and is best-effort — failures are surfaced in the report.

## CLI Commands

| Command | Description |
|---|---|
| `bs version` | Print version and exit |
| `bs doctor <recipe> [--dry-run]` | Validate recipe + probe endpoint + check binaries |
| `bs run <recipe>` | Execute one recipe end-to-end |
| `bs run-all <glob>` | Execute all matching recipes serially |
| `bs matrix <recipe> --axis k=v1,v2` | Sweep axes → one cell per combo |
| `bs compare <dirA> <dirB>` | Delta table between two result dirs |
| `bs report <result_dir>` | Regenerate summaries from artifacts |
| `bs init <name>` | Scaffold a new recipe |
| `bs convert-logits <dir> --tokenizer <hf-id> --model-name <name> --prompts <file> --output <dir>` | One-time `f_*.fp16` → safetensors + manifest |

## Supported Backends

- **vLLM** (any version with OpenAI-compatible server): `backend.type: vllm` (default).
- **llama.cpp server** (any version with `/v1` routes): `backend.type: llamacpp`.
- **TGI** (HuggingFace text-generation-inference): `backend.type: tgi`.
- **External endpoint** (no lifecycle management): `backend.type: external`. `bs run` probes + runs scorers only.

## Tooling Notes

- **llm-perf**: The primary load generator. Install from source: `cargo install --git https://github.com/iopsystems/llm-perf --tag v0.1.16`. Or use the wrapper with `vllm-bench` fallback (no Rust required).
- **vllm-bench**: Built into vLLM. `vllm bench throughput --model <model> --url <endpoint> ...`. Used when llm-perf is unavailable.
- **lm-eval**: Install with `uv add --optional perplexity lm-eval==0.4.12`. Heavy dep (~GBs) due to torch + datasets.
- **safetensors**: Required by KLD scorer (no torch dependency at scorer runtime; torch only in `bs convert-logits`).
- **inspect-ai**: Install with `uv add --optional agentic inspect-ai==0.3.260`.
- **terminal-bench**: Experimental; requires Docker.

## Development

```bash
uv sync --extra dev
uv run pytest tests/ -q       # 182 tests
uv run ruff check .
uv run basedpyright benchmark_suite/ tests/
```

Atomic commits with conventional format (`<type>: <description>` + What/Why/Testing body). See `AGENTS.md`.

## Comparison Across Setups

Because all benchmarks hit OpenAI-compatible endpoints, you can compare across:
- Different serving stacks (vLLM vs llama.cpp vs TGI vs OpenAI API)
- Different quantizations (FP16 vs AWQ vs GPTQ vs EXL3)
- Different hardware (gfx1030 vs gfx1100 vs NVIDIA)
- Different model checkpoints (fp16 control vs quantized variant via KLD)

Run the same recipe against two endpoints and `bs compare` to see the delta.

## Limitations

- **GPU smoke deferred**: This v1 ships without an end-to-end GPU run on real hardware. CI runs the recipe-validation + `--dry-run` path. Real-hardware validation is a separate session.
- **Logprobs via /v1/completions only**: `/v1/chat/completions` does not expose per-token logprobs. Endpoints that only support chat-completions cannot be perplexity-scored.
- **Self-judge bias**: LLM-judge scores are biased when the judge and candidate are the same endpoint. Use a fixed external judge.
