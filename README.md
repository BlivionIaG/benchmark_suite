# benchmark_suite

Declarative benchmarking and quality scoring for OpenAI-compatible LLM endpoints, with first-class submission to the [localmaxxing.com](https://www.localmaxxing.com) public leaderboard.

[![CI](https://github.com/BlivionIaG/benchmark_suite/actions/workflows/ci.yml/badge.svg)](https://github.com/BlivionIaG/benchmark_suite/actions)

## What it is

A CLI tool (`bs`) that runs reproducible benchmarks against any OpenAI-compatible HTTP endpoint — vLLM, llama.cpp server, TGI, OpenAI API — and posts the results to [localmaxxing.com](https://www.localmaxxing.com) (a public inference leaderboard with structured hardware/software/model metadata).

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

# 7. Submit to the localmaxxing.com leaderboard (via the official `lmx` CLI)
#    First-time setup (one time only):
#      1. Install lmx: https://github.com/LottoLottoLotto/localmaxxing-cli/releases/latest
#      2. lmx auth login   # opens browser, generates API key, saves to ~/.config/localmaxxing/
#    Then:
bs submit results/qwen36_2026-08-30/dense_fardna2_rdna2_cg1_mtp0/

# 8. (optional) Inspect the JSON payload before submitting
bs export results/qwen36_2026-08-30/dense_fardna2_rdna2_cg1_mtp0/ -o /tmp/payload.json
lmx speed-test dry-run /tmp/payload.json
```

## Submitting to localmaxxing.com (via `lmx`)

`benchmark_suite` is a **recipe + result-dir** automation layer. The actual leaderboard submission is delegated to the official [LocalMaxxing CLI (`lmx`)](https://github.com/LottoLottoLotto/localmaxxing-cli) — the same binary the `lmx` users run directly.

`bs submit` builds the localmaxxing JSON payload from the recipe + `summary.json`, writes it to a temp file, and shells out to:

```bash
lmx speed-test submit /tmp/bs-submit-XXX.json
# or with --dry-run:
lmx speed-test dry-run /tmp/bs-submit-XXX.json
```

`bs` does **not** make HTTP calls directly. Authentication, request formatting, response parsing, retries, and rate-limit handling are all `lmx`'s responsibility.

### Install `lmx`

```bash
# Linux (amd64)
curl -fsSLO https://github.com/LottoLottoLotto/localmaxxing-cli/releases/latest/download/lmx-linux-amd64.tar.gz
curl -fsSLO https://github.com/LottoLottoLotto/localmaxxing-cli/releases/latest/download/checksums.txt
sha256sum --check --ignore-missing checksums.txt
tar -xzf lmx-linux-amd64.tar.gz
sudo mv lmx /usr/local/bin/

# Or build from source (Go 1.22+)
go install github.com/LottoLottoLotto/localmaxxing-cli/cmd/lmx@latest

# Verify
lmx --version
```

### Authenticate

```bash
# Recommended for agents/CI: set the env var
export LMX_API_KEY=bhk_...

# Or interactive device-flow login (opens browser)
lmx auth login

# Or save the key locally (no shell history leak)
printf '%s\n' "$LMX_API_KEY" | lmx auth --key-stdin
```

`lmx` checks `$LMX_API_KEY` first, then `~/.config/localmaxxing/config.json`. `bs submit` does not pass the key directly — `lmx` reads it.

### Submit a result dir

```bash
bs submit results/qwen36_2026-08-30/dense_fardna2_rdna2_cg1_mtp0/
# → submitted: abc-123
# → view at:    https://www.localmaxxing.com/speed-tests/abc-123
```

### Validate before posting

```bash
bs submit results/<cell>/ --dry-run
# → payload valid (lmx accepted)
```

`lmx speed-test dry-run` does **not** consume the 1/min rate limit and does **not** write to the leaderboard. Use it freely during recipe iteration.

### Inspect the payload

```bash
bs export results/<cell>/ -o /tmp/payload.json
# → wrote /tmp/payload.json

# Hand to lmx yourself if you prefer
lmx speed-test dry-run /tmp/payload.json
lmx speed-test submit /tmp/payload.json
```

### CLI flags

| Flag | Purpose |
|---|---|
| `--lmx-bin <path>` | Path to the `lmx` binary. Defaults to `$PATH` lookup. |
| `--endpoint <url>` | Override the localmaxxing base URL (passed as `--api-url` to lmx). |
| `--notes <text>` | Free-form notes (max 2000 chars). |
| `--dry-run` | Use `lmx speed-test dry-run` instead of `submit`. |

### Recipe → payload mapping

`bs submit` reads `summary.json` (produced by `bs run`) and the recipe's `hardware:` + `quantization:` blocks, then constructs the localmaxxing JSON payload:

| Recipe field | localmaxxing field |
|---|---|
| `backend.type` | `engineName` (`vllm` / `llama.cpp` / `tgi` / `external`) |
| `backend.model_path` (`org/name`) | `hfId` |
| `resources.dtype` + `quantization:` | `quantization` (`FP16` default; explicit for everything else) |
| `hardware:` (when `is_complete()`) | `hardware.hwClass=DISCRETE_GPU` + gpuName/gpuCount/vramGb/cpu/ramGb/os |
| `cell.render()` | `engineFlags.cellId` (preserves the parent project's cell-id convention) |
| `resources.tensor_parallel_size` | `engineFlags.tensorParallel` |
| `summary.scores[throughput].metrics.output_tok_s` | `tokSOut` |
| `summary.scores[throughput].metrics.input_tok_s` | `tokSPrefill` |
| `summary.scores[throughput].metrics.ttft_mean_ms` | `ttftMs` |
| `--notes` | `notes` (clamped to 2000 chars) |

See [`benchmark_suite/submission.py`](benchmark_suite/submission.py) for the exact mapping.

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
| `hardware` | GPU/CPU/RAM identity (maps to localmaxxing `hardware` field) |
| `quantization` | Free-form quantization label (e.g. `W4A16-G32`, `FP16`, `EXL3-3bpw`) |

Example recipe (`recipes/qwen36-27b-gptq-tp4.yaml`):

```yaml
meta:
  name: qwen36-27b-gptq-tp4
  description: Qwen3.6-27B-GPTQ-W4A16-G32 on 4x V620
  tags: [gfx1030, gptq, w4a16, tp4]

backend:
  type: vllm
  model_path: Qwen/Qwen3.6-27B-GPTQ-W4A16-G32  # HF id for localmaxxing

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

hardware:
  vendor: amd
  model: Radeon PRO V620
  count: 4
  vram_gb: 32
  cpu: EPYC 7452
  ram_gb: 256
  os: Ubuntu 22.04
  power_watts: 200

quantization: W4A16-G32
```

### Shipped recipes

| Recipe | Purpose |
|---|---|
| `recipes/qwen36-27b-gptq-tp4.yaml` | Dense W4A16 production cell on 4x V620 (gfx1030) — V2 runner, FPP cudagraphs, RDNA2 kernels |
| `recipes/qwen36-35b-a3b-fp16-tp4.yaml` | MoE FP16 reference — baseline for quantization comparison |
| `recipes/perplexity-compare.yaml` | Cross-platform wikitext perplexity against any external endpoint |
| `recipes/kld-vs-fp16-reference.yaml` | KLD divergence vs a converted fp16 reference cache |

`backend.type: external` recipes never spawn a server — `bs run` probes the endpoint and runs scorers only. Point `endpoint.url` at whatever is already running. Use `hardware.vendor: other` and an empty model name in these recipes if you're scoring an external service without a known GPU.

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
| `bs export <result_dir> -o <file>` | Write the localmaxxing JSON payload (no network, no `lmx`) |
| `bs submit <result_dir> [--dry-run]` | Shell out to `lmx speed-test submit` (or `dry-run`) |
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
- **lmx**: The official [LocalMaxxing CLI](https://github.com/LottoLottoLotto/localmaxxing-cli). Required for `bs submit`. See the "Submitting to localmaxxing.com" section above for install + auth instructions.
- **lm-eval**: Install with `uv add --optional perplexity lm-eval==0.4.12`. Heavy dep (~GBs) due to torch + datasets.
- **safetensors**: Required by KLD scorer (no torch dependency at scorer runtime; torch only in `bs convert-logits`).
- **inspect-ai**: Install with `uv add --optional agentic inspect-ai==0.3.260`.
- **terminal-bench**: Experimental; requires Docker.

## Development

```bash
uv sync --extra dev
uv run pytest tests/ -q       # 227 tests
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

Run the same recipe against two endpoints and `bs compare` to see the delta. Or post both to [localmaxxing.com](https://www.localmaxxing.com) for cross-user comparison.

## Limitations

- **GPU smoke deferred**: This v1 ships without an end-to-end GPU run on real hardware. CI runs the recipe-validation + `--dry-run` path. Real-hardware validation is a separate session.
- **Logprobs via /v1/completions only**: `/v1/chat/completions` does not expose per-token logprobs. Endpoints that only support chat-completions cannot be perplexity-scored.
- **Self-judge bias**: LLM-judge scores are biased when the judge and candidate are the same endpoint. Use a fixed external judge.
- **localmaxxing requires HuggingFace model IDs**: `backend.model_path` should be `org/name` for the leaderboard to validate it; absolute local paths fall back to the last segment (e.g. `/models/Qwen3-8B` → `Qwen3-8B`).
