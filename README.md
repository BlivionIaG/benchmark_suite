# benchmark_suite

Declarative benchmarking and quality scoring for OpenAI-compatible LLM endpoints (vLLM, llama.cpp, TGI, OpenAI API).

## Install

```bash
# Requires uv — install if missing
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync --extra dev
```

## Quickstart

```bash
# Validate a recipe + probe endpoint
bs doctor recipes/foo.yaml

# Run a recipe end-to-end
bs run recipes/foo.yaml

# Compare two result directories
bs compare results/cell_a results/cell_b
```

## Recipe Anatomy

A recipe is a YAML file with these top-level sections:

| Section | Purpose |
|---|---|
| `meta` | Name, description, version, tags |
| `backend` | Backend type (`vllm`, `llamacpp`, `tgi`, `external`) and model path |
| `endpoint` | URL, API key env var, model name, timeouts |
| `resources` | TP/PP size, GPU mem utilization, max model len, dtype |
| `runtime` | Environment overrides, server startup config, GPU lock |
| `bench` | Load profile (concurrency, num prompts, input/output len) + scorers |
| `report` | Output format, output dir, artifact retention |
| `cell` | Cell ID fields (family, attn, linear, cg, mtp, extra) |

### Canonical `cell` ID format

```
{family}_{attn}_{linear}_cg{cg}_mtp{mtp}[_extra...]
```

Example: `dense_fardna2_rdna2_cg1_mtp2`

## Scorer Reference

| Scorer | Tool | Output |
|---|---|---|
| `throughput` | llm-perf or vllm-bench | `output_tok_s`, `total_tok_s`, `ttft_mean_ms`, `tpot_mean_ms`, … |
| `perplexity` | lm-eval | `wikitext` ppl, per-task results |
| `kld` | llm-perf or logits_dir | KL divergence vs reference logits |
| `llm_judge` | native httpx or promptfoo | 0–10 correctness/clarity scores |
| `agentic` | inspect-ai or terminal-bench | Task pass rates |

## Contributing

- Write failing tests first (TDD)
- Atomic commits with conventional format
- `bs doctor` before any GPU run
- All PRs require pytest + ruff + basedpyright green

## CLI Commands

| Command | Description |
|---|---|
| `bs run <recipe>` | Execute one recipe |
| `bs run-all <glob>` | Execute all matching recipes serially |
| `bs matrix <recipe> --axis k=v1,v2` | Sweep axes → one cell per combo |
| `bs compare <dirA> <dirB>` | Delta table between two result dirs |
| `bs report <result_dir>` | Regenerate summaries from artifacts |
| `bs init <name>` | Scaffold a new recipe |
| `bs doctor <recipe>` | Validate recipe + probe endpoint + check binaries |
| `bs convert-logits <dir>` | One-time `f_*.fp16` → safetensors + manifest |
