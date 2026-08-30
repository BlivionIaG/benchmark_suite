# Agent Rules for benchmark_suite

This file is the **first thing an agent reads**. It's structured so you can answer "where do I edit X?" or "what's the convention for Y?" without grep.

## 1. What this repo is

`bs` — a CLI that runs reproducible benchmarks against any OpenAI-compatible LLM endpoint (vLLM, llama.cpp server, TGI, OpenAI API). Recipes are YAML; scorers are pluggable; results land in `results/<name>_<ts>/<cell_id>/`.

Five scorers ship (see `benchmark_suite/scoring/`):

| `kind` | What it measures | Tool |
|---|---|---|
| `throughput` | output tok/s, TTFT, TPOT, ITL | `llm-perf` (primary) / `vllm-bench` (fallback) |
| `perplexity` | lm-eval tasks (wikitext etc.) | lm-eval-harness via `/v1/completions` |
| `kld` | per-token KL divergence vs reference distribution | custom top-k KL on safetensors cache, or `llm-perf kl-divergence` |
| `llm_judge` | 0–10 quality score from a judge LLM | native httpx (default, no Node) or `promptfoo` |
| `agentic` | task pass rate | `inspect-ai` (primary) / `terminal-bench` (experimental) |

`PLAN.md` is the **source of truth for design decisions**. Read it before making schema or architecture changes.

## 2. Setup

```bash
# Toolchain
curl -LsSf https://astral.sh/uv/install.sh | sh

# Dev install (core deps + lint + test)
uv sync --extra dev

# Add scorers you actually use:
uv sync --extra perplexity    # pulls torch + datasets (~GB) — only for perplexity scorer
uv sync --extra judge         # promptfoo wrapper (Node still required for driver=promptfoo)
uv sync --extra agentic       # inspect-ai + terminal-bench
uv sync --all-extras          # everything

# Optional: install llm-perf from source (Rust toolchain required)
cargo install --git https://github.com/iopsystems/llm-perf --tag v0.1.16

# Verify
uv run bs version
```

System Python is 3.14.3 — too new for `lm-eval`'s dep tree. Always use `uv`'s managed 3.13 venv. **Do not** `pip install` into the system.

## 3. Where things live

```
benchmark_suite/
├── __init__.py             # __version__
├── cli.py                  # typer app — `bs` entry point (ALL commands)
├── recipe.py               # Pydantic v2 schema — Recipe, scorer configs, load_recipe
├── paths.py                # result-dir layout helpers, ts_slug
├── _matrix.py              # private — cartesian axis expansion for `bs matrix`
├── runner/
│   ├── endpoint.py         # probe_endpoint (health, /v1/models, logprobs capability)
│   ├── serve.py            # synthesize_server_cmd, managed_server context mgr, GPUFileLock
│   ├── llm_perf.py         # TOML gen, run_llm_perf_bench/logprobs/kl-divergence, JSON parse
│   └── vllm_bench.py       # vllm bench throughput wrapper + 11 legacy regexes
├── scoring/
│   ├── base.py             # Scorer ABC, ScoreRecord, ScorerRegistry, @scorer decorator
│   ├── throughput.py       # llm-perf / vllm-bench → ScoreRecord
│   ├── perplexity.py       # lm-eval subprocess → ScoreRecord
│   ├── kl_divergence.py    # safetensors top-k KL (vocab-checked, vocab-mismatch refuses)
│   ├── llm_judge.py        # native httpx rubric (or promptfoo subprocess)
│   └── agentic.py          # inspect-ai primary, terminal-bench best-effort
├── tools/
│   └── convert_kl_logits.py  # one-time torch.load(f_*.fp16) → safetensors + manifest.json
├── report.py               # write_summary (summary.csv + summary.json + README.md)
└── compare.py              # compare_results (delta.csv + delta.md)

recipes/                    # YAML recipe library (4 shipped)
tests/
├── conftest.py             # top-level pytest fixtures
├── test_recipe.py, test_paths.py, test_matrix.py   # schema tests
├── test_cli.py, test_compare.py, test_report.py    # integration tests
├── runner/                 # endpoint + serve + llm_perf + vllm_bench tests
├── scoring/                # per-scorer tests
└── fixtures/               # canned outputs (e.g., vllm_bench stdout sample)
```

### Key contracts (don't break these)

- **`recipe.py` schema is the API.** Every scorer config uses Pydantic v2 discriminated union on `kind`. Schema changes require updating PLAN.md + all recipes.
- **legacy summary.csv columns stay byte-for-byte.** Defined as `SUMMARY_CSV_COLUMNS` in `report.py`. Originally from parent's `benchmark_results/parse_matrix_results.py`. Adding a column is fine; renaming/removing breaks downstream tooling.
- **Cell-id format**: `{family}_{attn}_{linear}_cg{cg}_mtp{mtp}[_extra...]` (e.g. `dense_fardna2_rdna2_cg1_mtp2`). Rendered by `CellId.render()` in `recipe.py`. Convention from parent project.
- **No torch in scorer hot path.** `kl_divergence.py` uses numpy + safetensors only. `torch` is gated to `tools/convert_kl_logits.py`.

## 4. Common tasks

### Add a new scorer

1. Create `benchmark_suite/scoring/<name>.py`. Subclass `Scorer` (from `base.py`), set `kind: ClassVar[str]`, decorate with `@scorer`. Implement `score(self, recipe, *, result_dir, endpoint_url=None) -> ScoreRecord`.
2. Add config class to `recipe.py`: `class XyzScorer(BaseModel)` with `kind: Literal["xyz"] = "xyz"` and your fields. Add it to the `ScorerConfig` discriminated union at the bottom.
3. Update `benchmark_suite/scoring/__init__.py` to re-export the new class.
4. Write `tests/scoring/test_<name>.py` — TDD first.
5. Add the new kind's columns to `SUMMARY_CSV_COLUMNS` in `report.py` if the metrics should appear in summary.csv.
6. Verify: `uv run pytest tests/scoring/test_<name>.py -v && uv run ruff check . && uv run basedpyright benchmark_suite/ tests/`

### Add a new CLI command

1. Add a `@app.command()` function in `benchmark_suite/cli.py`. Use `typer.Option` for flags, `typer.Argument` for positional args.
2. If the command depends on runner/scorer logic, call into existing functions — don't duplicate. Heavy lifting lives in `runner/` and `scoring/`; the CLI is the composition root.
3. Add a `typer.testing.CliRunner` test in `tests/test_cli.py`.

### Add a new recipe

1. Copy `recipes/qwen36-27b-gptq-tp4.yaml` (or the closest match).
2. Edit `meta.name` (must be unique slug — `^[a-z0-9][a-z0-9_-]*$`), `meta.description`, `backend.model_path`, `endpoint.url`, `resources.*`, `cell.*`.
3. Keep `cell.{family,attn,linear,cg,mtp}` aligned with the model architecture (e.g. MoE → `family: moe`).
4. Verify: `uv run bs doctor recipes/<new>.yaml --dry-run` exits 0.
5. Run the recipe-validation test: `uv run pytest tests/test_recipes.py -v` (auto-discovers all `recipes/*.yaml`).

### Modify the recipe schema

**Touch with care.** PLAN.md §2 is the canonical reference. Steps:

1. Update the Pydantic model in `recipe.py`.
2. If adding a field: give it a default value (so existing recipes still load).
3. If removing a field: grep `recipes/*.yaml` first; either migrate recipes or keep the field with a deprecation comment.
4. Run all recipe tests + schema tests: `uv run pytest tests/test_recipe.py tests/test_recipes.py -v`.
5. Update PLAN.md §2 to match.

### Add a new tool

Tools are one-off CLI utilities outside the scorer/run cycle. Pattern: create `benchmark_suite/tools/<name>.py` with a public API + a thin typer subcommand in `cli.py` (or a standalone entry point in `pyproject.toml`).

## 5. Gotchas

- **Pydantic v2 discriminated union syntax**: use `Annotated[Union[A, B, ...], Field(discriminator="kind")]`. The literal value of `kind` selects the right subclass at validation time.
- **`/v1/completions` vs `/v1/chat/completions`**: OpenAI does NOT expose per-token logprobs on chat. Perplexity and KLD scorers must hit the completions endpoint. vLLM and most OpenAI-compatible servers support it; llama.cpp server exposes it as well.
- **Cross-tokenizer KLD is meaningless.** `kl_divergence.py` refuses to run if `manifest.vocab_size` ≠ endpoint tokenizer vocab size. To override, set `vocab_check: false` in the scorer config — but you'll get garbage numbers if the tokenizers really differ.
- **GPU file lock is per-process.** `runner.serve.GPUFileLock` serializes GPU-owning phases within a single Python process. Multi-process scenarios (e.g. parallel CI jobs) need process-group orchestration at a higher level.
- **`managed_server` yields `None` for `backend.type: external`.** Don't try to read `.pid` etc. on the handle without a None check.
- **`recipe.endpoint.url` is the bare base** (e.g. `http://127.0.0.1:8000`). `EndpointSection.base_url_v1` is the `/v1`-appended form. Don't append `/v1` yourself — use the property.
- **llm-perf TOML keys are best-effort.** Upstream schema may change between tags. The `runner/llm_perf.py` tests verify the TOML parses + the binary is invoked correctly, but won't catch upstream schema drift without an end-to-end run.
- **`bs doctor --dry-run` always exits 0.** Use it for CI smoke tests; use `bs doctor` (no flag) in dev to actually gate on broken state.

## 6. Security Rules

### Rule 1: No credentials in git

**NEVER commit API keys, tokens, passwords, or secrets.** Prohibited: `*secret*`, `*credential*`, `*api_key*`, `*token*`, `*password*` filenames; `.pem`/`.key`/`.p12`; `.env*`.

```bash
git diff --cached | grep -iE "api_key|secret|token|password" && echo BLOCKED || echo OK
```

## 7. Commit Rules

### Rule 2: Descriptive commits (required format)

```
<type>: <short description>

<detailed explanation>

- What changed: <specific files / decisions>
- Why: <reasoning>
- Testing: <how verified>
```

Types: `feat`, `fix`, `chore`, `docs`, `refactor`. No trailers.

### Rule 3: Atomic commits

One logical change per commit. Don't mix unrelated edits. If the diff touches >5 files, split it.

### Rule 4: Push after green

After `pytest + ruff + basedpyright` all pass, `git push origin master` immediately. Never leave green work uncommitted on a shared branch.

### Rule 5: Never force-push

Use `git revert` for history fixes. Force-pushing rewrites shared history.

## 8. Workflow Rules

### Rule 6: TDD flow

1. Write failing tests first.
2. Implement to green.
3. `uv run ruff check .` → clean.
4. `uv run basedpyright benchmark_suite/ tests/` → clean.
5. Atomic commit + push.

### Rule 7: Recipe edits go through schema validation

Don't hand-edit `recipes/*.yaml` in ways that bypass `load_recipe`. Run `uv run bs doctor <recipe>` after any recipe change.

## 9. Pre-commit checklist

Before each commit:

- [ ] No credentials in `git diff --cached`
- [ ] `uv run pytest tests/ -q` — exit 0, all tests green
- [ ] `uv run ruff check .` — exit 0
- [ ] `uv run basedpyright benchmark_suite/ tests/` — exit 0 (strict mode, **both source AND tests**)
- [ ] If recipe was added/edited: `uv run pytest tests/test_recipes.py -v` — exit 0
- [ ] If new scorer added: re-exported in `benchmark_suite/scoring/__init__.py`
- [ ] If schema changed: PLAN.md §2 updated
- [ ] Commit message follows Rule 2 format (What / Why / Testing body)
- [ ] `git push origin master` after commit

## 10. Validation status (v1)

- 182 tests passing
- ruff + basedpyright strict: clean
- 4 recipes shipped (gfx1030 production + cross-platform perplexity + KLD)
- CI runs on `ubuntu-latest` (CPU-only) via `.github/workflows/ci.yml`
- **GPU smoke deferred**: this v1 was built while real GPUs were busy with bug investigations. Real-hardware validation is a separate session.
