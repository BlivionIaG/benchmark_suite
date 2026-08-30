# Agent Rules for benchmark_suite

## Security Rules (CRITICAL)

### Rule 1: No Credentials in Git

**NEVER commit API keys, tokens, passwords, or secrets to this repository.**

Prohibited patterns: files matching `*secret*`, `*credential*`, `*api_key*`, `*token*`, `*password*`; extensions `.pem`, `.key`, `.p12`; `.env` files.

Credentials belong in your shell config, not in the repo.

**Verification before commit:**
```bash
git diff --cached | grep -i "api_key\|secret\|token\|password"
```

## Commit Rules

### Rule 2: Descriptive Commits Required

Every commit must clearly describe what was changed and why.

Format:
```
<type>: <short description>

<detailed explanation>

- What changed: <specific changes>
- Why: <reasoning>
- Testing: <how verified>
```

Types: `feat`, `fix`, `chore`, `docs`, `refactor`.

### Rule 3: Atomic Commits

Each commit represents a single logical change:
- ✅ One feature = one commit
- ✅ One bug fix = one commit
- ❌ Multiple unrelated changes in one commit

## Workflow Rules

### Rule 4: Commit and Push Working Changes

When a modification works, commit and push immediately.

### Rule 5: TDD Flow

For new features or bug fixes:
1. Write failing tests first
2. Implement to make tests pass
3. Lint and type-check
4. Commit and push

### Rule 6: Never Force Push

**Never use `git push --force`.** Use `git revert` or create a new commit to fix history.

## Pre-commit Checklist

Before each commit:
- [ ] No credentials in modified files
- [ ] `ruff check .` passes
- [ ] `basedpyright benchmark_suite/` passes (strict mode)
- [ ] `pytest tests/ -q` passes
- [ ] Commit message follows Rule 2 format
