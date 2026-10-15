# Documentation Freshness Checklist

Use this checklist whenever behavior, commands, engine settings, or persistence
logic changes.

## When to update documentation

Update docs when you change:

- CLI commands in `four_in_a_row/cli.py`
- GUI behavior and commands in `four_in_a_row/gui.py`
- engine behavior or defaults in `four_in_a_row/engines.py` or `four_in_a_row/ai_*.py`
- SGF import/export behavior in `four_in_a_row/sgf.py`
- architecture boundaries between modules

## Files to review for consistency

- `README.md`
- `doc/index.md`
- `doc/software_architecture.md`
- `doc/engines.md`
- `doc/engine_random.md`
- `doc/engine_minimax.md`
- `doc/engine_negamax.md`
- `doc/engine_alphabeta.md`
- `doc/engine_mcts.md`
- `doc/sgf.md`

## Tests that back documentation claims

Run targeted tests:

```powershell
uv run pytest tests/test_cli.py tests/gui tests/test_engines.py tests/test_negamax_module.py tests/test_history_and_sgf.py
```

These tests validate key claims about:

- CLI command behavior and argument validation
- GUI click/AI/post-game behavior and entrypoints
- engine dispatch and evaluation contract
- minimax/negamax parity on curated positions
- SGF parsing/export validation and roundtrip

## Docs render validation

Always rebuild docs after updates:

```powershell
uv run zensical build
```

Expected output location:

- `site/`

## CI validation

Docs build is also validated in CI via:

- `.github/workflows/docs.yml`

Pull requests now also run a lightweight freshness gate. If a PR changes any of:

- `four_in_a_row/cli.py`
- `four_in_a_row/gui.py`
- `four_in_a_row/engines.py`
- `four_in_a_row/sgf.py`
- `four_in_a_row/ai_*.py`

then at least one documentation file must also change:

- `README.md`
- `doc/index.md`
- `doc/software_architecture.md`
- `doc/engines.md`
- `doc/sgf.md`
- `doc/docs_freshness.md`

If CI fails, compare local output first:

1. `uv sync --dev`
2. `uv run zensical build`

You can also run the freshness check locally for a set of changed files:

```powershell
uv run python scripts/check_docs_freshness.py four_in_a_row/cli.py README.md
```

## Quick pre-merge checklist

- [ ] Commands in `README.md` match `four_in_a_row/cli.py`
- [ ] Engine names/defaults in docs match `four_in_a_row/engines.py` and `ai_*.py`
- [ ] SGF docs match `four_in_a_row/sgf.py`
- [ ] Targeted tests pass
- [ ] Docs build succeeds and renders expected pages
