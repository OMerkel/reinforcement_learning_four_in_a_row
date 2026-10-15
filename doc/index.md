# Four in a Row Documentation

Welcome to the project documentation for **Four in a Row**.

This folder is the source for the rendered documentation site and is designed to
work well both in the repository and when rendered with Zensical into `./site`.

Documentation source of truth lives in `./doc`. Update these files first, then
rebuild the generated site output in `./site`.

## What this documentation covers

The documentation focuses on the implemented Python codebase:

- terminal gameplay and command flow
- graphical gameplay and event flow
- the internal game model and history tracking
- AI engines for move suggestions
- SGF import and export for persistence

## Documentation map

### [Reinforcement Learning](reinforcement_learning.md)

Start here to learn why zero-prior values, legal-action masks, terminal rewards,
opponent-relative updates, reproducible self-play, checkpoints, and the CLI form
one learning loop. The staged plan distinguishes the tabular baseline from
future neural learning and evidence-based assessment of strong play.

New to reinforcement learning? Start with the
[beginner's guide](rl_beginners_guide.md), then read the
[PUCT explanation](puct.md) for policy/value search.

### [Software Architecture](software_architecture.md)

Start here for the high-level system overview.

The [requirements register](requirements.md) groups formal FR/NFR scenarios
by architectural owner and links them to named tests and coverage KPIs.

Topics include:

- project goals
- module responsibilities
- runtime flow through CLI and GUI
- domain model responsibilities
- testing strategy

For the GUI-specific architecture path, see the
[GUI Usage and Flow section](software_architecture.md#gui-usage-and-flow).

### [Artificial Intelligence - The Engines](engines.md)

Read this page for the AI engine layer.

Topics include:

- available engines
- the suggestion contract
- evaluation semantics
- search strategy overview

Looking for API import examples? See the
[Import patterns section](engines.md#import-patterns).

#### Per-engine deep-dives

Each engine has a dedicated page with search behavior, diagrams, and tuning guidance:

- [Random engine](engine_random.md) — instant, stochastic baseline
- [Minimax engine](engine_minimax.md) — exhaustive depth-limited tree search
- [Negamax engine](engine_negamax.md) — minimax-equivalent with score negation
- [Alpha-Beta engine](engine_alphabeta.md) — minimax with branch pruning
- [MCTS engine](engine_mcts.md) — UCB-guided sampling with exploration tuning

### [SGF - The Smart Game Format](sgf.md)

Read this page for persistence and interchange.

Topics include:

- why SGF is used in this project
- how game data is stored
- import and export behavior
- coordinate mapping and validation notes

### [Documentation Freshness Checklist](docs_freshness.md)

Use this page before merging changes that affect behavior or user-facing docs.

Topics include:

- what to update when CLI, GUI, engine, or SGF behavior changes
- which tests to run as docs-backed validation
- which docs build command verifies rendering

## Project source map

The main implementation lives in `four_in_a_row/`:

| File | Responsibility |
| --- | --- |
| `four_in_a_row/__main__.py` | Primary module entry point for `python -m four_in_a_row` |
| `main.py` | Compatibility wrapper entry point |
| `four_in_a_row/cli.py` | Interactive terminal loop and commands |
| `four_in_a_row/gui.py` | Graphical UI loop and click-driven interaction |
| `four_in_a_row/game.py` | Board state, rules, move application and winner detection |
| `four_in_a_row/history.py` | Undo and replay management |
| `four_in_a_row/engines.py` | Stable engine façade and dispatch API |
| `four_in_a_row/ai_random.py` | Random AI engine |
| `four_in_a_row/ai_minimax.py` | Minimax AI engine |
| `four_in_a_row/ai_negamax.py` | Negamax AI engine |
| `four_in_a_row/ai_alphabeta.py` | Alpha-beta AI engine |
| `four_in_a_row/ai_mcts_ucb.py` | MCTS + UCB AI engine |
| `four_in_a_row/sgf.py` | SGF import and export |
| `four_in_a_row/rl.py` | Rules-only self-play Q-learning and checkpoint resume |
| `four_in_a_row/rl_cli.py` | Training, frozen evaluation, configurable-seat play |
| `scripts/quality.py` | Separate coverage gates and bidirectional requirement KPIs |

## Typical usage flow

1. Start the program.
2. Choose CLI or GUI interaction.
3. Play moves interactively.
4. Ask an engine for a suggested move or run engine turns.
5. Undo or replay moves if needed.
6. Save or load a game as SGF.

Note: `undo` and `replay` are CLI command-loop features. The GUI focuses on
click-driven play and post-game commands.

## Recommended reading order

If you are new to the project, read the pages in this order:

1. this page
2. [Software Architecture](software_architecture.md)
3. [Artificial Intelligence - The Engines](engines.md)
4. [SGF - The Smart Game Format](sgf.md)
5. [Documentation Freshness Checklist](docs_freshness.md)

## Rendering with Zensical

The documentation source lives in `./doc` and is configured to render into `./site`.

Typical commands:

```powershell
uv run zensical build
uv run zensical serve
```

After building, open the generated files in `./site` or serve them locally.

## License

This project is licensed under the terms in [../LICENSE](../LICENSE).
