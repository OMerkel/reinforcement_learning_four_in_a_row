# Artificial Intelligence - The Engines

This page documents the move suggestion engines implemented in the project.

Important design choice: the CLI game loop remains human-centric. In the CLI,
engines can either analyze the current position via `suggest <engine>` or apply
their proposed move via `play <engine>`, but there is no long-running
engine-vs-engine mode in CLI.

The GUI (`four_in_a_row/gui.py`) additionally supports automated engine-vs-
engine progression.

## Where the engines live

The engine implementation is split across dedicated modules:

- `four_in_a_row/ai_random.py`
- `four_in_a_row/ai_minimax.py`
- `four_in_a_row/ai_negamax.py`
- `four_in_a_row/ai_alphabeta.py`
- `four_in_a_row/ai_mcts_ucb.py`

The file `four_in_a_row/engines.py` remains as a stable façade that exports the
public `EngineSuggestion` type plus dispatcher and convenience functions.

The engine layer depends on:

- `GameState` from `four_in_a_row/game.py`
- position evaluation helpers such as `evaluate_state()`
- terminal result scoring via `terminal_score()`

## Available engines

The project currently includes five engine strategies:

- **Random**: picks a legal move uniformly at random
- **Minimax**: depth-limited full game tree search
- **Negamax**: minimax-equivalent search using score negation per ply
- **Alpha-Beta**: minimax with pruning to skip dominated branches
- **MCTS**: Monte Carlo Tree Search with UCB-based child selection

### Engine comparison at a glance

| Engine | Deterministic | Speed (relative) | PV depth | Uses `evaluate_state()` | Notes |
| --- | --- | --- | --- | --- | --- |
| Random | no (seeded: yes) | instant | 1 | partially (on result state) | baseline; useful for smoke tests |
| Minimax | yes | exponential in depth | deep | yes (leaf scoring) | readable exhaustive tree |
| Negamax | yes | exponential in depth | deep | yes (leaf scoring) | same quality as minimax |
| Alpha-Beta | yes | faster than minimax | deep | yes (leaf scoring) | recommended for depth-limited play |
| MCTS | no (seeded: yes) | linear in iterations | 1 | rollout fallback only | stochastic; best for sampling |

Depth is measured in plies (half-moves). "Deep" means the engine propagates PV
from the root to the depth cutoff. MCTS returns only the selected root move in
its PV because its selection policy operates on tree statistics rather than a
deterministic continuation.

## Defaults at a glance

Most frequently used defaults:

- CLI `suggest`/`play` with minimax, negamax, alphabeta: `depth=5`
- CLI `suggest`/`play` with mcts: `iterations=800`
- API `suggest_move(...)`: `depth=5`, `iterations=800`

For the full list (including module-level helpers), see
[Engine defaults](#engine-defaults).

## Engine contract

All engines return an `EngineSuggestion` value with the following fields:

- `engine`: engine identifier such as `"random"` or `"alphabeta"`
- `move`: the selected move as a **0-based column index**
- `pv`: principal variation, represented as a list of 0-based column indices
- `evaluation`: a scalar score in the range `[-1.0, +1.0]`

PV depth differs by engine:

- `minimax_suggestion()`, `negamax_suggestion()`, and `alphabeta_suggestion()`
  build the PV by prepending each move to the best child PV at every recursive
  level. The resulting `pv` therefore spans from the root move down to the depth
  cutoff or a terminal state.
- `random_suggestion()` and `mcts_suggestion()` return a one-element PV
  containing only the selected root move. Neither performs a deterministic
  recursive search that would produce a deeper continuation.

In the CLI, these columns are displayed to humans as `1..7`. The internal API
and tests continue to use 0-based columns.

All public engine helpers (`random_suggestion`, `minimax_suggestion`,
`negamax_suggestion`, `alphabeta_suggestion`, `mcts_suggestion`) treat the
input `GameState` as read-only and compute on cloned states internally.

Common validation behavior:

- if no legal moves exist, engines raise `ValueError("No legal moves")`
- search limits must be positive (`depth >= 1`, `iterations >= 1`)

### Evaluation semantics

The score is normalized so it can be displayed consistently across engines:

- `+1.0` means a winning result for Player 1
- `-1.0` means a winning result for Player 2
- values near `0.0` mean the position is currently balanced or uncertain

For tree-search engines, the score reflects the best line the engine found under
its configured search limits. For MCTS, the value is the average rollout result
of the selected child.

### Static board evaluation: `evaluate_state()`

The shared heuristic lives in `four_in_a_row/game.py` as `evaluate_state()`.
It is the main non-terminal position evaluator for the deterministic search
engines.

Engines use it as follows:

- `random_suggestion()` applies one random move on a cloned state and then uses
  `evaluate_state()` on that resulting position.
- `minimax_suggestion()` uses `evaluate_state()` at depth cutoffs and on
  terminal-or-draw short-circuit paths.
- `alphabeta_suggestion()` uses the same evaluation approach as minimax, but
  with pruning to reduce how often deeper branches need to be evaluated.
- `negamax_suggestion()` also uses `evaluate_state()`, but multiplies it by a
  side-to-move sign internally so the recurrence stays in negamax form.
- `mcts_suggestion()` is different: its rollout quality signal comes primarily
  from terminal win/loss outcomes via `terminal_score()`. It only uses
  `evaluate_state()` in the documented fallback path when the root ends the
  search without children.

#### What the heuristic actually measures

`evaluate_state()` first checks whether the state is already terminal:

- Player 1 win -> `+1.0`
- Player 2 win -> `-1.0`
- draw -> `0.0`

For non-terminal states, it combines two lightweight heuristics.

1. Center-column preference.
   The function counts how many Player 1 and Player 2 pieces are in the center
   column and adds a small weighted bonus for the side that controls it.

2. Open-line window scoring.
   The function iterates over every horizontal, vertical, and diagonal 4-cell
   window and scores only windows that are still "open" for one player.
   Mixed windows containing pieces from both players contribute `0.0`.

Current window weights are:

- `+0.12` for a Player 1 window with 3 stones and 1 empty cell
- `+0.03` for a Player 1 window with 2 stones and 2 empty cells
- `-0.12` for a Player 2 window with 3 stones and 1 empty cell
- `-0.03` for a Player 2 window with 2 stones and 2 empty cells

These weights were chosen by hand to balance threat assessment and are not
empirical results from parameter tuning. Contributors who modify these values
should re-run the engine tests to catch any regressions:

```powershell
uv run pytest tests/test_engine_*.py -v
```

The final score is clamped into `[-1.0, +1.0]`.

#### Why center matters more than edges

The implementation does not assign an explicit "corner penalty" or "edge
penalty". Instead, the bias emerges from the features it counts:

- center control gets an explicit bonus
- central cells participate in more possible 4-in-a-row windows
- edge and corner cells usually participate in fewer winning lines

As a result, edge and corner occupancy tends to be less valuable in this
heuristic unless it directly contributes to strong open windows.

#### Why this matters for engine quality and speed

The quality of minimax, negamax, and alpha-beta depends heavily on the quality
of `evaluate_state()`, because depth-limited search relies on heuristic leaf
scores whenever it cannot solve the position completely.

The efficiency of those engines also depends strongly on evaluation cost:

- deeper searches call `evaluate_state()` many times
- alpha-beta reduces the number of visited nodes, but the remaining leaf
  evaluations are still on the critical path
- even small changes in evaluation complexity can noticeably change practical
  search depth and response time

Because of that, `evaluate_state()` is a good target for profiling if engine
performance or move quality becomes a priority. Both CPU time and allocation
behavior are worth measuring, especially if additional heuristics are added.

To collect a quick profiling snapshot, run:

```powershell
uv run python -m cProfile -s cumtime -c "
import four_in_a_row.game as g, four_in_a_row.ai_alphabeta as ab
s = g.GameState.new(1)
ab.alphabeta_suggestion(s, depth=6)
"
```

This prints a cumulative-time profile sorted by the most expensive functions.
Look for `evaluate_state` and `apply_move` in the output — those are the typical
hotspots. Replace `ai_alphabeta` and `alphabeta_suggestion` with the engine you
want to profile.

#### How MCTS differs

MCTS is much less dependent on static board evaluation than the deterministic
tree-search engines.

- Its selection phase uses UCB over accumulated rollout outcomes.
- Its simulation phase judges positions by eventually reaching terminal wins,
  losses, or draws.
- Its backpropagation phase stores those rollout results as visit/value
  statistics on nodes.

So while deterministic engines judge many frontier states through
`evaluate_state()`, MCTS derives most of its quality signal from terminal node
results discovered by repeated playouts. In this codebase, that signal is based
on `terminal_score()` rather than a deep static heuristic.

## Dispatch entry point

The public convenience API is `suggest_move(state, engine, ...)`.

`suggest_move()` and the top-level helper functions in `engines.py` delegate to
the dedicated `ai_*` modules.

Dispatch details currently implemented in `engines.py`:

- `random` uses optional `rng`
- `minimax`, `negamax`, `alphabeta` use `depth`
- `mcts` uses `iterations` and optional `rng`
- `exploration` is configurable via `mcts_suggestion(...)` directly (not via
  `suggest_move(...)`)

Supported engine names:

- `random`
- `minimax`
- `negamax`
- `alphabeta`
- `alpha-beta`
- `alpha_beta`
- `mcts`

Unknown names raise `ValueError`.

The CLI help and command parser accept the same alpha-beta aliases.

## Engine defaults

The project uses the following default values:

| API / Command path | Default values |
| --- | --- |
| `suggest_move(..., depth=5, iterations=800)` | `depth=5`, `iterations=800` |
| `minimax_suggestion(state, depth=4)` | `depth=4` |
| `negamax_suggestion(state, depth=4)` | `depth=4` |
| `alphabeta_suggestion(state, depth=6)` | `depth=6` |
| `mcts_suggestion(state, iterations=500, exploration=sqrt(2))` | `iterations=500`, `exploration=sqrt(2)` |
| CLI `suggest <engine> [v]` / `play <engine> [v]` | `depth=5` for minimax/negamax/alphabeta, `iterations=800` for mcts |

This means the CLI intentionally uses slightly different default search limits
than some direct module-level engine functions.

## Import patterns

You can import engine functionality from the package root:

```python
from four_in_a_row import (
  EngineSuggestion,
  suggest_move,
  random_suggestion,
  minimax_suggestion,
  negamax_suggestion,
  alphabeta_suggestion,
  mcts_suggestion,
)
```

You can also import the split modules directly:

```python
from four_in_a_row import ai_random, ai_minimax, ai_negamax, ai_alphabeta, ai_mcts_ucb

suggestion = ai_minimax.minimax_suggestion(state, depth=4)
```

For reproducible tests with stochastic engines:

```python
import random

from four_in_a_row import mcts_suggestion, random_suggestion

random_pick = random_suggestion(state, rng=random.Random(7))
mcts_pick = mcts_suggestion(state, iterations=200, rng=random.Random(7))
```

## Per-engine details

Each engine has its own dedicated page covering search behavior, diagrams,
validation rules, and tuning guidance:

- [Random engine](engine_random.md) — instant, stochastic baseline
- [Minimax engine](engine_minimax.md) — exhaustive depth-limited tree search
- [Negamax engine](engine_negamax.md) — minimax-equivalent with score negation; includes Minimax vs Negamax comparison and benchmark
- [Alpha-Beta engine](engine_alphabeta.md) — minimax with branch pruning
- [Monte Carlo Tree Search (MCTS)](engine_mcts.md) — UCB-guided sampling with full UCB math and tuning guidance

## Determinism and testing

The test suite covers the core contract of all engines.

Verified behaviors include:

- random suggestions are legal and return a one-step PV
- minimax finds an immediate win in a known winning position
- negamax finds an immediate win in the same known winning position
- alpha-beta finds the same immediate win
- MCTS returns a legal move and bounded evaluation
- dispatch through `suggest_move()` works for all registered engines
- invalid search parameters raise `ValueError`
- requesting a move from a full board raises `ValueError`

Where randomness is involved, the tests inject seeded RNG instances so results
remain reproducible.

## Performance notes

The engines are intentionally simple and readable.

- **Random** is constant-cost relative to legal move count
- **Minimax** grows exponentially with depth
- **Negamax** has the same asymptotic complexity as minimax
- **Alpha-Beta** improves on minimax by pruning branches
- **MCTS** trades exact tree evaluation for repeated randomized sampling

For interactive usage:

- choose `random` for instant output
- choose `minimax` for simple deterministic analysis at small depths
- choose `negamax` if you want minimax-equivalent behavior with simpler
  recursion structure
- choose `alphabeta` for deeper deterministic search
- choose `mcts` when you want probabilistic search guided by many playouts

### Rough performance rules of thumb

**MCTS iteration budget vs. search depth:**

Applied to Four in a Row:

- `depth=4` (minimax/negamax/alphabeta) typically explores ~5,000–50,000 nodes
- `iterations=500` (MCTS) typically explores ~500–2,000 nodes but with rollout
  value refinement
- the CLI defaults use `depth=5` for deterministic engines and `iterations=800`
  for MCTS, giving broadly similar response time on typical hardware

When tuning the iteration budget for MCTS, start with 500 and increase in steps
of 200–500 until the response time feels right for your use case.

## User-facing interpretation

When the CLI displays a suggested move, keep these points in mind:

- the move is shown using the game's internal 0-based column model before any
  CLI-specific formatting
- the PV is an explanation of the line the engine prefers
- the evaluation is an AI estimate, not a proof of optimal play unless a search fully
  resolves the relevant portion of the game tree
