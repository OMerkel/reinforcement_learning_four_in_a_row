# Negamax engine

`negamax_suggestion()` performs the same game-theoretic search as minimax, but
uses a single maximizing recurrence and negates the score at each deeper ply.

## Core idea

Instead of writing separate maximize/minimize branches, negamax relies on:

$$
  \operatorname{value}(s) = \max_{m \in \operatorname{legal}(s)}\left(-\operatorname{value}(\operatorname{child}(s, m))\right)
$$

At each move, perspective switches to the opponent, so child scores are negated
when propagated back up.

## How this project handles evaluation perspective

`evaluate_state()` is Player-1-centered. Negamax internally converts this into
"score from side-to-move perspective" by multiplying with a sign:

- `+1` when `current_player` is Player 1
- `-1` when `current_player` is Player 2

At the root, it converts back so `EngineSuggestion.evaluation` remains in the
same Player-1-centered convention used by the other engines.

## Validation

- `depth` must be at least `1`
- if there are no legal moves, the function raises `ValueError`

## Minimax vs Negamax

Both approaches are equivalent in decision quality when they use the same depth
and evaluation function.

Key difference in implementation style:

- **Minimax** alternates explicit maximizing and minimizing logic based on turn.
- **Negamax** always maximizes and negates child score each ply.

Practical implications:

- negamax often yields shorter, more uniform recursive code
- minimax can be more explicit to read when learning max/min game trees
- both can be extended with alpha-beta pruning; negamax then uses negated
  alpha/beta windows when recursing

### Quick benchmark

You can compare minimax and negamax recursion call counts and timing with:

```powershell
uv run python scripts/benchmark_minimax_negamax.py --max-depth 5 --repeats 3
```

The benchmark script lives in `scripts/benchmark_minimax_negamax.py`.
