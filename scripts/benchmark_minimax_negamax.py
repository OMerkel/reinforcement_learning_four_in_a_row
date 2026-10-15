"""Benchmarking script to compare minimax and negamax engines."""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import four_in_a_row.ai_minimax as minimax_module
import four_in_a_row.ai_negamax as negamax_module
from four_in_a_row.ai_minimax import minimax_suggestion
from four_in_a_row.ai_negamax import negamax_suggestion
from four_in_a_row.game import GameState


@dataclass
class EngineResult:
    """Structured result of an engine benchmark."""
    name: str
    depth: int
    calls: int
    avg_ms: float
    move: int
    eval_value: float


def _state_from_moves(moves: list[int]) -> GameState:
    """Create a GameState from a list of moves."""
    state = GameState.new()
    for move in moves:
        state.apply_move(move)
    return state


def _measure(
    *,
    name: str,
    module,
    search_symbol: str,
    suggest_fn,
    state: GameState,
    depth: int,
    repeats: int,
) -> EngineResult:
    """Measure the performance of an engine for a given state and depth.
    Args:
        name: A label for the engine (e.g., "minimax" or "negamax").
        module: The module containing the search function to wrap.
        search_symbol: The name of the search function to wrap
                       (e.g., "_minimax_search").
        suggest_fn: The suggestion function to call (e.g., minimax_suggestion).
        state: The GameState to use for the benchmark.
        depth: The search depth to use for the benchmark.
        repeats: The number of times to repeat the
                 suggestion call for averaging.
    Returns:
        An EngineResult containing the benchmark results.
    """
    original = getattr(module, search_symbol)
    call_counter = {"count": 0}

    def wrapped(search_state, search_depth):
        call_counter["count"] += 1
        return original(search_state, search_depth)

    setattr(module, search_symbol, wrapped)
    try:
        start = time.perf_counter()
        suggestion = None
        for _ in range(repeats):
            suggestion = suggest_fn(state, depth=depth)
        elapsed = time.perf_counter() - start
    finally:
        setattr(module, search_symbol, original)

    assert suggestion is not None
    return EngineResult(
        name=name,
        depth=depth,
        calls=call_counter["count"] // repeats,
        avg_ms=(elapsed * 1000.0) / repeats,
        move=suggestion.move,
        eval_value=suggestion.evaluation,
    )


def main() -> None:
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark minimax vs negamax"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum search depth to benchmark",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Runs per engine-depth pair",
    )
    args = parser.parse_args()

    if args.max_depth < 1:
        raise ValueError("--max-depth must be >= 1")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    positions = {
        "opening": _state_from_moves([]),
        "midgame": _state_from_moves([3, 2, 3, 2, 4, 1, 4, 1]),
        "tactical": _state_from_moves([0, 0, 1, 1, 2, 2]),
    }

    for label, state in positions.items():
        print(f"\nPosition: {label}")
        print("depth  engine    calls    avg_ms   move   eval")
        print("-----  --------  -------  -------  -----  -------")
        for depth in range(1, args.max_depth + 1):
            minimax_result = _measure(
                name="minimax",
                module=minimax_module,
                search_symbol="_minimax_search",
                suggest_fn=minimax_suggestion,
                state=state,
                depth=depth,
                repeats=args.repeats,
            )
            negamax_result = _measure(
                name="negamax",
                module=negamax_module,
                search_symbol="_negamax_search",
                suggest_fn=negamax_suggestion,
                state=state,
                depth=depth,
                repeats=args.repeats,
            )

            for result in (minimax_result, negamax_result):
                print(
                    f"{result.depth:>5}  {result.name:<8}  {result.calls:>7}  "
                    f"{result.avg_ms:>7.2f}  {result.move + 1:>5}  "
                    f"{result.eval_value:+.3f}"
                )


if __name__ == "__main__":
    main()
