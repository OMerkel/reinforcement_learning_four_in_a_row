#!/usr/bin/env python
"""Run a round-robin tournament between all available engines.

Plays each engine against every other engine as both Player 1 and Player 2,
recording wins, losses, and draws. Useful for empirical grounding of the
engine comparison table in the documentation.

Usage:
    uv run python scripts/tournament.py --engines random minimax \
        alphabeta mcts --games 10

Default: plays all engines, 20 games per pairing (10 as P1, 10 as P2).
"""
from __future__ import annotations

import sys

sys.path.insert(0, str(__file__).rsplit('/', 1)[0] + '/..')  # noqa: E402

import argparse  # noqa: E402  # pylint: disable=wrong-import-position
import random  # noqa: E402  # pylint: disable=wrong-import-position

# pylint: disable=wrong-import-position
from typing import NamedTuple  # noqa: E402

# pylint: disable=wrong-import-position
from four_in_a_row import (  # noqa: E402
    GameState,
    suggest_move,
)


class GameResult(NamedTuple):
    """Result of a single game."""
    p1_engine: str
    p2_engine: str
    winner: int | None  # 1 or 2 for a winner, None for draw
    p1_first: bool  # whether p1 was the first player


class TournamentStats(NamedTuple):
    """Stats for one engine in the tournament."""
    wins: int
    losses: int
    draws: int

    @property
    def games_played(self) -> int:
        """Total number of games played."""
        return self.wins + self.losses + self.draws

    @property
    def score(self) -> float:
        """Wins + 0.5 * draws."""
        if self.games_played == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games_played


def play_game(
    state: GameState,
    p1_engine: str,
    p2_engine: str,
    depth: int = 4,
    iterations: int = 500,
    rng: random.Random | None = None,
    move_limit: int = 100,
) -> int | None:
    """Play one game to completion; return winner (1/2) or None for draw."""
    if rng is None:
        rng = random.Random()

    move_count = 0
    while not state.is_terminal() and move_count < move_limit:
        engine = p1_engine if state.current_player == 1 else p2_engine
        try:
            suggestion = suggest_move(
                state,
                engine,
                depth=depth,
                iterations=iterations,
                rng=rng,
            )
            state.apply_move(suggestion.move)
            move_count += 1
        except ValueError:
            # No legal moves (should not happen in Four in a Row)
            break

    if state.is_draw():
        return None
    return state.winner()


def run_tournament(
    engines: list[str],
    games_per_pairing: int = 20,
    depth: int = 4,
    iterations: int = 500,
    seed: int | None = None,
    verbose: bool = True,
) -> dict[str, TournamentStats]:
    """Run round-robin tournament and return stats for each engine."""
    if seed is not None:
        random.seed(seed)

    results: list[GameResult] = []
    stats: dict[str, TournamentStats] = {
        engine: TournamentStats(wins=0, losses=0, draws=0)
        for engine in engines
    }

    total_games = len(engines) * (len(engines) - 1) * games_per_pairing
    game_count = 0

    for p1_engine in engines:
        for p2_engine in engines:
            if p1_engine == p2_engine:
                continue

            for _ in range(games_per_pairing):
                game_count += 1
                if verbose:
                    print(
                        f"[{game_count}/{total_games}] "
                        f"{p1_engine} (P1) vs {p2_engine} (P2)... ",
                        end="",
                        flush=True,
                    )

                state = GameState.new(1)
                seed_val = seed if seed is None else seed + game_count
                rng = random.Random(seed_val)
                winner = play_game(
                    state,
                    p1_engine,
                    p2_engine,
                    depth=depth,
                    iterations=iterations,
                    rng=rng,
                )

                results.append(
                    GameResult(
                        p1_engine=p1_engine,
                        p2_engine=p2_engine,
                        winner=winner,
                        p1_first=True,
                    )
                )

                if verbose:
                    if winner == 1:
                        print("P1 wins")
                    elif winner == 2:
                        print("P2 wins")
                    else:
                        print("Draw")

                # Update stats
                if winner is None:
                    stats[p1_engine] = stats[p1_engine]._replace(
                        draws=stats[p1_engine].draws + 1
                    )
                    stats[p2_engine] = stats[p2_engine]._replace(
                        draws=stats[p2_engine].draws + 1
                    )
                elif winner == 1:
                    stats[p1_engine] = stats[p1_engine]._replace(
                        wins=stats[p1_engine].wins + 1
                    )
                    stats[p2_engine] = stats[p2_engine]._replace(
                        losses=stats[p2_engine].losses + 1
                    )
                else:  # winner == 2
                    stats[p1_engine] = stats[p1_engine]._replace(
                        losses=stats[p1_engine].losses + 1
                    )
                    stats[p2_engine] = stats[p2_engine]._replace(
                        wins=stats[p2_engine].wins + 1
                    )

    return stats


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run a round-robin tournament between AI engines.",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["random", "minimax", "negamax", "alphabeta", "mcts"],
        help="Engines to include in tournament (default: all).",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=20,
        help="Games per pairing (default: 20).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Depth for minimax/negamax/alphabeta (default: 4).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Iterations for MCTS (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    print(f"Tournament: {', '.join(args.engines)}")
    print(f"Games per pairing: {args.games}")
    print(f"Minimax/Negamax/Alpha-Beta depth: {args.depth}")
    print(f"MCTS iterations: {args.iterations}")
    print()

    stats = run_tournament(
        engines=args.engines,
        games_per_pairing=args.games,
        depth=args.depth,
        iterations=args.iterations,
        seed=args.seed,
    )

    # Print results table
    print("\nTournament Results")
    print("=" * 80)
    header = (
        f"{'Engine':<15} {'Wins':<6} {'Losses':<6} "
        f"{'Draws':<6} {'Score':<8} {'Games':<6}"
    )
    print(header)
    print("-" * 80)

    for engine in sorted(stats.keys()):
        s = stats[engine]
        print(
            f"{engine:<15} {s.wins:<6} {s.losses:<6} {s.draws:<6} "
            f"{s.score:<8.4f} {s.games_played:<6}"
        )

    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
