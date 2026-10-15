"""Small, evaluation-only tactical diagnostics based on the game rules."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .game import GameState, opponent
from .rl import QAgent

if TYPE_CHECKING:
    from .rl_nn import NeuralAgent
    from .rl_puct import PUCTAgent


def winning_actions(state: GameState, player: int) -> frozenset[int]:
    """Enumerate legal immediate wins for either token on a copied board."""
    opponent(player)
    result = set()
    for action in state.legal_moves():
        following = state.clone()
        following.current_player = player
        following.apply_move(action)
        if following.winner() == player:
            result.add(action)
    return frozenset(result)


def tactical_actions(state: GameState, category: str) -> frozenset[int]:
    """Compute rule-derived acceptable actions for a diagnostic category."""
    player = state.current_player
    if category == "win":
        return winning_actions(state, player)
    safe = set()
    forks = set()
    defenses = set()
    for action in state.legal_moves():
        following = state.clone()
        following.apply_move(action)
        if following.is_terminal():
            safe.add(action)
            defenses.add(action)
        elif not winning_actions(following, following.current_player):
            safe.add(action)
            if len(winning_actions(following, player)) >= 2:
                forks.add(action)
            if category == "fork_defense" and not permits_opponent_fork(
                following, player
            ):
                defenses.add(action)
    if category in ("block", "safety"):
        return frozenset(safe)
    if category == "fork":
        return frozenset(forks)
    if category == "fork_defense":
        return frozenset(defenses)
    raise ValueError(f"Unknown tactical category: {category}")


def permits_opponent_fork(state: GameState, player: int) -> bool:
    """Detect an opponent reply with two wins and no immediate counter-win."""
    enemy = opponent(player)
    for action in state.legal_moves():
        following = state.clone()
        following.apply_move(action)
        if following.is_terminal():
            continue
        if (
            not winning_actions(following, player)
            and len(winning_actions(following, enemy)) >= 2
        ):
            return True
    return False


@dataclass(frozen=True)
class TacticalCase:
    """An immutable legal move history and pinned acceptable action set."""

    name: str
    category: str
    moves: tuple[int, ...]
    starting_player: int
    expected: frozenset[int]

    def state(self) -> GameState:
        """Rebuild the fixture and reject terminal or mismatched positions."""
        state = GameState.new(starting_player=self.starting_player)
        for action in self.moves:
            if state.is_terminal():
                raise ValueError("Tactical fixture contains post-game moves")
            state.apply_move(action)
        if state.is_terminal():
            raise ValueError("Tactical fixture must be nonterminal")
        if not self.expected or self.expected != tactical_actions(
            state, self.category
        ):
            raise ValueError("Tactical fixture answers do not match the rules")
        return state


def tactical_cases(
    *,
    include_fork_defense: bool = True,
) -> tuple[TacticalCase, ...]:
    """Return versioned fixtures with reflection and token-color variants."""
    bases: tuple[tuple[str, tuple[int, ...], frozenset[int]], ...] = (
        ("win", (0, 1, 0, 1, 0, 2), frozenset({0})),
        ("block", (0, 1, 2, 1, 3, 1), frozenset({1})),
        (
            "safety",
            (3, 1, 6, 4, 5, 4, 4, 6, 2, 0, 3, 6, 5),
            frozenset({0, 1, 2, 4, 6}),
        ),
        (
            "fork",
            (0, 2, 0, 6, 3, 2, 0, 6, 6, 0, 3, 4, 4, 2, 2, 6),
            frozenset({3}),
        ),
    )
    if include_fork_defense:
        bases += (
            (
                "fork_defense-a",
                (6, 5, 0, 2, 1, 3, 1, 0, 6, 3, 4, 5, 4, 1, 4, 1, 1, 4, 3),
                frozenset({5, 6}),
            ),
            (
                "fork_defense-b",
                (5, 2, 0, 5, 2, 3, 4, 5, 6, 6, 5, 5, 6, 5),
                frozenset({3, 4, 6}),
            ),
        )
    result = []
    for label, moves, expected in bases:
        category = label.split("-", maxsplit=1)[0]
        for starting in (1, 2):
            for mirrored in (False, True):
                history = (
                    tuple(6 - action for action in moves) if mirrored else moves
                )
                answers = (
                    frozenset(6 - action for action in expected)
                    if mirrored
                    else expected
                )
                orientation = "mirror" if mirrored else "normal"
                result.append(
                    TacticalCase(
                        f"{label}-p{starting}-{orientation}",
                        category,
                        history,
                        starting,
                        answers,
                    )
                )
    return tuple(result)


def benchmark(
    agent: QAgent | NeuralAgent | PUCTAgent,
    *,
    seed: int = 2000,
    cases: tuple[TacticalCase, ...] | None = None,
) -> dict[str, Any]:
    """Score a copied, nonlearning policy on versioned diagnostic fixtures."""
    selected = tactical_cases() if cases is None else cases
    if not selected:
        raise ValueError("Tactical suite must contain at least one case")
    frozen = copy.deepcopy(agent)
    frozen.rng.seed(seed)
    categories: dict[str, dict[str, int | float]] = {}
    failures = []
    correct = 0
    for case in selected:
        state = case.state()
        action = frozen.select(state, epsilon=0)
        passed = action in case.expected
        count = categories.setdefault(case.category, {"correct": 0, "total": 0})
        count["total"] += 1
        count["correct"] += int(passed)
        correct += int(passed)
        if not passed:
            failures.append(
                {
                    "case": case.name,
                    "category": case.category,
                    "moves": list(case.moves),
                    "starting_player": case.starting_player,
                    "chosen_column": action + 1,
                    "expected_columns": [
                        column + 1 for column in sorted(case.expected)
                    ],
                }
            )
    for count in categories.values():
        count["accuracy"] = count["correct"] / count["total"]
    legacy_cases = sum(
        int(count["total"])
        for category, count in categories.items()
        if category != "fork_defense"
    )
    legacy_correct = sum(
        int(count["correct"])
        for category, count in categories.items()
        if category != "fork_defense"
    )
    return {
        "suite": (
            "connect-four-tactics-v2"
            if "fork_defense" in categories
            else "connect-four-tactics-v1"
        ),
        "legacy_cases": legacy_cases,
        "legacy_correct": legacy_correct,
        "seed": seed,
        "cases": len(selected),
        "correct": correct,
        "accuracy": correct / len(selected),
        "categories": categories,
        "failures": failures,
    }
