"""Monte Carlo Tree Search engine for Four in a Row."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .constants import PLAYER_ONE, PLAYER_TWO
from .game import GameState, evaluate_state, terminal_score
from .suggestion import EngineSuggestion


@dataclass
class _MctsNode:
    state: GameState
    parent: _MctsNode | None = None
    move: int | None = None
    children: list[_MctsNode] = field(default_factory=list)
    untried_moves: list[int] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0

    def __post_init__(self) -> None:
        if not self.untried_moves:
            self.untried_moves = self.state.legal_moves().copy()

    def ucb_value(self, exploration: float) -> float:
        """Calculate the UCB value for this node.
        Args:
            exploration: The exploration constant (e.g., sqrt(2)).
        Returns:
            The UCB value for this node.
        """
        if self.visits == 0:
            return math.inf
        parent_visits = max(1, self.parent.visits if self.parent else 1)
        mean = self.total_value / self.visits
        if self.parent and self.parent.state.current_player == PLAYER_TWO:
            mean = -mean
        return mean + exploration * math.sqrt(
            math.log(parent_visits) / self.visits
        )


def _random_rollout(state: GameState, rng: random.Random) -> float:
    sandbox = state.clone()
    while not sandbox.is_terminal():
        sandbox.apply_move(rng.choice(sandbox.legal_moves()))
    return terminal_score(sandbox)


def mcts_suggestion(
    state: GameState,
    iterations: int = 500,
    exploration: float = math.sqrt(2.0),
    rng: random.Random | None = None,
) -> EngineSuggestion:
    """Suggest a move using Monte Carlo Tree Search.
    Args:
        state: The current game state.
        iterations: The number of MCTS iterations to perform (default: 500).
        exploration: The exploration constant for UCB (default: sqrt(2)).
        rng: Optional random number generator for reproducibility.
    Returns:
        An EngineSuggestion with the best move,
        principal variation, and evaluation.
    Raises:
        ValueError: If there are no legal moves or
        if iterations is less than 1.
    """
    legal = state.legal_moves()
    if not legal:
        raise ValueError("No legal moves")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    random_gen = rng or random.Random()
    root = _MctsNode(state=state.clone())

    for _ in range(iterations):
        node = root

        while not node.untried_moves and \
                node.children and \
                not node.state.is_terminal():
            node = max(
                node.children,
                key=lambda child: child.ucb_value(exploration)
            )

        if node.untried_moves and not node.state.is_terminal():
            move = random_gen.choice(node.untried_moves)
            node.untried_moves.remove(move)
            child_state = node.state.clone()
            child_state.apply_move(move)
            child = _MctsNode(state=child_state, parent=node, move=move)
            node.children.append(child)
            node = child

        score = _random_rollout(node.state, random_gen)

        current: _MctsNode | None = node
        while current is not None:
            current.visits += 1
            current.total_value += score
            current = current.parent

    if not root.children:
        move = legal[0]
        fallback = state.clone()
        fallback.apply_move(move)
        return EngineSuggestion(
            engine="mcts", move=move, pv=[move],
            evaluation=evaluate_state(fallback)
        )

    if state.current_player == PLAYER_ONE:
        best = max(
            root.children,
            key=lambda child: child.total_value / max(1, child.visits)
        )
    else:
        best = min(
            root.children,
            key=lambda child: child.total_value / max(1, child.visits)
        )

    evaluation = best.total_value / max(1, best.visits)
    return EngineSuggestion(
        engine="mcts", move=best.move or legal[0],
        pv=[best.move or legal[0]], evaluation=evaluation
    )
