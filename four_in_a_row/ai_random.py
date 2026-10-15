"""Random move suggestion engine for Four in a Row."""
from __future__ import annotations

import random

from .game import GameState, evaluate_state
from .suggestion import EngineSuggestion


def random_suggestion(
    state: GameState, rng: random.Random | None = None
) -> EngineSuggestion:
    """Suggest a move using a random strategy.
    Args:
        state: The current game state.
        rng: An optional random number generator (default: None).
    Returns:
        An EngineSuggestion with the chosen move,
        principal variation, and evaluation.
    Raises:
        ValueError: If there are no legal moves.
    """
    legal = state.legal_moves()
    if not legal:
        raise ValueError("No legal moves")
    random_gen = rng or random.Random()
    move = random_gen.choice(legal)
    sandbox = state.clone()
    sandbox.apply_move(move)
    return EngineSuggestion(
        engine="random", move=move, pv=[move],
        evaluation=evaluate_state(sandbox)
    )
