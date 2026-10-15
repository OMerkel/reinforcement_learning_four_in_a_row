"""Engine façade and shared suggestion type.

This module preserves the public engine API while delegating concrete
implementations to dedicated modules:

- ai_random.py
- ai_minimax.py
- ai_negamax.py
- ai_alphabeta.py
- ai_mcts_ucb.py
"""
from __future__ import annotations

import random

from .ai_alphabeta import alphabeta_suggestion as _alphabeta_suggestion
from .ai_mcts_ucb import mcts_suggestion as _mcts_suggestion
from .ai_minimax import minimax_suggestion as _minimax_suggestion
from .ai_negamax import negamax_suggestion as _negamax_suggestion
from .ai_random import random_suggestion as _random_suggestion
from .game import GameState
from .suggestion import EngineSuggestion


def random_suggestion(
    state: GameState,
    rng: random.Random | None = None,
) -> EngineSuggestion:
    """Suggest a move using a random engine.
    Args:
        state: The current game state.
        rng: Optional random number generator for reproducibility.
    Returns:
        An EngineSuggestion with the suggested move and evaluation.
    Raises: ValueError if there are no legal moves.
    """
    return _random_suggestion(state, rng=rng)


def minimax_suggestion(state: GameState, depth: int = 4) -> EngineSuggestion:
    """Suggest a move using the minimax engine.
    Args:
        state: The current game state.
        depth: The search depth (default: 4).
    Returns:
        An EngineSuggestion with the suggested move and evaluation.
    Raises:
        ValueError if there are no legal moves.
    """
    return _minimax_suggestion(state, depth=depth)


def alphabeta_suggestion(state: GameState, depth: int = 6) -> EngineSuggestion:
    """Suggest a move using the alphabeta engine.
    Args:
        state: The current game state.
        depth: The search depth (default: 6).
    Returns:
        An EngineSuggestion with the suggested move and evaluation.
    Raises:
        ValueError if there are no legal moves.
    """
    return _alphabeta_suggestion(state, depth=depth)


def negamax_suggestion(state: GameState, depth: int = 4) -> EngineSuggestion:
    """Suggest a move using the negamax engine.
    Args:
        state: The current game state.
        depth: The search depth (default: 4).
    Returns:
        An EngineSuggestion with the suggested move and evaluation.
    Raises:
        ValueError if there are no legal moves.
    """
    return _negamax_suggestion(state, depth=depth)


def mcts_suggestion(
    state: GameState,
    iterations: int = 500,
    exploration: float = 2.0**0.5,
    rng: random.Random | None = None,
) -> EngineSuggestion:
    """Suggest a move using the MCTS engine.
    Args:
        state: The current game state.
        iterations: The number of MCTS iterations (default: 500).
        exploration: The exploration constant for UCB (default: sqrt(2)).
        rng: Optional random number generator for reproducibility.
    Returns:
        An EngineSuggestion with the suggested move and evaluation.
    Raises:
        ValueError if there are no legal moves.
    """
    return _mcts_suggestion(
            state,
            iterations=iterations,
            exploration=exploration,
            rng=rng,
    )


def suggest_move(
    state: GameState,
    engine: str,
    *,
    depth: int = 5,
    iterations: int = 800,
    rng: random.Random | None = None,
) -> EngineSuggestion:
    """Suggest a move using the specified engine and parameters.
    Args:
        state: The current game state.
        engine: The name of the engine to use ("random", "minimax",
                "negamax", "alphabeta", "mcts").
        depth: The search depth for applicable engines (default: 5).
        iterations: The number of iterations for MCTS (default: 800).
        rng: Optional random number generator for reproducibility.
    Returns:
        An EngineSuggestion with the suggested move and evaluation.
    Raises:
        ValueError if the engine name is unknown or
        if parameters are invalid.
    """
    normalized = engine.strip().lower()
    if normalized == "random":
        return random_suggestion(state, rng=rng)
    if normalized == "minimax":
        return minimax_suggestion(state, depth=depth)
    if normalized == "negamax":
        return negamax_suggestion(state, depth=depth)
    if normalized in {"alphabeta", "alpha-beta", "alpha_beta"}:
        return alphabeta_suggestion(state, depth=depth)
    if normalized == "mcts":
        return mcts_suggestion(state, iterations=iterations, rng=rng)
    raise ValueError(f"Unknown engine: {engine}")
