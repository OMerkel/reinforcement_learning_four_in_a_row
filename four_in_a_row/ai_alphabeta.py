"""Alpha-beta pruning engine for Four in a Row."""
from __future__ import annotations

import math

from .constants import PLAYER_ONE
from .game import GameState, evaluate_state
from .suggestion import EngineSuggestion


def _alphabeta_search(
        state: GameState, depth: int, alpha: float, beta: float
) -> tuple[float, list[int]]:
    if depth == 0 or state.is_terminal():
        return evaluate_state(state), []

    legal = state.legal_moves()
    if not legal:
        return evaluate_state(state), []

    maximizing = state.current_player == PLAYER_ONE
    best_value = -math.inf if maximizing else math.inf
    best_pv: list[int] = []

    for move in legal:
        child = state.clone()
        child.apply_move(move)
        value, pv = _alphabeta_search(child, depth - 1, alpha, beta)
        if maximizing:
            if value > best_value:
                best_value = value
                best_pv = [move] + pv
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break
        else:
            if value < best_value:
                best_value = value
                best_pv = [move] + pv
            beta = min(beta, best_value)
            if alpha >= beta:
                break
    return best_value, best_pv


def alphabeta_suggestion(
    state: GameState,
    depth: int = 6,
) -> EngineSuggestion:
    """Suggest a move using alpha-beta pruning.
    Args:
        state: The current game state.
        depth: The maximum search depth (default: 6).
    Returns:
        An EngineSuggestion with the best move,
        principal variation, and evaluation.
    Raises:
        ValueError: If depth is less than 1 or
        if there are no legal moves.
    """

    if depth < 1:
        raise ValueError("depth must be >= 1")
    score, pv = _alphabeta_search(state, depth, -math.inf, math.inf)
    if not pv:
        raise ValueError("No legal moves")
    return EngineSuggestion(
        engine="alphabeta", move=pv[0], pv=pv, evaluation=score
    )
