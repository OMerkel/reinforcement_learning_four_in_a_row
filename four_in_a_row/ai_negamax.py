"""Negamax search engine for Four in a Row."""
from __future__ import annotations

from .constants import PLAYER_ONE
from .game import GameState, evaluate_state
from .suggestion import EngineSuggestion


def _player_sign(player: int) -> float:
    return 1.0 if player == PLAYER_ONE else -1.0


def _negamax_search(state: GameState, depth: int) -> tuple[float, list[int]]:
    if depth == 0 or state.is_terminal():
        return _player_sign(state.current_player) * evaluate_state(state), []

    legal = state.legal_moves()
    if not legal:
        return _player_sign(state.current_player) * evaluate_state(state), []

    best_value = float("-inf")
    best_pv: list[int] = []

    for move in legal:
        child = state.clone()
        child.apply_move(move)
        child_value, child_pv = _negamax_search(child, depth - 1)
        value = -child_value
        if value > best_value:
            best_value = value
            best_pv = [move] + child_pv

    return best_value, best_pv


def negamax_suggestion(
    state: GameState, depth: int = 4
) -> EngineSuggestion:
    """Suggest a move using negamax search.
    Args:
        state: The current game state.
        depth: The maximum search depth (default: 4).
    Returns:
        An EngineSuggestion with the best move,
        principal variation, and evaluation.
    Raises:
        ValueError: If depth is less than 1 or
        if there are no legal moves.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")

    score_from_side_to_move, pv = _negamax_search(state, depth)
    if not pv:
        raise ValueError("No legal moves")

    evaluation = score_from_side_to_move * _player_sign(state.current_player)
    return EngineSuggestion(
        engine="negamax", move=pv[0], pv=pv, evaluation=evaluation
    )
