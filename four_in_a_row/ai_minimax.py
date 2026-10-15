"""Minimax search engine for Four in a Row."""
from __future__ import annotations

from .constants import PLAYER_ONE
from .game import GameState, evaluate_state
from .suggestion import EngineSuggestion


def _best_for_player(
        scores: list[tuple[int, float, list[int]]], player: int
) -> tuple[int, float, list[int]]:
    if player == PLAYER_ONE:
        return max(scores, key=lambda item: item[1])
    return min(scores, key=lambda item: item[1])


def _minimax_search(state: GameState, depth: int) -> tuple[float, list[int]]:
    if depth == 0 or state.is_terminal():
        return evaluate_state(state), []

    legal = state.legal_moves()
    if not legal:
        return evaluate_state(state), []

    candidates: list[tuple[int, float, list[int]]] = []
    for move in legal:
        child = state.clone()
        child.apply_move(move)
        value, pv = _minimax_search(child, depth - 1)
        candidates.append((move, value, [move] + pv))

    _, value, pv = _best_for_player(candidates, state.current_player)
    return value, pv


def minimax_suggestion(
        state: GameState, depth: int = 4
) -> EngineSuggestion:
    """Suggest a move using minimax search.
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
    score, pv = _minimax_search(state, depth)
    if not pv:
        raise ValueError("No legal moves")
    return EngineSuggestion(
        engine="minimax", move=pv[0], pv=pv, evaluation=score
    )
