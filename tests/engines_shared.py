"""Shared helpers for engine-focused tests."""

from four_in_a_row.constants import PLAYER_ONE
from four_in_a_row.game import GameState


def winning_position_for_p1() -> GameState:
    """Return a state where Player 1 has an immediate winning move in col 4."""
    state = GameState.new()
    for move in [0, 0, 1, 1, 2, 2]:
        state.apply_move(move)
    assert state.current_player == PLAYER_ONE
    return state


def no_legal_moves_state() -> GameState:
    """Return a terminal state with no legal moves left."""
    state = GameState.new()
    for column in [
        0, 0, 1, 1, 2, 2, 4, 3, 3, 4, 4, 5, 3, 5, 5, 6, 6,
        6, 0, 1, 2, 2, 1, 0, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5,
        6, 6, 5, 4, 3, 2, 1, 0,
    ]:
        if column in state.legal_moves():  # pragma: no cover
            state.apply_move(column)
        if state.is_terminal() and not state.legal_moves():
            break

    while state.legal_moves():  # pragma: no cover
        state.apply_move(state.legal_moves()[0])
    return state
