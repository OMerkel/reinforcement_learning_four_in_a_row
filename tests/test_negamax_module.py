# Requirements: FR-NEGAMAX-001
"""Tests for the negamax module, comparing its output to
the minimax module on curated positions.

These tests cover the following features:
- Negamax suggestion: Testing that the negamax_suggestion function returns
  the same move, principal variation, and evaluation as the minimax_suggestion
  function on a variety of game states and search depths.

The tests use pytest for assertions and parameterization.
Each test includes a docstring that describes the feature being tested,
the scenario, and the expected outcomes."""
import pytest

from four_in_a_row.ai_minimax import minimax_suggestion
from four_in_a_row.ai_negamax import negamax_suggestion
from four_in_a_row.game import GameState

pytestmark = pytest.mark.regression


def _state_from_moves(moves: list[int]) -> GameState:
    state = GameState.new()
    for move in moves:
        state.apply_move(move)
    return state


@pytest.mark.parametrize(
    ("moves", "depth"),
    [
        ([], 1),
        ([], 2),
        ([3, 2, 3, 2], 2),
        ([3, 2, 4, 2, 3], 3),
        ([0, 0, 1, 1, 2, 2], 3),
    ],
)
def test_negamax_matches_minimax_on_curated_positions(
    moves: list[int], depth: int
) -> None:
    """Test that negamax_suggestion matches
    minimax_suggestion on curated positions.

    Feature: Negamax suggestion matches minimax
    Scenario: Comparing negamax and minimax suggestions on curated positions

    Given a game state derived from a sequence of moves
    When requesting suggestions from both negamax and
    minimax with the same depth
    Then the suggested move, principal variation, and evaluation should match
    """
    state = _state_from_moves(moves)
    minimax = minimax_suggestion(state, depth=depth)
    negamax = negamax_suggestion(state, depth=depth)

    assert negamax.move == minimax.move
    assert negamax.pv == minimax.pv
    assert negamax.evaluation == pytest.approx(minimax.evaluation, abs=1e-9)


def test_negamax_reproducible_for_same_position_and_depth() -> None:
    """Test that negamax_suggestion is reproducible on the same position.

    Feature: Negamax deterministic behavior
    Scenario: Running negamax_suggestion multiple times on the same state

    Given a curated non-terminal game state
    When requesting negamax suggestions repeatedly with the same depth
    Then the suggested move, PV, and evaluation are identical
    """
    state = _state_from_moves([3, 2, 3, 4, 2, 4])

    first = negamax_suggestion(state, depth=4)
    second = negamax_suggestion(state, depth=4)

    assert first.move == second.move
    assert first.pv == second.pv
    assert first.evaluation == pytest.approx(second.evaluation, abs=1e-9)


def test_negamax_does_not_mutate_state() -> None:
    """Test that negamax_suggestion does not mutate the input state.

    Feature: Negamax state safety
    Scenario: Calling negamax_suggestion on a shared GameState

    Given a game state with existing moves
    When requesting a negamax suggestion
    Then the game state's move stack and current player remain unchanged
    """
    state = _state_from_moves([0, 1, 0, 1, 2])
    before_stack = list(state.move_stack)
    before_player = state.current_player

    _ = negamax_suggestion(state, depth=3)

    assert state.move_stack == before_stack
    assert state.current_player == before_player
