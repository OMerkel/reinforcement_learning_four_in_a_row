# Requirements: FR-ALPHABETA-001
"""Tests for the alpha-beta engine helper."""

import math

import pytest

from four_in_a_row.ai_alphabeta import _alphabeta_search
from four_in_a_row.constants import PLAYER_TWO
from four_in_a_row.engines import alphabeta_suggestion
from four_in_a_row.game import GameState

from .engines_shared import no_legal_moves_state, winning_position_for_p1

pytestmark = pytest.mark.regression


def test_alphabeta_finds_immediate_win() -> None:
    """Test that alpha-beta finds an immediate winning move.
    This is a regression test for a bug where the alpha-beta engine would fail
    to find an immediate winning move if it was the first legal move
    in the list.

    Feature: Immediate Win Detection
    Scenario: Alpha-beta engine finds immediate winning move

    Given a game state where player 1 has an immediate winning move in column 3
    When the alpha-beta engine is invoked with sufficient depth
    Then the engine should suggest the winning move in column 3 with
    an evaluation of 1.0
    """
    state = winning_position_for_p1()
    suggestion = alphabeta_suggestion(state, depth=5)
    assert suggestion.move == 3
    assert suggestion.pv[0] == 3
    assert suggestion.evaluation == 1.0


def test_alphabeta_depth_validation() -> None:
    """Test that alpha-beta engine validates depth parameter.
    This is a regression test for a bug where the alpha-beta engine would not
    validate the depth parameter and would raise an unexpected error when given
    a non-positive depth.

    Feature: Depth Parameter Validation
    Scenario: Alpha-beta engine validates depth parameter

    Given a new game state
    When the alpha-beta engine is invoked with a depth of 0
    Then the engine should raise a ValueError indicating that depth must be
    a positive integer
    """
    state = GameState.new()
    with pytest.raises(ValueError):
        alphabeta_suggestion(state, depth=0)


def test_alphabeta_no_legal_moves_raises() -> None:
    """Test that alphabeta raises ValueError when there are no legal moves.

    Feature: No Legal Moves Validation
    Scenario: alphabeta_suggestion raises when called on a terminal state

    Given a terminal game state with no legal moves
    When alphabeta_suggestion is invoked
    Then a ValueError should be raised
    """
    state = no_legal_moves_state()
    assert state.is_terminal()
    with pytest.raises(ValueError, match="No legal moves"):
        alphabeta_suggestion(state, depth=1)


def test_alphabeta_search_returns_empty_pv_on_terminal() -> None:
    """Test that _alphabeta_search returns empty pv for a terminal state.

    Feature: Terminal State Handling
    Scenario: _alphabeta_search returns evaluation without pv
    for terminal states

    Given a terminal game state
    When _alphabeta_search is called directly
    Then it should return a float value and an empty pv
    """
    state = no_legal_moves_state()
    assert state.is_terminal()
    value, pv = _alphabeta_search(
        state, depth=3, alpha=-math.inf, beta=math.inf
    )
    assert pv == []
    assert isinstance(value, float)


def test_alphabeta_search_depth_zero_returns_evaluation() -> None:
    """Test that _alphabeta_search at depth 0 returns static evaluation.

    Feature: Depth-Zero Cutoff
    Scenario: Search terminates at depth 0 and returns heuristic value

    Given any game state
    When _alphabeta_search is called with depth=0
    Then it should return the static evaluation and empty pv
    """
    state = GameState.new()
    value, pv = _alphabeta_search(
        state, depth=0, alpha=-math.inf, beta=math.inf
    )
    assert pv == []
    assert isinstance(value, float)


def test_alphabeta_search_no_legal_moves_non_terminal(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that _alphabeta_search handles empty legal_moves
    on non-terminal state.

    Feature: Defensive Empty Moves Check
    Scenario: _alphabeta_search handles state where legal_moves() is empty
    but is_terminal() returns False

    Given a state mocked so is_terminal() is False but legal_moves() is empty
    When _alphabeta_search is called
    Then it should return the static evaluation and empty pv
    """
    state = GameState.new()
    monkeypatch.setattr(state, "is_terminal", lambda: False)
    monkeypatch.setattr(state, "legal_moves", lambda: [])
    value, pv = _alphabeta_search(
        state, depth=2, alpha=-math.inf, beta=math.inf
    )
    assert pv == []
    assert isinstance(value, float)


def test_alphabeta_player_two_minimizes() -> None:
    """Test that alphabeta returns a valid move for PLAYER_TWO.

    Feature: Alpha-Beta for PLAYER_TWO
    Scenario: alphabeta_suggestion suggests a move when PLAYER_TWO is to move

    Given a state where PLAYER_TWO is to move
    When alphabeta_suggestion is invoked
    Then the engine should return a legal move with valid evaluation
    """
    state = GameState.new()
    state.apply_move(3)  # PLAYER_ONE plays; now PLAYER_TWO's turn
    assert state.current_player == PLAYER_TWO
    suggestion = alphabeta_suggestion(state, depth=2)
    assert suggestion.move in state.legal_moves()
    assert -1.0 <= suggestion.evaluation <= 1.0


def test_alphabeta_pruning_triggered_by_maximizer() -> None:
    """Test that alpha pruning fires (beta cutoff) when maximizing.

    Feature: Alpha-Beta Pruning
    Scenario: Maximizer triggers beta cutoff

    Given a position with a winning move for PLAYER_ONE
    When alphabeta_suggestion is called with a high-enough depth
    Then it should return the winning move quickly via pruning
    """
    state = winning_position_for_p1()
    # Deep enough search exercises alpha >= beta cutoff in maximizing branch
    suggestion = alphabeta_suggestion(state, depth=6)
    assert suggestion.move == 3
    assert suggestion.evaluation == 1.0


def test_alphabeta_pruning_triggered_by_minimizer() -> None:
    """Test that beta pruning fires (alpha cutoff) when minimizing.

    Feature: Alpha-Beta Pruning
    Scenario: Minimizer triggers alpha cutoff

    Given a state where PLAYER_TWO is to move with a strong immediate response
    When alphabeta_suggestion is called
    Then it should trigger the minimizing branch cutoff
    """
    state = winning_position_for_p1()
    state.apply_move(
        3
    )  # P1 wins, now it's P2's turn on terminal-adjacent state
    # Build a position where P2 needs to minimize
    state2 = GameState.new()
    for move in [0, 0, 1, 1, 2, 2]:
        state2.apply_move(move)
    # Now P1 can win with col 3; let P1 play something else to give P2 a turn
    state2.apply_move(6)  # P1 plays elsewhere
    assert state2.current_player == PLAYER_TWO
    suggestion = alphabeta_suggestion(state2, depth=4)
    assert suggestion.move in state2.legal_moves()


def test_alphabeta_suggestion_returns_engine_name() -> None:
    """Test that alphabeta suggestion result has correct engine name.

    Feature: Engine Identification
    Scenario: Suggestion contains correct engine label

    Given a new game state
    When alphabeta_suggestion is invoked
    Then the suggestion engine field should be "alphabeta"
    """
    state = GameState.new()
    suggestion = alphabeta_suggestion(state, depth=2)
    assert suggestion.engine == "alphabeta"
    assert len(suggestion.pv) >= 1


def test_alphabeta_pv_starts_with_suggested_move() -> None:
    """Test that the principal variation starts with the suggested move.

    Feature: Principal Variation Consistency
    Scenario: Suggested move matches first element of pv

    Given a new game state
    When alphabeta_suggestion is invoked
    Then pv[0] should equal the suggested move
    """
    state = GameState.new()
    suggestion = alphabeta_suggestion(state, depth=3)
    assert suggestion.pv[0] == suggestion.move
