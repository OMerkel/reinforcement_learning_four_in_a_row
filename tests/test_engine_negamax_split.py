# Requirements: FR-NEGAMAX-001
"""Tests for the negamax engine helper."""

import pytest

from four_in_a_row.ai_negamax import _negamax_search, _player_sign
from four_in_a_row.constants import PLAYER_ONE, PLAYER_TWO
from four_in_a_row.engines import negamax_suggestion
from four_in_a_row.game import GameState

from .engines_shared import no_legal_moves_state, winning_position_for_p1

pytestmark = pytest.mark.regression


def test_negamax_finds_immediate_win() -> None:
    """Test that negamax finds an immediate winning move.

    Feature: Immediate Win Detection
    Scenario: Negamax engine finds immediate winning move

    Given a game state where player 1 has an immediate winning move in column 3
    When the negamax engine is invoked with sufficient depth
    Then the engine should suggest the winning move in column 3 with
    an evaluation of 1.0
    """
    state = winning_position_for_p1()
    suggestion = negamax_suggestion(state, depth=3)
    assert suggestion.move == 3
    assert suggestion.pv[0] == 3
    assert suggestion.evaluation == 1.0


def test_negamax_depth_validation() -> None:
    """Test that negamax engine validates depth parameter.

    Feature: Depth Parameter Validation
    Scenario: Negamax engine validates depth parameter

    Given a new game state
    When the negamax engine is invoked with a depth of 0
    Then the engine should raise a ValueError indicating that depth must be
    a positive integer
    """
    state = GameState.new()
    with pytest.raises(ValueError):
        negamax_suggestion(state, depth=0)


def test_negamax_no_legal_moves_raises() -> None:
    """Test that negamax raises ValueError when there are no legal moves.

    Feature: No Legal Moves Validation
    Scenario: negamax_suggestion raises when called on a terminal state

    Given a terminal game state with no legal moves
    When negamax_suggestion is invoked
    Then a ValueError should be raised
    """
    state = no_legal_moves_state()
    assert state.is_terminal()
    with pytest.raises(ValueError, match="No legal moves"):
        negamax_suggestion(state, depth=1)


def test_negamax_search_returns_empty_pv_on_terminal() -> None:
    """Test that _negamax_search returns empty pv for a terminal state.

    Feature: Terminal State Handling
    Scenario: _negamax_search returns evaluation without pv for terminal states

    Given a terminal game state
    When _negamax_search is called directly
    Then it should return a float value and an empty pv
    """
    state = no_legal_moves_state()
    assert state.is_terminal()
    value, pv = _negamax_search(state, depth=3)
    assert pv == []
    assert isinstance(value, float)


def test_negamax_search_depth_zero_returns_evaluation() -> None:
    """Test that _negamax_search at depth 0 returns static evaluation.

    Feature: Depth-Zero Cutoff
    Scenario: Search terminates at depth 0 and returns heuristic value

    Given any game state
    When _negamax_search is called with depth=0
    Then it should return the static evaluation and empty pv
    """
    state = GameState.new()
    value, pv = _negamax_search(state, depth=0)
    assert pv == []
    assert isinstance(value, float)


def test_negamax_search_no_legal_moves_non_terminal(monkeypatch) -> None:
    """Test that _negamax_search handles empty legal_moves
    on non-terminal state.

    Feature: Defensive Empty Moves Check
    Scenario: _negamax_search handles state where legal_moves() is empty
    but is_terminal() returns False

    Given a state mocked so is_terminal() is False but legal_moves() is empty
    When _negamax_search is called
    Then it should return the static evaluation and empty pv
    """
    state = GameState.new()
    monkeypatch.setattr(state, "is_terminal", lambda: False)
    monkeypatch.setattr(state, "legal_moves", lambda: [])
    value, pv = _negamax_search(state, depth=2)
    assert pv == []
    assert isinstance(value, float)


def test_negamax_player_two_minimizes() -> None:
    """Test that negamax returns a valid move for PLAYER_TWO.

    Feature: Negamax for PLAYER_TWO
    Scenario: Negamax suggests a move when PLAYER_TWO is to move

    Given a state where PLAYER_TWO is to move
    When negamax_suggestion is invoked
    Then the engine should return a legal move with valid evaluation
    """
    state = GameState.new()
    state.apply_move(3)  # PLAYER_ONE plays; now PLAYER_TWO's turn
    assert state.current_player == PLAYER_TWO
    suggestion = negamax_suggestion(state, depth=2)
    assert suggestion.move in state.legal_moves()
    assert -1.0 <= suggestion.evaluation <= 1.0


def test_player_sign_returns_positive_for_player_one() -> None:
    """Test that _player_sign returns 1.0 for PLAYER_ONE.

    Feature: Player Sign Calculation
    Scenario: _player_sign for PLAYER_ONE

    Given PLAYER_ONE
    When _player_sign is called
    Then the result should be 1.0
    """
    assert _player_sign(PLAYER_ONE) == 1.0


def test_player_sign_returns_negative_for_player_two() -> None:
    """Test that _player_sign returns -1.0 for PLAYER_TWO.

    Feature: Player Sign Calculation
    Scenario: _player_sign for PLAYER_TWO

    Given PLAYER_TWO
    When _player_sign is called
    Then the result should be -1.0
    """
    assert _player_sign(PLAYER_TWO) == -1.0


def test_negamax_suggestion_returns_engine_name() -> None:
    """Test that negamax suggestion result has correct engine name.

    Feature: Engine Identification
    Scenario: Suggestion contains correct engine label

    Given a new game state
    When negamax_suggestion is invoked
    Then the suggestion engine field should be "negamax"
    """
    state = GameState.new()
    suggestion = negamax_suggestion(state, depth=2)
    assert suggestion.engine == "negamax"
    assert len(suggestion.pv) >= 1


def test_negamax_pv_starts_with_suggested_move() -> None:
    """Test that the principal variation starts with the suggested move.

    Feature: Principal Variation Consistency
    Scenario: Suggested move matches first element of pv

    Given a new game state
    When negamax_suggestion is invoked
    Then pv[0] should equal the suggested move
    """
    state = GameState.new()
    suggestion = negamax_suggestion(state, depth=3)
    assert suggestion.pv[0] == suggestion.move
