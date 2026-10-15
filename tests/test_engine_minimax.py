# Requirements: FR-MINIMAX-001
"""Tests for the minimax engine helper."""

import pytest

from four_in_a_row.ai_minimax import _best_for_player, _minimax_search
from four_in_a_row.constants import PLAYER_ONE, PLAYER_TWO
from four_in_a_row.engines import minimax_suggestion
from four_in_a_row.game import GameState

from .engines_shared import no_legal_moves_state, winning_position_for_p1

pytestmark = pytest.mark.regression


def test_minimax_finds_immediate_win() -> None:
    """Test that minimax finds an immediate winning move.

    Feature: Immediate Win Detection
    Scenario: Minimax engine finds immediate winning move

    Given a game state where player 1 has an immediate winning move in column 3
    When the minimax engine is invoked with sufficient depth
    Then the engine should suggest the winning move in column 3 with
    an evaluation of 1.0
    """
    state = winning_position_for_p1()
    suggestion = minimax_suggestion(state, depth=3)
    assert suggestion.move == 3
    assert suggestion.pv[0] == 3
    assert suggestion.evaluation == 1.0


def test_minimax_depth_validation() -> None:
    """Test that minimax engine validates depth parameter.

    Feature: Depth Parameter Validation
    Scenario: Minimax engine validates depth parameter

    Given a new game state
    When the minimax engine is invoked with a depth of 0
    Then the engine should raise a ValueError indicating that depth must be
    a positive integer
    """
    state = GameState.new()
    with pytest.raises(ValueError):
        minimax_suggestion(state, depth=0)


def test_minimax_no_legal_moves_raises() -> None:
    """Test that minimax raises ValueError when there are no legal moves.

    Feature: No Legal Moves Validation
    Scenario: minimax_suggestion raises when called on a terminal state

    Given a terminal game state with no legal moves
    When minimax_suggestion is invoked
    Then a ValueError should be raised
    """
    state = no_legal_moves_state()
    assert state.is_terminal()
    with pytest.raises(ValueError, match="No legal moves"):
        minimax_suggestion(state, depth=1)


def test_minimax_search_returns_empty_pv_on_terminal() -> None:
    """Test that _minimax_search returns empty pv for a terminal state.

    Feature: Terminal State Handling
    Scenario: _minimax_search returns evaluation without pv for terminal states

    Given a terminal game state
    When _minimax_search is called directly
    Then it should return an evaluation and an empty pv
    """
    state = no_legal_moves_state()
    assert state.is_terminal()
    value, pv = _minimax_search(state, depth=3)
    assert pv == []
    assert isinstance(value, float)


def test_minimax_search_depth_zero_returns_evaluation() -> None:
    """Test that _minimax_search at depth 0 returns static evaluation.

    Feature: Depth-Zero Cutoff
    Scenario: Search terminates at depth 0 and returns heuristic value

    Given any game state
    When _minimax_search is called with depth=0
    Then it should return the static evaluation and empty pv
    """
    state = GameState.new()
    value, pv = _minimax_search(state, depth=0)
    assert pv == []
    assert isinstance(value, float)


def test_minimax_player_two_minimizes() -> None:
    """Test that minimax minimizes for PLAYER_TWO.

    Feature: Minimax for PLAYER_TWO
    Scenario: Minimax selects move that minimizes evaluation for PLAYER_TWO

    Given a state where PLAYER_TWO is to move
    When minimax_suggestion is invoked
    Then the engine should return a legal move
    """
    state = GameState.new()
    state.apply_move(3)  # PLAYER_ONE plays; now PLAYER_TWO's turn
    assert state.current_player == PLAYER_TWO
    suggestion = minimax_suggestion(state, depth=2)
    assert suggestion.move in state.legal_moves()
    assert -1.0 <= suggestion.evaluation <= 1.0


def test_best_for_player_returns_max_for_player_one() -> None:
    """Test that _best_for_player returns max score for PLAYER_ONE.

    Feature: Best Move Selection
    Scenario: _best_for_player selects maximum-valued candidate for PLAYER_ONE

    Given a list of move candidates with different scores
    When _best_for_player is called for PLAYER_ONE
    Then it should return the candidate with the highest score
    """
    candidates = [(0, -0.5, [0]), (1, 0.3, [1]), (2, 0.8, [2])]
    move, value, pv = _best_for_player(candidates, PLAYER_ONE)
    assert move == 2
    assert value == 0.8
    assert pv == [2]


def test_best_for_player_returns_min_for_player_two() -> None:
    """Test that _best_for_player returns min score for PLAYER_TWO.

    Feature: Best Move Selection
    Scenario: _best_for_player selects minimum-valued candidate for PLAYER_TWO

    Given a list of move candidates with different scores
    When _best_for_player is called for PLAYER_TWO
    Then it should return the candidate with the lowest score
    """
    candidates = [(0, -0.5, [0]), (1, 0.3, [1]), (2, 0.8, [2])]
    move, value, pv = _best_for_player(candidates, PLAYER_TWO)
    assert move == 0
    assert value == -0.5
    assert pv == [0]


def test_minimax_suggestion_returns_engine_name() -> None:
    """Test that minimax suggestion result has correct engine name.

    Feature: Engine Identification
    Scenario: Suggestion contains correct engine label

    Given a new game state
    When minimax_suggestion is invoked
    Then the suggestion engine field should be "minimax"
    """
    state = GameState.new()
    suggestion = minimax_suggestion(state, depth=2)
    assert suggestion.engine == "minimax"
    assert len(suggestion.pv) >= 1


def test_minimax_pv_starts_with_suggested_move() -> None:
    """Test that the principal variation starts with the suggested move.

    Feature: Principal Variation Consistency
    Scenario: Suggested move matches first element of pv

    Given a new game state
    When minimax_suggestion is invoked
    Then pv[0] should equal the suggested move
    """
    state = GameState.new()
    suggestion = minimax_suggestion(state, depth=3)
    assert suggestion.pv[0] == suggestion.move


def test_minimax_search_no_legal_moves_non_terminal(monkeypatch) -> None:
    """Test that _minimax_search handles empty legal_moves
    on non-terminal state.

    Feature: Defensive Empty Moves Check
    Scenario: _minimax_search handles state where legal_moves() is empty
    but is_terminal() returns False

    Given a state mocked so is_terminal() is False but legal_moves() is empty
    When _minimax_search is called
    Then it should return the static evaluation and empty pv
    """
    state = GameState.new()
    monkeypatch.setattr(state, "is_terminal", lambda: False)
    monkeypatch.setattr(state, "legal_moves", lambda: [])
    value, pv = _minimax_search(state, depth=2)
    assert pv == []
    assert isinstance(value, float)
