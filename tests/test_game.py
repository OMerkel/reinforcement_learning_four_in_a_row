# Requirements: FR-GAME-001
"""Unit tests for the GameState class and related functions in game.py."""
import pytest

from four_in_a_row.constants import PLAYER_ONE, PLAYER_TWO
from four_in_a_row.game import (
    GameState,
    InvalidMoveError,
    evaluate_state,
    opponent,
    terminal_score,
)

pytestmark = pytest.mark.fast


def test_apply_and_undo_move() -> None:
    """Test that applying a move updates the game state correctly and
    that undoing a move restores the previous state.

    Feature: Apply and undo move
    Scenario: Applying and undoing a move

    Given the initial game state
    When applying a move in column 0
    Then the move is applied correctly
    """
    state = GameState.new()
    row = state.apply_move(0)
    assert row == 0
    assert state.current_player == PLAYER_TWO
    assert state.move_stack == [0]

    undone = state.undo_move()
    assert undone == 0
    assert state.current_player == PLAYER_ONE
    assert not state.move_stack


def test_invalid_move_column_range() -> None:
    """Test that applying a move in an invalid column raises an error.

    Feature: Invalid move
    Scenario: Applying a move in an invalid column

    Given the initial game state
    When applying a move in column 10
    Then an InvalidMoveError is raised
    """
    state = GameState.new()
    with pytest.raises(InvalidMoveError):
        state.apply_move(10)


def test_new_state_invalid_starting_player() -> None:
    """Test that creating a new game state with
    an invalid starting player raises an error.

    Feature: Invalid starting player
    Scenario: Creating a new game state with an invalid starting player

    Given an invalid starting player
    When creating a new game state
    Then a ValueError is raised
    """
    with pytest.raises(ValueError):
        GameState.new(starting_player=3)


def test_undo_without_moves_raises() -> None:
    """Test that undoing a move without any moves raises an error.

    Feature: Undo move
    Scenario: Undoing a move without any moves

    Given the initial game state
    When undoing a move without any moves
    Then an InvalidMoveError is raised
    """
    state = GameState.new()
    with pytest.raises(InvalidMoveError):
        state.undo_move()


def test_column_full_raises() -> None:
    """Test that applying a move in a full column raises an error.

    Feature: Column full
    Scenario: Applying a move in a full column

    Given a column that is full
    When applying a move in that column
    Then an InvalidMoveError is raised
    """
    state = GameState.new()
    for _ in range(6):
        state.apply_move(0)
    with pytest.raises(InvalidMoveError):
        state.apply_move(0)


def test_horizontal_winner() -> None:
    """Test that a horizontal win is detected correctly.

    Feature: Horizontal win
    Scenario: Player one achieves a horizontal win

    Given a game state with a horizontal line of three for player one
    When player one makes the winning move
    Then player one is the winner
    """
    state = GameState.new()
    moves = [0, 0, 1, 1, 2, 2, 3]
    for move in moves:
        state.apply_move(move)
    assert state.winner() == PLAYER_ONE
    assert terminal_score(state) == 1.0


def test_vertical_winner() -> None:
    """Test that a vertical win is detected correctly.

    Feature: Vertical win
    Scenario: Player one achieves a vertical win

    Given a game state with a vertical line of three for player one
    When player one makes the winning move
    Then player one is the winner
    """
    state = GameState.new()
    moves = [0, 1, 0, 1, 0, 1, 0]
    for move in moves:
        state.apply_move(move)
    assert state.winner() == PLAYER_ONE


def test_diagonal_winner() -> None:
    """Test that a diagonal win is detected correctly.

    Feature: Diagonal win
    Scenario: Player one achieves a diagonal win

    Given a game state with a diagonal line of three for player one
    When player one makes the winning move
    Then player one is the winner
    """
    state = GameState.new()
    moves = [0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3]
    for move in moves:
        state.apply_move(move)
    assert state.winner() == PLAYER_ONE


def test_evaluate_state_in_range() -> None:
    """Test that the evaluation of a game state is within the valid range.

    Feature: State evaluation
    Scenario: Evaluating a game state

    Given a game state
    When evaluating the state
    Then the evaluation is between -1.0 and 1.0
    """
    state = GameState.new()
    state.apply_move(3)
    value = evaluate_state(state)
    assert -1.0 <= value <= 1.0


def test_opponent_invalid_player() -> None:
    """Test that requesting the opponent of an invalid player raises an error.

    Feature: Opponent
    Scenario: Requesting the opponent of an invalid player

    Given an invalid player
    When requesting the opponent
    Then a ValueError is raised
    """
    with pytest.raises(ValueError):
        opponent(0)
