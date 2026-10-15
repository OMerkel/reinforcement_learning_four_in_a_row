# Requirements: FR-HISTORY-001, FR-SGF-001
"""Tests for the history and SGF functionality of the four_in_a_row package.
These tests cover the following features:

- HistoryManager: Testing the undo and replay functionality of
  the HistoryManager class.
- SGF import/export: Testing the export_sgf and import_sgf functions for
  correct serialization and deserialization of game states, as well as
  error handling for invalid SGF content.

The tests use pytest for assertions and exception handling. Each test includes
a docstring that describes the feature being tested, the scenario, and the
expected outcomes.
"""
import pytest

from four_in_a_row.game import GameState
from four_in_a_row.history import HistoryManager
from four_in_a_row.sgf import SgfError, export_sgf, import_sgf

pytestmark = pytest.mark.fast


def test_history_undo_replay() -> None:
    """Test that the HistoryManager can undo and replay moves correctly.

    Feature: HistoryManager undo and replay
    Scenario: Recording moves, undoing, and replaying

    Given a game state and a history manager
    When recording a sequence of moves
    Then the move stack should reflect those moves
    When undoing the last move
    Then the move stack should reflect the undone move
    When replaying the moves
    Then the move stack should be restored to the original sequence
    """
    state = GameState.new()
    history = HistoryManager()

    for column in [0, 1, 0]:
        player = state.current_player
        row = state.apply_move(column)
        history.record_move(player, column, row)

    assert state.move_stack == [0, 1, 0]
    history.undo(state)
    assert state.move_stack == [0, 1]
    history.replay(state)
    assert state.move_stack == [0, 1, 0]


def test_history_empty_undo_and_replay_raise() -> None:
    """Test that undoing or replaying with an empty history raises an error.

    Feature: HistoryManager empty undo and replay
    Scenario: Attempting to undo or replay with no recorded moves

    Given a game state and an empty history manager
    When checking if undo or replay is possible
    Then both should return False
    When attempting to undo or replay
    Then an exception should be raised
    """
    state = GameState.new()
    history = HistoryManager()
    assert history.can_undo() is False
    assert history.can_replay() is False
    with pytest.raises(ValueError):
        history.undo(state)
    with pytest.raises(ValueError):
        history.replay(state)


def test_export_import_sgf_roundtrip() -> None:
    """Test that exporting and then importing SGF content preserves
    the game state.

    Feature: SGF export and import
    Scenario: Exporting and importing SGF content

    Given a sequence of moves
    When exporting the moves to SGF format
    Then the SGF content should be generated correctly
    When importing the SGF content
    Then the starting player and moves should match the original sequence
    """
    moves = [0, 1, 2, 3]
    content = export_sgf(moves)
    start_player, parsed_moves = import_sgf(content)
    assert start_player == 1
    assert parsed_moves == moves


def test_import_sgf_invalid_size() -> None:
    """Test that importing SGF content with an
    invalid board size raises an error.

    Feature: SGF import
    Scenario: Importing SGF content with an invalid board size

    Given SGF content with an invalid board size
    When importing the SGF content
    Then an SgfError should be raised
    """
    with pytest.raises(SgfError):
        import_sgf("(;GM[41]SZ[8:8];B[a])")


def test_import_sgf_defaults_to_player_one_when_pl_missing() -> None:
    """Test that importing SGF without a `PL` property defaults to Player 1.

    Feature: SGF default starting player
    Scenario: Importing SGF content with no starting-player property

    Given SGF content without a `PL[...]` property
    When importing the SGF content
    Then the starting player defaults to Player 1
    """
    start_player, parsed_moves = import_sgf("(;GM[41]SZ[7:6];B[a];W[b])")

    assert start_player == 1
    assert parsed_moves == [0, 1]


def test_import_sgf_invalid_order() -> None:
    """Test that importing SGF content with an
    invalid move order raises an error.

    Feature: SGF import
    Scenario: Importing SGF content with an invalid move order

    Given SGF content with an invalid move order
    When importing the SGF content
    Then an SgfError should be raised
    """
    with pytest.raises(SgfError):
        import_sgf("(;GM[41]SZ[7:6]PL[B];W[a])")


def test_export_sgf_validation() -> None:
    """Test that exporting SGF content with invalid parameters raises an error.

    Feature: SGF export
    Scenario: Exporting SGF content with invalid parameters

    Given invalid SGF parameters
    When exporting the SGF content
    Then an SgfError should be raised
    """
    with pytest.raises(SgfError):
        export_sgf([0], starting_player=3)
    with pytest.raises(SgfError):
        export_sgf([99])


def test_import_sgf_out_of_range_column() -> None:
    """Test that importing SGF content with an
    out-of-range column raises an error.

    Feature: SGF import
    Scenario: Importing SGF content with an out-of-range column

    Given SGF content with an out-of-range column
    When importing the SGF content
    Then an SgfError should be raised
    """
    with pytest.raises(SgfError):
        import_sgf("(;GM[41]SZ[7:6]PL[B];B[z])")


def test_history_record_move_validation() -> None:
    """Test that record_move validates player and board coordinates.

    Feature: History validation
    Scenario: Recording invalid move metadata

    Given a history manager
    When invalid player, column, or row values are recorded
    Then a ValueError should be raised
    """
    history = HistoryManager()

    with pytest.raises(ValueError):
        history.record_move(player=0, column=0, row=0)
    with pytest.raises(ValueError):
        history.record_move(player=1, column=-1, row=0)
    with pytest.raises(ValueError):
        history.record_move(player=1, column=0, row=99)


def test_import_sgf_rejects_unplayable_move_sequence() -> None:
    """Test that SGF import rejects sequences that overfill a column.

    Feature: SGF import validation
    Scenario: Importing an unplayable but syntactically valid move list

    Given SGF content that repeatedly plays into a full column
    When importing the SGF content
    Then an SgfError should be raised
    """
    with pytest.raises(SgfError):
        import_sgf(
            "(;GM[41]SZ[7:6]PL[B]"
            ";B[a];W[b];B[a];W[b];B[a];W[b]"
            ";B[a];W[b];B[a];W[b];B[a];W[b];B[a])"
        )


def test_history_undo_mismatch_raises_runtime_error(monkeypatch) -> None:
    """Test that undo raises RuntimeError if game state doesn't match history.

    Feature: History mismatch detection
    Scenario: Undoing when game state column doesn't match recorded column

    Given a game state with a recorded move at column 2
    When the game state is corrupted to return a different column
    Then undo() should raise RuntimeError with mismatch message
    """
    state = GameState.new()
    history = HistoryManager()

    # Record and apply a move at column 2
    player = state.current_player
    row = state.apply_move(2)
    history.record_move(player, 2, row)

    # Mock undo_move to return a different column (simulating state corruption)
    monkeypatch.setattr(state, "undo_move", lambda: 1)

    with pytest.raises(
        RuntimeError, match="History mismatch: undone column differs"
    ):
        history.undo(state)


def test_history_replay_player_mismatch_raises_runtime_error(
    monkeypatch
) -> None:
    """Test that replay raises RuntimeError if current_player doesn't match.

    Feature: History mismatch detection
    Scenario: Replaying when game state current_player doesn't match record

    Given a game state with recorded and undone moves
    When the game state current_player is changed
    Then replay() should raise RuntimeError with player mismatch message
    """
    state = GameState.new()
    history = HistoryManager()

    # Record a move, undo it
    player = state.current_player
    row = state.apply_move(0)
    history.record_move(player, 0, row)
    history.undo(state)

    # Mock current_player to return a different player
    monkeypatch.setattr(
        type(state), "current_player", property(lambda self: 2)
    )

    with pytest.raises(
        RuntimeError, match="History mismatch: player to replay"
    ):
        history.replay(state)


def test_history_replay_row_mismatch_raises_runtime_error(monkeypatch) -> None:
    """Test that replay raises RuntimeError if replay row doesn't match record.

    Feature: History mismatch detection
    Scenario: Replaying when apply_move returns different row than recorded

    Given a game state with recorded and undone moves
    When apply_move is mocked to return a different row
    Then replay() should raise RuntimeError with row mismatch message
    """
    state = GameState.new()
    history = HistoryManager()

    # Record a move, undo it
    player = state.current_player
    row = state.apply_move(0)
    history.record_move(player, 0, row)
    history.undo(state)

    # Mock apply_move to return a different row
    monkeypatch.setattr(state, "apply_move", lambda col: 5)

    with pytest.raises(
        RuntimeError, match="History mismatch: replayed row differs"
    ):
        history.replay(state)
