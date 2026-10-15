"""History management for Four in a Row game."""
from __future__ import annotations

from dataclasses import dataclass, field

from .constants import BOARD_HEIGHT, BOARD_WIDTH, PLAYER_ONE, PLAYER_TWO
from .game import GameState, InvalidMoveError


@dataclass(frozen=True)
class MoveRecord:
    """Record of a single move in the game history."""
    player: int
    column: int
    row: int


@dataclass
class HistoryManager:
    """Manager for tracking the history of moves in a game,
    allowing undo and replay functionality."""
    applied: list[MoveRecord] = field(default_factory=list)
    undone: list[MoveRecord] = field(default_factory=list)

    def record_move(self, player: int, column: int, row: int) -> None:
        """Record a move in the history.
        This should be called after applying a move to the game state.

        Args:
            player: The player who made the move (1 or 2).
            column: The column where the move was made (0-6).
            row: The row where the piece landed (0-5).

        Raises:
            ValueError: If player is not 1 or 2, or
            if column/row are out of bounds.
        """
        if player not in (PLAYER_ONE, PLAYER_TWO):
            raise ValueError("player must be 1 or 2")
        if column < 0 or column >= BOARD_WIDTH:
            raise ValueError(
                f"column must be in range 0..{BOARD_WIDTH - 1}"
            )
        if row < 0 or row >= BOARD_HEIGHT:
            raise ValueError(
                f"row must be in range 0..{BOARD_HEIGHT - 1}"
            )
        self.applied.append(MoveRecord(player=player, column=column, row=row))
        self.undone.clear()

    def can_undo(self) -> bool:
        """Check if there are moves that can be undone.

        Returns: True if there are moves in the applied history,
                 False otherwise.
        """
        return bool(self.applied)

    def can_replay(self) -> bool:
        """Check if there are moves that can be replayed.

        Returns: True if there are moves in the undone history,
                 False otherwise.
        """
        return bool(self.undone)

    def undo(self, state: GameState) -> MoveRecord:
        """Undo the last move in the history and
        update the game state accordingly.

        Args:
            state: The current game state to update.
        Returns: The MoveRecord of the undone move.
        Raises:
            ValueError: If there are no moves to undo.
            RuntimeError: If the undone move does not match the game state.
        """
        if not self.applied:
            raise InvalidMoveError("No moves in history to undo")
        record = self.applied.pop()
        undone_column = state.undo_move()
        if undone_column != record.column:
            raise RuntimeError(
                "History mismatch: undone column differs from game state"
            )
        self.undone.append(record)
        return record

    def replay(self, state: GameState) -> MoveRecord:
        """Replay the last undone move in the history and
        update the game state accordingly.

        Args:
            state: The current game state to update.
        Returns: The MoveRecord of the replayed move.
        Raises:
            ValueError: If there are no moves to replay.
            RuntimeError: If the replayed move does not match the game state.
        """
        if not self.undone:
            raise InvalidMoveError("No moves in history to replay")
        record = self.undone.pop()
        if state.current_player != record.player:
            raise RuntimeError(
                "History mismatch: player to replay does not match game state"
            )
        row = state.apply_move(record.column)
        if row != record.row:
            raise RuntimeError(
                "History mismatch: replayed row differs from record"
            )
        self.applied.append(record)
        return record
