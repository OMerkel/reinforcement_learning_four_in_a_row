"""Game logic for Four in a Row (Connect Four)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    CONNECT_N,
    EMPTY,
    PLAYER_ONE,
    PLAYER_TWO,
)


class InvalidMoveError(ValueError):
    """Raised when an invalid move is attempted."""


def opponent(player: int) -> int:
    """Returns the opponent of the given player.

    Args:        player: The player (1 or 2).
    Returns:     The opponent player (2 or 1).
    Raises:      ValueError: If the input player is not 1 or 2.
    """
    if player == PLAYER_ONE:
        return PLAYER_TWO
    if player == PLAYER_TWO:
        return PLAYER_ONE
    raise ValueError(f"Invalid player: {player}")


@dataclass
class GameState:
    """Represents the state of a Four in a Row game."""
    board: list[list[int]] = field(default_factory=lambda: [
        [EMPTY] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)
    ])
    current_player: int = PLAYER_ONE
    move_stack: list[int] = field(default_factory=list)

    @classmethod
    def new(cls, starting_player: int = PLAYER_ONE) -> GameState:
        """Creates a new game state with an empty board and
        the specified starting player.
        Args:        starting_player: The player to start the game (1 or 2).
        Returns:     A new GameState instance.
        Raises:      ValueError: If starting_player is not 1 or 2.
        """
        if starting_player not in (PLAYER_ONE, PLAYER_TWO):
            raise ValueError("starting_player must be 1 or 2")
        return cls(current_player=starting_player)

    def clone(self) -> GameState:
        """Creates a deep copy of the current game state.
        Returns:     A new GameState instance with the same board,
                     current player, and move stack.
        """
        return GameState(
            board=[row.copy() for row in self.board],
            current_player=self.current_player,
            move_stack=self.move_stack.copy(),
        )

    def legal_moves(self) -> list[int]:
        """Returns list of legal moves (column indices) for current state.
        A move is legal if the top cell of the column is empty.

        Returns:     A list of column indices where a move can be made.
        """
        return [column for column in range(BOARD_WIDTH)
                if self.board[BOARD_HEIGHT - 1][column] == EMPTY]

    def apply_move(self, column: int) -> int:
        """Applies a move for the current player in the specified column.
        Args:        column: The column index where the move is to be made.
        Returns:     The row index where the piece was placed.
        Raises:      InvalidMoveError: If the move is invalid (column
                     out of range or full).
        """
        if column < 0 or column >= BOARD_WIDTH:
            raise InvalidMoveError(
                f"Column must be in range 0..{BOARD_WIDTH - 1}"
            )
        for row in range(BOARD_HEIGHT):
            if self.board[row][column] == EMPTY:
                self.board[row][column] = self.current_player
                self.move_stack.append(column)
                self.current_player = opponent(self.current_player)
                return row
        raise InvalidMoveError("Column is full")

    def undo_move(self) -> int:
        """Undoes the last move made in the game.
        Returns:     The column index of the undone move.
        Raises:      InvalidMoveError: If there are no moves to undo.
        """
        if not self.move_stack:
            raise InvalidMoveError("No moves to undo")
        column = self.move_stack.pop()
        for row in range(BOARD_HEIGHT - 1, -1, -1):
            if self.board[row][column] != EMPTY:
                self.board[row][column] = EMPTY
                self.current_player = opponent(self.current_player)
                return column
        raise RuntimeError(  # pragma: no cover
            "Invalid state: move stack does not match board"
        )

    def winner(self) -> int:
        """Determines the winner of the game, if any.
        Returns:     The player number (1 or 2) if there is a winner, or
                     0 if there is no winner.
        """
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                player = self.board[row][col]
                if player == EMPTY:
                    continue
                if self._has_connect_n_from(row, col, 1, 0, player):
                    return player
                if self._has_connect_n_from(row, col, 0, 1, player):
                    return player
                if self._has_connect_n_from(row, col, 1, 1, player):
                    return player
                if self._has_connect_n_from(row, col, 1, -1, player):
                    return player
        return EMPTY

    def is_draw(self) -> bool:
        """Checks if the game is a draw (no winner and no legal moves).
        Returns:     True if the game is a draw, False otherwise.
        """
        return self.winner() == EMPTY and not self.legal_moves()

    def is_terminal(self) -> bool:
        """Checks if the game has reached a terminal state (win or draw).
        Returns:     True if the game is over, False otherwise.
        """
        return self.winner() != EMPTY or self.is_draw()

    def render(self) -> str:
        """Returns a string representation of the game board for display.

        Returns:     A multi-line string representing the game board, with
                     rows from top to bottom and columns labeled 1-7.
        """
        markers = {EMPTY: ".", PLAYER_ONE: "X", PLAYER_TWO: "O"}
        lines: list[str] = []
        for row in range(BOARD_HEIGHT - 1, -1, -1):
            lines.append(" ".join(markers[self.board[row][column]]
                                  for column in range(BOARD_WIDTH)))
        lines.append("1 2 3 4 5 6 7")
        return "\n".join(lines)

    def _has_connect_n_from(
            self, row: int, col: int, dr: int, dc: int, player: int
    ) -> bool:
        """Checks if there are CONNECT_N pieces of the same player starting
        from (row, col) in the direction specified by (dr, dc).

        Args:        row: The starting row index.
                     col: The starting column index.
                     dr: The row direction (delta row).
                     dc: The column direction (delta column).
                     player: The player number to check for (1 or 2).
        Returns:     True if there are CONNECT_N pieces of the player in a row
                     in the specified direction, False otherwise.
        """
        for offset in range(CONNECT_N):
            nr = row + dr * offset
            nc = col + dc * offset
            if nr < 0 or nr >= BOARD_HEIGHT or nc < 0 or nc >= BOARD_WIDTH:
                return False
            if self.board[nr][nc] != player:
                return False
        return True


def terminal_score(state: GameState) -> float:
    """Evaluates the terminal score of the game state.

    Args:        state: The GameState to evaluate.
    Returns:     1.0 if PLAYER_ONE wins,
                 -1.0 if PLAYER_TWO wins,
                 or 0.0 if there is no winner.
    """
    winner = state.winner()
    if winner == PLAYER_ONE:
        return 1.0
    if winner == PLAYER_TWO:
        return -1.0
    return 0.0


def evaluate_state(state: GameState) -> float:
    """Evaluates the given game state from the perspective of PLAYER_ONE.
    The score is positive if the state is favorable for PLAYER_ONE, negative
    if it is favorable for PLAYER_TWO, and zero if it is neutral.

    Args:        state: The GameState to evaluate.
    Returns:     A float score representing the favorability of the state for
                 PLAYER_ONE.
    """
    outcome = terminal_score(state)
    if outcome != 0.0 or state.is_draw():
        return outcome

    score = 0.0

    center_column = BOARD_WIDTH // 2
    center_count_p1 = sum(1 for row in range(BOARD_HEIGHT)
                          if state.board[row][center_column] == PLAYER_ONE)
    center_count_p2 = sum(1 for row in range(BOARD_HEIGHT)
                          if state.board[row][center_column] == PLAYER_TWO)
    score += 0.03 * (center_count_p1 - center_count_p2)

    def window_value(window: list[int]) -> float:
        """Evaluates a 4-cell window for potential winning combinations.

        Args:        window: A list of 4 integers representing
                     a segment of the board.
        Returns:     A float score based on the contents of the window:
                     +0.12 if PLAYER_ONE has 3 pieces and 1 empty cell,
                     +0.03 if PLAYER_ONE has 2 pieces and 2 empty cells,
                     -0.12 if PLAYER_TWO has 3 pieces and 1 empty cell,
                     -0.03 if PLAYER_TWO has 2 pieces and 2 empty cells,
                     or 0.0 otherwise.
        """
        p1 = window.count(PLAYER_ONE)
        p2 = window.count(PLAYER_TWO)
        empty = window.count(EMPTY)
        if p1 and p2:
            return 0.0
        if p1 == 3 and empty == 1:
            return 0.12
        if p1 == 2 and empty == 2:
            return 0.03
        if p2 == 3 and empty == 1:
            return -0.12
        if p2 == 2 and empty == 2:
            return -0.03
        return 0.0

    def iter_windows() -> list[list[int]]:
        """Generates all possible 4-cell windows on the board.
        Returns:     A list of lists, where each inner list is a 4-cell window
                     from the board (horizontal, vertical, or diagonal).
        """
        windows: list[list[int]] = []
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH - 3):
                windows.append([state.board[row][col + i] for i in range(4)])
        for col in range(BOARD_WIDTH):
            for row in range(BOARD_HEIGHT - 3):
                windows.append([state.board[row + i][col] for i in range(4)])
        for row in range(BOARD_HEIGHT - 3):
            for col in range(BOARD_WIDTH - 3):
                windows.append([state.board[row + i][col + i]
                                for i in range(4)])
        for row in range(3, BOARD_HEIGHT):
            for col in range(BOARD_WIDTH - 3):
                windows.append([state.board[row - i][col + i]
                                for i in range(4)])
        return windows

    for window in iter_windows():
        score += window_value(window)

    return max(-1.0, min(1.0, score))
