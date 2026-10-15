"""SGF import/export functionality for four-in-a-row game records."""
from __future__ import annotations

import re

from .constants import BOARD_WIDTH, PLAYER_ONE, PLAYER_TWO
from .game import GameState, InvalidMoveError

MOVE_PATTERN = re.compile(r";([BW])\[([a-z])\]")
PLAYER_TO_SGF = {PLAYER_ONE: "B", PLAYER_TWO: "W"}
SGF_TO_PLAYER = {"B": PLAYER_ONE, "W": PLAYER_TWO}


class SgfError(ValueError):
    """Custom exception for SGF import/export errors."""


def _column_to_sgf(column: int) -> str:
    if column < 0 or column >= BOARD_WIDTH:
        raise SgfError(f"Column out of range for SGF export: {column}")
    return chr(ord("a") + column)


def _sgf_to_column(coord: str) -> int:
    column = ord(coord) - ord("a")
    if column < 0 or column >= BOARD_WIDTH:
        raise SgfError(f"Column out of range in SGF input: {coord}")
    return column


def export_sgf(
        move_columns: list[int], starting_player: int = PLAYER_ONE
) -> str:
    """Export a game record to SGF format.
    Args:
        move_columns: A list of column indices (0-6) representing the moves.
        starting_player: The player who made the first move (1 or 2).
    Returns:
        A string containing the SGF representation of the game.
    Raises:
        SgfError: If starting_player is not 1 or 2, or
        if any column is out of range.
    """
    if starting_player not in (PLAYER_ONE, PLAYER_TWO):
        raise SgfError("starting_player must be 1 or 2")

    node_moves: list[str] = []
    player = starting_player
    for column in move_columns:
        node_moves.append(
            f";{PLAYER_TO_SGF[player]}[{_column_to_sgf(column)}]"
        )
        player = PLAYER_TWO if player == PLAYER_ONE else PLAYER_ONE

    pl = PLAYER_TO_SGF[starting_player]
    root = f"(;GM[41]FF[4]CA[UTF-8]AP[four-in-a-row:1.0]SZ[7:6]PL[{pl}]"
    return root + "".join(node_moves) + ")\n"


def _validate_replayable_moves(
        starting_player: int, moves: list[int]
) -> None:
    state = GameState.new(starting_player=starting_player)
    for column in moves:
        try:
            state.apply_move(column)
        except InvalidMoveError as exc:
            raise SgfError(
                f"Illegal move sequence in SGF content: {exc}"
            ) from exc


def import_sgf(content: str) -> tuple[int, list[int]]:
    """Import a game record from SGF format.
    Args:
        content: A string containing the SGF representation of the game.
    Returns:
        A tuple containing the starting player (1 or 2) and a list of
        column indices (0-6) representing the moves.
    Raises:
        SgfError: If the SGF content is invalid or unsupported.
    """
    if "SZ[7:6]" not in content:
        raise SgfError("Unsupported SGF board size; expected SZ[7:6]")

    pl_match = re.search(r"PL\[([BW])\]", content)
    starting_player = (
        PLAYER_ONE if pl_match is None else SGF_TO_PLAYER[pl_match.group(1)]
    )

    moves: list[int] = []
    expected_player = starting_player
    for match in MOVE_PATTERN.finditer(content):
        player_char, coord = match.groups()
        player = SGF_TO_PLAYER[player_char]
        if player != expected_player:
            raise SgfError("Invalid move order in SGF content")
        moves.append(_sgf_to_column(coord))
        expected_player = PLAYER_TWO if expected_player == PLAYER_ONE \
            else PLAYER_ONE

    _validate_replayable_moves(starting_player, moves)
    return starting_player, moves
