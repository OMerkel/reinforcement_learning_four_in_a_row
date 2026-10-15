"""Command-line interface for Four in a Row game."""
from __future__ import annotations

import argparse
from pathlib import Path

from .constants import PLAYER_ONE, PLAYER_TWO
from .engines import EngineSuggestion, suggest_move
from .game import GameState, InvalidMoveError, opponent
from .history import HistoryManager
from .sgf import export_sgf, import_sgf

HELP_TEXT = """
Commands (with usage):
    1..7
        Usage: <column>
        Meaning: Drop a token into column 1..7.

    move <n>
        Usage: move <column>
        Parameters:
            <column>  Column number in range 1..7.
        Meaning: Same as entering a number directly.

    new
        Usage: new
        Meaning: Start a fresh game and clear undo/replay history.

    undo [n]
        Usage: undo [count]
        Parameters:
            [count]   Optional positive integer (default: 1).
        Meaning: Undo one or more already applied moves.

    replay [n]
        Usage: replay [count]
        Parameters:
            [count]   Optional positive integer (default: 1).
        Meaning: Replay one or more previously undone moves.

    suggest <engine> [value]
        Usage: suggest <engine> [value]
        Parameters:
            <engine>  random | minimax | negamax | alphabeta | alpha-beta |
                                alpha_beta | mcts (default: alphabeta).
            [value]   Positive integer.
                                - minimax/negamax/alphabeta: search depth
                                  (default: 5)
                                - mcts: iteration count (default: 800)
        Meaning: Print suggested move, principal variation (PV), and
                 evaluation.

    play <engine> [value]
        Usage: play <engine> [value]
        Parameters:
            <engine>  Same engine options as suggest.
            [value]   Same meaning/defaults as suggest.
        Meaning: Compute a suggestion and apply that move immediately.

    save <file.sgf>
        Usage: save <path/to/file.sgf>
        Parameters:
            <path>    Target SGF file path.
        Meaning: Export current move sequence to SGF.

    load <file.sgf>
        Usage: load <path/to/file.sgf>
        Parameters:
            <path>    Source SGF file path.
        Meaning: Import SGF moves and rebuild game state/history.

    show
        Usage: show
        Meaning: Re-render the board without changing game state.

    help
        Usage: help
        Meaning: Print this command reference.

    quit
        Usage: quit
        Meaning: Exit the game.
""".strip()


def _judgement(score: float) -> str:
    if score > 0.2:
        return "Likely advantage for Player 1"
    if score < -0.2:
        return "Likely advantage for Player 2"
    return "Position looks balanced"


def _rebuild_from_moves(
        starting_player: int, moves: list[int]
) -> tuple[GameState, HistoryManager]:
    state = GameState.new(starting_player=starting_player)
    history = HistoryManager()
    for column in moves:
        player = state.current_player
        row = state.apply_move(column)
        history.record_move(player, column, row)
    return state, history


def _print_suggestion(suggestion: EngineSuggestion, output_fn) -> None:
    pv_human = [str(column + 1) for column in suggestion.pv]
    output_fn(
        f"Engine={suggestion.engine} move={suggestion.move + 1} "
        f"PV={' '.join(pv_human)} "
        f"eval={suggestion.evaluation:+.3f} "
        f"({_judgement(suggestion.evaluation)})"
    )


def _parse_positive_int(parts: list[str], default_value: int) -> int:
    if len(parts) < 3:
        return default_value
    parsed = int(parts[2])
    if parsed < 1:
        raise ValueError("value must be >= 1")
    return parsed


def _parse_command_count(
        parts: list[str], default_value: int, *, arg_index: int, name: str
) -> int:
    if len(parts) <= arg_index:
        return default_value
    parsed = int(parts[arg_index])
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1")
    return parsed


def _short_moves_current(history: HistoryManager) -> str:
    return f"#{len(history.applied)}"


def _short_moves_progress(history: HistoryManager) -> str:
    current = len(history.applied)
    total = current + len(history.undone)
    return f"#{current}/{total}"


def _short_moves_prompt(history: HistoryManager) -> str:
    if history.can_replay():
        return _short_moves_progress(history)
    return _short_moves_current(history)


def run_cli(input_fn=input, output_fn=print) -> int:
    """Run the command-line interface for Four in a Row."""
    state = GameState.new()
    history = HistoryManager()

    output_fn("Four in a Row (7x6)")
    output_fn(
        "Both players are human. "
        "Use 'suggest <engine>' to get advisor help. "
        "Use 'play <engine>' to let engine play for you."
    )
    output_fn(
        "CLI columns are shown as 1-7; engine APIs use 0-based columns "
        "internally."
    )
    output_fn(HELP_TEXT)

    while True:
        output_fn("\n" + state.render())
        winner = state.winner()
        game_finished = False
        if winner == PLAYER_ONE:
            output_fn("Player 1 (X) wins.")
            game_finished = True
        if winner == PLAYER_TWO:
            output_fn("Player 2 (O) wins.")
            game_finished = True
        if state.is_draw():
            output_fn("Draw.")
            game_finished = True

        if game_finished:
            output_fn(
                "Match finished. Use 'new' to start another game or 'quit' "
                "to exit."
            )

        if game_finished:
            prompt = f"Match finished ({_short_moves_prompt(history)}) > "
        else:
            symbol = "X" if state.current_player == PLAYER_ONE else "O"
            prompt = (
                f"Player {state.current_player} ({symbol}) "
                f"({_short_moves_prompt(history)}) > "
            )
        command = input_fn(prompt).strip()
        if not command:
            continue

        parts = command.split()
        keyword = parts[0].lower()

        try:
            if keyword in {"quit", "exit"}:
                output_fn("Bye.")
                return 0

            if keyword == "help":
                output_fn(HELP_TEXT)
                continue

            if keyword == "new":
                state = GameState.new()
                history = HistoryManager()
                output_fn("Started a new game.")
                continue

            if keyword == "show":
                continue

            if keyword == "undo":
                count = _parse_command_count(
                    parts, 1, arg_index=1, name="undo count"
                )
                for _ in range(count):
                    history.undo(state)
                output_fn(f"Undo state: {_short_moves_progress(history)}")
                continue

            if keyword == "replay":
                count = _parse_command_count(
                    parts, 1, arg_index=1, name="replay count"
                )
                for _ in range(count):
                    history.replay(state)
                continue

            if game_finished:
                raise ValueError(
                    "Match is finished. Use 'undo', 'replay', 'new', "
                    "'show', 'save', 'load', 'help', or 'quit'."
                )

            if keyword.isdigit() or keyword == "move":
                move_text = keyword if keyword.isdigit() else parts[1]
                column = int(move_text) - 1
                player = state.current_player
                row = state.apply_move(column)
                history.record_move(player=player, column=column, row=row)
                continue

            if keyword in {"suggest", "play"}:
                engine = "alphabeta" if len(parts) < 2 else parts[1]
                if engine.lower() == "mcts":
                    iterations = _parse_positive_int(parts, 800)
                    suggestion = suggest_move(
                        state, engine=engine, iterations=iterations
                    )
                elif engine.lower() in {
                    "minimax",
                    "negamax",
                    "alphabeta",
                    "alpha-beta",
                    "alpha_beta"
                }:
                    depth = _parse_positive_int(parts, 5)
                    suggestion = suggest_move(
                        state, engine=engine, depth=depth
                    )
                else:
                    suggestion = suggest_move(state, engine=engine)

                if keyword == "suggest":
                    _print_suggestion(suggestion, output_fn)
                else:
                    _print_suggestion(suggestion, output_fn)
                    player = state.current_player
                    row = state.apply_move(suggestion.move)
                    history.record_move(
                        player=player, column=suggestion.move, row=row
                    )
                    output_fn(
                        f"Engine played column {suggestion.move + 1} "
                        f"({suggestion.engine})."
                    )
                continue

            if keyword == "save":
                if len(parts) < 2:
                    raise ValueError("Usage: save <file.sgf>")
                moves_count = len(history.applied)
                if moves_count % 2 == 0:
                    starting_player = state.current_player
                else:
                    starting_player = opponent(state.current_player)

                content = export_sgf(
                    [record.column for record in history.applied],
                    starting_player=starting_player,
                )
                Path(parts[1]).write_text(content, encoding="utf-8")
                output_fn(f"Saved to {parts[1]}")
                continue

            if keyword == "load":
                if len(parts) < 2:
                    raise ValueError("Usage: load <file.sgf>")
                content = Path(parts[1]).read_text(encoding="utf-8")
                starting_player, moves = import_sgf(content)
                state, history = _rebuild_from_moves(starting_player, moves)
                output_fn(f"Loaded {len(moves)} moves from {parts[1]}")
                continue

            raise ValueError("Unknown command. Type 'help'.")
        except (InvalidMoveError, ValueError, IndexError) as exc:
            output_fn(f"Error: {exc}")


def main() -> None:
    """Entry point for the Four in a Row CLI."""
    parser = argparse.ArgumentParser(
        description="Play Four in a Row in the terminal"
    )
    parser.parse_args()
    run_cli()


if __name__ == "__main__":
    main()
