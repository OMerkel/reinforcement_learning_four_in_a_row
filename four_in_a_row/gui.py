"""Matplotlib GUI for Four in a Row game."""
from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.text import Text

from .cli import _judgement, _print_suggestion
from .constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    CONNECT_N,
    EMPTY,
    PLAYER_ONE,
    PLAYER_TWO,
)
from .engines import suggest_move
from .game import GameState, InvalidMoveError, opponent
from .history import HistoryManager
from .sgf import export_sgf

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]

PLAYER_KIND_HUMAN = "human"
ENGINE_RANDOM = "random"
ENGINE_MINIMAX = "minimax"
ENGINE_NEGAMAX = "negamax"
ENGINE_ALPHABETA = "alphabeta"
ENGINE_MCTS = "mcts"

PLAYER_CHOICES = [
    PLAYER_KIND_HUMAN,
    ENGINE_RANDOM,
    ENGINE_MINIMAX,
    ENGINE_NEGAMAX,
    ENGINE_ALPHABETA,
    ENGINE_MCTS,
]


@dataclass(frozen=True)
class PlayerConfig:
    """Configuration for one side (human or AI engine)."""

    kind: str
    strength: int | None = None


@dataclass
class MatchStats:
    """Track basic move and timing stats for stdout logging."""

    moves_p1: int = 0
    moves_p2: int = 0
    ai_turns: int = 0
    ai_total_seconds: float = 0.0
    ai_max_seconds: float = 0.0


def _normalize_kind(kind: str) -> str:
    normalized = kind.strip().lower()
    if normalized in {"alpha-beta", "alpha_beta"}:
        return ENGINE_ALPHABETA
    if normalized not in PLAYER_CHOICES:
        raise ValueError(f"Unknown player kind: {kind}")
    return normalized


def _default_strength(kind: str) -> int | None:
    if kind == ENGINE_MCTS:
        return 800
    if kind in {ENGINE_MINIMAX, ENGINE_NEGAMAX, ENGINE_ALPHABETA}:
        return 5
    return None


def _strength_label(kind: str) -> str:
    return "iterations" if kind == ENGINE_MCTS else "depth"


def _describe_config(config: PlayerConfig) -> str:
    if config.kind == PLAYER_KIND_HUMAN:
        return "human"
    if config.strength is None:
        return f"{config.kind} (default)"
    return f"{config.kind} ({_strength_label(config.kind)}={config.strength})"


def _ask_player_config(
        player: int,
        preselected_kind: str | None,
        preselected_strength: int | None,
        *,
        input_fn: InputFn,
        output_fn: OutputFn,
) -> PlayerConfig:
    kind = preselected_kind
    if kind is None:
        output_fn(
            "Choose player type for "
            f"Player {player} [{'/'.join(PLAYER_CHOICES)}] (default human):"
        )
        answer = input_fn(f"Player {player} type > ").strip()
        kind = PLAYER_KIND_HUMAN if not answer else answer
    kind = _normalize_kind(kind)

    if kind == PLAYER_KIND_HUMAN:
        return PlayerConfig(kind=kind)

    strength = preselected_strength
    default_strength = _default_strength(kind)
    if strength is None and default_strength is not None:
        output_fn(
            f"Choose {_strength_label(kind)} for Player {player} "
            f"{kind} (default {default_strength}):"
        )
        answer = input_fn(
            f"Player {player} {
                _strength_label(kind)} > ").strip()
        if answer:
            strength = int(answer)
        else:
            strength = default_strength

    if strength is not None and strength < 1:
        raise ValueError("strength must be >= 1")
    return PlayerConfig(kind=kind, strength=strength)


def _resolve_player_configs(
        p1: str | None,
        p2: str | None,
        p1_strength: int | None,
        p2_strength: int | None,
        *,
        input_fn: InputFn,
        output_fn: OutputFn,
) -> dict[int, PlayerConfig]:
    config_p1 = _ask_player_config(
        1,
        p1,
        p1_strength,
        input_fn=input_fn,
        output_fn=output_fn,
    )
    config_p2 = _ask_player_config(
        2,
        p2,
        p2_strength,
        input_fn=input_fn,
        output_fn=output_fn,
    )
    return {
        PLAYER_ONE: config_p1,
        PLAYER_TWO: config_p2,
    }


class FourInARowGui:
    """Interactive matplotlib GUI controller for a single match."""

    def __init__(
            self,
            *,
            players: dict[int, PlayerConfig],
            input_fn: InputFn,
            output_fn: OutputFn,
    ) -> None:
        self.players = players
        self.input_fn = input_fn
        self.output_fn = output_fn

        self.state = GameState.new()
        self.history = HistoryManager()
        self.stats = MatchStats()

        self.figure: Figure
        self.axis: Axes
        self.pieces: list[list[Circle]] = []
        self.status_text: Text | None = None
        self.finished = False
        self._finished_prompt_pending = False

    def _is_engine_vs_engine(self) -> bool:
        return all(
            config.kind != PLAYER_KIND_HUMAN
            for config in self.players.values()
        )

    def run(self) -> int:
        """Create the window and run the GUI event loop."""
        self._create_plot()
        self._log_match_configuration()
        if self._is_engine_vs_engine():
            self.axis.set_title("Four in a Row - Engine vs Engine (read-only)")
        self._render_board()

        self.figure.canvas.mpl_connect("button_press_event", self._on_click)

        if self._is_engine_vs_engine():
            self.output_fn(
                "Engine vs engine mode: board is read-only (mouse clicks do "
                "not control moves)."
            )
            plt.show(block=False)
            self._handle_ai_until_human_turn()
            plt.show()
        else:
            # If player 1 is an engine, start playing immediately.
            self._handle_ai_until_human_turn()
            self._handle_post_game_prompt_if_needed()
            plt.show()

        self._print_final_statistics()
        return 0

    def _create_plot(self) -> None:
        self.figure, self.axis = plt.subplots(figsize=(7, 6))
        manager = self.figure.canvas.manager
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("Four in a Row")
        self.axis.set_xlim(0, BOARD_WIDTH)
        self.axis.set_ylim(0, BOARD_HEIGHT)
        self.axis.set_aspect("equal")
        self.axis.set_facecolor("blue")
        self.axis.set_xticks([column + 0.5 for column in range(BOARD_WIDTH)])
        self.axis.set_xticklabels([str(column + 1)
                                  for column in range(BOARD_WIDTH)])
        self.axis.set_yticks([])
        self.axis.set_title("Four in a Row - Click a column to play")
        self.axis.tick_params(axis="x", colors="white")

        self.status_text = self.axis.text(
            0.02,
            1.02,
            "",
            transform=self.axis.transAxes,
            color="white",
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )

        for row in range(BOARD_HEIGHT):
            row_circles: list[Circle] = []
            for column in range(BOARD_WIDTH):
                piece = Circle(
                    (column + 0.5, row + 0.5),
                    radius=0.42,
                    facecolor="white",
                    edgecolor="navy",
                    linewidth=1.2,
                )
                self.axis.add_patch(piece)
                row_circles.append(piece)
            self.pieces.append(row_circles)

    def _current_player_name(self) -> str:
        if self.state.current_player == PLAYER_ONE:
            return "Player 1 (red)"
        return "Player 2 (yellow)"

    def _render_board(self) -> None:
        if self.status_text is None:
            raise RuntimeError("Status text is not initialized")

        was_finished = self.finished
        status_text = self.status_text
        colors = {
            EMPTY: "white",
            PLAYER_ONE: "red",
            PLAYER_TWO: "yellow",
        }
        winning_cells = self._winning_cells()
        for row in range(BOARD_HEIGHT):
            for column in range(BOARD_WIDTH):
                cell = self.state.board[row][column]
                piece = self.pieces[row][column]
                piece.set_facecolor(colors[cell])
                # Reset border style each frame, then emphasize winning line.
                piece.set_edgecolor("navy")
                piece.set_linewidth(1.2)
                if (row, column) in winning_cells:
                    piece.set_edgecolor("lime")
                    piece.set_linewidth(3.0)

        winner = self.state.winner()
        if winner == PLAYER_ONE:
            self.finished = True
            status_text.set_text("Player 1 (red) wins")
        elif winner == PLAYER_TWO:
            self.finished = True
            status_text.set_text("Player 2 (yellow) wins")
        elif self.state.is_draw():
            self.finished = True
            status_text.set_text("Draw")
        else:
            player_config = self.players[self.state.current_player]
            if player_config.kind == PLAYER_KIND_HUMAN:
                extra = "click a column"
            else:
                extra = f"{player_config.kind} thinking"
            status_text.set_text(f"{self._current_player_name()} - {extra}")

        self.figure.canvas.draw_idle()
        if self.finished and not was_finished:
            self._finished_prompt_pending = True
            if winner == PLAYER_ONE:
                self.output_fn("Result: Player 1 (red, X) wins")
            elif winner == PLAYER_TWO:
                self.output_fn("Result: Player 2 (yellow, O) wins")
            else:
                self.output_fn("Result: Draw")

    def _winning_cells(self) -> set[tuple[int, int]]:
        winner = self.state.winner()
        if winner == EMPTY:
            return set()

        directions = (
            (1, 0),   # vertical
            (0, 1),   # horizontal
            (1, 1),   # diagonal up-right
            (1, -1),  # diagonal up-left
        )
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                if self.state.board[row][col] != winner:
                    continue
                for dr, dc in directions:
                    cells: list[tuple[int, int]] = []
                    for offset in range(CONNECT_N):
                        nr = row + dr * offset
                        nc = col + dc * offset
                        if nr < 0 or nr >= BOARD_HEIGHT:
                            break
                        if nc < 0 or nc >= BOARD_WIDTH:
                            break
                        if self.state.board[nr][nc] != winner:
                            break
                        cells.append((nr, nc))
                    if len(cells) == CONNECT_N:
                        return set(cells)
        return set()

    def _apply_move(self, column: int) -> None:
        player = self.state.current_player
        row = self.state.apply_move(column)
        self.history.record_move(player=player, column=column, row=row)

        if player == PLAYER_ONE:
            self.stats.moves_p1 += 1
        else:
            self.stats.moves_p2 += 1

        symbol = "X" if player == PLAYER_ONE else "O"
        self.output_fn(
            f"Move {len(self.history.applied):02d}: Player {player} "
            f"({symbol}) -> column {column + 1}, row {row + 1}"
        )
        self.output_fn(self.state.render())

    def _play_ai_turn(self) -> None:
        player = self.state.current_player
        config = self.players[player]
        if config.kind == PLAYER_KIND_HUMAN:
            return

        started = time.perf_counter()
        if config.kind == ENGINE_MCTS:
            strength = config.strength
            if strength is None:
                strength = _default_strength(config.kind)
            if strength is None:
                raise RuntimeError("MCTS strength could not be resolved")
            suggestion = suggest_move(
                self.state,
                engine=config.kind,
                iterations=strength,
            )
        elif config.kind in {ENGINE_MINIMAX, ENGINE_NEGAMAX, ENGINE_ALPHABETA}:
            strength = config.strength
            if strength is None:
                strength = _default_strength(config.kind)
            if strength is None:
                raise RuntimeError(
                    f"Search depth could not be resolved for {config.kind}"
                )
            suggestion = suggest_move(
                self.state,
                engine=config.kind,
                depth=strength,
            )
        else:
            suggestion = suggest_move(self.state, engine=config.kind)

        elapsed = time.perf_counter() - started
        self.stats.ai_turns += 1
        self.stats.ai_total_seconds += elapsed
        self.stats.ai_max_seconds = max(self.stats.ai_max_seconds, elapsed)

        _print_suggestion(suggestion, self.output_fn)
        self.output_fn(
            f"AI {config.kind} chose column {suggestion.move + 1} "
            f"for Player {player} in {elapsed:.3f}s "
            f"({_judgement(suggestion.evaluation)})"
        )
        self._apply_move(suggestion.move)

    def _handle_ai_until_human_turn(self) -> None:
        while not self.finished:
            current = self.players[self.state.current_player]
            if current.kind == PLAYER_KIND_HUMAN:
                break
            self._play_ai_turn()
            self._render_board()
            self.figure.canvas.flush_events()
            plt.pause(0.001)
        self._handle_post_game_prompt_if_needed()

    def _on_click(self, event) -> None:
        if self.finished:
            self._handle_post_game_prompt_if_needed()
            return
        if event.inaxes != self.axis:
            return
        if event.xdata is None:
            return

        current = self.players[self.state.current_player]
        if current.kind != PLAYER_KIND_HUMAN:
            self.output_fn("Ignored click: waiting for engine move.")
            return

        column = int(event.xdata)
        if column < 0 or column >= BOARD_WIDTH:
            return

        try:
            self._apply_move(column)
        except InvalidMoveError as exc:
            self.output_fn(f"Invalid move: {exc}")
            return

        self._render_board()
        if not self.finished:
            self._handle_ai_until_human_turn()
            self._render_board()
        self._handle_post_game_prompt_if_needed()

    def _reset_game(self) -> None:
        self.state = GameState.new()
        self.history = HistoryManager()
        self.stats = MatchStats()
        self.finished = False
        self._finished_prompt_pending = False
        self.output_fn("Started a new game.")
        self._render_board()
        self._handle_ai_until_human_turn()

    def _handle_post_game_prompt_if_needed(self) -> None:
        if not self._finished_prompt_pending:
            return
        self._finished_prompt_pending = False
        while self.finished:
            self.output_fn(
                "Match finished. Enter 'help', 'save <file.sgf>', "
                "'new' or 'quit'."
            )
            raw_command = self.input_fn("GUI match finished > ").strip()
            command = raw_command.lower()
            if command == "new":
                self._reset_game()
                return
            if command in {"quit", "exit"}:
                self.output_fn("Bye.")
                plt.close(self.figure)
                return
            if command == "help":
                self.output_fn("Post-game commands:")
                self.output_fn("  help               Show this help")
                self.output_fn(
                    "  save <file.sgf>    Export current match to SGF"
                )
                self.output_fn("  new                Start a new game")
                self.output_fn("  quit               Close the GUI")
                continue
            if command.startswith("save"):
                parts = raw_command.split(maxsplit=1)
                if len(parts) < 2:
                    self.output_fn("Usage: save <file.sgf>")
                    continue
                moves_count = len(self.history.applied)
                if moves_count % 2 == 0:
                    starting_player = self.state.current_player
                else:
                    starting_player = opponent(self.state.current_player)

                content = export_sgf(
                    [record.column for record in self.history.applied],
                    starting_player=starting_player,
                )
                Path(parts[1]).write_text(content, encoding="utf-8")
                self.output_fn(f"Saved to {parts[1]}")
                continue
            if command == "":
                continue
            self.output_fn(
                "Unknown command. Use 'help', 'save <file.sgf>', "
                "'new' or 'quit'."
            )

    def _log_match_configuration(self) -> None:
        self.output_fn("Four in a Row GUI (matplotlib)")
        self.output_fn("Board background: blue")
        self.output_fn(
            f"Player 1 (red): {
                _describe_config(
                    self.players[PLAYER_ONE])}")
        self.output_fn(
            f"Player 2 (yellow): {_describe_config(self.players[PLAYER_TWO])}"
        )
        self.output_fn("Human moves: click a column on the board.")

    def _print_final_statistics(self) -> None:
        winner = self.state.winner()
        if winner == PLAYER_ONE:
            result = "Player 1 (red) wins"
        elif winner == PLAYER_TWO:
            result = "Player 2 (yellow) wins"
        elif self.state.is_draw():
            result = "draw"
        else:
            result = "unfinished"

        total_moves = len(self.history.applied)
        self.output_fn("Match finished.")
        self.output_fn(f"Result: {result}")
        self.output_fn(f"Total moves: {total_moves}")
        self.output_fn(f"Moves by Player 1: {self.stats.moves_p1}")
        self.output_fn(f"Moves by Player 2: {self.stats.moves_p2}")
        self.output_fn(f"AI turns: {self.stats.ai_turns}")
        if self.stats.ai_turns > 0:
            avg = self.stats.ai_total_seconds / self.stats.ai_turns
            self.output_fn(
                f"AI time: total={self.stats.ai_total_seconds:.3f}s "
                f"avg={avg:.3f}s max={self.stats.ai_max_seconds:.3f}s"
            )


def run_gui(
        *,
        p1: str | None = None,
        p2: str | None = None,
        p1_strength: int | None = None,
        p2_strength: int | None = None,
        input_fn: InputFn = input,
        output_fn: OutputFn = print,
) -> int:
    """Run the matplotlib GUI with CLI-like configuration fallback."""
    players = _resolve_player_configs(
        p1,
        p2,
        p1_strength,
        p2_strength,
        input_fn=input_fn,
        output_fn=output_fn,
    )
    app = FourInARowGui(
        players=players,
        input_fn=input_fn,
        output_fn=output_fn)
    return app.run()


def main() -> None:
    """Entry point for the Four in a Row GUI."""
    parser = argparse.ArgumentParser(
        description="Play Four in a Row with a matplotlib GUI"
    )
    parser.add_argument(
        "--p1",
        choices=PLAYER_CHOICES + ["alpha-beta", "alpha_beta"],
        help="Player 1 mode (red): human or engine",
    )
    parser.add_argument(
        "--p2",
        choices=PLAYER_CHOICES + ["alpha-beta", "alpha_beta"],
        help="Player 2 mode (yellow): human or engine",
    )
    parser.add_argument(
        "--p1-strength",
        type=int,
        help="Player 1 engine strength (depth or mcts iterations)",
    )
    parser.add_argument(
        "--p2-strength",
        type=int,
        help="Player 2 engine strength (depth or mcts iterations)",
    )
    args = parser.parse_args()

    run_gui(
        p1=args.p1,
        p2=args.p2,
        p1_strength=args.p1_strength,
        p2_strength=args.p2_strength,
    )


if __name__ == "__main__":
    main()
