# Requirements: FR-CLI-001
"""Tests for the command-line interface (CLI) of the Four in a Row game.
These tests simulate user interactions with the CLI by
providing predefined command sequences and capturing the outputs.
The tests cover various scenarios, including:
- Quitting immediately
- Requesting suggestions from different engines
- Playing against an engine and undoing moves
- Saving and loading game states
- Handling invalid commands and arguments
Each test verifies that the CLI responds correctly to the given commands
and that the outputs contain the expected information.
"""
import argparse
import runpy
import sys
from pathlib import Path

import pytest

import four_in_a_row.cli as cli_module
from four_in_a_row.cli import run_cli
from four_in_a_row.sgf import export_sgf

pytestmark = pytest.mark.regression


def _run_with_commands(commands: list[str]) -> list[str]:
    outputs: list[str] = []
    iterator = iter(commands)

    def fake_input(_prompt: str) -> str:
        return next(iterator)

    run_cli(input_fn=fake_input, output_fn=outputs.append)
    return outputs


def _run_with_commands_and_prompts(
    commands: list[str],
) -> tuple[list[str], list[str]]:
    outputs: list[str] = []
    prompts: list[str] = []
    iterator = iter(commands)

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return next(iterator)

    run_cli(input_fn=fake_input, output_fn=outputs.append)
    return outputs, prompts


def test_cli_quit_immediately() -> None:
    """Test that the CLI quits immediately when the 'quit' command is issued.

    Feature: Quit
    Scenario: Quitting the CLI immediately

    Given the CLI is running
    When the 'quit' command is issued
    Then the CLI should display a goodbye message and exit
    """
    outputs = _run_with_commands(["quit"])
    assert any("Four in a Row" in line for line in outputs)
    assert any("Bye." in line for line in outputs)


def test_cli_supports_suggest_and_move() -> None:
    """Test that the CLI supports the 'suggest' command and applying a move.

    Feature: Suggest and move
    Scenario: Requesting a suggestion and applying the suggested move

    Given the CLI is running
    When the 'suggest random' command is issued
    Then a suggestion from the random engine is displayed
    When the suggested move is applied
    Then the move is applied correctly
    """
    outputs = _run_with_commands(["suggest random", "4", "quit"])
    assert any("Engine=random" in line for line in outputs)


def test_cli_supports_suggest_negamax_with_depth() -> None:
    """Test that the CLI supports the 'suggest negamax' command with
    a specified depth.

    Feature: Suggest negamax with depth
    Scenario: Requesting a negamax suggestion with a specific depth

    Given the CLI is running
    When the 'suggest negamax 3' command is issued
    Then a suggestion from the negamax engine with depth 3 is displayed
    """
    outputs = _run_with_commands(["suggest negamax 3", "quit"])
    assert any("Engine=negamax" in line for line in outputs)


def test_cli_supports_alpha_beta_aliases() -> None:
    """Test that the CLI accepts alpha-beta engine aliases.

    Feature: Alpha-beta aliases
    Scenario: Requesting suggestions and plays with alias engine names

    Given the CLI is running
    When using `alpha-beta` and `alpha_beta`
    Then the alpha-beta engine is invoked successfully
    """
    outputs = _run_with_commands([
        "suggest alpha-beta 3",
        "play alpha_beta 3",
        "quit",
    ])
    assert any("Engine=alphabeta" in line for line in outputs)
    assert any("Engine played column" in line for line in outputs)


def test_cli_supports_mcts_default_iterations() -> None:
    """Test that the CLI uses the default MCTS iteration count.

    Feature: MCTS default iterations
    Scenario: Requesting an MCTS suggestion without an explicit iteration count

    Given the CLI is running
    When the `suggest mcts` command is issued
    Then a suggestion from the MCTS engine is displayed
    """
    outputs = _run_with_commands(["suggest mcts", "quit"])
    assert any("Engine=mcts" in line for line in outputs)


def test_cli_rejects_suggest_negamax_zero_depth() -> None:
    """Test that the CLI rejects the 'suggest negamax' command with zero depth.

    Feature: Suggest negamax with invalid depth
    Scenario: Requesting a negamax suggestion with zero depth

    Given the CLI is running
    When the 'suggest negamax 0' command is issued
    Then an error message is displayed
    """
    outputs = _run_with_commands(["suggest negamax 0", "quit"])
    assert any("value must be >= 1" in line for line in outputs)


def test_cli_play_applies_engine_move() -> None:
    """Test that the CLI applies a move suggested by an engine when the
    'play' command is used.

    Feature: Play engine move
    Scenario: Playing a move suggested by an engine

    Given the CLI is running
    When the 'play random' command is issued
    Then a move from the random engine is applied
    """
    outputs = _run_with_commands(["play random", "undo", "quit"])
    assert any("Engine played column" in line for line in outputs)
    assert not any("Error: No moves to undo" in line for line in outputs)


def test_cli_rejects_play_negamax_zero_depth() -> None:
    """Test that the CLI rejects the 'play negamax' command with zero depth.

    Feature: Play negamax with invalid depth
    Scenario: Playing a move suggested by negamax with zero depth

    Given the CLI is running
    When the 'play negamax 0' command is issued
    Then an error message is displayed
    """
    outputs = _run_with_commands(["play negamax 0", "quit"])
    assert any("value must be >= 1" in line for line in outputs)


def test_cli_save_and_load(tmp_path: Path) -> None:
    """Test that the CLI supports saving and loading games.

    Feature: Save and load
    Scenario: Saving and loading a game

    Given the CLI is running
    When the 'save' command is issued with a valid file path
    Then the game state is saved to that file
    When the 'load' command is issued with the same file path
    Then the game state is loaded from that file
    """
    save_path = tmp_path / "sample.sgf"
    outputs = _run_with_commands([
        f"save {save_path}", f"load {save_path}", "quit"
    ])
    assert save_path.exists()
    assert any("Saved to" in line for line in outputs)
    assert any("Loaded" in line for line in outputs)


def test_cli_load_prebuilt_sgf(tmp_path: Path) -> None:
    """Test that the CLI can load a prebuilt SGF file.

    Feature: Load prebuilt SGF
    Scenario: Loading a prebuilt SGF file

    Given the CLI is running
    When the 'load' command is issued with a valid SGF file path
    Then the game state is loaded from that file
    """
    save_path = tmp_path / "prebuilt.sgf"
    save_path.write_text(export_sgf([0, 1, 0]), encoding="utf-8")
    outputs = _run_with_commands([
        f"load {save_path}", "undo", "replay", "quit"
    ])
    assert any("Loaded 3 moves" in line for line in outputs)


def test_cli_save_preserves_loaded_starting_player(tmp_path: Path) -> None:
    """Test that the CLI preserves the starting player when
    saving a loaded game.

    Feature: Save preserves loaded starting player
    Scenario: Saving a loaded game preserves the starting player

    Given the CLI is running
    When the 'load' command is issued with a valid SGF file path
    And the 'save' command is issued with a valid file path
    Then the starting player is preserved in the saved file
    """
    source_path = tmp_path / "from_p2.sgf"
    target_path = tmp_path / "saved.sgf"
    source_path.write_text(
        export_sgf([0, 1, 0], starting_player=2),
        encoding="utf-8"
    )

    outputs = _run_with_commands([
        f"load {source_path}", f"save {target_path}", "quit"
    ])

    assert any("Loaded 3 moves" in line for line in outputs)
    assert any("Saved to" in line for line in outputs)
    saved = target_path.read_text(encoding="utf-8")
    assert "PL[W]" in saved


def test_cli_help_show_and_invalid_commands() -> None:
    """Test that the CLI displays help, shows the board, and
    handles invalid commands.

    Feature: Help, show, and invalid commands
    Scenario: Using help, show, and invalid commands

    Given the CLI is running
    When the 'help' command is issued
    Then the list of commands is displayed
    When the 'show' command is issued
    Then the current board state is displayed
    When an invalid command is issued
    Then an error message is displayed
    """
    outputs = _run_with_commands(["help", "show", "bogus", "quit"])
    assert any("Commands (with usage):" in line for line in outputs)
    assert any("alpha-beta" in line for line in outputs)
    assert any("new" in line for line in outputs)
    assert any("Usage: move <column>" in line for line in outputs)
    assert any("Parameters:" in line for line in outputs)
    assert any("Meaning:" in line for line in outputs)
    assert any("Unknown command" in line for line in outputs)


def test_cli_argument_errors() -> None:
    """Test that the CLI displays error messages for
    commands with missing or invalid arguments.

    Feature: Argument errors
    Scenario: Using commands with missing or invalid arguments

    Given the CLI is running
    When commands with missing or invalid arguments are issued
    Then error messages are displayed
    """
    outputs = _run_with_commands([
        "move", "save", "load", "undo", "replay", "suggest mcts 0", "quit"
    ])
    assert sum(1 for line in outputs if line.startswith("Error:")) >= 5


def test_cli_rejects_non_positive_undo_and_replay_counts() -> None:
    """Test that undo/replay reject non-positive counts.

    Feature: Undo/replay validation
    Scenario: Using invalid non-positive counts

    Given the CLI is running
    When `undo 0` or `replay 0` is issued
    Then an error message is displayed instead of silently doing nothing
    """
    outputs = _run_with_commands(["undo 0", "replay 0", "quit"])
    assert any("undo count must be >= 1" in line for line in outputs)
    assert any("replay count must be >= 1" in line for line in outputs)


def test_cli_allows_new_after_match_finished() -> None:
    """Test that the CLI can start a new game after a finished match.

    Feature: Post-finish restart
    Scenario: Winning sequence followed by `new`

    Given a running CLI game
    When Player 1 wins and then `new` is entered
    Then a new game starts instead of exiting immediately
    """
    outputs = _run_with_commands([
        "1", "2", "1", "2", "1", "2", "1", "new", "quit"
    ])
    assert any("Player 1 (X) wins." in line for line in outputs)
    assert any("Match finished. Use 'new'" in line for line in outputs)
    assert any("Started a new game." in line for line in outputs)


def test_cli_rejects_move_command_when_match_finished() -> None:
    """Test that move commands are rejected after a finished match.

    Feature: Post-finish command validation
    Scenario: Issuing move while finished

    Given a finished game
    When a move command is entered
    Then the CLI asks for `new` or `quit` style commands
    """
    outputs = _run_with_commands([
        "1", "2", "1", "2", "1", "2", "1", "4", "quit"
    ])
    assert any("Player 1 (X) wins." in line for line in outputs)
    assert any("Match is finished. Use 'undo'" in line for line in outputs)


def test_cli_allows_undo_count_after_match_finished() -> None:
    """Test that undo count works after a finished match and play resumes.

    Feature: Post-finish undo
    Scenario: Win, undo one move, continue playing

    Given a finished game
    When `undo 1` is issued
    Then the command succeeds and a subsequent move is accepted
    """
    outputs = _run_with_commands([
        "1", "2", "1", "2", "1", "2", "1", "undo 1", "4", "quit"
    ])
    assert any("Player 1 (X) wins." in line for line in outputs)
    assert not any("Error: Match is finished" in line for line in outputs)
    assert not any("Error:" in line for line in outputs)


def test_cli_allows_replay_command_after_match_finished() -> None:
    """Test that replay command is accepted after a finished match.

    Feature: Post-finish replay acceptance
    Scenario: Win, then issue replay with no undone moves

    Given a finished game
    When `replay` is issued
    Then replay is handled by replay logic (not finished-state gate)
    """
    outputs = _run_with_commands([
        "1", "2", "1", "2", "1", "2", "1", "replay", "quit"
    ])
    assert any("Player 1 (X) wins." in line for line in outputs)
    assert any("No moves in history to replay" in line for line in outputs)
    assert not any("Error: Match is finished" in line for line in outputs)


def test_cli_player_two_can_win() -> None:
    """Test that Player 2 winner branch is reported.

    Feature: Player 2 victory
    Scenario: Player 2 wins the game

    Given a running CLI game
    When Player 2 wins
    Then the CLI reports Player 2 as the winner
    """
    outputs = _run_with_commands([
        "2", "1", "2", "1", "3", "1", "3", "1", "quit"
    ])
    assert any("Player 2 (O) wins." in line for line in outputs)


def test_cli_handles_blank_command_and_positive_undo_replay_counts() -> None:
    """Test blank commands plus explicit positive undo/replay counts.

    Feature: Blank commands and positive undo/replay counts
    Scenario: Handling blank commands and positive undo/replay counts

    Given the CLI is running
    When a blank command is entered
    And `undo` and `replay` commands with positive counts are issued
    Then the CLI processes the commands correctly without errors
    """
    outputs = _run_with_commands([
        "",
        "1",
        "2",
        "undo 2",
        "replay 2",
        "quit",
    ])
    assert any("Bye." in line for line in outputs)
    assert not any("Error:" in line for line in outputs)


def test_cli_prompt_includes_short_current_move_count() -> None:
    """Test prompt includes compact current move count notation.

    Feature: Prompt move counter
    Scenario: Prompt reflects current played move count

    Given a running CLI game
    When a move is played and then undone
    Then prompt text includes compact move counts for current state
    """
    _outputs, prompts = _run_with_commands_and_prompts([
        "1",
        "undo",
        "quit",
    ])
    assert any("(#0)" in prompt for prompt in prompts)
    assert any("(#1)" in prompt for prompt in prompts)
    assert any("(#0/1)" in prompt for prompt in prompts)


def test_cli_undo_prints_short_current_and_total_move_count() -> None:
    """Test undo reports compact current/total move progress notation.

    Feature: Undo move progress display
    Scenario: Showing current and total move counts after undo

    Given moves have been played
    When undo is issued
    Then CLI prints current and total move counts in compact format
    """
    outputs = _run_with_commands([
        "1",
        "2",
        "undo",
        "quit",
    ])
    assert any("Undo state: #1/2" in line for line in outputs)


def test_cli_suggestion_can_report_player_two_advantage() -> None:
    """Test suggestion output can include negative-eval judgement text.

    Feature: Suggestion output
    Scenario: Reporting player two advantage

    Given the CLI is running
    When a suggestion command is issued
    Then the output can include negative-eval judgement text
    indicating likely advantage for Player 2
    """
    original = cli_module.suggest_move

    def _fake_suggest_move(*_args, **_kwargs):
        return cli_module.EngineSuggestion(
            engine="random",
            move=0,
            pv=[0],
            evaluation=-0.5,
        )

    cli_module.suggest_move = _fake_suggest_move
    outputs = _run_with_commands(["suggest random", "quit"])
    cli_module.suggest_move = original
    assert any("Likely advantage for Player 2" in line for line in outputs)


def test_cli_draw_branch_via_stubbed_state(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test draw handling branch by stubbing initial game state.

    Feature: Draw handling
    Scenario: Handling a draw state

    Given the CLI is running with a stubbed game state that is a draw
    When the CLI renders the board and checks for game status
    Then the CLI reports a draw
    """

    class _DrawState:
        current_player = 1

        def render(self) -> str:
            """Simulate rendering a draw board state."""
            return "draw board"

        def winner(self) -> int:
            """Simulate a game state with no winner."""
            return 0

        def is_draw(self) -> bool:
            """Simulate a draw state."""
            return True

        def apply_move(self, _column: int) -> int:
            """This should not be called since the game is already a draw."""
            raise AssertionError(  # pragma: no cover
                "apply_move should not be called"
            )

    monkeypatch.setattr(
        cli_module.GameState,
        "new",
        staticmethod(_DrawState),
    )

    outputs = _run_with_commands(["quit"])
    assert any("Draw." in line for line in outputs)
    assert any("Match finished. Use 'new'" in line for line in outputs)


def test_cli_main_and_dunder_main_entrypoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test cli main function and __main__ guard execution path."""
    called = {"run": 0}

    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(),
    )
    monkeypatch.setattr(
        cli_module,
        "run_cli",
        lambda *args, **kwargs: called.__setitem__("run", called["run"] + 1),
    )

    cli_module.main()
    assert called["run"] == 1

    monkeypatch.setattr("builtins.input", lambda _prompt: "quit")
    monkeypatch.delitem(sys.modules, "four_in_a_row.cli", raising=False)
    monkeypatch.setattr(sys, "argv", ["four_in_a_row.cli"])
    runpy.run_module("four_in_a_row.cli", run_name="__main__")
    assert called["run"] == 1
