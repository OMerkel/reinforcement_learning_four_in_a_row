# Requirements: FR-GUI-001
"""GUI tests for post-game prompt command handling."""
# pylint: disable=protected-access

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import four_in_a_row.gui as gui_module


def test_reset_game_and_post_game_prompt_commands(
    monkeypatch: pytest.MonkeyPatch,
    make_app,
    input_from,
) -> None:
    """Test the handling of post-game prompt commands and game reset.

    Feature: Post-game prompt command handling
    Scenario: Handle post-game commands and reset game

    Given a GUI app that has finished a game
    When the post-game prompt is handled with various commands
    Then the appropriate actions should be taken for each command
    """
    app, outputs = make_app(inputs=["", "weird", "new"])
    app._create_plot()
    try:
        app.finished = True
        app._finished_prompt_pending = True
        ai_called = {"count": 0}
        monkeypatch.setattr(
            app,
            "_handle_ai_until_human_turn",
            lambda: ai_called.__setitem__("count", ai_called["count"] + 1),
        )
        app._handle_post_game_prompt_if_needed()
        assert app.finished is False
        assert ai_called["count"] == 1
        assert any("Unknown command" in line for line in outputs)
        assert any("Started a new game." in line for line in outputs)

        app.finished = True
        app._finished_prompt_pending = True
        app.input_fn = input_from(["exit"])
        closed = {"value": 0}
        monkeypatch.setattr(
            gui_module.plt,
            "close",
            lambda _fig: closed.__setitem__("value", closed["value"] + 1),
        )
        app._handle_post_game_prompt_if_needed()
        assert closed["value"] == 1
        assert any("Bye." in line for line in outputs)
    finally:
        plt.close(app.figure)


def test_post_game_prompt_supports_help_and_save(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_app,
) -> None:
    """Test that the post-game prompt supports help and save commands.

    Feature: Post-game prompt command handling
    Scenario: Support help and save commands

    Given a GUI app that has finished a game
    When the post-game prompt is handled with help and save commands
    Then the appropriate actions should be taken for each command
    """
    save_path = tmp_path / "gui_finished.sgf"
    app, outputs = make_app(inputs=["help", f"save {save_path}", "quit"])
    app._create_plot()
    try:
        app._apply_move(0)
        app._apply_move(1)
        app.finished = True
        app._finished_prompt_pending = True

        closed = {"value": 0}
        monkeypatch.setattr(
            gui_module.plt,
            "close",
            lambda _fig: closed.__setitem__("value", closed["value"] + 1),
        )

        app._handle_post_game_prompt_if_needed()

        assert closed["value"] == 1
        assert save_path.exists()
        sgf_content = save_path.read_text(encoding="utf-8")
        assert "SZ[7:6]" in sgf_content
        assert any("Post-game commands:" in line for line in outputs)
        assert any("Saved to" in line for line in outputs)
    finally:
        plt.close(app.figure)


def test_post_game_prompt_save_requires_path(
    monkeypatch: pytest.MonkeyPatch,
    make_app,
) -> None:
    """Test that the save command requires a file path.

    Feature: Post-game prompt command handling
    Scenario: Save command requires a file path

    Given a GUI app that has finished a game
    When the post-game prompt is handled with a save command without a path
    Then the appropriate usage message should be displayed
    """
    app, outputs = make_app(inputs=["save", "quit"])
    app._create_plot()
    try:
        app.finished = True
        app._finished_prompt_pending = True
        monkeypatch.setattr(gui_module.plt, "close", lambda _fig: None)

        app._handle_post_game_prompt_if_needed()

        assert any("Usage: save <file.sgf>" in line for line in outputs)
        assert any("Bye." in line for line in outputs)
    finally:
        plt.close(app.figure)


def test_post_game_prompt_save_odd_move_count_uses_opponent_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_app,
) -> None:
    """Test that the save command uses the opponent's start color
    for odd move counts.

    Feature: Post-game prompt command handling
    Scenario: Save command uses opponent's start color for odd move counts

    Given a GUI app that has finished a game with an odd number of moves
    When the post-game prompt is handled with a save command
    Then the saved SGF file should use the opponent's start color
    """
    save_path = tmp_path / "odd_moves.sgf"
    app, outputs = make_app(inputs=[f"save {save_path}", "quit"])
    app._create_plot()
    try:
        app._apply_move(0)
        app.finished = True
        app._finished_prompt_pending = True
        monkeypatch.setattr(gui_module.plt, "close", lambda _fig: None)

        app._handle_post_game_prompt_if_needed()

        assert save_path.exists()
        content = save_path.read_text(encoding="utf-8")
        assert "PL[B]" in content
        assert any("Saved to" in line for line in outputs)
    finally:
        plt.close(app.figure)


def test_handle_post_game_prompt_returns_when_not_pending(make_app) -> None:
    """Test that the post-game prompt handler returns immediately
    when not pending.

    Feature: Post-game prompt handling
    Scenario: Post-game prompt handler returns when not pending

    Given a GUI app
    When the post-game prompt handler is called and not pending
    Then it should return immediately without output
    """
    app, outputs = make_app()
    app._finished_prompt_pending = False
    app.finished = True
    app._handle_post_game_prompt_if_needed()
    assert not outputs


def test_handle_post_game_prompt_pending_but_not_finished(make_app) -> None:
    """Test that the post-game prompt handler returns immediately
    when pending but the game is not finished.

    Feature: Post-game prompt handling
    Scenario: Post-game prompt handler returns when pending but not finished

    Given a GUI app
    When the post-game prompt handler is called and pending but not finished
    Then it should return immediately without output
    """
    app, outputs = make_app()
    app._finished_prompt_pending = True
    app.finished = False
    app._handle_post_game_prompt_if_needed()
    assert app._finished_prompt_pending is False
    assert not outputs
