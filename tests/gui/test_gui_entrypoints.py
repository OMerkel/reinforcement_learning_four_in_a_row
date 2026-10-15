# Requirements: FR-GUI-001
"""GUI tests for lifecycle orchestration and entrypoints."""
# pylint: disable=protected-access

import argparse
import runpy
import sys
from argparse import Namespace

import matplotlib.pyplot as plt
import pytest

import four_in_a_row.gui as gui_module
from four_in_a_row.constants import PLAYER_ONE, PLAYER_TWO
from four_in_a_row.gui import (
    ENGINE_NEGAMAX,
    ENGINE_RANDOM,
    PLAYER_KIND_HUMAN,
    PlayerConfig,
    run_gui,
)


def test_log_and_final_statistics_branches(
    monkeypatch: pytest.MonkeyPatch,
    make_app,
) -> None:
    """Test logging of match configuration and final statistics formatting.

    Feature: Logging and final statistics
    Scenario: Log match configuration and print final statistics
    with various outcomes

    Given a GUI app with default human players
    When the match configuration is logged
    Then the output should include the board background color
    """
    app, outputs = make_app()
    app._log_match_configuration()
    assert any("Board background: blue" in line for line in outputs)

    app.stats.ai_turns = 2
    app.stats.ai_total_seconds = 1.0
    app.stats.ai_max_seconds = 0.7
    monkeypatch.setattr(app.state, "winner", lambda: PLAYER_ONE)
    app._print_final_statistics()
    assert any("Result: Player 1 (red) wins" in line for line in outputs)
    assert any("AI time:" in line for line in outputs)

    outputs.clear()
    monkeypatch.setattr(app.state, "winner", lambda: PLAYER_TWO)
    app._print_final_statistics()
    assert any("Result: Player 2 (yellow) wins" in line for line in outputs)

    outputs.clear()
    monkeypatch.setattr(app.state, "winner", lambda: 0)
    monkeypatch.setattr(app.state, "is_draw", lambda: True)
    app._print_final_statistics()
    assert any("Result: draw" in line for line in outputs)

    outputs.clear()
    monkeypatch.setattr(app.state, "is_draw", lambda: False)
    app._print_final_statistics()
    assert any("Result: unfinished" in line for line in outputs)


def test_run_calls_lifecycle_methods(
    monkeypatch: pytest.MonkeyPatch,
    make_app,
) -> None:
    """Test that the run method calls the expected lifecycle methods in order.

    Feature: Run method lifecycle
    Scenario: Run method calls lifecycle methods in order

    Given a GUI app
    When the run method is called
    Then the expected lifecycle methods should be called in order
    """
    app, _outputs = make_app()
    called = {"show": 0, "ai": 0, "post": 0, "final": 0}
    monkeypatch.setattr(
        gui_module.plt,
        "show",
        lambda: called.__setitem__("show", 1),
    )
    monkeypatch.setattr(
        app,
        "_handle_ai_until_human_turn",
        lambda: called.__setitem__("ai", 1),
    )
    monkeypatch.setattr(
        app,
        "_handle_post_game_prompt_if_needed",
        lambda: called.__setitem__("post", 1),
    )
    monkeypatch.setattr(
        app,
        "_print_final_statistics",
        lambda: called.__setitem__("final", 1),
    )
    assert app.run() == 0
    assert called == {"show": 1, "ai": 1, "post": 1, "final": 1}
    plt.close(app.figure)


def test_run_engine_vs_engine_shows_board(
    monkeypatch: pytest.MonkeyPatch,
    make_app,
) -> None:
    """Test that the run method shows the board in engine vs engine mode.

    Feature: Engine vs engine mode
    Scenario: Run method shows the board in engine vs engine mode

    Given a GUI app with engine players
    When the run method is called
    Then the board should be shown
    """
    players = {
        PLAYER_ONE: PlayerConfig(kind=ENGINE_RANDOM),
        PLAYER_TWO: PlayerConfig(kind=ENGINE_NEGAMAX, strength=2),
    }
    app, outputs = make_app(players)

    show_calls: list[dict[str, object]] = []
    called = {"ai": 0, "final": 0}

    monkeypatch.setattr(
        gui_module.plt,
        "show",
        lambda *args, **kwargs: show_calls.append(
            {"args": args, "kwargs": kwargs}
        ),
    )
    monkeypatch.setattr(gui_module.plt, "pause", lambda _seconds: None)
    monkeypatch.setattr(
        app,
        "_handle_ai_until_human_turn",
        lambda: called.__setitem__("ai", called["ai"] + 1),
    )
    monkeypatch.setattr(
        app,
        "_print_final_statistics",
        lambda: called.__setitem__("final", called["final"] + 1),
    )

    assert app.run() == 0

    assert called["ai"] == 1
    assert called["final"] == 1
    assert len(show_calls) == 2
    assert show_calls[0]["kwargs"] == {"block": False}
    assert any("Engine vs engine mode" in line for line in outputs)
    plt.close(app.figure)


def test_engine_vs_engine_post_game_prompt_is_available(
    monkeypatch: pytest.MonkeyPatch,
    make_app,
) -> None:
    """Test that the post-game prompt is available in engine vs engine mode.

    Feature: Engine vs engine mode
    Scenario: Post-game prompt is available in engine vs engine mode

    Given a GUI app with engine players
    When the game finishes
    Then the post-game prompt should be available
    """
    players = {
        PLAYER_ONE: PlayerConfig(kind=ENGINE_RANDOM),
        PLAYER_TWO: PlayerConfig(kind=ENGINE_NEGAMAX, strength=2),
    }
    app, outputs = make_app(players, ["quit"])
    app._create_plot()
    try:
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
        assert any("Match finished. Enter 'help'" in line for line in outputs)
        assert any("Bye." in line for line in outputs)
    finally:
        plt.close(app.figure)


def test_run_gui_and_main_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the run_gui function and main entrypoint.

    Feature: GUI entrypoints
    Scenario: Run GUI and main entrypoint

    Given the run_gui function and main entrypoint
    When run_gui is called with player configurations
    Then it should resolve player configs and run the GUI
    When the main entrypoint is executed
    Then it should parse command-line arguments and call run_gui
    """
    calls: dict[str, object] = {}

    def fake_resolve(*args, **kwargs):
        _ = args, kwargs
        calls["resolve"] = True
        return {
            PLAYER_ONE: PlayerConfig(kind=PLAYER_KIND_HUMAN),
            PLAYER_TWO: PlayerConfig(kind=ENGINE_RANDOM),
        }

    class DummyApp:
        """A dummy FourInARowGui class for testing the run_gui function."""

        def __init__(self, *, players, input_fn, output_fn):
            calls["players"] = players
            _ = input_fn, output_fn

        def run(self) -> int:
            """A dummy run method that simulates running the GUI."""
            return 7

    monkeypatch.setattr(gui_module, "_resolve_player_configs", fake_resolve)
    monkeypatch.setattr(gui_module, "FourInARowGui", DummyApp)

    result = run_gui(p1="human", p2="random")
    assert result == 7
    assert calls["resolve"] is True

    captured_main: dict[str, object] = {}

    def fake_run_gui(**kwargs):
        captured_main.update(kwargs)
        return 0

    monkeypatch.setattr(gui_module, "run_gui", fake_run_gui)
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: Namespace(
            p1="human",
            p2="mcts",
            p1_strength=2,
            p2_strength=200,
        ),
    )
    gui_module.main()
    assert captured_main["p1"] == "human"
    assert captured_main["p2"] == "mcts"
    assert captured_main["p1_strength"] == 2
    assert captured_main["p2_strength"] == 200


def test_module_dunder_main_executes_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that executing the module as __main__ runs the main function.

    Feature: Module execution
    Scenario: Module __main__ execution

    Given the four_in_a_row.gui module
    When it is executed as __main__
    Then the main function should be called and the GUI should run
    """
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    monkeypatch.delitem(sys.modules, "four_in_a_row.gui", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["four_in_a_row.gui", "--p1", "human", "--p2", "human"],
    )
    runpy.run_module("four_in_a_row.gui", run_name="__main__")
