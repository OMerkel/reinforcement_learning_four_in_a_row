# Requirements: FR-GUI-001
"""GUI tests for move application and AI turn flow."""
# pylint: disable=protected-access

import matplotlib.pyplot as plt
import pytest

import four_in_a_row.gui as gui_module
from four_in_a_row.constants import PLAYER_ONE, PLAYER_TWO
from four_in_a_row.gui import (
    ENGINE_MCTS,
    ENGINE_NEGAMAX,
    ENGINE_RANDOM,
    PLAYER_KIND_HUMAN,
    FourInARowGui,
    PlayerConfig,
)
from four_in_a_row.suggestion import EngineSuggestion


def test_apply_move_updates_stats_for_both_players(make_app) -> None:
    """Test that applying moves updates the statistics for both players.

    Feature: Move application and statistics
    Scenario: Applying moves updates player statistics

    Given a GUI app
    When moves are applied for both players
    Then the statistics for both players should be updated accordingly
    """
    app, outputs = make_app()
    app._apply_move(0)
    app._apply_move(1)
    assert app.stats.moves_p1 == 1
    assert app.stats.moves_p2 == 1
    assert len(app.history.applied) == 2
    assert any("Move 01" in line for line in outputs)


def test_play_ai_turn_variants_and_errors(
    monkeypatch: pytest.MonkeyPatch,
    input_from,
) -> None:
    """Test the _play_ai_turn function with various AI configurations
    and error scenarios.

    Feature: AI turn handling
    Scenario: Play AI turn with different configurations and handle errors

    Given a GUI app with AI players
    When _play_ai_turn is called with different AI configurations
    Then the correct moves should be suggested or errors should be raised
    """
    outputs: list[str] = []
    players = {
        PLAYER_ONE: PlayerConfig(kind=ENGINE_MCTS),
        PLAYER_TWO: PlayerConfig(kind=PLAYER_KIND_HUMAN),
    }
    app = FourInARowGui(
        players=players,
        input_fn=input_from([]),
        output_fn=outputs.append,
    )

    calls: list[tuple[str, int, int]] = []

    def fake_suggest_move(state, engine, *, depth=5, iterations=800, rng=None):
        _ = state, rng
        calls.append((engine, depth, iterations))
        return EngineSuggestion(engine=engine, move=3, pv=[3], evaluation=0.1)

    monkeypatch.setattr(gui_module, "suggest_move", fake_suggest_move)
    app._play_ai_turn()
    assert calls[-1] == (ENGINE_MCTS, 5, 800)

    app.players[app.state.current_player] = PlayerConfig(
        kind=ENGINE_NEGAMAX,
        strength=4,
    )
    app._play_ai_turn()
    assert calls[-1] == (ENGINE_NEGAMAX, 4, 800)

    app.players[app.state.current_player] = PlayerConfig(kind=ENGINE_RANDOM)
    app._play_ai_turn()
    assert calls[-1][0] == ENGINE_RANDOM

    app.players[app.state.current_player] = PlayerConfig(
        kind=PLAYER_KIND_HUMAN
    )
    before = len(app.history.applied)
    app._play_ai_turn()
    assert len(app.history.applied) == before

    monkeypatch.setattr(gui_module, "_default_strength", lambda _kind: None)
    app.players[PLAYER_ONE] = PlayerConfig(kind=ENGINE_MCTS, strength=None)
    app.state.current_player = PLAYER_ONE
    with pytest.raises(
        RuntimeError, match="MCTS strength could not be resolved"
    ):
        app._play_ai_turn()

    app.players[PLAYER_ONE] = PlayerConfig(kind=ENGINE_NEGAMAX, strength=None)
    with pytest.raises(
        RuntimeError, match="Search depth could not be resolved"
    ):
        app._play_ai_turn()


def test_play_ai_turn_mcts_uses_explicit_strength(
    monkeypatch: pytest.MonkeyPatch,
    input_from,
) -> None:
    """Test that the _play_ai_turn function uses the explicitly
    set strength for MCTS.

    Feature: MCTS strength configuration
    Scenario: MCTS uses explicitly set strength

    Given a GUI app with an MCTS player with an explicitly set strength
    When _play_ai_turn is called
    Then the MCTS engine should be called with the explicitly set strength
    """
    outputs: list[str] = []
    players = {
        PLAYER_ONE: PlayerConfig(kind=ENGINE_MCTS, strength=123),
        PLAYER_TWO: PlayerConfig(kind=PLAYER_KIND_HUMAN),
    }
    app = FourInARowGui(
        players=players,
        input_fn=input_from([]),
        output_fn=outputs.append,
    )

    calls: list[tuple[str, int, int]] = []

    def fake_suggest_move(state, engine, *, depth=5, iterations=800, rng=None):
        _ = state, rng
        calls.append((engine, depth, iterations))
        return EngineSuggestion(engine=engine, move=3, pv=[3], evaluation=0.1)

    monkeypatch.setattr(gui_module, "suggest_move", fake_suggest_move)
    app._play_ai_turn()
    assert calls[-1] == (ENGINE_MCTS, 5, 123)


def test_handle_ai_until_human_turn_and_post_prompt_called(
    monkeypatch: pytest.MonkeyPatch,
    make_app,
) -> None:
    """Test that the _handle_ai_until_human_turn function calls the
    _handle_post_game_prompt_if_needed method after the AI turn.

    Feature: AI turn handling and post-game prompt
    Scenario: _handle_ai_until_human_turn calls post-game prompt

    Given a GUI app with an AI player
    When _handle_ai_until_human_turn is called and the game finishes
    Then the _handle_post_game_prompt_if_needed method should be called
    """
    players = {
        PLAYER_ONE: PlayerConfig(kind=ENGINE_NEGAMAX, strength=2),
        PLAYER_TWO: PlayerConfig(kind=PLAYER_KIND_HUMAN),
    }
    app, _outputs = make_app(players)
    app._create_plot()
    try:
        monkeypatch.setattr(gui_module.plt, "pause", lambda _seconds: None)
        monkeypatch.setattr(
            gui_module,
            "suggest_move",
            lambda _state, engine, **_kw: EngineSuggestion(
                engine=engine, move=0, pv=[0], evaluation=0.1
            ),
        )
        called = {"post": 0}

        def fake_post() -> None:
            called["post"] += 1

        monkeypatch.setattr(
            app, "_handle_post_game_prompt_if_needed", fake_post
        )
        app._handle_ai_until_human_turn()
        assert app.state.current_player == PLAYER_TWO
        assert called["post"] == 1
    finally:
        plt.close(app.figure)


def test_handle_ai_until_human_turn_skips_loop_when_already_finished(
    monkeypatch: pytest.MonkeyPatch,
    make_app,
) -> None:
    """Test that the _handle_ai_until_human_turn function skips the AI loop
    when the game is already finished.

    Feature: AI turn handling
    Scenario: Skip AI loop when game is finished

    Given a GUI app with a finished game
    When _handle_ai_until_human_turn is called
    Then the AI loop should be skipped
    And the post-game prompt should be called
    """
    app, _outputs = make_app()
    app._create_plot()
    try:
        app.finished = True
        called = {"ai": 0, "post": 0}

        monkeypatch.setattr(
            app,
            "_play_ai_turn",
            lambda: called.__setitem__("ai", called["ai"] + 1),
        )
        monkeypatch.setattr(
            app,
            "_handle_post_game_prompt_if_needed",
            lambda: called.__setitem__("post", called["post"] + 1),
        )

        app._handle_ai_until_human_turn()
        assert called["ai"] == 0
        assert called["post"] == 1
    finally:
        plt.close(app.figure)
