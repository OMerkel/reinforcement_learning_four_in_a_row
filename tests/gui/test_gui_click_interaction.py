# Requirements: FR-GUI-001
"""GUI tests for board click interaction handling."""
# pylint: disable=protected-access

from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest

from four_in_a_row.constants import PLAYER_ONE
from four_in_a_row.game import InvalidMoveError
from four_in_a_row.gui import (
    ENGINE_RANDOM,
    PLAYER_KIND_HUMAN,
    PlayerConfig,
)


def test_on_click_branches(
    monkeypatch: pytest.MonkeyPatch,
    make_app,
) -> None:
    """Test the on_click method with various click scenarios.

    Feature: Click interaction handling
    Scenario: Handle different click scenarios in on_click method

    Given a GUI app
    When on_click is called with different scenarios
    Then the appropriate branches should be executed without errors
    """
    app, outputs = make_app()
    app._create_plot()
    try:
        called = {"post": 0}
        monkeypatch.setattr(
            app,
            "_handle_post_game_prompt_if_needed",
            lambda: called.__setitem__("post", called["post"] + 1),
        )

        app.finished = True
        app._on_click(SimpleNamespace(inaxes=app.axis, xdata=0.0))
        assert called["post"] == 1

        app.finished = False
        app._on_click(SimpleNamespace(inaxes=None, xdata=1.0))
        app._on_click(SimpleNamespace(inaxes=app.axis, xdata=None))

        app.players[PLAYER_ONE] = PlayerConfig(kind=ENGINE_RANDOM)
        app._on_click(SimpleNamespace(inaxes=app.axis, xdata=1.0))
        assert any("Ignored click" in line for line in outputs)

        app.players[PLAYER_ONE] = PlayerConfig(kind=PLAYER_KIND_HUMAN)
        app._on_click(SimpleNamespace(inaxes=app.axis, xdata=-1.0))
        app._on_click(SimpleNamespace(inaxes=app.axis, xdata=7.4))

        monkeypatch.setattr(
            app,
            "_apply_move",
            lambda _c: (_ for _ in ()).throw(InvalidMoveError("bad")),
        )
        app._on_click(SimpleNamespace(inaxes=app.axis, xdata=1.0))
        assert any("Invalid move" in line for line in outputs)
    finally:
        plt.close(app.figure)


def test_on_click_normal_flow_calls_ai_and_post(
    monkeypatch: pytest.MonkeyPatch,
    make_app,
) -> None:
    """Test that the on_click method calls the AI handling
    and post-game prompt in the normal flow.

    Feature: Click interaction handling
    Scenario: Normal flow calls AI and post-game prompt

    Given a GUI app with a human player
    When on_click is called
    Then the AI handling and post-game prompt should be called
    """
    app, _outputs = make_app()
    app._create_plot()
    try:
        called = {"ai": 0, "post": 0}

        monkeypatch.setattr(
            app,
            "_handle_ai_until_human_turn",
            lambda: called.__setitem__("ai", called["ai"] + 1),
        )
        monkeypatch.setattr(
            app,
            "_handle_post_game_prompt_if_needed",
            lambda: called.__setitem__("post", called["post"] + 1),
        )

        app._on_click(SimpleNamespace(inaxes=app.axis, xdata=3.0))
        assert called["ai"] == 1
        assert called["post"] == 1
    finally:
        plt.close(app.figure)


def test_on_click_winning_move_skips_ai_followup(
    monkeypatch: pytest.MonkeyPatch,
    make_app,
) -> None:
    """Test that the on_click method skips the AI follow-up
    when a winning move is made.

    Feature: Click interaction handling
    Scenario: Winning move skips AI follow-up

    Given a GUI app with a human player
    When on_click is called with a winning move
    Then the AI follow-up should be skipped
    And the post-game prompt should be called
    """
    app, _outputs = make_app()
    app._create_plot()
    try:
        for move in [0, 1, 0, 1, 0, 1]:
            app._apply_move(move)

        called = {"ai": 0, "post": 0}
        monkeypatch.setattr(
            app,
            "_handle_ai_until_human_turn",
            lambda: called.__setitem__("ai", called["ai"] + 1),
        )
        monkeypatch.setattr(
            app,
            "_handle_post_game_prompt_if_needed",
            lambda: called.__setitem__("post", called["post"] + 1),
        )

        app._on_click(SimpleNamespace(inaxes=app.axis, xdata=0.0))

        assert app.finished is True
        assert called["ai"] == 0
        assert called["post"] == 1
    finally:
        plt.close(app.figure)
