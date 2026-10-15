# Requirements: FR-GUI-001
"""GUI tests for plot creation and board rendering behavior."""
# pylint: disable=protected-access

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_rgba

import four_in_a_row.gui as gui_module
from four_in_a_row.constants import PLAYER_ONE, PLAYER_TWO
from four_in_a_row.gui import ENGINE_NEGAMAX, PLAYER_KIND_HUMAN, PlayerConfig


def test_create_plot_and_current_player_name(make_app) -> None:
    """Test that the plot is created correctly and
    the current player name is displayed.

    Feature: Plot creation and current player name
    Scenario: Create plot and display current player name

    Given a GUI app
    When the plot is created
    Then the current player name should be displayed correctly
    """
    app, _outputs = make_app()
    app._create_plot()
    try:
        assert app.axis.get_facecolor()[:3] == pytest.approx((0.0, 0.0, 1.0))
        assert len(app.pieces) == 6
        assert len(app.pieces[0]) == 7
        assert app._current_player_name() == "Player 1 (red)"
        app.state.current_player = PLAYER_TWO
        assert app._current_player_name() == "Player 2 (yellow)"
    finally:
        plt.close(app.figure)


def test_create_plot_handles_missing_window_title_manager(
    monkeypatch: pytest.MonkeyPatch,
    make_app,
) -> None:
    """Test that the plot creation handles missing window title manager.

    Feature: Plot creation
    Scenario: Handle missing window title manager

    Given a GUI app
    When the plot is created with a missing window title manager
    Then the plot should be created without errors
    """
    original_subplots = gui_module.plt.subplots

    def fake_subplots(*args, **kwargs):
        figure, axis = original_subplots(*args, **kwargs)
        figure.canvas.manager = None
        return figure, axis

    monkeypatch.setattr(gui_module.plt, "subplots", fake_subplots)
    app, _outputs = make_app()
    app._create_plot()
    try:
        assert len(app.pieces) == 6
        assert len(app.pieces[0]) == 7
    finally:
        plt.close(app.figure)


def test_render_board_requires_status_text(make_app) -> None:
    """Test that rendering the board requires the
    status text to be initialized.

    Feature: Board rendering
    Scenario: Render board requires status text

    Given a GUI app
    When the board is rendered without initializing the status text
    Then a RuntimeError should be raised
    """
    app, _outputs = make_app()
    with pytest.raises(RuntimeError, match="Status text is not initialized"):
        app._render_board()


def test_render_board_winner_and_draw_paths(
    monkeypatch: pytest.MonkeyPatch,
    make_app,
) -> None:
    """Test the rendering of the board when there is a winner or a draw.

    Feature: Board rendering
    Scenario: Render board with winner and draw

    Given a GUI app
    When the board is rendered with a winner or a draw
    Then the appropriate status text should be displayed
    """
    app, outputs = make_app()
    app._create_plot()
    try:
        for _ in range(4):
            app._apply_move(0)
            app._apply_move(1)
        app._render_board()
        assert app.finished is True
        assert app._finished_prompt_pending is True
        assert any("Result: Player 1 (red" in line for line in outputs)

        app._finished_prompt_pending = False
        app._render_board()
        assert app._finished_prompt_pending is False

        app2, outputs2 = make_app()
        app2._create_plot()
        app2.finished = False
        monkeypatch.setattr(app2.state, "winner", lambda: 0)
        monkeypatch.setattr(app2.state, "is_draw", lambda: True)
        app2._render_board()
        assert app2.finished is True
        assert any("Result: Draw" in line for line in outputs2)
        plt.close(app2.figure)
    finally:
        plt.close(app.figure)


def test_render_board_player_two_wins_branch(make_app) -> None:
    """Test the rendering of the board when player two wins.

    Feature: Board rendering
    Scenario: Render board with player two winning

    Given a GUI app
    When the board is rendered with player two winning
    Then the appropriate status text should be displayed
    """
    app, outputs = make_app()
    app._create_plot()
    try:
        winning_sequence = [1, 0, 1, 0, 2, 0, 3, 0]
        for move in winning_sequence:
            app._apply_move(move)

        app._render_board()
        assert app.status_text is not None
        assert app.status_text.get_text() == "Player 2 (yellow) wins"
        assert any("Result: Player 2 (yellow" in line for line in outputs)
    finally:
        plt.close(app.figure)


def test_render_board_highlights_winning_row(make_app) -> None:
    """Test the rendering of the board highlights the winning row.

    Feature: Board rendering
    Scenario: Highlight winning row

    Given a GUI app
    When the board is rendered with a winning row
    Then the winning row should be highlighted
    """
    app, _outputs = make_app()
    app._create_plot()
    try:
        for move in [0, 0, 1, 1, 2, 2, 3]:
            app._apply_move(move)

        app._render_board()

        winning_rgba = to_rgba("lime")
        default_rgba = to_rgba("navy")
        for column in (0, 1, 2, 3):
            piece = app.pieces[0][column]
            assert piece.get_edgecolor() == pytest.approx(winning_rgba)
            assert piece.get_linewidth() == pytest.approx(3.0)

        non_winning = app.pieces[0][4]
        assert non_winning.get_edgecolor() == pytest.approx(default_rgba)
        assert non_winning.get_linewidth() == pytest.approx(1.2)
    finally:
        plt.close(app.figure)


def test_winning_cells_no_win_covers_boundary_breaks(
    monkeypatch: pytest.MonkeyPatch,
    make_app,
) -> None:
    """Test that the _winning_cells method returns an empty set
    when there is no winner, even if the board has pieces in a winning
    configuration that would break boundary checks.

    Feature: Winning cells calculation
    Scenario: Winning cells returns empty set when no winner

    Given a GUI app with pieces in a winning configuration but no winner
    When the _winning_cells method is called
    Then it should return an empty set without errors
    """
    app, _outputs = make_app()
    app.state.board[5][0] = PLAYER_ONE
    app.state.board[0][0] = PLAYER_ONE
    app.state.board[2][2] = PLAYER_ONE
    assert app.state.winner() == 0

    monkeypatch.setattr(app.state, "winner", lambda: PLAYER_ONE)
    cells = app._winning_cells()

    assert cells == set()


def test_render_board_non_terminal_status_text(make_app) -> None:
    """Test that the status text is updated correctly
    when the game is not finished.

    Feature: Board rendering
    Scenario: Update status text for non-terminal game

    Given a GUI app with players configured
    When the board is rendered and the game is not finished
    Then the status text should indicate the current player's turn
    """
    players = {
        PLAYER_ONE: PlayerConfig(kind=PLAYER_KIND_HUMAN),
        PLAYER_TWO: PlayerConfig(kind=ENGINE_NEGAMAX, strength=2),
    }
    app, _outputs = make_app(players)
    app._create_plot()
    try:
        app._render_board()
        assert app.status_text is not None
        assert "click a column" in app.status_text.get_text()
        app.state.current_player = PLAYER_TWO
        app._render_board()
        assert "negamax thinking" in app.status_text.get_text()
    finally:
        plt.close(app.figure)
