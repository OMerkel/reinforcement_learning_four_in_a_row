# Requirements: FR-ENGINE-001
"""Tests for engine dispatch behavior in suggest_move."""

import random

import pytest

from four_in_a_row.engines import suggest_move
from four_in_a_row.game import GameState

from .engines_shared import winning_position_for_p1

pytestmark = pytest.mark.regression


def test_suggest_move_dispatch() -> None:
    """Test that suggest_move dispatches to the correct engine and
    returns a valid suggestion.

    Feature: Engine Dispatch
    Scenario: suggest_move dispatches to the correct engine

    Given a new game state
    When suggest_move is invoked with each supported engine
    Then the returned suggestion should be valid
    """
    state = GameState.new()
    for engine in ["random", "minimax", "negamax", "alphabeta", "mcts"]:
        suggestion = suggest_move(
            state,
            engine=engine,
            depth=2,
            iterations=30,
            rng=random.Random(0),
        )
        assert suggestion.move in state.legal_moves()
        assert -1.0 <= suggestion.evaluation <= 1.0


def test_suggest_move_supports_alpha_beta_aliases() -> None:
    """Test that suggest_move supports different aliases for
    the alpha-beta engine.

    Feature: Engine Alias Support
    Scenario: suggest_move supports alpha-beta engine aliases

    Given a winning position for player 1
    When suggest_move is invoked with each alpha-beta alias
    Then the returned suggestion should indicate the correct engine and move
    """
    state = winning_position_for_p1()
    for engine in ["alphabeta", "alpha-beta", "alpha_beta"]:
        suggestion = suggest_move(state, engine=engine, depth=3)
        assert suggestion.engine == "alphabeta"
        assert suggestion.move == 3
        assert suggestion.evaluation == 1.0


def test_suggest_move_unknown_engine() -> None:
    """Test that suggest_move raises an error for unknown engines.

    Feature: Engine Validation
    Scenario: suggest_move raises an error for unknown engines

    Given a new game state
    When suggest_move is invoked with an unknown engine
    Then a ValueError should be raised
    """
    state = GameState.new()
    with pytest.raises(ValueError):
        suggest_move(state, engine="nope")
