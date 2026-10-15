# Requirements: FR-RANDOM-001
"""Tests for the random engine helper."""

import random

import pytest

from four_in_a_row.engines import random_suggestion
from four_in_a_row.game import GameState

from .engines_shared import no_legal_moves_state

pytestmark = pytest.mark.regression


def test_random_suggestion_legal() -> None:
    """Test that random_suggestion returns a legal move and valid evaluation.

    Feature: Random Suggestion Validity
    Scenario: random_suggestion returns legal move and valid evaluation

    Given a new game state
    When random_suggestion is invoked
    Then the returned suggestion should be a legal move with a valid evaluation
    """
    state = GameState.new()
    suggestion = random_suggestion(state, rng=random.Random(7))
    assert suggestion.move in state.legal_moves()
    assert suggestion.pv == [suggestion.move]
    assert -1.0 <= suggestion.evaluation <= 1.0


def test_random_suggestion_deterministic_with_seed() -> None:
    """Test that random_suggestion is deterministic with a fixed seed.

    Feature: Random Suggestion Determinism
    Scenario: random_suggestion is deterministic with a fixed seed

    Given a new game state
    When random_suggestion is invoked with the same seed
    Then the returned suggestions should be identical
    """
    state = GameState.new()
    first = random_suggestion(state, rng=random.Random(1234))
    second = random_suggestion(state, rng=random.Random(1234))

    assert first.move == second.move
    assert first.pv == second.pv
    assert first.evaluation == second.evaluation


def test_random_suggestion_no_legal_moves() -> None:
    """Test that random_suggestion raises an error when
    there are no legal moves.

    Feature: Random Suggestion Validation
    Scenario: random_suggestion raises an error when
    no legal moves are available

    Given a game state with no legal moves
    When random_suggestion is invoked
    Then a ValueError should be raised
    """
    state = no_legal_moves_state()
    with pytest.raises(ValueError):
        random_suggestion(state)
