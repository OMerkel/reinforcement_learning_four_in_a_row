# Requirements: FR-GUI-001
"""GUI tests for configuration parsing and normalization."""

from collections.abc import Callable

import pytest

from four_in_a_row.constants import PLAYER_ONE, PLAYER_TWO
from four_in_a_row.gui import (
    ENGINE_ALPHABETA,
    ENGINE_MCTS,
    ENGINE_MINIMAX,
    ENGINE_NEGAMAX,
    ENGINE_RANDOM,
    PLAYER_KIND_HUMAN,
    PlayerConfig,
    _ask_player_config,
    _default_strength,
    _describe_config,
    _normalize_kind,
    _resolve_player_configs,
    _strength_label,
)


def test_normalizers_and_descriptors() -> None:
    """Test normalization and description of player configurations.

    Feature: Configuration normalization and description
    Scenario: Normalize and describe player configurations

    Given various player configuration inputs
    When they are normalized and described
    Then the normalization and description should be correct
    """
    assert _normalize_kind("alpha-beta") == ENGINE_ALPHABETA
    assert _normalize_kind("alpha_beta") == ENGINE_ALPHABETA
    assert _normalize_kind("  mcts ") == ENGINE_MCTS
    with pytest.raises(ValueError, match="Unknown player kind"):
        _normalize_kind("stockfish")

    assert _default_strength(ENGINE_MCTS) == 800
    assert _default_strength(ENGINE_MINIMAX) == 5
    assert _default_strength(PLAYER_KIND_HUMAN) is None

    assert _strength_label(ENGINE_MCTS) == "iterations"
    assert _strength_label(ENGINE_NEGAMAX) == "depth"

    assert _describe_config(PlayerConfig(kind=PLAYER_KIND_HUMAN)) == "human"
    assert _describe_config(
        PlayerConfig(kind=ENGINE_RANDOM)
    ) == "random (default)"
    assert _describe_config(PlayerConfig(kind=ENGINE_NEGAMAX, strength=4)) == (
        "negamax (depth=4)"
    )


def test_ask_player_config_paths(
    input_from: Callable[[list[str]], Callable[[str], str]],
) -> None:
    """Test the _ask_player_config function with various input paths.

    Feature: Player configuration
    Scenario: Ask player configuration

    Given various preselected kinds and strengths
    When _ask_player_config is called with different input paths
    Then the returned PlayerConfig should be correct or
    an error should be raised
    """
    outputs: list[str] = []

    cfg = _ask_player_config(
        1,
        preselected_kind=None,
        preselected_strength=None,
        input_fn=input_from(["", ""]),
        output_fn=outputs.append,
    )
    assert cfg == PlayerConfig(kind=PLAYER_KIND_HUMAN)

    cfg2 = _ask_player_config(
        2,
        preselected_kind=ENGINE_MCTS,
        preselected_strength=None,
        input_fn=input_from([""]),
        output_fn=outputs.append,
    )
    assert cfg2 == PlayerConfig(kind=ENGINE_MCTS, strength=800)

    cfg3 = _ask_player_config(
        2,
        preselected_kind=ENGINE_NEGAMAX,
        preselected_strength=3,
        input_fn=input_from([]),
        output_fn=outputs.append,
    )
    assert cfg3 == PlayerConfig(kind=ENGINE_NEGAMAX, strength=3)

    with pytest.raises(ValueError, match="strength must be >= 1"):
        _ask_player_config(
            2,
            preselected_kind=ENGINE_NEGAMAX,
            preselected_strength=0,
            input_fn=input_from([]),
            output_fn=outputs.append,
        )


def test_resolve_player_configs_mixed_sources(
    input_from: Callable[[list[str]], Callable[[str], str]],
) -> None:
    """Test resolving player configs with mixed sources.

    Feature: Player configuration resolution
    Scenario: Resolve player configs with mixed sources

    Given preselected kinds and strengths for players
    When _resolve_player_configs is called with input for missing values
    Then the resolved PlayerConfig should be correct
    """
    outputs: list[str] = []
    result = _resolve_player_configs(
        p1="human",
        p2=None,
        p1_strength=None,
        p2_strength=None,
        input_fn=input_from([ENGINE_NEGAMAX, "3"]),
        output_fn=outputs.append,
    )
    assert result[PLAYER_ONE] == PlayerConfig(kind=PLAYER_KIND_HUMAN)
    assert result[PLAYER_TWO] == PlayerConfig(kind=ENGINE_NEGAMAX, strength=3)
