# Requirements: FR-TACTIC-001
# Requirements: FR-FORK-DEFENSE-001
"""Evaluation-only tactical fixtures and frozen policy diagnostics."""

import pytest

from four_in_a_row.game import GameState
from four_in_a_row.rl import QAgent
from four_in_a_row.rl_benchmark import (
    TacticalCase,
    benchmark,
    tactical_actions,
    tactical_cases,
    winning_actions,
)


def test_fixture_labels_are_valid_and_color_symmetric():
    """Pin legal fixtures and verify their answers using only game rules."""
    cases = tactical_cases()
    assert len(cases) == 24
    assert len({case.name for case in cases}) == 24
    for case in cases:
        state = case.state()
        assert not state.is_terminal()
        assert case.expected == tactical_actions(state, case.category)
        assert case.expected <= set(state.legal_moves())
    assert {case.category for case in cases} == {
        "win",
        "block",
        "safety",
        "fork",
        "fork_defense",
    }


def test_benchmark_scores_and_freezes_a_policy(monkeypatch):
    """Report category failures without changing policy state or weights."""
    agent = QAgent(seed=11)
    initial_rng = agent.rng.getstate()
    monkeypatch.setattr(agent, "select", lambda state, **kwargs: 0)
    report = benchmark(agent, seed=2000)
    assert report["cases"] == 24
    assert report["correct"] + len(report["failures"]) == 24
    assert set(report["categories"]) == {
        "win",
        "block",
        "safety",
        "fork",
        "fork_defense",
    }
    assert report["categories"]["win"]["correct"] == 2
    assert not agent.table and agent.episodes == 0
    assert agent.rng.getstate() == initial_rng


def test_rules_oracle_passes_every_tactical_case(monkeypatch):
    """Check the scoring harness independently of neural playing strength."""
    agent = QAgent()
    answers = {
        QAgent.key(case.state()): min(case.expected)
        for case in tactical_cases()
    }
    monkeypatch.setattr(
        agent, "select", lambda state, **kwargs: answers[QAgent.key(state)]
    )
    report = benchmark(agent)
    assert report["accuracy"] == 1
    assert not report["failures"]


def test_empty_suite_and_malformed_fixtures_are_rejected():
    """Reject empty, terminal, mismatched, or invalid diagnostic inputs."""
    with pytest.raises(ValueError):
        benchmark(QAgent(), cases=())
    with pytest.raises(ValueError):
        TacticalCase("invalid", "win", (8,), 1, frozenset({0})).state()
    with pytest.raises(ValueError):
        TacticalCase(
            "finished", "win", (0, 1, 0, 1, 0, 1, 0), 1, frozenset({0})
        ).state()
    state = GameState.new()
    with pytest.raises(ValueError):
        winning_actions(state, 0)
    with pytest.raises(ValueError):
        tactical_actions(state, "unknown")
    assert not winning_actions(state, 1)
    with pytest.raises(ValueError):
        TacticalCase("mismatch", "win", (), 1, frozenset({0})).state()
    with pytest.raises(ValueError):
        TacticalCase(
            "postgame", "win", (0, 1, 0, 1, 0, 1, 0, 2), 1, frozenset({0})
        ).state()
    for column in [0, 1, 0, 1, 0, 2]:
        state.apply_move(column)
    assert 0 in tactical_actions(state, "safety")


def test_fork_defense_rejects_moves_that_allow_double_threats():
    """Do not mistake creating our own fork for preventing the opponent's."""
    state = GameState.new()
    for action in [6, 5, 0, 2, 1, 3, 1, 0, 6, 3, 4, 5, 4, 1, 4, 1, 1, 4, 3]:
        state.apply_move(action)
    assert not winning_actions(state, state.current_player)
    assert not winning_actions(state, 3 - state.current_player)
    assert tactical_actions(state, "fork_defense") == frozenset({5, 6})


def test_old_tactical_subset_remains_comparable():
    """Keep prior scores identifiable when extending the diagnostic suite."""
    legacy = tactical_cases(include_fork_defense=False)
    assert len(legacy) == 16
    assert all(case.category != "fork_defense" for case in legacy)
    report = benchmark(QAgent(), cases=legacy)
    assert report["suite"] == "connect-four-tactics-v1"
    assert report["legacy_cases"] == 16
    assert report["legacy_correct"] == report["correct"]
