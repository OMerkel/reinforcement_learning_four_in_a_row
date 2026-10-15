# Requirements: FR-RL-001, FR-RL-002, FR-RL-003, FR-RL-004, FR-RL-005
# Requirements: NFR-RL-001
"""Contract tests for rules-only, zero-prior self-play learning."""

import json
import random

import pytest

from four_in_a_row.game import GameState
from four_in_a_row.rl import QAgent, train


def test_cold_start_and_player_relative_encoding():
    """Unseen values are zero and token-color swaps preserve the key."""
    agent = QAgent(seed=7)
    state = GameState.new()
    assert agent.values(state) == [0.0] * 7
    assert not agent.table
    state.apply_move(0)
    assert agent.key(state).startswith("2")
    swapped = state.clone()
    swapped.current_player = 1
    swapped.board = [
        [3 - cell if cell else 0 for cell in row] for row in state.board
    ]
    assert agent.key(state) == agent.key(swapped)


def test_mask_and_random_ties():
    """Never select full columns, even when their values are highest."""
    agent = QAgent(seed=7, epsilon=0)
    state = GameState.new()
    for _ in range(6):
        state.apply_move(0)
    agent.table[agent.key(state)] = [1.0] + [0.0] * 6
    assert {agent.select(state) for _ in range(100)} == set(range(1, 7))
    assert all(agent.select(state, epsilon=1) != 0 for _ in range(30))


def test_opponent_bootstrap_and_terminal_reward():
    """Negate opponent estimates and stop bootstrapping at terminal boards."""
    agent = QAgent(alpha=1, gamma=0.9)
    state = GameState.new()
    following = state.clone()
    following.apply_move(0)
    agent.table[agent.key(following)] = [0.5] * 7
    agent.update(state, 0, 0, following)
    assert agent.values(state)[0] == pytest.approx(-0.45)
    winning = GameState.new()
    for column in [0, 1, 0, 1, 0, 1]:
        winning.apply_move(column)
    terminal = winning.clone()
    terminal.apply_move(0)
    agent.update(winning, 0, 1, terminal)
    assert agent.values(winning)[0] == 1
    with pytest.raises(ValueError, match="terminal"):
        agent.select(terminal)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0},
        {"alpha": 2},
        {"gamma": -1},
        {"gamma": 2},
        {"epsilon": -1},
        {"epsilon": 2},
        {"alpha": float("nan")},
    ],
)
def test_invalid_hyperparameters(kwargs):
    """Reject out-of-range learning, discount, and exploration settings."""
    with pytest.raises(ValueError):
        QAgent(**kwargs)


def test_invalid_actions_and_rewards():
    """Reject unavailable actions and non-finite or out-of-range rewards."""
    agent = QAgent()
    state = GameState.new()
    with pytest.raises(ValueError):
        agent.select(state, epsilon=2)
    with pytest.raises(ValueError):
        agent.update(state, 7, 0, state)
    with pytest.raises(ValueError):
        agent.update(state, 0, float("nan"), state)
    with pytest.raises(ValueError):
        agent.update(state, 0, 2, state)


def test_seeded_self_play_and_checkpoint_resume(tmp_path):
    """Preserve seeded self-play and exact continuation across JSON resume."""
    first = QAgent(seed=13)
    second = QAgent(seed=13)
    assert train(first, 12) == train(second, 12)
    assert first.table == second.table
    assert first.table and any(any(values) for values in first.table.values())
    checkpoint = tmp_path / "nested" / "agent.json"
    first.save(checkpoint)
    resumed = QAgent.load(checkpoint)
    assert resumed.table == first.table
    assert resumed.episodes == 12
    assert train(first, 3) == train(resumed, 3)
    assert resumed.table == first.table
    assert resumed.rng.getstate() == first.rng.getstate()
    with pytest.raises(ValueError):
        train(first, 0)


@pytest.mark.parametrize(
    "change",
    [
        {"version": 999},
        {"table": {"bad": [0] * 7}},
        {"table": {"0" * 42: [0]}},
        {"table": {"0" * 42: [float("inf")] * 7}},
        {"episodes": -1},
        {"rng": [3, [], None]},
        {"alpha": 0},
        {"table": []},
        {"table": {"x" * 42: [0] * 7}},
        {"table": {"0" * 42: ["bad"] * 7}},
        {"episodes": 1.5},
    ],
)
def test_reject_invalid_checkpoint(tmp_path, change):
    """Validate JSON checkpoint schema, dimensions, values, and counters."""
    checkpoint = tmp_path / "agent.json"
    QAgent().save(checkpoint)
    data = json.loads(checkpoint.read_text())
    data.update(change)
    checkpoint.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        QAgent.load(checkpoint)


def test_learning_does_not_use_heuristics(monkeypatch):
    """Self-play obtains rewards from rules rather than a heuristic teacher."""
    def forbidden(*_args, **_kwargs):
        pytest.fail("Training must not use heuristic evaluation or search")

    monkeypatch.setattr("four_in_a_row.game.evaluate_state", forbidden)
    agent = QAgent(seed=3)
    result = train(agent, 2)
    assert result.episodes == 2
    assert result.player_one_wins + result.player_two_wins + result.draws == 2
    assert agent.rng.getstate() != random.Random(0).getstate()


def test_checkpoint_missing_fields(tmp_path):
    """Report missing checkpoint fields as a validation error."""
    checkpoint = tmp_path / "broken.json"
    checkpoint.write_text("{}")
    with pytest.raises(ValueError, match="checkpoint"):
        QAgent.load(checkpoint)


def test_draw_has_zero_bootstrap_and_is_counted(monkeypatch):
    """Draw transitions have zero targets and contribute to episode totals."""
    draw = GameState(
        board=[
            [1, 2, 1, 2, 1, 2, 1],
            [1, 2, 1, 2, 1, 2, 1],
            [2, 1, 2, 1, 2, 1, 2],
            [2, 1, 2, 1, 2, 1, 2],
            [1, 2, 1, 2, 1, 2, 1],
            [1, 2, 1, 2, 1, 2, 1],
        ]
    )
    assert draw.is_draw()
    before = draw.clone()
    before.board[5][6] = 0
    agent = QAgent(alpha=1)
    agent.update(before, 6, 0, draw)
    assert agent.values(before)[6] == 0
    with pytest.raises(ValueError):
        agent.update(draw, 6, 0, draw)
    monkeypatch.setattr(GameState, "new", lambda **kwargs: draw.clone())
    assert train(agent, 1).draws == 1
