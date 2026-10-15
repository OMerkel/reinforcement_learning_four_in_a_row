# Requirements: FR-NEURAL-001, FR-NN-SAVE-001, FR-OUTCOME-001
"""Neural learning mechanics, bounded storage, and resume contracts."""

import copy
from typing import Any

import pytest
import torch

from four_in_a_row.game import GameState
from four_in_a_row.rl import QAgent, train
from four_in_a_row.rl_nn import (
    NeuralAgent,
    Transition,
    double_dqn_targets,
    encode,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    """Use one CPU thread in tests and restore the previous setting."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def winning_transition():
    """Build a legal one-move vertical win for terminal-target tests."""
    before = GameState.new()
    for column in [0, 1, 0, 1, 0, 1]:
        before.apply_move(column)
    following = before.clone()
    following.apply_move(0)
    return before, following


def small_agent(**kwargs):
    """Create a small training fixture with caller-selected parameters."""
    return NeuralAgent(
        batch_size=4,
        warmup=4,
        replay_capacity=32,
        train_every=1,
        target_sync=3,
        **kwargs,
    )


def test_neural_zero_prior_and_encoding():
    """Seeded untrained weights and mover-relative observation planes agree."""
    first = small_agent(seed=7)
    second = small_agent(seed=7)
    assert not hasattr(first, "table")
    assert len(first.replay) == 0
    assert first.values(GameState.new()) == second.values(GameState.new())
    assert first.parameter_count < 30000
    state = GameState.new()
    state.apply_move(0)
    planes = encode(state)
    assert planes.shape == (3, 6, 7)
    assert planes.dtype == torch.float32
    assert planes[1, 0, 0] == 1
    assert torch.all(planes.sum(dim=0) == 1)
    swapped = state.clone()
    swapped.current_player = 1
    swapped.board = [
        [3 - cell if cell else 0 for cell in row] for row in state.board
    ]
    assert torch.equal(encode(state), encode(swapped))


def test_double_dqn_opponent_target():
    """Select legal actions online and negate their target-network values."""
    online = torch.tensor([[99.0, 0.8, 0.7], [9, 8, 7], [1, 2, 3]])
    target = torch.tensor([[99.0, 0.4, 0.9], [float("nan")] * 3, [8, 9, 10]])
    legal = torch.tensor([[False, True, True], [False] * 3, [False] * 3])
    result = double_dqn_targets(
        online,
        target,
        torch.tensor([0.0, 1.0, 0.0]),
        legal,
        torch.tensor([False, True, True]),
        0.9,
    )
    assert result.tolist() == pytest.approx([-0.36, 1, 0])
    assert not result.requires_grad
    with pytest.raises(ValueError):
        double_dqn_targets(
            online,
            target,
            torch.zeros(3),
            legal,
            torch.tensor([False, False, True]),
            1,
        )


def test_neural_legal_action_mask(monkeypatch):
    """Mask full columns during both greedy selection and exploration."""
    agent = small_agent(seed=4)
    state = GameState.new()
    for _ in range(6):
        state.apply_move(0)
    monkeypatch.setattr(agent, "values", lambda _: [99.0] + [0.0] * 6)
    assert {agent.select(state, epsilon=0) for _ in range(100)} == set(
        range(1, 7)
    )
    assert all(agent.select(state, epsilon=1) in range(1, 7) for _ in range(30))
    _, terminal = winning_transition()
    with pytest.raises(ValueError, match="terminal"):
        agent.select(terminal)
    with pytest.raises(ValueError):
        agent.select(state, epsilon=2)


def test_replay_reflection_and_target_sync():
    """Reflect whole transitions, cap replay, and synchronize on schedule."""
    agent = NeuralAgent(
        batch_size=2, warmup=2, replay_capacity=4, train_every=1, target_sync=2
    )
    before, following = winning_transition()
    old_target = copy.deepcopy(agent.target.state_dict())
    agent.update(before, 0, 1, following)
    assert len(agent.replay) == 2
    original, reflected = agent.replay
    assert reflected.action == 6 - original.action
    mirrored = before.clone()
    mirrored.board = [list(reversed(row)) for row in before.board]
    assert reflected.position == QAgent.key(mirrored)
    assert all(
        torch.equal(value, agent.target.state_dict()[key])
        for key, value in old_target.items()
    )
    assert any(
        not torch.equal(value, agent.online.state_dict()[key])
        for key, value in old_target.items()
    )
    agent.update(before, 0, 1, following)
    assert agent.updates == 2
    assert all(
        torch.equal(value, agent.target.state_dict()[key])
        for key, value in agent.online.state_dict().items()
    )
    agent.update(before, 0, 1, following)
    assert len(agent.replay) == 4


def test_tiny_terminal_dataset_can_be_fit():
    """Verify optimizer learning before running larger self-play experiments."""
    agent = NeuralAgent(
        learning_rate=0.01,
        batch_size=2,
        warmup=2,
        replay_capacity=16,
        train_every=1,
    )
    before, following = winning_transition()
    initial_error = (1 - agent.values(before)[0]) ** 2
    for _ in range(80):
        agent.update(before, 0, 1, following)
    assert agent.values(before)[0] > 0.9
    assert (1 - agent.values(before)[0]) ** 2 < initial_error / 20
    assert agent.last_loss is not None


def test_neural_checkpoint_resume_and_fixed_model_size(tmp_path):
    """Resume exactly and export a compact inference-only model."""
    agent = small_agent(seed=9)
    train(agent, 2)
    checkpoint = tmp_path / "nested" / "agent.pt"
    agent.save(checkpoint)
    resumed = NeuralAgent.load(checkpoint, for_training=True)
    assert train(agent, 2) == train(resumed, 2)
    assert agent.episodes == resumed.episodes == 4
    assert agent.steps == resumed.steps and agent.updates == resumed.updates
    assert agent.rng.getstate() == resumed.rng.getstate()
    assert list(agent.replay) == list(resumed.replay)
    assert all(
        torch.equal(value, resumed.online.state_dict()[key])
        for key, value in agent.online.state_dict().items()
    )
    assert all(
        torch.equal(value, resumed.target.state_dict()[key])
        for key, value in agent.target.state_dict().items()
    )
    model = tmp_path / "model.pt"
    agent.save(model, inference_only=True)
    original_size = model.stat().st_size
    restored = NeuralAgent.load(model)
    assert restored.values(GameState.new()) == agent.values(GameState.new())
    assert not restored.replay
    with pytest.raises(ValueError, match="Inference-only"):
        NeuralAgent.load(model, for_training=True)
    before, following = winning_transition()
    with pytest.raises(ValueError, match="Inference-only"):
        restored.update(before, 0, 1, following)
    train(agent, 2)
    agent.save(model, inference_only=True)
    assert abs(model.stat().st_size - original_size) < 4096
    assert model.stat().st_size < 150000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0},
        {"gamma": -1},
        {"epsilon": 2},
        {"batch_size": 0},
        {"replay_capacity": 2, "batch_size": 4},
        {"warmup": -1},
        {"train_every": 0},
        {"target_sync": 0},
        {"learning_rate": float("nan")},
        {"seed": "bad"},
        {"batch_size": 1.5},
        {"warmup": 5001},
    ],
)
def test_neural_invalid_configuration(kwargs):
    """Reject invalid learning settings and replay/update capacities."""
    with pytest.raises(ValueError):
        NeuralAgent(**kwargs)


def test_neural_invalid_transition_and_checkpoint(tmp_path):
    """Reject illegal experience and unsupported checkpoint versions."""
    agent = small_agent()
    before, following = winning_transition()
    for action, reward in [(7, 0), (0, float("nan")), (0, 2)]:
        with pytest.raises(ValueError):
            agent.update(before, action, reward, following)
    with pytest.raises(ValueError):
        agent.update(following, 0, 1, following)
    path = tmp_path / "bad.pt"
    torch.save({"version": 99, "algorithm": "bad"}, path)
    with pytest.raises(ValueError):
        NeuralAgent.load(path)


def test_neural_training_uses_only_rules(monkeypatch):
    """Neural self-play must not consult a heuristic evaluator."""

    def forbidden(*_args, **_kwargs):
        pytest.fail("Neural training must not use a heuristic teacher")

    monkeypatch.setattr("four_in_a_row.game.evaluate_state", forbidden)
    agent = small_agent()
    assert train(agent, 2).episodes == 2
    assert agent.updates > 0


@pytest.mark.parametrize(
    "changes",
    [
        {"position": "bad"},
        {"action": 7},
        {"legal": (False,)},
        {"terminal": False, "legal": (False,) * 7},
        {"legal": (1,) * 7},
        {"terminal": "bad"},
    ],
)
def test_invalid_replay_record(changes):
    """Validate replay structure independently of saved network tensors."""
    fields: dict[str, Any] = {
        "position": "0" * 42,
        "action": 0,
        "reward": 0.0,
        "following": "0" * 42,
        "legal": (True,) * 7,
        "terminal": False,
    }
    fields.update(changes)
    with pytest.raises(ValueError):
        Transition(**fields).validate()


@pytest.mark.parametrize(
    "change",
    [
        "training_ready",
        "episodes",
        "loss",
        "online_nan",
        "target_nan",
        "capacity",
        "replay_record",
        "shape",
        "missing",
        "unsafe_object",
    ],
)
def test_reject_malformed_neural_payload(tmp_path, change):
    """Reject corrupt metadata, tensors, replay, and unrestricted objects."""
    agent = small_agent()
    before, following = winning_transition()
    agent.update(before, 0, 1, following)
    path = tmp_path / "agent.pt"
    agent.save(path)
    data = torch.load(path, weights_only=True)
    if change == "training_ready":
        data["training_ready"] = "yes"
    elif change == "episodes":
        data["episodes"] = -1
    elif change == "loss":
        data["last_loss"] = float("nan")
    elif change in ("online_nan", "target_nan"):
        name = "online" if change == "online_nan" else "target"
        data[name][next(iter(data[name]))].fill_(float("nan"))
    elif change == "capacity":
        data["replay"] *= 100
    elif change == "replay_record":
        data["replay"][0]["position"] = "bad"
    elif change == "shape":
        data["online"][next(iter(data["online"]))] = torch.zeros(1)
    elif change == "missing":
        data = {}
    else:
        data = GameState.new()
    torch.save(data, path)
    with pytest.raises(ValueError):
        NeuralAgent.load(path)


def test_episode_outcomes_teach_both_players():
    """Every completed-game action gets a mover-relative outcome target."""
    agent = small_agent(gamma=1.0, outcome_weight=0.5)
    state = GameState.new()
    for column in [0, 1, 0, 1, 0, 1]:
        before = state.clone()
        state.apply_move(column)
        agent.update(before, column, 0, state)
    assert not agent.replay
    before = state.clone()
    state.apply_move(0)
    agent.update(before, 0, 1, state)
    assert len(agent.replay) == 14
    assert [item.return_target for item in list(agent.replay)[::2]] == [
        1,
        -1,
        1,
        -1,
        1,
        -1,
        1,
    ]
    assert not agent.episode
    assert agent.updates > 0


def test_discounted_episode_outcomes_and_legacy_mode():
    """Discount by remaining plies and preserve explicit TD-only updates."""
    agent = small_agent(gamma=0.9, outcome_weight=1)
    state = GameState.new()
    for column in [0, 1, 0, 1, 0, 1, 0]:
        before = state.clone()
        state.apply_move(column)
        agent.update(before, column, float(bool(state.winner())), state)
    returns = [item.return_target for item in list(agent.replay)[::2]]
    assert returns == pytest.approx(
        [
            0.9**6,
            -(0.9**5),
            0.9**4,
            -(0.9**3),
            0.9**2,
            -0.9,
            1,
        ]
    )
    legacy = small_agent(outcome_weight=0)
    empty = GameState.new()
    following = empty.clone()
    following.apply_move(0)
    legacy.update(empty, 0, 0, following)
    assert len(legacy.replay) == 2
    assert all(item.return_target is None for item in legacy.replay)


def test_outcome_resume_and_partial_episode_guard(tmp_path):
    """Persist outcome settings and reject ambiguous mid-game snapshots."""
    agent = small_agent(outcome_weight=0.75)
    train(agent, 2)
    path = tmp_path / "outcomes.pt"
    agent.save(path)
    restored = NeuralAgent.load(path, for_training=True)
    assert restored.outcome_weight == 0.75
    assert list(restored.replay) == list(agent.replay)
    assert train(agent, 2) == train(restored, 2)
    assert all(
        torch.equal(value, restored.online.state_dict()[key])
        for key, value in agent.online.state_dict().items()
    )
    before = GameState.new()
    following = before.clone()
    following.apply_move(0)
    agent.update(before, 0, 0, following)
    with pytest.raises(ValueError, match="episode"):
        agent.save(tmp_path / "partial.pt")
    agent.save(tmp_path / "partial-model.pt", inference_only=True)


def test_version_one_snapshot_preserves_td_only_training(tmp_path):
    """Load old replay and configuration without silently changing targets."""
    agent = small_agent(outcome_weight=0)
    train(agent, 2)
    path = tmp_path / "legacy.pt"
    agent.save(path)
    payload = torch.load(path, weights_only=True)
    payload["version"] = 1
    payload["config"].pop("outcome_weight")
    for item in payload["replay"]:
        item.pop("return_target")
    torch.save(payload, path)
    restored = NeuralAgent.load(path, for_training=True)
    assert restored.outcome_weight == 0
    assert list(restored.replay) == list(agent.replay)
    assert train(agent, 1) == train(restored, 1)


def test_invalid_outcome_settings_and_returns():
    """Reject mixing weights and return targets outside their valid range."""
    with pytest.raises(ValueError):
        small_agent(outcome_weight=2)
    transition = Transition(
        "0" * 42, 0, 0, "0" * 42, (True,) * 7, False, return_target=float("nan")
    )
    with pytest.raises(ValueError):
        transition.validate()


def test_outcome_collection_is_bounded_and_draw_returns_are_zero():
    """Bound pending history and label drawn games without false losses."""
    agent = small_agent(outcome_weight=0.5)
    state = GameState.new()
    following = state.clone()
    following.apply_move(0)
    for _ in range(42):
        agent.update(state, 0, 0, following)
    with pytest.raises(ValueError, match="42"):
        agent.update(state, 0, 0, following)
    agent.complete_episode(0)
    assert not agent.episode
    assert len(agent.replay) == 32
    assert all(item.return_target == 0 for item in agent.replay)


def test_full_episode_fits_winning_and_losing_moves():
    """Direct outcomes teach losses as well as terminal winning actions."""
    agent = NeuralAgent(
        learning_rate=0.005,
        gamma=1,
        outcome_weight=1,
        batch_size=14,
        warmup=14,
        replay_capacity=56,
        train_every=1,
    )
    for _ in range(60):
        state = GameState.new()
        for column in [0, 1, 0, 1, 0, 1, 0]:
            before = state.clone()
            state.apply_move(column)
            agent.update(before, column, float(bool(state.winner())), state)
    winning, _ = winning_transition()
    losing = GameState.new()
    for column in [0, 1, 0, 1, 0]:
        losing.apply_move(column)
    assert agent.values(winning)[0] > 0.8
    assert agent.values(losing)[1] < -0.8


def test_td_only_remains_default_until_strength_improves():
    """Do not promote an experimental mode without stronger-play evidence."""
    assert NeuralAgent().outcome_weight == 0
