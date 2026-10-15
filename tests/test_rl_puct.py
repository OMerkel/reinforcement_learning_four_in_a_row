# Requirements: FR-PUCT-001, FR-PUCT-SAVE-001
# Requirements: FR-POOL-001
# Requirements: NFR-EVAL-DIVERSITY-001
"""Policy/value prediction, PUCT search, self-play, and resume contracts."""

import json
import math

import pytest
import torch

from four_in_a_row.game import GameState
from four_in_a_row.rl import QAgent
from four_in_a_row.rl_cli import evaluate, load_agent, main, match
from four_in_a_row.rl_puct import (
    PolicySample,
    PUCTAgent,
    SearchNode,
    train_puct,
)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    """Keep small CPU network tests efficient and repeatable."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_agent(**kwargs):
    """Build a low-budget agent for search and optimizer tests."""
    return PUCTAgent(
        simulations=16,
        batch_size=4,
        warmup=4,
        replay_capacity=32,
        updates_per_episode=1,
        **kwargs,
    )


def test_policy_value_shapes_and_legal_priors():
    """Return seven masked probabilities and one mover-relative value."""
    agent = small_agent(seed=7)
    state = GameState.new()
    for _ in range(6):
        state.apply_move(0)
    policy, value = agent.predict(state)
    assert len(policy) == 7 and policy[0] == 0
    assert sum(policy) == pytest.approx(1)
    assert all(probability >= 0 for probability in policy)
    assert -1 <= value <= 1
    assert agent.parameter_count < 30000
    assert small_agent(seed=7).predict(state) == (policy, value)


def test_puct_explores_under_visited_prior():
    """Reproduce the guide's exploration versus exploitation example."""
    first = SearchNode(GameState.new(), prior=0.5, visits=50, total_value=-30)
    second = SearchNode(GameState.new(), prior=0.3, visits=5, total_value=-1)
    assert first.score(100, 1.5) == pytest.approx(0.7470588235)
    assert second.score(100, 1.5) == pytest.approx(0.95)
    assert second.score(100, 1.5) > first.score(100, 1.5)
    assert SearchNode(GameState.new()).mean_value == 0


def test_search_visits_and_terminal_backup_find_a_win(monkeypatch):
    """Exact winning outcomes back up positively to the previous mover."""
    agent = small_agent(seed=2)
    monkeypatch.setattr(
        agent,
        "predict",
        lambda state: (
            [
                (
                    1 / len(state.legal_moves())
                    if column in state.legal_moves()
                    else 0
                )
                for column in range(7)
            ],
            0.0,
        ),
    )
    state = GameState.new()
    for action in [0, 1, 0, 1, 0, 2]:
        state.apply_move(action)
    before = state.clone()
    result = agent.search(state, simulations=64)
    assert result.move == 0
    assert sum(result.visits) == 64
    assert sum(result.policy) == pytest.approx(1)
    assert result.value > 0
    assert state == before
    state.apply_move(0)
    with pytest.raises(ValueError, match="terminal"):
        agent.search(state)


def test_search_blocks_opponent_and_masks_full_columns(monkeypatch):
    """Lookahead rejects immediate losses and never expands full columns."""
    agent = small_agent(seed=1)
    monkeypatch.setattr(
        agent,
        "predict",
        lambda state: (
            [
                (
                    1 / len(state.legal_moves())
                    if column in state.legal_moves()
                    else 0
                )
                for column in range(7)
            ],
            0.0,
        ),
    )
    state = GameState.new()
    for action in [0, 1, 2, 1, 3, 1]:
        state.apply_move(action)
    assert agent.search(state, simulations=256).move == 1
    filled = GameState.new()
    for _ in range(6):
        filled.apply_move(0)
    result = agent.search(filled)
    assert result.visits[0] == result.policy[0] == 0
    assert result.move != 0


def test_self_play_labels_and_exact_resume(tmp_path, monkeypatch):
    """Persist learned heads, optimizer, bounded replay, and local RNG."""

    def forbidden(*_args, **_kwargs):
        pytest.fail("PUCT must not use upstream heuristic evaluation")

    monkeypatch.setattr("four_in_a_row.game.evaluate_state", forbidden)
    agent = small_agent(seed=9)
    assert train_puct(agent, 2).episodes == 2
    assert len(agent.replay) <= 32 and agent.updates > 0
    assert all(sample.outcome in (-1, 0, 1) for sample in agent.replay)
    path = tmp_path / "agent.pt"
    agent.save(path)
    resumed = PUCTAgent.load(path, for_training=True)
    assert train_puct(agent, 2) == train_puct(resumed, 2)
    assert list(agent.replay) == list(resumed.replay)
    assert agent.rng.getstate() == resumed.rng.getstate()
    assert all(
        torch.equal(value, resumed.network.state_dict()[key])
        for key, value in agent.network.state_dict().items()
    )
    model = tmp_path / "model.pt"
    agent.save(model, inference_only=True)
    restored = PUCTAgent.load(model)
    assert restored.predict(GameState.new()) == agent.predict(GameState.new())
    assert model.stat().st_size < 150000
    with pytest.raises(ValueError, match="Inference-only"):
        train_puct(restored, 1)


def test_reflection_and_tiny_policy_value_fit():
    """Fit search visits and positive or negative game outcomes separately."""
    agent = small_agent(learning_rate=0.01)
    state = GameState.new()
    state.apply_move(0)
    sample = PolicySample(
        QAgent.key(state), (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0), -1.0
    )
    mirror = sample.reflected()
    assert mirror.policy[4] == 1 and mirror.outcome == -1
    agent.replay.extend([sample, mirror] * 2)
    for _ in range(80):
        agent.optimize()
    policy, value = agent.predict(state)
    assert policy[2] > 0.9 and value < -0.9
    assert agent.last_policy_loss is not None
    assert agent.last_value_loss is not None


def test_training_noise_sampling_and_network_only_selection():
    """Separate root noise and visit sampling from frozen policy selection."""
    agent = small_agent(seed=4)
    result = agent.search(GameState.new(), temperature=1, add_noise=True)
    assert result.move in range(7)
    assert sum(result.policy) == pytest.approx(1)
    agent.use_search = False
    assert agent.select(GameState.new(), epsilon=0) in range(7)
    assert agent.select(GameState.new(), epsilon=1) in range(7)
    with pytest.raises(ValueError):
        agent.select(GameState.new(), epsilon=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"simulations": 0},
        {"c_puct": -1},
        {"learning_rate": math.nan},
        {"batch_size": 0},
        {"replay_capacity": 1, "batch_size": 4},
        {"warmup": -1},
        {"noise_fraction": 2},
        {"dirichlet_alpha": 0},
    ],
)
def test_invalid_puct_configuration(kwargs):
    """Reject invalid budgets, priors, and optimizer/replay settings."""
    with pytest.raises(ValueError):
        PUCTAgent(**kwargs)


def test_puct_cli_train_evaluate_export_and_play(tmp_path, capsys):
    """Expose search-assisted and network-only checkpoint workflows."""
    checkpoint = tmp_path / "puct.pt"
    model = tmp_path / "model.pt"
    assert (
        main(
            [
                "train",
                "--algorithm",
                "puct",
                "--episodes",
                "2",
                "--simulations",
                "8",
                "--batch-size",
                "4",
                "--warmup",
                "4",
                "--replay-capacity",
                "32",
                "--checkpoint",
                str(checkpoint),
                "--export-model",
                str(model),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["algorithm"] == "puct"
    assert report["simulations"] == 8
    assert isinstance(load_agent(model), PUCTAgent)
    for flags in (["--simulations", "16"], ["--no-search"]):
        assert (
            main(
                ["evaluate", "--checkpoint", str(model), "--games", "2", *flags]
            )
            == 0
        )
        result = json.loads(capsys.readouterr().out)
        assert result["search_simulations"] == (
            16 if "--simulations" in flags else 0
        )
    assert (
        main(
            [
                "play",
                "--p1",
                "rl",
                "--p2",
                "random",
                "--checkpoint",
                str(model),
                "--simulations",
                "8",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out
    assert (
        main(
            [
                "benchmark",
                "--checkpoint",
                str(model),
                "--untrained",
                "--no-search",
                "--seeds",
                "9000",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert len(result["policies"]) == 2
    assert main(["train", "--resume", str(checkpoint), "--episodes", "1"]) == 0
    assert PUCTAgent.load(checkpoint).episodes == 3


@pytest.mark.parametrize(
    "change",
    [
        "version",
        "flag",
        "counter",
        "loss",
        "nan",
        "capacity",
        "sample",
        "missing",
    ],
)
def test_reject_invalid_puct_checkpoint(tmp_path, change):
    """Validate checkpoint metadata, weights, capacity, and replay labels."""
    agent = small_agent()
    train_puct(agent, 1)
    path = tmp_path / "bad.pt"
    agent.save(path)
    data = torch.load(path, weights_only=True)
    if change == "version":
        data["version"] = 9
    elif change == "flag":
        data["training_ready"] = "bad"
    elif change == "counter":
        data["episodes"] = -1
    elif change == "loss":
        data["last_loss"] = math.nan
    elif change == "nan":
        data["network"][next(iter(data["network"]))].fill_(math.nan)
    elif change == "capacity":
        data["replay"] *= 100
    elif change == "sample":
        data["replay"][0]["position"] = "bad"
    else:
        data = {}
    torch.save(data, path)
    with pytest.raises(ValueError):
        PUCTAgent.load(path)


@pytest.mark.parametrize(
    "sample",
    [
        PolicySample("bad", (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 1),
        PolicySample("0" * 42, (0.0,) * 7, 1),
        PolicySample("0" * 42, (1.0,) + (0.0,) * 6, math.nan),
        PolicySample("1" * 42, (1.0,) + (0.0,) * 6, 1),
    ],
)
def test_reject_invalid_policy_samples(sample):
    """Never train on malformed distributions, illegal columns, or values."""
    with pytest.raises(ValueError):
        sample.validate()


def test_draw_search_and_training_boundaries(monkeypatch):
    """Use exact zero values for draws and reject impossible search calls."""
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
    before = draw.clone()
    before.board[5][6] = 0
    agent = small_agent()
    assert agent.search(before).value == 0
    assert agent.search(before).move == 6
    for kwargs in ({"simulations": 0}, {"temperature": -1}):
        with pytest.raises(ValueError):
            agent.search(GameState.new(), **kwargs)
    with pytest.raises(ValueError):
        agent.predict(draw)
    with pytest.raises(ValueError):
        agent.select(draw)
    with pytest.raises(ValueError):
        train_puct(agent, 0)
    monkeypatch.setattr(GameState, "new", lambda **_kwargs: draw.clone())
    assert train_puct(agent, 1).draws == 1


def test_puct_cli_rejects_wrong_controls_and_corruption(tmp_path, capsys):
    """Reject backend mismatches and malformed tensor inputs cleanly."""
    legacy = tmp_path / "q.json"
    QAgent().save(legacy)
    assert (
        main(
            [
                "evaluate",
                "--checkpoint",
                str(legacy),
                "--games",
                "2",
                "--no-search",
            ]
        )
        == 2
    )
    agent_path = tmp_path / "puct.pt"
    small_agent().save(agent_path)
    assert (
        main(
            [
                "evaluate",
                "--checkpoint",
                str(agent_path),
                "--games",
                "2",
                "--simulations",
                "0",
            ]
        )
        == 2
    )
    assert (
        main(["train", "--resume", str(agent_path), "--outcome-weight", "0.5"])
        == 2
    )
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"bad tensor data")
    assert main(["evaluate", "--checkpoint", str(bad), "--games", "2"]) == 2
    assert "error:" in capsys.readouterr().err


def test_historical_pool_freezes_and_caps_archived_weights():
    """Archived policies cannot track optimizer changes or grow forever."""
    agent = small_agent(
        opponent_pool_size=2, snapshot_interval=1, historical_fraction=1
    )
    agent.archive_policy()
    archived = {
        key: value.clone() for key, value in agent.opponent_pool[0].items()
    }
    train_puct(agent, 1)
    assert len(agent.opponent_pool) == 2
    assert agent.historical_games == 1
    assert all(
        torch.equal(value, agent.opponent_pool[0][key])
        for key, value in archived.items()
    )
    assert any(
        not torch.equal(value, agent.network.state_dict()[key])
        for key, value in archived.items()
    )
    opponent = agent.sample_opponent()
    assert opponent is not None and not opponent.training_ready
    assert opponent.rng is agent.rng
    assert all(
        not value.requires_grad for value in opponent.network.parameters()
    )
    train_puct(agent, 2)
    assert len(agent.opponent_pool) == 2
    assert agent.historical_games == 3


def test_historical_pool_resume_is_exact(tmp_path):
    """Resume reproduces opponent sampling, game outcomes, and training."""
    agent = small_agent(
        seed=17,
        opponent_pool_size=3,
        snapshot_interval=2,
        historical_fraction=0.5,
    )
    train_puct(agent, 3)
    path = tmp_path / "history.pt"
    agent.save(path)
    restored = PUCTAgent.load(path, for_training=True)
    assert len(restored.opponent_pool) == len(agent.opponent_pool)
    assert train_puct(agent, 3) == train_puct(restored, 3)
    assert agent.historical_games == restored.historical_games
    assert agent.rng.getstate() == restored.rng.getstate()
    assert list(agent.replay) == list(restored.replay)
    assert all(
        torch.equal(value, restored.network.state_dict()[key])
        for key, value in agent.network.state_dict().items()
    )
    model = tmp_path / "model.pt"
    agent.save(model, inference_only=True)
    assert not PUCTAgent.load(model).opponent_pool


def test_disabled_pool_and_invalid_history_settings():
    """Keep old self-play unchanged unless a valid pool is enabled."""
    agent = small_agent()
    agent.archive_policy()
    assert agent.sample_opponent() is None
    assert not agent.opponent_pool
    for settings in (
        {"opponent_pool_size": -1},
        {"snapshot_interval": 0},
        {"historical_fraction": 2},
    ):
        with pytest.raises(ValueError):
            small_agent(**settings)


def test_pool_cli_and_resume_controls(tmp_path, capsys):
    """Retain recorded pool settings and history across CLI resume."""
    path = tmp_path / "pool.pt"
    assert (
        main(
            [
                "train",
                "--algorithm",
                "puct",
                "--episodes",
                "2",
                "--simulations",
                "8",
                "--batch-size",
                "4",
                "--warmup",
                "4",
                "--replay-capacity",
                "32",
                "--opponent-pool-size",
                "2",
                "--snapshot-interval",
                "1",
                "--historical-fraction",
                "1",
                "--checkpoint",
                str(path),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["historical_games"] == 2
    assert summary["opponent_pool_entries"] == 2
    assert main(["train", "--resume", str(path), "--episodes", "1"]) == 0
    assert PUCTAgent.load(path).historical_games == 3
    assert (
        main(
            [
                "train",
                "--resume",
                str(path),
                "--episodes",
                "1",
                "--opponent-pool-size",
                "0",
            ]
        )
        == 0
    )
    assert not PUCTAgent.load(path).opponent_pool


def test_opening_evaluation_is_paired_and_frozen():
    """Evaluate both seats on identical legal prefixes without training."""
    agent = small_agent()
    agent.use_search = False
    rng = agent.rng.getstate()
    report = evaluate(agent, 4, seed=41000, opening_plies=4)
    assert report["opening_plies"] == 4
    assert report["opening_moves"][0] == report["opening_moves"][1]
    assert report["opening_moves"][2] == report["opening_moves"][3]
    assert all(len(moves) == 4 for moves in report["opening_moves"])
    assert agent.rng.getstate() == rng and not agent.replay
    for plies in (-1, 7):
        with pytest.raises(ValueError):
            evaluate(agent, 2, opening_plies=plies)


def test_version_one_puct_snapshot_keeps_original_self_play(tmp_path):
    """Do not silently enable historical opponents for existing snapshots."""
    agent = small_agent()
    train_puct(agent, 1)
    path = tmp_path / "legacy.pt"
    agent.save(path)
    data = torch.load(path, weights_only=True)
    data["version"] = 1
    data.pop("opponent_pool")
    data.pop("historical_games")
    for field in (
        "opponent_pool_size",
        "snapshot_interval",
        "historical_fraction",
    ):
        data["config"].pop(field)
    torch.save(data, path)
    restored = PUCTAgent.load(path, for_training=True)
    assert restored.opponent_pool_size == 0
    assert train_puct(agent, 2) == train_puct(restored, 2)
    assert all(
        torch.equal(value, restored.network.state_dict()[key])
        for key, value in agent.network.state_dict().items()
    )


@pytest.mark.parametrize("change", ["counter", "capacity", "weights"])
def test_corrupted_historical_pool_is_rejected(tmp_path, change):
    """Validate historical counters, pool limits, and frozen tensor values."""
    agent = small_agent(opponent_pool_size=1)
    agent.archive_policy()
    path = tmp_path / "history.pt"
    agent.save(path)
    data = torch.load(path, weights_only=True)
    if change == "counter":
        data["historical_games"] = 100
    elif change == "capacity":
        data["opponent_pool"] *= 2
    else:
        weights = data["opponent_pool"][0]
        weights[next(iter(weights))].fill_(math.nan)
    torch.save(data, path)
    with pytest.raises(ValueError):
        PUCTAgent.load(path)


def test_history_training_does_not_read_diagnostic_labels(monkeypatch):
    """Historical self-play remains rules-only despite new tactical fixtures."""

    def forbidden(*_args, **_kwargs):
        pytest.fail("Training must not use tactical benchmark answers")

    monkeypatch.setattr("four_in_a_row.rl_benchmark.tactical_cases", forbidden)
    agent = small_agent(opponent_pool_size=2, historical_fraction=1)
    assert train_puct(agent, 2).episodes == 2


def test_wrong_pool_controls_and_opening_prefixes_are_rejected(tmp_path):
    """Reject pool options for Q agents and post-game evaluation prefixes."""
    path = tmp_path / "q.json"
    QAgent().save(path)
    assert (
        main(
            [
                "train",
                "--resume",
                str(path),
                "--episodes",
                "1",
                "--opponent-pool-size",
                "2",
            ]
        )
        == 2
    )
    with pytest.raises(ValueError):
        match(lambda _: 0, lambda _: 1, opening_moves=(0, 1, 0, 1, 0, 1, 0, 2))
