# Requirements: FR-PLAY-001, FR-EVAL-001, FR-RL-004, FR-RL-005, NFR-ENV-001
# Requirements: FR-NEURAL-001, FR-NN-SAVE-001
# Requirements: FR-OUTCOME-001, FR-TACTIC-001
"""Application and subprocess end-to-end tests for RL workflows."""

import json
import random
import runpy
import subprocess
import sys
from itertools import cycle
from pathlib import Path

import pytest
import torch

from four_in_a_row.game import GameState
from four_in_a_row.rl import QAgent
from four_in_a_row.rl_cli import evaluate, human_move, main, make_player, match
from four_in_a_row.rl_nn import NeuralAgent


def test_train_evaluate_play_e2e(tmp_path):
    """Exercise legacy training, resume, evaluation, and play subprocesses."""
    assert Path(sys.prefix).name == ".venv"
    assert Path("uv.lock").is_file()
    checkpoint = tmp_path / "agent.json"
    commands = [
        [
            "train",
            "--algorithm",
            "tabular",
            "--episodes",
            "4",
            "--checkpoint",
            str(checkpoint),
        ],
        [
            "train",
            "--episodes",
            "2",
            "--resume",
            str(checkpoint),
            "--checkpoint",
            str(checkpoint),
        ],
        ["evaluate", "--games", "4", "--checkpoint", str(checkpoint)],
        ["play", "--p1", "rl", "--p2", "rl", "--checkpoint", str(checkpoint)],
    ]
    for arguments in commands:
        result = subprocess.run(
            [sys.executable, "-m", "four_in_a_row.rl_cli", *arguments],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip()
    assert QAgent.load(checkpoint).episodes == 6


def test_evaluation_is_balanced_and_frozen(tmp_path, capsys):
    """Use both seats equally without altering the source JSON checkpoint."""
    checkpoint = tmp_path / "agent.json"
    QAgent(seed=9).save(checkpoint)
    before = checkpoint.read_bytes()
    assert (
        main(["evaluate", "--games", "4", "--checkpoint", str(checkpoint)]) == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["wins"] + report["losses"] + report["draws"] == 4
    assert report["games_as_first"] == report["games_as_second"] == 2
    assert 0 <= report["score"] <= 1
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize(
    "players", [("human", "human"), ("human", "random"), ("random", "human")]
)
def test_human_combinations(monkeypatch, capsys, players):
    """Allow human players in either or both seats."""
    columns = cycle(str(column) for column in range(1, 8))
    monkeypatch.setattr("builtins.input", lambda _: next(columns))
    assert main(["play", "--p1", players[0], "--p2", players[1]]) == 0
    assert "Winner" in capsys.readouterr().out or players != ("human", "human")


@pytest.mark.parametrize(
    "arguments",
    [
        ["train", "--episodes", "0"],
        ["evaluate", "--games", "3"],
        ["play", "--p1", "rl"],
        ["train", "--epsilon", "2"],
        ["evaluate"],
        ["play", "--strength", "0"],
        ["train", "--resume", "does-not-exist.json"],
    ],
)
def test_invalid_cli_arguments(arguments, capsys):
    """Report semantic CLI errors without writing training artifacts."""
    assert main(arguments) == 2
    assert "error:" in capsys.readouterr().err


def test_in_process_train_resume_and_module_entrypoint(
    tmp_path, capsys, monkeypatch
):
    """Preserve legacy resume and module entry-point exit status."""
    checkpoint = str(tmp_path / "agent.json")
    assert (
        main(
            [
                "train",
                "--algorithm",
                "tabular",
                "--episodes",
                "2",
                "--checkpoint",
                checkpoint,
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "train",
                "--episodes",
                "2",
                "--resume",
                checkpoint,
                "--checkpoint",
                checkpoint,
            ]
        )
        == 0
    )
    assert QAgent.load(checkpoint).episodes == 4
    monkeypatch.setattr(
        sys, "argv", ["rl_cli", "play", "--p1", "random", "--p2", "random"]
    )
    monkeypatch.delitem(sys.modules, "four_in_a_row.rl_cli", raising=False)
    with pytest.raises(SystemExit) as result:
        runpy.run_module("four_in_a_row.rl_cli", run_name="__main__")
    assert result.value.code == 0
    assert capsys.readouterr().out


def test_human_input_recovery_and_interrupt(monkeypatch, capsys):
    """Retry invalid columns and allow a human to interrupt a match."""
    responses = iter(["nonsense", "8", "2"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    assert human_move(GameState.new()) == 1
    assert "not legal" in capsys.readouterr().out
    monkeypatch.setattr("builtins.input", lambda _: "quit")
    assert main(["play", "--p1", "human", "--p2", "human"]) == 0
    assert "interrupted" in capsys.readouterr().out


def test_evaluation_outcomes_and_in_memory_freeze(monkeypatch):
    """Count outcomes and preserve the learner's table, counters, and RNG."""
    outcomes = iter([0, 2, 2, 0])
    monkeypatch.setattr(
        "four_in_a_row.rl_cli.match", lambda *args: next(outcomes)
    )
    agent = QAgent(seed=9)
    before = agent.rng.getstate()
    result = evaluate(agent, 4)
    assert (result["wins"], result["losses"], result["draws"]) == (1, 1, 2)
    assert result["score"] == 0.5
    interval = result["win_rate_ci95"]
    assert isinstance(interval, list)
    assert interval[0] < 0.25 < interval[1]
    assert not agent.table and agent.episodes == 0
    assert agent.rng.getstate() == before
    with pytest.raises(ValueError):
        evaluate(agent, 4, opponent="human")
    with pytest.raises(ValueError):
        make_player("unknown", None, random.Random(0), 1)


@pytest.mark.parametrize("opponent", ["alphabeta", "mcts"])
def test_search_opponents_are_available(opponent):
    """Expose the existing search engines as frozen evaluation opponents."""
    result = evaluate(QAgent(), 2, opponent=opponent, strength=1)
    assert result["games"] == 2


def test_match_draw_rendering(monkeypatch, capsys):
    """Render a drawn match without requesting a move."""
    state = GameState.new()
    monkeypatch.setattr(state, "is_terminal", lambda: True)
    monkeypatch.setattr(GameState, "new", lambda: state)
    assert match(lambda _: 0, lambda _: 0, render=True) == 0
    assert "Draw" in capsys.readouterr().out


def test_different_seat_checkpoints_and_missing_checkpoint(tmp_path, capsys):
    """Load separate seat agents and report absent checkpoint files."""
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    QAgent(seed=1).save(first)
    QAgent(seed=2).save(second)
    assert (
        main(
            [
                "play",
                "--p1",
                "rl",
                "--p2",
                "rl",
                "--p1-checkpoint",
                str(first),
                "--p2-checkpoint",
                str(second),
            ]
        )
        == 0
    )
    assert "Winner" in capsys.readouterr().out
    assert main(["evaluate", "--checkpoint", "does-not-exist.json"]) == 2


def test_nn_cli_e2e(tmp_path):
    """Exercise neural training, export, resume, evaluation, and AI play."""
    checkpoint = tmp_path / "agent.pt"
    model = tmp_path / "model.pt"
    commands = [
        [
            "train",
            "--episodes",
            "3",
            "--batch-size",
            "4",
            "--warmup",
            "4",
            "--replay-capacity",
            "32",
            "--train-every",
            "1",
            "--target-sync",
            "3",
            "--checkpoint",
            str(checkpoint),
            "--export-model",
            str(model),
        ],
        [
            "train",
            "--episodes",
            "1",
            "--resume",
            str(checkpoint),
            "--checkpoint",
            str(checkpoint),
        ],
        ["evaluate", "--games", "4", "--checkpoint", str(model)],
        ["play", "--p1", "rl", "--p2", "rl", "--checkpoint", str(model)],
    ]
    for arguments in commands:
        result = subprocess.run(
            [sys.executable, "-m", "four_in_a_row.rl_cli", *arguments],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip()
    agent = NeuralAgent.load(checkpoint, for_training=True)
    assert agent.episodes == 4 and agent.updates > 0
    assert model.stat().st_size < 150000


def test_neural_cli_default_export_and_guards(tmp_path, monkeypatch, capsys):
    """Protect existing snapshots and exports unless overwrite is explicit."""
    monkeypatch.chdir(tmp_path)
    model = "model.pt"
    assert main(["train", "--episodes", "1", "--export-model", model]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["algorithm"] == "neural"
    assert report["parameters"] < 30000
    assert Path("checkpoints/nn_agent.pt").exists()
    checkpoint = Path("checkpoints/nn_agent.pt")
    original = checkpoint.read_bytes()
    assert main(["train", "--episodes", "1"]) == 2
    assert checkpoint.read_bytes() == original
    assert (
        main(
            [
                "train",
                "--episodes",
                "1",
                "--checkpoint",
                "other.pt",
                "--export-model",
                model,
            ]
        )
        == 2
    )
    assert main(["train", "--episodes", "1", "--resume", model]) == 2
    assert main(["train", "--episodes", "1", "--checkpoint", "bad.json"]) == 2
    assert (
        main(
            [
                "train",
                "--episodes",
                "1",
                "--checkpoint",
                model,
                "--export-model",
                model,
            ]
        )
        == 2
    )
    assert (
        main(["train", "--algorithm", "tabular", "--export-model", model]) == 2
    )
    assert (
        main(
            [
                "train",
                "--episodes",
                "1",
                "--overwrite",
                "--export-model",
                model,
            ]
        )
        == 0
    )
    assert main(["train", "--algorithm", "tabular", "--episodes", "1"]) == 0


def test_neural_evaluation_preserves_weights_and_replay():
    """Frozen evaluation cannot change weights, replay, updates, or RNG."""
    torch.set_num_threads(1)
    agent = NeuralAgent(batch_size=4, warmup=4, replay_capacity=32)
    before = {
        key: value.clone() for key, value in agent.online.state_dict().items()
    }
    rng = agent.rng.getstate()
    result = evaluate(agent, 2)
    assert result["games"] == 2
    assert all(
        torch.equal(value, agent.online.state_dict()[key])
        for key, value in before.items()
    )
    assert not agent.replay and agent.updates == 0
    assert agent.rng.getstate() == rng


def test_missing_neural_extra_has_actionable_error(
    tmp_path, monkeypatch, capsys
):
    """Explain how to install optional neural dependencies when unavailable."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert main(["train", "--episodes", "1"]) == 2
    assert "uv sync --extra rl --dev" in capsys.readouterr().err


def test_tactical_benchmark_cli_and_outcome_resume(tmp_path, capsys):
    """Expose frozen diagnostics and explicit outcome-mode upgrades."""
    path = tmp_path / "agent.pt"
    upgraded = tmp_path / "outcomes.pt"
    agent = NeuralAgent(
        outcome_weight=0,
        batch_size=4,
        warmup=4,
        replay_capacity=32,
        train_every=1,
    )
    agent.save(path)
    initial = path.read_bytes()
    assert (
        main(
            [
                "benchmark",
                "--checkpoint",
                str(path),
                "--seeds",
                "2000",
                "2001",
                "--untrained",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert len(report["policies"]) == 2
    assert all(len(policy["runs"]) == 2 for policy in report["policies"])
    assert path.read_bytes() == initial
    assert (
        main(
            [
                "train",
                "--episodes",
                "2",
                "--resume",
                str(path),
                "--checkpoint",
                str(upgraded),
                "--outcome-weight",
                "0.75",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["outcome_weight"] == 0.75
    assert summary["outcome_weight_previous"] == 0
    assert NeuralAgent.load(upgraded).outcome_weight == 0.75
    assert path.read_bytes() == initial
    assert (
        main(
            [
                "benchmark",
                "--checkpoint",
                str(upgraded),
                "--compare",
                str(path),
                "--seeds",
                "2002",
            ]
        )
        == 0
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["benchmark"],
        ["train", "--outcome-weight", "2"],
        ["train", "--algorithm", "tabular", "--outcome-weight", "0.5"],
    ],
)
def test_invalid_strength_workflow_arguments(arguments, capsys):
    """Reject missing models and invalid or inapplicable target settings."""
    assert main(arguments) == 2
    assert "error:" in capsys.readouterr().err
