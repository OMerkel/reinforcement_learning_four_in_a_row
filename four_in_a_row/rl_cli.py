"""Training, frozen evaluation, and independently configured player seats."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import random
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeIs

from .engines import suggest_move
from .game import GameState
from .rl import QAgent, train
from .rl_benchmark import benchmark

if TYPE_CHECKING:
    from .rl_nn import NeuralAgent
    from .rl_puct import PUCTAgent

type Agent = QAgent | NeuralAgent | PUCTAgent

Player = Callable[[GameState], int]
PLAYER_KINDS = ("human", "rl", "random", "alphabeta", "mcts")


def neural_class() -> type[NeuralAgent]:
    """Load the optional CPU backend without requiring it for tabular play."""
    try:
        torch = import_module("torch")
    except ImportError as error:
        raise ValueError(
            "Neural RL requires: uv sync --extra rl --dev"
        ) from error
    torch.set_num_threads(1)
    agent_class: type[NeuralAgent] = import_module(
        ".rl_nn", __package__
    ).NeuralAgent
    return agent_class


def load_agent(path: str | Path, *, for_training: bool = False) -> Agent:
    """Detect and validate a legacy JSON or restricted tensor checkpoint."""
    with Path(path).open("rb") as stream:
        tabular = stream.read(1) == b"{"
    if tabular:
        return QAgent.load(path)
    backend = neural_class()
    try:
        payload = import_module("torch").load(
            path, weights_only=True, map_location="cpu"
        )
        if (
            isinstance(payload, dict)
            and payload.get("algorithm") == "puct-policy-value"
        ):
            return puct_class().load(path, for_training=for_training)
    except (
        RuntimeError,
        TypeError,
        IndexError,
        pickle.UnpicklingError,
        EOFError,
    ) as error:
        raise ValueError("Invalid tensor checkpoint") from error
    return backend.load(path, for_training=for_training)


def puct_class() -> type[PUCTAgent]:
    """Load the optional policy/value backend after checking CPU PyTorch."""
    neural_class()
    agent_class: type[PUCTAgent] = import_module(
        ".rl_puct", __package__
    ).PUCTAgent
    return agent_class


def is_puct(agent: Agent) -> TypeIs[PUCTAgent]:
    """Distinguish the new backend without importing Torch for JSON agents."""
    return not isinstance(agent, QAgent) and isinstance(agent, puct_class())


def configure_search(
    agent: Agent,
    simulations: int | None,
    use_search: bool = True,
) -> None:
    """Apply explicit PUCT inference controls without altering saved files."""
    if is_puct(agent):
        if simulations is not None:
            if simulations < 1:
                raise ValueError("simulations must be positive")
            agent.simulations = simulations
        agent.use_search = use_search
    elif simulations is not None or not use_search:
        raise ValueError("Search controls require a PUCT policy/value agent")


def human_move(state: GameState) -> int:
    """Prompt until the human chooses a legal column or exits."""
    while True:
        text = input(
            f"Player {state.current_player}, column 1..7 (quit to exit): "
        )
        if text.strip().lower() == "quit":
            raise KeyboardInterrupt
        try:
            column = int(text) - 1
        except ValueError:
            print("Enter a column number.")
            continue
        if column in state.legal_moves():
            return column
        print("That column is not legal.")


def make_player(
    kind: str,
    checkpoint: str | None,
    rng: random.Random,
    strength: int,
    *,
    simulations: int | None = None,
    use_search: bool = True,
) -> Player:
    """Construct an independently configured human, learned, or search seat."""
    if strength < 1:
        raise ValueError("strength must be positive")
    if kind == "human":
        return human_move
    if kind == "rl":
        if not checkpoint:
            raise ValueError(
                "An RL player requires --checkpoint or a seat checkpoint"
            )
        agent = load_agent(checkpoint)
        configure_search(agent, simulations, use_search)
        agent.rng = rng
        return lambda state: agent.select(state, epsilon=0)
    if kind not in PLAYER_KINDS:
        raise ValueError(f"Unknown player: {kind}")
    return lambda state: (
        suggest_move(
            state,
            kind,
            depth=strength,
            iterations=strength,
            rng=rng,
        ).move
    )


def match(
    first: Player,
    second: Player,
    *,
    render: bool = False,
    opening_moves: Sequence[int] = (),
) -> int:
    """Play one complete match and return the winning token or zero."""
    state = GameState.new()
    for column in opening_moves:
        if state.is_terminal():
            raise ValueError("Opening prefix must not contain post-game moves")
        state.apply_move(column)
    while not state.is_terminal():
        if render:
            print()
            print(state.render())
        player = first if state.current_player == 1 else second
        state.apply_move(player(state))
    if render:
        print()
        print(state.render())
        print(f"Winner: player {state.winner()}" if state.winner() else "Draw")
    return state.winner()


def evaluate(
    agent: Agent,
    games: int,
    *,
    opponent: str = "random",
    strength: int = 2,
    seed: int = 1000,
    opening_plies: int = 0,
) -> dict[str, Any]:
    """Measure a frozen policy with balanced seats and win-rate uncertainty."""
    if games < 2 or games % 2:
        raise ValueError("games must be a positive even number (at least 2)")
    if (
        not isinstance(opening_plies, int)
        or isinstance(opening_plies, bool)
        or not 0 <= opening_plies <= 6
    ):
        raise ValueError("opening_plies must be an integer in [0, 6]")
    if opponent not in ("random", "alphabeta", "mcts"):
        raise ValueError(
            "Evaluation opponent must be random, alphabeta, or mcts"
        )
    frozen = copy.deepcopy(agent)
    started = time.perf_counter()
    frozen.rng = random.Random(seed)
    rival = make_player(opponent, None, random.Random(seed + 1), strength)
    opening_rng = random.Random(seed + 2)
    openings: list[list[int]] = []
    opening: list[int] = []

    def learner(state: GameState) -> int:
        return frozen.select(state, epsilon=0)

    wins = losses = draws = 0
    for index in range(games):
        seat = 1 + index % 2
        if opening_plies:
            if index % 2 == 0:
                prefix = GameState.new()
                for _ in range(opening_plies):
                    prefix.apply_move(opening_rng.choice(prefix.legal_moves()))
                opening = prefix.move_stack.copy()
            openings.append(opening.copy())
            first, second = (learner, rival) if seat == 1 else (rival, learner)
            winner = match(first, second, opening_moves=opening)
        else:
            winner = (
                match(learner, rival) if seat == 1 else match(rival, learner)
            )
        if winner == 0:
            draws += 1
        elif winner == seat:
            wins += 1
        else:
            losses += 1
    rate = wins / games
    z_score = 1.959963984540054
    denominator = 1 + z_score**2 / games
    center = (rate + z_score**2 / (2 * games)) / denominator
    half_width = (
        z_score
        * math.sqrt(rate * (1 - rate) / games + z_score**2 / (4 * games**2))
        / denominator
    )
    return {
        "opening_plies": opening_plies,
        "opening_moves": openings,
        "search_simulations": (
            frozen.simulations if is_puct(frozen) and frozen.use_search else 0
        ),
        "selection_mode": (
            "puct"
            if is_puct(frozen) and frozen.use_search
            else "network" if is_puct(frozen) else "q"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "games": games,
        "games_as_first": games // 2,
        "games_as_second": games // 2,
        "opponent": opponent,
        "strength": strength,
        "seed": seed,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": rate,
        "win_rate_ci95": [
            max(0.0, center - half_width),
            min(1.0, center + half_width),
        ],
        "score": (wins + 0.5 * draws) / games,
    }


def parser() -> argparse.ArgumentParser:
    """Define training, evaluation, and player configuration options."""
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument(
        "--algorithm", choices=("neural", "tabular", "puct"), default="neural"
    )
    training.add_argument("--episodes", type=int, default=1000)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--alpha", type=float, default=0.2)
    training.add_argument("--gamma", type=float, default=1.0)
    training.add_argument("--epsilon", type=float, default=0.2)
    training.add_argument("--resume")
    training.add_argument("--checkpoint")
    training.add_argument("--learning-rate", type=float, default=0.001)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--replay-capacity", type=int, default=5000)
    training.add_argument("--warmup", type=int, default=128)
    training.add_argument("--train-every", type=int, default=4)
    training.add_argument("--target-sync", type=int, default=200)
    training.add_argument("--outcome-weight", type=float)
    training.add_argument("--simulations", type=int)
    training.add_argument("--c-puct", type=float, default=1.5)
    training.add_argument("--updates-per-episode", type=int, default=4)
    training.add_argument("--opponent-pool-size", type=int)
    training.add_argument("--snapshot-interval", type=int)
    training.add_argument("--historical-fraction", type=float)
    training.add_argument("--export-model")
    training.add_argument("--overwrite", action="store_true")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint")
    evaluation.add_argument("--games", type=int, default=100)
    evaluation.add_argument("--seed", type=int, default=1000)
    evaluation.add_argument(
        "--opponent", choices=PLAYER_KINDS[2:], default="random"
    )
    evaluation.add_argument("--strength", type=int, default=2)
    evaluation.add_argument("--opening-plies", type=int, default=0)
    diagnostics = commands.add_parser("benchmark")
    diagnostics.add_argument("--checkpoint")
    diagnostics.add_argument("--compare", nargs="*", default=[])
    diagnostics.add_argument(
        "--seeds", nargs="+", type=int, default=[2000, 2001, 2002]
    )
    diagnostics.add_argument("--untrained", action="store_true")
    playing = commands.add_parser("play")
    playing.add_argument("--p1", choices=PLAYER_KINDS, default="human")
    playing.add_argument("--p2", choices=PLAYER_KINDS, default="random")
    playing.add_argument("--checkpoint")
    playing.add_argument("--p1-checkpoint")
    playing.add_argument("--p2-checkpoint")
    playing.add_argument("--seed", type=int, default=0)
    playing.add_argument("--strength", type=int, default=2)
    for command in (evaluation, diagnostics, playing):
        command.add_argument("--simulations", type=int)
        command.add_argument("--no-search", action="store_true")
    return root


def benchmark_checkpoints(arguments: argparse.Namespace) -> dict[str, Any]:
    """Compare immutable checkpoints and optional cold policies over seeds."""
    if not arguments.checkpoint:
        raise ValueError("Benchmark requires --checkpoint")
    policies = []
    for path in [arguments.checkpoint, *arguments.compare]:
        agent = load_agent(path)
        configure_search(agent, arguments.simulations, not arguments.no_search)
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        variants = [("checkpoint", agent)]
        if arguments.untrained:
            cold: Agent = (
                puct_class()(
                    seed=agent.seed,
                    simulations=agent.simulations,
                    c_puct=agent.c_puct,
                    use_search=agent.use_search,
                )
                if is_puct(agent)
                else type(agent)(seed=agent.seed)
            )
            variants.append(("untrained", cold))
        for kind, policy in variants:
            runs = [benchmark(policy, seed=seed) for seed in arguments.seeds]
            policies.append(
                {
                    "checkpoint": path,
                    "model_sha256": digest,
                    "kind": kind,
                    "search_simulations": (
                        policy.simulations
                        if is_puct(policy) and policy.use_search
                        else 0
                    ),
                    "initialization_seed": policy.seed,
                    "runs": runs,
                    "mean_accuracy": sum(run["accuracy"] for run in runs)
                    / len(runs),
                }
            )
    return {"policies": policies}


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one CLI workflow and return a shell-friendly status code."""
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "train":
            if arguments.outcome_weight is not None and not (
                0 <= arguments.outcome_weight <= 1
            ):
                raise ValueError("outcome_weight must be in [0, 1]")
            agent: Agent
            outcome_previous = None
            if arguments.resume:
                agent = load_agent(arguments.resume, for_training=True)
            elif arguments.algorithm == "tabular":
                agent = QAgent(
                    alpha=arguments.alpha,
                    gamma=arguments.gamma,
                    epsilon=arguments.epsilon,
                    seed=arguments.seed,
                )
            elif arguments.algorithm == "puct":
                agent = puct_class()(
                    seed=arguments.seed,
                    learning_rate=arguments.learning_rate,
                    simulations=(
                        arguments.simulations
                        if arguments.simulations is not None
                        else 64
                    ),
                    c_puct=arguments.c_puct,
                    batch_size=arguments.batch_size,
                    replay_capacity=arguments.replay_capacity,
                    warmup=arguments.warmup,
                    updates_per_episode=arguments.updates_per_episode,
                )
            else:
                agent = neural_class()(
                    learning_rate=arguments.learning_rate,
                    gamma=arguments.gamma,
                    epsilon=arguments.epsilon,
                    seed=arguments.seed,
                    batch_size=arguments.batch_size,
                    replay_capacity=arguments.replay_capacity,
                    warmup=arguments.warmup,
                    train_every=arguments.train_every,
                    target_sync=arguments.target_sync,
                    outcome_weight=(
                        arguments.outcome_weight
                        if arguments.outcome_weight is not None
                        else 0.0
                    ),
                )
            tabular = isinstance(agent, QAgent)
            pool_options = (
                arguments.opponent_pool_size,
                arguments.snapshot_interval,
                arguments.historical_fraction,
            )
            if any(value is not None for value in pool_options):
                if not is_puct(agent):
                    raise ValueError("Historical pool settings require PUCT")
                agent.configure_pool(
                    (
                        arguments.opponent_pool_size
                        if arguments.opponent_pool_size is not None
                        else agent.opponent_pool_size
                    ),
                    (
                        arguments.snapshot_interval
                        if arguments.snapshot_interval is not None
                        else agent.snapshot_interval
                    ),
                    (
                        arguments.historical_fraction
                        if arguments.historical_fraction is not None
                        else agent.historical_fraction
                    ),
                )
            if arguments.outcome_weight is not None:
                if isinstance(agent, QAgent) or is_puct(agent):
                    raise ValueError("--outcome-weight requires a DQN agent")
                if arguments.resume:
                    outcome_previous = agent.outcome_weight
                    agent.outcome_weight = arguments.outcome_weight
            checkpoint = (
                arguments.checkpoint
                or arguments.resume
                or (
                    "checkpoints/q_agent.json"
                    if tabular
                    else (
                        "checkpoints/puct_agent.pt"
                        if is_puct(agent)
                        else "checkpoints/nn_agent.pt"
                    )
                )
            )
            destination = Path(checkpoint)
            if not tabular and destination.suffix.lower() == ".json":
                raise ValueError(
                    "Use a .pt path for neural checkpoints, not .json"
                )
            if (
                destination.exists()
                and not arguments.overwrite
                and (
                    not arguments.resume
                    or destination.resolve() != Path(arguments.resume).resolve()
                )
            ):
                raise ValueError(
                    "Checkpoint exists; use --resume or --overwrite"
                )
            if arguments.export_model:
                if tabular:
                    raise ValueError("--export-model requires a neural agent")
                if (
                    Path(arguments.export_model).resolve()
                    == destination.resolve()
                ):
                    raise ValueError(
                        "Model export must not replace the resume checkpoint"
                    )
                if (
                    Path(arguments.export_model).exists()
                    and not arguments.overwrite
                ):
                    raise ValueError("Model export exists; use --overwrite")
            if is_puct(agent):
                if arguments.simulations is not None:
                    configure_search(agent, arguments.simulations)
                stats = import_module(".rl_puct", __package__).train_puct(
                    agent, arguments.episodes
                )
            else:
                stats = train(agent, arguments.episodes)
            agent.save(destination)
            summary: dict[str, int | float | str | None] = {
                **asdict(stats),
                "total_episodes": agent.episodes,
                "checkpoint": str(destination),
                "checkpoint_bytes": destination.stat().st_size,
                "algorithm": (
                    "tabular"
                    if tabular
                    else "puct" if is_puct(agent) else "neural"
                ),
            }
            if isinstance(agent, QAgent):
                summary["states"] = len(agent.table)
            else:
                summary.update(
                    parameters=agent.parameter_count,
                    replay_size=len(agent.replay),
                    updates=agent.updates,
                    loss=agent.last_loss,
                )
                if is_puct(agent):
                    summary.update(
                        simulations=agent.simulations,
                        policy_loss=agent.last_policy_loss,
                        value_loss=agent.last_value_loss,
                        historical_games=agent.historical_games,
                        opponent_pool_entries=len(agent.opponent_pool),
                        opponent_pool_size=agent.opponent_pool_size,
                        snapshot_interval=agent.snapshot_interval,
                        historical_fraction=agent.historical_fraction,
                    )
                else:
                    summary["outcome_weight"] = agent.outcome_weight
                if outcome_previous is not None:
                    summary["outcome_weight_previous"] = outcome_previous
                if arguments.export_model:
                    agent.save(arguments.export_model, inference_only=True)
                    summary["model"] = arguments.export_model
                    summary["model_bytes"] = (
                        Path(arguments.export_model).stat().st_size
                    )
            print(json.dumps(summary))
        elif arguments.command == "evaluate":
            if not arguments.checkpoint:
                raise ValueError("Evaluation requires --checkpoint")
            agent = load_agent(arguments.checkpoint)
            configure_search(
                agent, arguments.simulations, not arguments.no_search
            )
            report = evaluate(
                agent,
                arguments.games,
                opponent=arguments.opponent,
                strength=arguments.strength,
                seed=arguments.seed,
                opening_plies=arguments.opening_plies,
            )
            report["model_sha256"] = hashlib.sha256(
                Path(arguments.checkpoint).read_bytes(),
            ).hexdigest()
            print(json.dumps(report))
        elif arguments.command == "benchmark":
            print(json.dumps(benchmark_checkpoints(arguments)))
        else:
            first = make_player(
                arguments.p1,
                arguments.p1_checkpoint or arguments.checkpoint,
                random.Random(arguments.seed),
                arguments.strength,
                simulations=arguments.simulations,
                use_search=not arguments.no_search,
            )
            second = make_player(
                arguments.p2,
                arguments.p2_checkpoint or arguments.checkpoint,
                random.Random(arguments.seed + 1),
                arguments.strength,
                simulations=arguments.simulations,
                use_search=not arguments.no_search,
            )
            match(first, second, render=True)
        return 0
    except (KeyboardInterrupt, EOFError):
        print("Game interrupted.")
        return 0
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
