"""Small CPU Double DQN with masked zero-sum targets and bounded replay."""

from __future__ import annotations

import copy
import math
import pickle
import random
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional

from .game import GameState
from .rl import QAgent


def encode_key(position: str) -> Tensor:
    """Expand a mover-relative board key into three one-hot planes."""
    return torch.tensor(
        [[cell == symbol for cell in position] for symbol in "120"],
        dtype=torch.float32,
    ).reshape(3, 6, 7)


def encode(state: GameState) -> Tensor:
    """Own, opponent, and empty planes, from the mover's perspective."""
    return encode_key(QAgent.key(state))


def double_dqn_targets(
    online_next: Tensor,
    target_next: Tensor,
    rewards: Tensor,
    legal: Tensor,
    terminal: Tensor,
    gamma: float,
) -> Tensor:
    """Select legal actions online and negate their frozen opponent values."""
    if torch.any(~terminal & ~legal.any(dim=1)):
        raise ValueError("Nonterminal transitions need a legal next action")
    with torch.no_grad():
        actions = online_next.masked_fill(~legal, -torch.inf).argmax(dim=1)
        opponent_values = target_next.gather(1, actions[:, None]).squeeze(1)
        return torch.where(terminal, rewards, rewards - gamma * opponent_values)


@dataclass(frozen=True)
class Transition:
    """One mover-relative experience with its next-action mask and outcome."""

    position: str
    action: int
    reward: float
    following: str
    legal: tuple[bool, ...]
    terminal: bool
    return_target: float | None = None

    def reflected(self) -> Transition:
        """Mirror both boards, the action, and the next-action mask."""
        indices = [
            row + column
            for row in range(0, 42, 7)
            for column in range(6, -1, -1)
        ]
        position = "".join(self.position[index] for index in indices)
        following = "".join(self.following[index] for index in indices)
        return Transition(
            position,
            6 - self.action,
            self.reward,
            following,
            self.legal[::-1],
            self.terminal,
            self.return_target,
        )

    def validate(self) -> None:
        """Validate replay keys, actions, rewards, masks, and outcomes."""
        if any(
            len(key) != 42 or not set(key) <= set("012")
            for key in (self.position, self.following)
        ):
            raise ValueError("Invalid replay position")
        if (
            not isinstance(self.action, int)
            or isinstance(self.action, bool)
            or not 0 <= self.action < 7
            or self.position[35 + self.action] != "0"
        ):
            raise ValueError("Invalid replay action")
        if not math.isfinite(self.reward) or not -1 <= self.reward <= 1:
            raise ValueError("Invalid replay reward")
        if self.return_target is not None and (
            not math.isfinite(self.return_target)
            or not -1 <= self.return_target <= 1
        ):
            raise ValueError("Invalid episode return")
        if (
            len(self.legal) != 7
            or any(not isinstance(value, bool) for value in self.legal)
            or not isinstance(self.terminal, bool)
            or (not self.terminal and not any(self.legal))
        ):
            raise ValueError("Invalid replay legal mask or outcome")


@dataclass
class NeuralAgent:
    """Fixed-size CPU Double DQN with bounded replay and exact local resume."""

    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 0.2
    seed: int = 0
    batch_size: int = 64
    replay_capacity: int = 5000
    warmup: int = 128
    train_every: int = 4
    target_sync: int = 200
    outcome_weight: float = 0.0
    episodes: int = field(default=0, init=False)
    steps: int = field(default=0, init=False)
    updates: int = field(default=0, init=False)
    last_loss: float | None = field(default=None, init=False)
    training_ready: bool = field(default=True, init=False)
    rng: random.Random = field(init=False, repr=False)
    online: nn.Sequential = field(init=False, repr=False)
    target: nn.Sequential = field(init=False, repr=False)
    optimizer: torch.optim.Adam = field(init=False, repr=False)
    replay: deque[Transition] = field(init=False, repr=False)
    episode: list[tuple[int, Transition]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not 0 <= self.gamma <= 1 or not 0 <= self.epsilon <= 1:
            raise ValueError("gamma and epsilon must be in [0, 1]")
        if not 0 <= self.outcome_weight <= 1:
            raise ValueError("outcome_weight must be in [0, 1]")
        counts = (
            self.batch_size,
            self.replay_capacity,
            self.train_every,
            self.target_sync,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in counts
        ):
            raise ValueError(
                "Batch, capacity, and update periods must be positive"
            )
        if (
            not isinstance(self.warmup, int)
            or isinstance(self.warmup, bool)
            or not 0 <= self.warmup <= self.replay_capacity
            or self.batch_size > self.replay_capacity
        ):
            raise ValueError("Replay must fit batch_size and warmup")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        self.rng = random.Random(self.seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.online = nn.Sequential(
                nn.Flatten(),
                nn.Linear(126, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 7),
                nn.Tanh(),
            )
        self.target = copy.deepcopy(self.online)
        self.target.requires_grad_(False)
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=self.learning_rate
        )
        self.replay = deque(maxlen=self.replay_capacity)

    @property
    def parameter_count(self) -> int:
        """Return the fixed number of online-network trainable parameters."""
        return sum(parameter.numel() for parameter in self.online.parameters())

    def values(self, state: GameState) -> list[float]:
        """Predict mover-relative action values without creating experience."""
        with torch.no_grad():
            result: list[float] = self.online(encode(state)[None])[0].tolist()
            return result

    def select(self, state: GameState, *, epsilon: float | None = None) -> int:
        """Choose only legal actions using exploration or predicted values."""
        if state.is_terminal():
            raise ValueError("Cannot select a move in a terminal state")
        exploration = self.epsilon if epsilon is None else epsilon
        if not 0 <= exploration <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        legal = state.legal_moves()
        if self.rng.random() < exploration:
            return self.rng.choice(legal)
        values = self.values(state)
        best = max(values[column] for column in legal)
        return self.rng.choice(
            [column for column in legal if values[column] == best]
        )

    def update(
        self,
        state: GameState,
        action: int,
        reward: float,
        following: GameState,
    ) -> None:
        """Store an observed transition and mirror, then update on schedule."""
        if not self.training_ready:
            raise ValueError("Inference-only checkpoint cannot resume training")
        if state.is_terminal() or action not in state.legal_moves():
            raise ValueError("Learning action must be legal in an active state")
        legal = following.legal_moves()
        transition = Transition(
            QAgent.key(state),
            action,
            reward,
            QAgent.key(following),
            tuple(column in legal for column in range(7)),
            following.is_terminal(),
        )
        transition.validate()
        if self.outcome_weight:
            if len(self.episode) >= 42:
                raise ValueError("An episode cannot exceed 42 moves")
            self.episode.append((state.current_player, transition))
            if transition.terminal:
                self.complete_episode(following.winner())
        else:
            self.replay.extend((transition, transition.reflected()))
        self.steps += 1
        if (
            len(self.replay) >= max(self.batch_size, self.warmup)
            and self.steps % self.train_every == 0
        ):
            self.optimize()

    def complete_episode(self, winner: int) -> None:
        """Label all actions from game outcomes without tactical teachers."""
        for index, (player, transition) in enumerate(self.episode):
            outcome = 0.0 if not winner else (1.0 if player == winner else -1.0)
            remaining = len(self.episode) - index - 1
            labelled = replace(
                transition, return_target=outcome * self.gamma**remaining
            )
            self.replay.extend((labelled, labelled.reflected()))
        self.episode.clear()

    def optimize(self) -> float:
        """Fit one replay minibatch and periodically synchronize targets."""
        batch = self.rng.sample(list(self.replay), self.batch_size)
        positions = torch.stack([encode_key(item.position) for item in batch])
        following = torch.stack([encode_key(item.following) for item in batch])
        actions = torch.tensor([item.action for item in batch])
        rewards = torch.tensor([item.reward for item in batch])
        legal = torch.tensor([item.legal for item in batch])
        terminal = torch.tensor([item.terminal for item in batch])
        with torch.no_grad():
            targets = double_dqn_targets(
                self.online(following),
                self.target(following),
                rewards,
                legal,
                terminal,
                self.gamma,
            )
            returns = torch.tensor(
                [
                    (
                        item.return_target
                        if item.return_target is not None
                        else float("nan")
                    )
                    for item in batch
                ]
            )
            targets = torch.where(
                torch.isfinite(returns),
                (1 - self.outcome_weight) * targets
                + self.outcome_weight * returns,
                targets,
            )
        predicted = (
            self.online(positions).gather(1, actions[:, None]).squeeze(1)
        )
        loss = functional.smooth_l1_loss(predicted, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.last_loss = float(loss.detach())
        return self.last_loss

    def save(self, path: str | Path, *, inference_only: bool = False) -> None:
        """Atomically save a training snapshot or compact playing model."""
        if self.episode and not inference_only:
            raise ValueError("Save training snapshots at episode boundaries")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        config = {
            name: getattr(self, name)
            for name in (
                "learning_rate",
                "gamma",
                "epsilon",
                "seed",
                "batch_size",
                "replay_capacity",
                "warmup",
                "train_every",
                "target_sync",
                "outcome_weight",
            )
        }
        payload = {
            "version": 2,
            "algorithm": "masked-negamax-double-dqn",
            "config": config,
            "online": self.online.state_dict(),
            "episodes": self.episodes,
            "steps": self.steps,
            "updates": self.updates,
            "last_loss": self.last_loss,
            "training_ready": self.training_ready and not inference_only,
            "rng": self.rng.getstate(),
        }
        if payload["training_ready"]:
            payload.update(
                target=self.target.state_dict(),
                optimizer=self.optimizer.state_dict(),
                replay=[asdict(item) for item in self.replay],
            )
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)

    @classmethod
    def load(
        cls, path: str | Path, *, for_training: bool = False
    ) -> NeuralAgent:
        """Load tensors with restricted, CPU-only deserialization."""
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (
                payload["version"] not in (1, 2)
                or payload["algorithm"] != "masked-negamax-double-dqn"
            ):
                raise ValueError("Unsupported neural checkpoint format")
            ready = payload["training_ready"]
            if not isinstance(ready, bool):
                raise ValueError("Invalid training_ready flag")
            if for_training and not ready:
                raise ValueError(
                    "Inference-only checkpoint cannot resume training"
                )
            config = dict(payload["config"])
            if payload["version"] == 1:
                config.setdefault("outcome_weight", 0.0)
            agent = cls(**config)
            agent.online.load_state_dict(payload["online"])
            if any(
                not torch.isfinite(parameter).all()
                for parameter in agent.online.parameters()
            ):
                raise ValueError("Non-finite network parameters")
            for name in ("episodes", "steps", "updates"):
                value = payload[name]
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                ):
                    raise ValueError("Invalid neural training counter")
                setattr(agent, name, value)
            agent.last_loss = payload["last_loss"]
            if agent.last_loss is not None and (
                not math.isfinite(agent.last_loss) or agent.last_loss < 0
            ):
                raise ValueError("Invalid training loss")
            agent.rng.setstate(payload["rng"])
            agent.training_ready = ready
            if ready:
                agent.target.load_state_dict(payload["target"])
                if any(
                    not torch.isfinite(parameter).all()
                    for parameter in agent.target.parameters()
                ):
                    raise ValueError("Non-finite target network parameters")
                agent.optimizer.load_state_dict(payload["optimizer"])
                records = payload["replay"]
                if len(records) > agent.replay_capacity:
                    raise ValueError(
                        "Checkpoint replay exceeds configured capacity"
                    )
                for record in records:
                    record["legal"] = tuple(record["legal"])
                    transition = Transition(**record)
                    transition.validate()
                    agent.replay.append(transition)
            else:
                agent.target.load_state_dict(agent.online.state_dict())
            return agent
        except (
            KeyError,
            TypeError,
            RuntimeError,
            OverflowError,
            pickle.UnpicklingError,
            EOFError,
        ) as error:
            raise ValueError("Invalid neural checkpoint") from error
