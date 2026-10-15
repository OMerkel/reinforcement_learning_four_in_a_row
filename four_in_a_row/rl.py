"""Tabular negamax Q-learning from rules and self-play,
without handcrafted heuristics.

The important distinction is handcrafted heuristics
versus learned behavior."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .game import GameState


@dataclass
class QAgent:
    """Player-relative action values, with zero values for unseen positions."""

    alpha: float = 0.2
    gamma: float = 1.0
    epsilon: float = 0.2
    seed: int = 0
    episodes: int = 0
    table: dict[str, list[float]] = field(default_factory=dict)
    rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= self.gamma <= 1 or not 0 <= self.epsilon <= 1:
            raise ValueError("gamma and epsilon must be in [0, 1]")
        self.rng = random.Random(self.seed)

    @staticmethod
    def key(state: GameState) -> str:
        """Encode empty/self/opponent cells as 0/1/2, bottom row first."""
        return "".join(
            "0" if cell == 0 else "1" if cell == state.current_player else "2"
            for row in state.board
            for cell in row
        )

    def values(self, state: GameState) -> list[float]:
        """Return a copy of action values without inserting unseen positions."""
        return self.table.get(self.key(state), [0.0] * 7).copy()

    def select(self, state: GameState, *, epsilon: float | None = None) -> int:
        """Choose a legal epsilon-greedy action with random tie-breaking."""
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
        """Update one action using a terminal or negated opponent target."""
        if state.is_terminal() or action not in state.legal_moves():
            raise ValueError(
                "Learning action must be legal in an active state"
            )
        if not math.isfinite(reward) or not -1 <= reward <= 1:
            raise ValueError("Reward must be finite and in [-1, 1]")
        target = reward
        if not following.is_terminal():
            values = self.values(following)
            target -= self.gamma * max(
                values[col] for col in following.legal_moves()
            )
        current = self.table.setdefault(self.key(state), [0.0] * 7)
        current[action] += self.alpha * (target - current[action])

    def save(self, path: str | Path) -> None:
        """Atomically persist values, parameters, counters, and RNG state."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "algorithm": "tabular-negamax-q",
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "seed": self.seed,
            "episodes": self.episodes,
            "table": self.table,
            "rng": self.rng.getstate(),
        }
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(data, allow_nan=False), encoding="utf-8"
        )
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> QAgent:
        """Load versioned JSON, never executable pickle data."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if (
                data["version"] != 1
                or data["algorithm"] != "tabular-negamax-q"
            ):
                raise ValueError("Unsupported checkpoint format")
            agent = cls(
                alpha=data["alpha"],
                gamma=data["gamma"],
                epsilon=data["epsilon"],
                seed=data["seed"],
            )
            episodes = data["episodes"]
            if (
                not isinstance(episodes, int)
                or isinstance(episodes, bool)
                or episodes < 0
            ):
                raise ValueError("Invalid episode count")
            table = data["table"]
            if not isinstance(table, dict):
                raise ValueError("Invalid action-value table")
            for key, values in table.items():
                if len(key) != 42 or not set(key) <= set("012"):
                    raise ValueError("Invalid position encoding")
                if not isinstance(values, list) or len(values) != 7:
                    raise ValueError("Expected seven action values")
                if any(
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or not -1 <= value <= 1
                    for value in values
                ):
                    raise ValueError("Invalid action value")
            version, state, gaussian = data["rng"]
            agent.rng.setstate((version, tuple(state), gaussian))
            agent.table = table
            agent.episodes = episodes
            return agent
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError("Invalid checkpoint") from error


@dataclass(frozen=True)
class TrainingStats:
    """Completed episode count and outcomes by token color."""

    episodes: int
    player_one_wins: int
    player_two_wins: int
    draws: int


class Learner(Protocol):
    """Policy and update contract shared by tabular and neural trainers."""

    episodes: int

    def select(
        self, state: GameState, *, epsilon: float | None = None
    ) -> int:
        """Choose an action in the current position."""

    def update(
        self,
        state: GameState,
        action: int,
        reward: float,
        following: GameState,
    ) -> None:
        """Learn from an observed move and resulting position."""


def train(agent: Learner, episodes: int) -> TrainingStats:
    """Alternate the starting token; learn after each self-play transition."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    outcomes = [0, 0, 0]
    for _ in range(episodes):
        state = GameState.new(starting_player=1 + agent.episodes % 2)
        while not state.is_terminal():
            before = state.clone()
            action = agent.select(state)
            state.apply_move(action)
            reward = 1.0 if state.winner() == before.current_player else 0.0
            agent.update(before, action, reward, state)
        outcomes[state.winner()] += 1
        agent.episodes += 1
    return TrainingStats(episodes, outcomes[1], outcomes[2], outcomes[0])
