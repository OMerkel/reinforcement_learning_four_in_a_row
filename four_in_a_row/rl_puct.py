"""Policy/value self-play with legal, mover-relative PUCT tree search."""

from __future__ import annotations

import math
import pickle
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar

import torch
from torch import Tensor, nn
from torch.nn import functional

from .game import GameState
from .rl import QAgent, TrainingStats
from .rl_nn import encode, encode_key


class PolicyValueNetwork(nn.Module):
    """Shared board features with seven policy logits and one bounded value."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(126, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.policy = nn.Linear(64, 7)
        self.value = nn.Linear(64, 1)

    def forward(self, boards: Tensor) -> tuple[Tensor, Tensor]:
        """Return batched logits and mover-relative value predictions."""
        features = self.encoder(boards)
        return self.policy(features), torch.tanh(self.value(features)).squeeze(
            1
        )


@dataclass
class SearchNode:
    """Search statistics expressed from this node's mover perspective."""

    state: GameState
    prior: float = 1.0
    visits: int = 0
    total_value: float = 0.0
    children: dict[int, SearchNode] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        """Return the observed mean or the neutral unvisited convention."""
        return self.total_value / self.visits if self.visits else 0.0

    def score(self, parent_visits: int, exploration: float) -> float:
        """Combine the parent's exploitation value with a prior bonus."""
        bonus = exploration * self.prior * math.sqrt(max(1, parent_visits))
        return -self.mean_value + bonus / (1 + self.visits)

    def expand(self, policy: list[float]) -> None:
        """Create only legal child boards with normalized policy priors."""
        for action in self.state.legal_moves():
            following = self.state.clone()
            following.apply_move(action)
            self.children[action] = SearchNode(following, prior=policy[action])


@dataclass(frozen=True)
class SearchResult:
    """Chosen move, raw visit policy, per-column counts, and root value."""

    move: int
    policy: tuple[float, ...]
    visits: tuple[int, ...]
    value: float


@dataclass(frozen=True)
class PolicySample:
    """Search visit targets and the completed game's mover-relative outcome."""

    position: str
    policy: tuple[float, ...]
    outcome: float

    def reflected(self) -> PolicySample:
        """Mirror board and policy columns without changing the outcome."""
        indices = [
            row + column
            for row in range(0, 42, 7)
            for column in range(6, -1, -1)
        ]
        return PolicySample(
            "".join(self.position[index] for index in indices),
            self.policy[::-1],
            self.outcome,
        )

    def validate(self) -> None:
        """Reject malformed boards, probability targets, and value labels."""
        if len(self.position) != 42 or not set(self.position) <= set("012"):
            raise ValueError("Invalid policy sample position")
        if (
            len(self.policy) != 7
            or any(
                not math.isfinite(value) or not 0 <= value <= 1
                for value in self.policy
            )
            or not math.isclose(sum(self.policy), 1, abs_tol=1e-5)
        ):
            raise ValueError("Invalid policy sample probabilities")
        if any(
            self.position[35 + action] != "0" and probability > 0
            for action, probability in enumerate(self.policy)
        ):
            raise ValueError("Policy assigns probability to an illegal column")
        if not math.isfinite(self.outcome) or not -1 <= self.outcome <= 1:
            raise ValueError("Invalid policy sample outcome")


@dataclass
class PUCTAgent:
    """Small CPU policy/value learner with independently budgeted PUCT."""

    algorithm: ClassVar[str] = "puct-policy-value"
    seed: int = 0
    learning_rate: float = 0.001
    simulations: int = 64
    c_puct: float = 1.5
    batch_size: int = 32
    replay_capacity: int = 2000
    warmup: int = 64
    updates_per_episode: int = 4
    dirichlet_alpha: float = 0.3
    noise_fraction: float = 0.25
    exploration_moves: int = 8
    weight_decay: float = 0.0001
    use_search: bool = True
    opponent_pool_size: int = 0
    snapshot_interval: int = 100
    historical_fraction: float = 0.5
    episodes: int = field(default=0, init=False)
    updates: int = field(default=0, init=False)
    historical_games: int = field(default=0, init=False)
    training_ready: bool = field(default=True, init=False)
    last_loss: float | None = field(default=None, init=False)
    last_policy_loss: float | None = field(default=None, init=False)
    last_value_loss: float | None = field(default=None, init=False)
    rng: random.Random = field(init=False, repr=False)
    network: PolicyValueNetwork = field(init=False, repr=False)
    optimizer: torch.optim.Adam = field(init=False, repr=False)
    replay: deque[PolicySample] = field(init=False, repr=False)
    opponent_pool: list[dict[str, Tensor]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.configure_pool(
            self.opponent_pool_size,
            self.snapshot_interval,
            self.historical_fraction,
        )
        counts = (
            self.simulations,
            self.batch_size,
            self.replay_capacity,
            self.updates_per_episode,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in counts
        ):
            raise ValueError("PUCT budgets and replay counts must be positive")
        if (
            not isinstance(self.warmup, int)
            or self.warmup < 0
            or max(self.warmup, self.batch_size) > self.replay_capacity
            or not isinstance(self.exploration_moves, int)
            or self.exploration_moves < 0
        ):
            raise ValueError("Invalid warmup or exploratory move count")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (self.learning_rate, self.c_puct, self.dirichlet_alpha)
        ):
            raise ValueError("Learning rate and exploration must be positive")
        if not 0 <= self.noise_fraction <= 1 or not 0 <= self.weight_decay:
            raise ValueError("Invalid root noise or weight decay")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        self.rng = random.Random(self.seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.network = PolicyValueNetwork()
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.replay = deque(maxlen=self.replay_capacity)

    def configure_pool(self, size: int, interval: int, fraction: float) -> None:
        """Record valid history settings without resetting learned weights."""
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("Pool size must be a nonnegative integer")
        if (
            not isinstance(interval, int)
            or isinstance(interval, bool)
            or interval < 1
        ):
            raise ValueError("Snapshot interval must be a positive integer")
        if not 0 <= fraction <= 1:
            raise ValueError("Historical fraction must be in [0, 1]")
        self.opponent_pool_size = size
        self.snapshot_interval = interval
        self.historical_fraction = fraction
        self.opponent_pool = self.opponent_pool[-size:] if size else []

    def archive_policy(self) -> None:
        """Archive frozen weights without claiming strength promotion."""
        if not self.opponent_pool_size:
            return
        self.opponent_pool.append(
            {
                name: value.detach().clone()
                for name, value in self.network.state_dict().items()
            }
        )
        capacity = self.opponent_pool_size
        self.opponent_pool = self.opponent_pool[-capacity:]

    def sample_opponent(self) -> PUCTAgent | None:
        """Sample a frozen actor with the trainer's checkpointed RNG."""
        if not self.opponent_pool_size:
            return None
        if not self.opponent_pool:
            self.archive_policy()
        if self.rng.random() >= self.historical_fraction:
            return None
        weights = self.rng.choice(self.opponent_pool)
        frozen = PUCTAgent(
            seed=self.seed, simulations=self.simulations, c_puct=self.c_puct
        )
        frozen.network.load_state_dict(weights)
        frozen.network.requires_grad_(False)
        frozen.network.eval()
        frozen.rng = self.rng
        frozen.training_ready = False
        return frozen

    @property
    def parameter_count(self) -> int:
        """Return the fixed-size two-head network's parameter count."""
        return sum(parameter.numel() for parameter in self.network.parameters())

    def predict(self, state: GameState) -> tuple[list[float], float]:
        """Predict a legal policy prior and nonterminal mover-relative value."""
        if state.is_terminal():
            raise ValueError("Predict only nonterminal positions")
        with torch.no_grad():
            logits, values = self.network(encode(state)[None])
            legal = state.legal_moves()
            mask = torch.tensor([action in legal for action in range(7)])
            probabilities: list[float] = torch.softmax(
                logits[0].masked_fill(~mask, -torch.inf), dim=0
            ).tolist()
            return probabilities, float(values[0])

    def search(
        self,
        state: GameState,
        *,
        simulations: int | None = None,
        temperature: float = 0,
        add_noise: bool = False,
    ) -> SearchResult:
        """Run fresh bounded PUCT and choose from the root visit counts."""
        if state.is_terminal():
            raise ValueError("Cannot search a terminal position")
        budget = self.simulations if simulations is None else simulations
        if (
            not isinstance(budget, int)
            or isinstance(budget, bool)
            or budget < 1
        ):
            raise ValueError("simulations must be a positive integer")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        root = SearchNode(state.clone())
        policy, _ = self.predict(state)
        root.expand(policy)
        if add_noise and self.noise_fraction:
            noise = [
                self.rng.gammavariate(self.dirichlet_alpha, 1)
                for _ in root.children
            ]
            total = sum(noise)
            for child, value in zip(root.children.values(), noise, strict=True):
                child.prior = (
                    1 - self.noise_fraction
                ) * child.prior + self.noise_fraction * value / total
        for _ in range(budget):
            node = root
            path = [node]
            while node.children:
                scores = {
                    action: child.score(node.visits, self.c_puct)
                    for action, child in node.children.items()
                }
                best = max(scores.values())
                action = self.rng.choice(
                    [
                        action
                        for action, score in scores.items()
                        if score == best
                    ]
                )
                node = node.children[action]
                path.append(node)
            if node.state.is_terminal():
                winner = node.state.winner()
                value = (
                    0.0
                    if not winner
                    else 1.0 if winner == node.state.current_player else -1.0
                )
            else:
                priors, value = self.predict(node.state)
                node.expand(priors)
            for visited in reversed(path):
                visited.visits += 1
                visited.total_value += value
                value = -value
        visits = tuple(
            root.children[action].visits if action in root.children else 0
            for action in range(7)
        )
        visit_policy = tuple(count / budget for count in visits)
        if temperature == 0:
            largest = max(visits)
            move = self.rng.choice(
                [
                    action
                    for action, count in enumerate(visits)
                    if count == largest
                ]
            )
        else:
            largest = max(visits)
            weights = [
                (count / largest) ** (1 / temperature) for count in visits
            ]
            move = self.rng.choices(range(7), weights=weights)[0]
        return SearchResult(move, visit_policy, visits, root.mean_value)

    def select(self, state: GameState, *, epsilon: float | None = None) -> int:
        """Select through PUCT or use the unaided policy head explicitly."""
        exploration = 0.0 if epsilon is None else epsilon
        if not 0 <= exploration <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        if state.is_terminal():
            raise ValueError("Cannot select from a terminal position")
        if exploration and self.rng.random() < exploration:
            return self.rng.choice(state.legal_moves())
        if self.use_search:
            return self.search(state).move
        policy, _ = self.predict(state)
        best = max(policy)
        return self.rng.choice(
            [action for action in state.legal_moves() if policy[action] == best]
        )

    def optimize(self) -> float:
        """Fit a minibatch of visit policies and final game outcomes."""
        batch = self.rng.sample(list(self.replay), self.batch_size)
        boards = torch.stack([encode_key(sample.position) for sample in batch])
        policies = torch.tensor([sample.policy for sample in batch])
        outcomes = torch.tensor([sample.outcome for sample in batch])
        logits, values = self.network(boards)
        legal = torch.tensor(
            [[cell == "0" for cell in sample.position[35:]] for sample in batch]
        )
        log_policy = functional.log_softmax(
            logits.masked_fill(~legal, -1e9), dim=1
        )
        policy_loss = -(policies * log_policy).sum(dim=1).mean()
        value_loss = functional.mse_loss(values, outcomes)
        loss = policy_loss + value_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10)
        self.optimizer.step()
        self.updates += 1
        self.last_policy_loss = float(policy_loss.detach())
        self.last_value_loss = float(value_loss.detach())
        self.last_loss = float(loss.detach())
        return self.last_loss

    def save(self, path: str | Path, *, inference_only: bool = False) -> None:
        """Atomically save a resumable agent or a compact policy/value model."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        config = {
            name: getattr(self, name)
            for name in (
                "seed",
                "learning_rate",
                "simulations",
                "c_puct",
                "batch_size",
                "replay_capacity",
                "warmup",
                "updates_per_episode",
                "dirichlet_alpha",
                "noise_fraction",
                "exploration_moves",
                "weight_decay",
                "use_search",
                "opponent_pool_size",
                "snapshot_interval",
                "historical_fraction",
            )
        }
        data = {
            "version": 2,
            "algorithm": self.algorithm,
            "config": config,
            "network": self.network.state_dict(),
            "episodes": self.episodes,
            "updates": self.updates,
            "historical_games": self.historical_games,
            "rng": self.rng.getstate(),
            "training_ready": self.training_ready and not inference_only,
            "last_loss": self.last_loss,
            "last_policy_loss": self.last_policy_loss,
            "last_value_loss": self.last_value_loss,
        }
        if data["training_ready"]:
            data.update(
                optimizer=self.optimizer.state_dict(),
                replay=[asdict(sample) for sample in self.replay],
                opponent_pool=self.opponent_pool,
            )
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(data, temporary)
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path, *, for_training: bool = False) -> PUCTAgent:
        """Validate restricted CPU checkpoint data without executing objects."""
        try:
            data = torch.load(path, weights_only=True, map_location="cpu")
            if (
                data["version"] not in (1, 2)
                or data["algorithm"] != cls.algorithm
            ):
                raise ValueError("Unsupported PUCT checkpoint format")
            if not isinstance(data["training_ready"], bool):
                raise ValueError("Invalid PUCT training flag")
            if for_training and not data["training_ready"]:
                raise ValueError("Inference-only model cannot resume training")
            agent = cls(**data["config"])
            agent.network.load_state_dict(data["network"])
            if any(
                not torch.isfinite(parameter).all()
                for parameter in agent.network.parameters()
            ):
                raise ValueError("Non-finite policy/value parameters")
            for name in ("episodes", "updates"):
                value = data[name]
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                ):
                    raise ValueError("Invalid PUCT counter")
                setattr(agent, name, value)
            historical = data.get("historical_games", 0)
            if (
                not isinstance(historical, int)
                or isinstance(historical, bool)
                or not 0 <= historical <= agent.episodes
            ):
                raise ValueError("Invalid historical game counter")
            agent.historical_games = historical
            for name in ("last_loss", "last_policy_loss", "last_value_loss"):
                value = data[name]
                if value is not None and (
                    not math.isfinite(value) or value < 0
                ):
                    raise ValueError("Invalid PUCT loss")
                setattr(agent, name, value)
            agent.rng.setstate(data["rng"])
            agent.training_ready = data["training_ready"]
            if agent.training_ready:
                agent.optimizer.load_state_dict(data["optimizer"])
                if len(data["replay"]) > agent.replay_capacity:
                    raise ValueError("PUCT replay exceeds capacity")
                for record in data["replay"]:
                    sample = PolicySample(
                        record["position"],
                        tuple(record["policy"]),
                        record["outcome"],
                    )
                    sample.validate()
                    agent.replay.append(sample)
                archived = data.get("opponent_pool", [])
                if len(archived) > agent.opponent_pool_size:
                    raise ValueError("Historical pool exceeds capacity")
                expected = agent.network.state_dict()
                for weights in archived:
                    if set(weights) != set(expected) or any(
                        not isinstance(value, Tensor)
                        or value.shape != expected[name].shape
                        or not torch.isfinite(value).all()
                        for name, value in weights.items()
                    ):
                        raise ValueError("Invalid archived policy weights")
                    agent.opponent_pool.append(
                        {
                            name: value.detach().clone()
                            for name, value in weights.items()
                        }
                    )
            return agent
        except (
            KeyError,
            TypeError,
            IndexError,
            RuntimeError,
            OverflowError,
            pickle.UnpicklingError,
            EOFError,
        ) as error:
            raise ValueError("Invalid PUCT checkpoint") from error


def train_puct(agent: PUCTAgent, episodes: int) -> TrainingStats:
    """Generate visit targets through self-play and value labels from rules."""
    if not agent.training_ready:
        raise ValueError("Inference-only model cannot resume training")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    outcomes = [0, 0, 0]
    for _ in range(episodes):
        historical = agent.sample_opponent()
        learner_seat = 1 + (agent.episodes // 2) % 2
        state = GameState.new(starting_player=1 + agent.episodes % 2)
        history: list[tuple[int, str, tuple[float, ...]]] = []
        while not state.is_terminal():
            actor = (
                historical
                if historical is not None
                and state.current_player != learner_seat
                else agent
            )
            result = actor.search(
                state,
                add_noise=actor is agent,
                temperature=(
                    1
                    if actor is agent
                    and len(state.move_stack) < agent.exploration_moves
                    else 0
                ),
            )
            if actor is agent:
                history.append(
                    (state.current_player, QAgent.key(state), result.policy)
                )
            state.apply_move(result.move)
        winner = state.winner()
        for player, position, policy in history:
            outcome = 0.0 if not winner else (1.0 if player == winner else -1.0)
            sample = PolicySample(position, policy, outcome)
            sample.validate()
            agent.replay.extend((sample, sample.reflected()))
        if len(agent.replay) >= max(agent.batch_size, agent.warmup):
            for _ in range(agent.updates_per_episode):
                agent.optimize()
        outcomes[winner] += 1
        agent.episodes += 1
        if historical is not None:
            agent.historical_games += 1
        if (
            agent.opponent_pool_size
            and agent.episodes % agent.snapshot_interval == 0
        ):
            agent.archive_policy()
    return TrainingStats(episodes, outcomes[1], outcomes[2], outcomes[0])
