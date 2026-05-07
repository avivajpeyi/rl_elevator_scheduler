from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from .dqn_agent import DQNAgent, DQNConfig
from .env import SimpleElevatorEnv
from .heuristic import NearestCallHeuristic
from .metrics import collect_metrics
from .replay_buffer import ReplayBuffer


NUM_FLOORS = 5
NUM_ELEVATORS = 1
EPISODE_STEPS = 500
TRAIN_EPISODES = 1000


@dataclass
class TrainConfig:
    num_floors: int = NUM_FLOORS
    episode_steps: int = EPISODE_STEPS
    train_episodes: int = TRAIN_EPISODES
    call_probability: float = 0.25
    replay_capacity: int = 20_000
    batch_size: int = 64
    hidden_size: int = 128
    learning_starts: int = 1_000
    train_every: int = 4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 800
    seed: int = 0


ProgressCallback = Callable[[str, int, int, Dict[str, float]], None]


def _warm_start_with_heuristic(replay_buffer: ReplayBuffer, config: TrainConfig, env: SimpleElevatorEnv) -> None:
    """Pre-fill replay buffer with heuristic transitions so DQN starts from good behavior."""
    heuristic = NearestCallHeuristic()
    observation, _ = env.reset(seed=config.seed + 99_999)
    heuristic.reset()
    steps = config.learning_starts * 2
    for _ in range(steps):
        action = heuristic.get_action(env)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(observation, action, reward, next_observation, done)
        observation = next_observation
        if done:
            observation, _ = env.reset()
            heuristic.reset()


def make_env(config: TrainConfig, seed: int | None = None) -> SimpleElevatorEnv:
    return SimpleElevatorEnv(
        num_floors=config.num_floors,
        episode_steps=config.episode_steps,
        call_probability=config.call_probability,
        seed=seed,
    )


def epsilon_for_episode(episode: int, config: TrainConfig) -> float:
    fraction = min(1.0, episode / max(1, config.epsilon_decay_episodes))
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def train_dqn(
    config: TrainConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Tuple[DQNAgent, List[Dict[str, float]]]:
    config = config or TrainConfig()
    env = make_env(config, seed=config.seed)
    agent = DQNAgent(
        observation_size=env.observation_size,
        action_size=env.action_size,
        config=DQNConfig(batch_size=config.batch_size, hidden_size=config.hidden_size),
        seed=config.seed,
    )
    replay_buffer = ReplayBuffer(capacity=config.replay_capacity, seed=config.seed)
    _warm_start_with_heuristic(replay_buffer, config, env)
    history: List[Dict[str, float]] = []
    global_step = 0

    for episode in range(config.train_episodes):
        observation, _ = env.reset(seed=config.seed + episode)
        terminated = False
        truncated = False
        episode_reward = 0.0
        losses = []
        epsilon = epsilon_for_episode(episode, config)

        while not (terminated or truncated):
            action = agent.select_action(observation, epsilon=epsilon)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(observation, action, reward, next_observation, done)
            global_step += 1
            should_train = global_step >= config.learning_starts and global_step % config.train_every == 0
            if should_train:
                loss = agent.optimize(replay_buffer)
                if loss is not None:
                    losses.append(loss)
            observation = next_observation
            episode_reward += reward

        metrics = collect_metrics(env, episode_reward)
        metrics["episode"] = float(episode)
        metrics["epsilon"] = float(epsilon)
        metrics["loss"] = float(sum(losses) / len(losses)) if losses else 0.0
        metrics["optimizer_steps"] = float(agent.training_steps)
        history.append(metrics)
        if progress_callback is not None:
            progress_callback("train", episode + 1, config.train_episodes, metrics)

    return agent, history


def evaluate_dqn(
    agent: DQNAgent,
    config: TrainConfig | None = None,
    episodes: int = 5,
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, float]:
    config = config or TrainConfig()
    totals: Dict[str, float] = {}
    for episode in range(episodes):
        env = make_env(config, seed=config.seed + 10_000 + episode)
        observation, _ = env.reset(seed=config.seed + 10_000 + episode)
        terminated = False
        truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action = agent.select_action(observation, epsilon=0.0)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        metrics = collect_metrics(env, total_reward)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        if progress_callback is not None:
            progress_callback("eval", episode + 1, episodes, metrics)
    return {key: value / episodes for key, value in totals.items()}
