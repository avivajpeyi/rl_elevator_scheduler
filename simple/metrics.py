from __future__ import annotations

from typing import Callable, Dict, Protocol

import numpy as np

from .env import SimpleElevatorEnv


class EnvPolicy(Protocol):
    def get_action(self, env: SimpleElevatorEnv) -> int:
        ...


def run_episode(env: SimpleElevatorEnv, policy: EnvPolicy | Callable[[SimpleElevatorEnv], int], seed: int | None = None) -> Dict[str, float]:
    if hasattr(policy, "reset"):
        policy.reset()  # type: ignore[attr-defined]
    env.reset(seed=seed)

    total_reward = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        if hasattr(policy, "get_action"):
            action = policy.get_action(env)  # type: ignore[attr-defined]
        else:
            action = policy(env)
        _, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += reward

    return collect_metrics(env, total_reward)


def evaluate_policy(
    make_env: Callable[[], SimpleElevatorEnv],
    policy: EnvPolicy | Callable[[SimpleElevatorEnv], int],
    episodes: int = 5,
    seed: int = 0,
) -> Dict[str, float]:
    episode_metrics = [run_episode(make_env(), policy, seed=seed + idx) for idx in range(episodes)]
    keys = episode_metrics[0].keys()
    return {key: float(np.mean([metrics[key] for metrics in episode_metrics])) for key in keys}


def collect_metrics(env: SimpleElevatorEnv, total_reward: float | None = None) -> Dict[str, float]:
    served = [passenger for passenger in env.passengers.values() if passenger.is_served]
    total_passengers = len(env.passengers)
    delivery_rate_pct = (len(served) / total_passengers * 100.0) if total_passengers else 0.0
    wait_times = [
        float(passenger.pickup_time - passenger.call_time)
        for passenger in served
        if passenger.pickup_time is not None
    ]
    trip_times = [
        float(passenger.dropoff_time - passenger.pickup_time)
        for passenger in served
        if passenger.pickup_time is not None and passenger.dropoff_time is not None
    ]
    return {
        "mean_wait_time": float(np.mean(wait_times)) if wait_times else 0.0,
        "mean_trip_time": float(np.mean(trip_times)) if trip_times else 0.0,
        "served_passengers": float(len(served)),
        "delivery_rate_pct": float(delivery_rate_pct),
        "total_reward": float(env.total_reward if total_reward is None else total_reward),
    }
