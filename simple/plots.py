from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _prepare_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _ensure_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_metrics(metrics: Mapping[str, float], output_path: str | Path) -> Path:
    plt = _prepare_matplotlib()
    path = _ensure_parent(output_path)
    metric_defs = [
        ("mean_wait_time", "Mean wait", "#4c78a8"),
        ("mean_trip_time", "Mean trip", "#f58518"),
        ("served_passengers", "Served", "#54a24b"),
        ("delivery_rate_pct", "Delivered %", "#e45756"),
        ("total_reward", "Reward", "#b279a2"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0))
    for ax, (metric_name, title, color) in zip(axes.flat, metric_defs):
        value = float(metrics.get(metric_name, 0.0))
        bars = ax.bar([title], [value], color=color, width=0.55)
        ax.set_title(title)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.grid(axis="y", alpha=0.25)
        _set_metric_ylim(ax, [value])
        _label_bars(ax, bars, [value], suffix="%" if metric_name == "delivery_rate_pct" else "")
    for ax in axes.flat[len(metric_defs) :]:
        ax.axis("off")

    fig.suptitle("Episode metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_training_history(history: Sequence[Mapping[str, float]], output_path: str | Path) -> Path:
    plt = _prepare_matplotlib()
    path = _ensure_parent(output_path)
    episodes = [int(item.get("episode", idx)) for idx, item in enumerate(history)]
    rewards = [float(item.get("total_reward", 0.0)) for item in history]
    waits = [float(item.get("mean_wait_time", 0.0)) for item in history]
    delivery_rates = [float(item.get("delivery_rate_pct", 0.0)) for item in history]
    reward_trend = _rolling_mean(rewards)
    wait_trend = _rolling_mean(waits)
    delivery_trend = _rolling_mean(delivery_rates)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
    axes[0].plot(episodes, rewards, color="#4c78a8", alpha=0.35, label="episode")
    axes[0].plot(episodes, reward_trend, color="#4c78a8", linewidth=2, label="rolling mean")
    axes[0].set_ylabel("reward")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(episodes, waits, color="#f58518", alpha=0.35, label="episode")
    axes[1].plot(episodes, wait_trend, color="#f58518", linewidth=2, label="rolling mean")
    axes[1].set_ylabel("mean wait")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    axes[2].plot(episodes, delivery_rates, color="#e45756", alpha=0.35, label="episode")
    axes[2].plot(episodes, delivery_trend, color="#e45756", linewidth=2, label="rolling mean")
    axes[2].set_xlabel("episode")
    axes[2].set_ylabel("delivered %")
    axes[2].set_ylim(-5, 105)
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    fig.suptitle("DQN training")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _rolling_mean(values: Sequence[float], window: int = 10) -> list[float]:
    if not values:
        return []
    effective_window = min(window, len(values))
    kernel = np.ones(effective_window) / effective_window
    padded = np.pad(np.asarray(values, dtype=float), (effective_window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid").tolist()


def plot_comparison(
    heuristic_metrics: Mapping[str, float],
    dqn_metrics: Mapping[str, float],
    output_path: str | Path,
) -> Path:
    plt = _prepare_matplotlib()
    path = _ensure_parent(output_path)
    metrics = [
        ("mean_wait_time", "Mean wait"),
        ("mean_trip_time", "Mean trip"),
        ("served_passengers", "Served"),
        ("delivery_rate_pct", "Delivered %"),
        ("total_reward", "Reward"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    for ax, (metric_name, title) in zip(axes.flat, metrics):
        values = [
            float(heuristic_metrics.get(metric_name, 0.0)),
            float(dqn_metrics.get(metric_name, 0.0)),
        ]
        bars = ax.bar(["Heuristic", "DQN"], values, color=["#4c78a8", "#54a24b"], width=0.55)
        ax.set_title(title)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.grid(axis="y", alpha=0.25)
        _set_metric_ylim(ax, values)
        _label_bars(ax, bars, values, suffix="%" if metric_name == "delivery_rate_pct" else "")
    for ax in axes.flat[len(metrics) :]:
        ax.axis("off")

    fig.suptitle("Heuristic vs DQN")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _set_metric_ylim(ax, values: Sequence[float]) -> None:
    low = min(list(values) + [0.0])
    high = max(list(values) + [0.0])
    span = max(high - low, 1.0)
    ax.set_ylim(low - span * 0.18, high + span * 0.28)


def _label_bars(ax, bars, values: Sequence[float], suffix: str = "") -> None:
    low = min(list(values) + [0.0])
    high = max(list(values) + [0.0])
    span = max(high - low, 1.0)
    for bar, value in zip(bars, values):
        label_y = value + span * 0.04 if value >= 0 else value - span * 0.08
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{value:.2f}{suffix}",
            ha="center",
            va=va,
            fontsize=9,
        )
