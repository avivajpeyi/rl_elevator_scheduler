from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simple.env import SimpleElevatorEnv
from simple.heuristic import NearestCallHeuristic
from simple.metrics import evaluate_policy
from simple.plots import plot_comparison, plot_training_history
from simple.progress import ProgressReporter
from simple.train_dqn import TrainConfig, evaluate_dqn, train_dqn


def print_metrics(label: str, metrics: dict) -> None:
    print(f"{label}:")
    print(f"  mean wait: {metrics['mean_wait_time']:.2f}")
    print(f"  mean trip: {metrics['mean_trip_time']:.2f}")
    print(f"  served: {metrics['served_passengers']:.2f}")
    print(f"  delivered: {metrics['delivery_rate_pct']:.2f}%")
    print(f"  reward: {metrics['total_reward']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare nearest-call heuristic and simple DQN.")
    parser.add_argument("--dqn-episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--floors", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--call-probability", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=300)
    parser.add_argument("--plot-path", type=Path, default=Path("plots/compare_simple.png"))
    parser.add_argument("--training-plot-path", type=Path, default=Path("plots/dqn_training.png"))
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Disable progress output.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N episodes.")
    args = parser.parse_args()

    def make_eval_env() -> SimpleElevatorEnv:
        return SimpleElevatorEnv(
            num_floors=args.floors,
            episode_steps=args.steps,
            call_probability=args.call_probability,
        )

    if not args.quiet:
        print(f"Evaluating heuristic for {args.eval_episodes} episodes...", flush=True)
    heuristic_metrics = evaluate_policy(
        make_eval_env,
        NearestCallHeuristic(),
        episodes=args.eval_episodes,
        seed=args.seed,
    )

    config = TrainConfig(
        num_floors=args.floors,
        episode_steps=args.steps,
        train_episodes=args.dqn_episodes,
        call_probability=args.call_probability,
        seed=args.seed,
        batch_size=8 if args.dqn_episodes <= 2 else args.batch_size,
        hidden_size=args.hidden_size,
        learning_starts=args.learning_starts,
        train_every=args.train_every,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
    )
    reporter = None if args.quiet else ProgressReporter(every=args.progress_every)
    if not args.quiet:
        print(f"Training DQN for {args.dqn_episodes} episodes...", flush=True)
    agent, history = train_dqn(config, progress_callback=reporter)
    if not args.quiet:
        print(f"Evaluating DQN for {args.eval_episodes} episodes...", flush=True)
    dqn_metrics = evaluate_dqn(agent, config, episodes=args.eval_episodes, progress_callback=reporter)

    print_metrics("Heuristic", heuristic_metrics)
    print()
    print_metrics("DQN", dqn_metrics)
    if not args.no_plot:
        if not args.quiet:
            print("Writing plots...", flush=True)
        comparison_path = plot_comparison(heuristic_metrics, dqn_metrics, args.plot_path)
        training_path = plot_training_history(history, args.training_plot_path)
        print()
        print(f"Saved comparison plot: {comparison_path}")
        print(f"Saved DQN training plot: {training_path}")


if __name__ == "__main__":
    main()
