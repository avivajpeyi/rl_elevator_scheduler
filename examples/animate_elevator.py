from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, patches
from matplotlib.gridspec import GridSpec

from simple.env import (
    ACTION_MOVE_DOWN,
    ACTION_MOVE_UP,
    ACTION_OPEN_DOORS,
    ACTION_STAY,
    DIRECTION_UP,
    SimpleElevatorEnv,
)
from simple.heuristic import NearestCallHeuristic


ACTION_LABELS = {
    ACTION_STAY: "Waiting",
    ACTION_MOVE_UP: "Moving Up",
    ACTION_MOVE_DOWN: "Moving Down",
    ACTION_OPEN_DOORS: "Doors Open",
}

LEFT_EDGE = -3.2
RIGHT_EDGE = 3.2
SHAFT_LEFT = -0.5
SHAFT_RIGHT = 0.5
CABIN_WIDTH = 0.84
CABIN_HEIGHT = 0.72
PASSENGER_WALK_FRAMES = 16
EXIT_WALK_FRAMES = 14


@dataclass(frozen=True)
class PassengerSnapshot:
    passenger_id: int
    origin: int
    destination: int
    direction: int
    call_time: int
    pickup_time: int | None
    dropoff_time: int | None


@dataclass(frozen=True)
class StepFrame:
    step: int
    action: int
    reward: float
    prev_floor: int
    elevator_floor: int
    pending_up: dict[int, list[int]]
    pending_down: dict[int, list[int]]
    onboard: list[int]
    boarded: list[int]
    dropped_off: list[int]
    passengers: dict[int, PassengerSnapshot]
    waiting_passengers: int
    served_passengers: int
    total_reward: float


def collect_episode(
    *,
    steps: int,
    num_floors: int,
    call_probability: float,
    seed: int,
) -> list[StepFrame]:
    env = SimpleElevatorEnv(
        num_floors=num_floors,
        episode_steps=steps,
        call_probability=call_probability,
        seed=seed,
    )
    policy = NearestCallHeuristic()
    _, _ = env.reset(seed=seed)
    policy.reset()

    frames: list[StepFrame] = []
    for _ in range(steps):
        action = policy.get_action(env)
        prev_floor = env.elevator_floor
        before_time = env.time
        before_pending = {
            "up": {floor: set(ids) for floor, ids in env.pending_up.items()},
            "down": {floor: set(ids) for floor, ids in env.pending_down.items()},
        }
        before_onboard = set(env.onboard)

        _, reward, terminated, truncated, info = env.step(action)

        boarded = sorted(
            passenger_id
            for passenger_id, passenger in env.passengers.items()
            if passenger.pickup_time == before_time
        )
        dropped_off = sorted(
            passenger_id
            for passenger_id, passenger in env.passengers.items()
            if passenger.dropoff_time == before_time
        )

        if action == ACTION_OPEN_DOORS and not boarded:
            boarded = sorted(before_pending["up"][prev_floor] | before_pending["down"][prev_floor])
        if action == ACTION_OPEN_DOORS and not dropped_off:
            dropped_off = sorted(
                passenger_id
                for passenger_id in before_onboard
                if env.passengers[passenger_id].destination == prev_floor
            )

        passenger_snapshots = {
            passenger_id: PassengerSnapshot(
                passenger_id=passenger_id,
                origin=passenger.origin,
                destination=passenger.destination,
                direction=passenger.direction,
                call_time=passenger.call_time,
                pickup_time=passenger.pickup_time,
                dropoff_time=passenger.dropoff_time,
            )
            for passenger_id, passenger in env.passengers.items()
        }

        frames.append(
            StepFrame(
                step=int(info["time"]),
                action=action,
                reward=float(reward),
                prev_floor=prev_floor,
                elevator_floor=int(info["elevator_floor"]),
                pending_up={floor: list(ids) for floor, ids in info["pending_up"].items()},
                pending_down={floor: list(ids) for floor, ids in info["pending_down"].items()},
                onboard=list(info["onboard"]),
                boarded=boarded,
                dropped_off=dropped_off,
                passengers=passenger_snapshots,
                waiting_passengers=int(info["waiting_passengers"]),
                served_passengers=int(info["served_passengers"]),
                total_reward=float(info["total_reward"]),
            )
        )
        if terminated or truncated:
            break
    return frames


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def smoothstep(value: float) -> float:
    value = clamp(value)
    return value * value * (3.0 - 2.0 * value)


def marker_for(passenger: PassengerSnapshot) -> str:
    return "^" if passenger.direction == DIRECTION_UP else "v"


def queue_positions(passenger_ids: Sequence[int], *, floor: int) -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {}
    for index, passenger_id in enumerate(passenger_ids):
        column = index % 5
        row = index // 5
        positions[passenger_id] = (SHAFT_LEFT - 0.16 - column * 0.13, floor + row * 0.09)
    return positions


def onboard_positions(passenger_ids: Sequence[int], elevator_y: float) -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {}
    for index, passenger_id in enumerate(passenger_ids):
        column = index % 4
        row = index // 4
        positions[passenger_id] = (-0.24 + column * 0.16, elevator_y - 0.22 + row * 0.13)
    return positions


def waiting_position(
    passenger_id: int,
    passenger: PassengerSnapshot,
    queue_position: tuple[float, float],
    *,
    step_index: int,
    subframe: int,
    subframes: int,
) -> tuple[float, float]:
    visual_age = ((step_index + 1) - passenger.call_time - 1) * subframes + subframe
    progress = smoothstep(visual_age / PASSENGER_WALK_FRAMES)
    start_x = LEFT_EDGE + 0.28
    target_x, target_y = queue_position
    return start_x + (target_x - start_x) * progress, target_y


def exit_position(
    passenger: PassengerSnapshot,
    *,
    age: int,
) -> tuple[float, float]:
    progress = smoothstep(age / EXIT_WALK_FRAMES)
    start_x = SHAFT_RIGHT + 0.18
    end_x = RIGHT_EDGE - 0.15
    return start_x + (end_x - start_x) * progress, passenger.destination


def build_animation(
    steps: list[StepFrame],
    *,
    num_floors: int,
    fps: int,
    subframes: int,
    output: Path,
) -> Path:
    if not steps:
        raise RuntimeError("No frames were collected from the episode.")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 13,
            "axes.titlesize": 18,
            "axes.labelsize": 13,
            "figure.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
        }
    )

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    grid = GridSpec(2, 2, figure=fig, width_ratios=[3.2, 1], height_ratios=[1, 1])
    ax_building = fig.add_subplot(grid[:, 0])
    ax_text = fig.add_subplot(grid[0, 1])
    ax_reward = fig.add_subplot(grid[1, 1])

    ax_building.set_title("Elevator Traffic Flow", weight="bold", pad=16)
    ax_building.set_xlim(LEFT_EDGE, RIGHT_EDGE)
    ax_building.set_ylim(-0.5, num_floors - 0.5)
    ax_building.set_xticks([])
    ax_building.set_yticks(range(num_floors))
    ax_building.set_yticklabels([f"Floor {floor}" for floor in range(num_floors)])
    ax_building.set_facecolor("#ffffff")
    ax_building.tick_params(axis="y", labelsize=17, length=0, pad=12, colors="#334155")
    ax_building.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax_building.set_axisbelow(True)

    for floor in range(num_floors + 1):
        ax_building.axhline(
            floor - 0.5,
            xmin=0.0,
            xmax=1.0,
            color="#e2e8f0",
            linewidth=1.0,
            zorder=0,
        )

    shaft = patches.Rectangle(
        (SHAFT_LEFT, -0.5),
        SHAFT_RIGHT - SHAFT_LEFT,
        num_floors,
        facecolor="#f1f5f9",
        edgecolor="#334155",
        linewidth=1.8,
        zorder=1,
    )
    ax_building.add_patch(shaft)

    cabin = patches.Rectangle(
        (-CABIN_WIDTH / 2, steps[0].prev_floor - CABIN_HEIGHT / 2),
        CABIN_WIDTH,
        CABIN_HEIGHT,
        facecolor="#256d93",
        edgecolor="#0f172a",
        linewidth=2.4,
        zorder=4,
    )
    ax_building.add_patch(cabin)

    ax_text.axis("off")
    ax_text.set_title("Status", weight="bold", loc="left", pad=16)
    step_text = ax_text.text(0.0, 0.86, "", transform=ax_text.transAxes, fontsize=19, weight="bold", color="#0f172a")
    action_text = ax_text.text(0.0, 0.64, "", transform=ax_text.transAxes, fontsize=18, color="#1e293b")
    passenger_text = ax_text.text(
        0.0,
        0.38,
        "",
        transform=ax_text.transAxes,
        fontsize=15,
        linespacing=1.55,
        color="#334155",
    )
    longest_wait_text = ax_text.text(
        0.0,
        0.13,
        "",
        transform=ax_text.transAxes,
        fontsize=17,
        weight="bold",
        color="#0f172a",
    )

    rewards = [frame.total_reward for frame in steps]
    min_reward = min(rewards + [0.0])
    max_reward = max(rewards + [0.0])
    padding = max(1.0, (max_reward - min_reward) * 0.12)
    ax_reward.set_title("Reward Trend", weight="bold", loc="left", pad=16)
    ax_reward.set_xlim(0, len(steps))
    ax_reward.set_ylim(min_reward - padding, max_reward + padding)
    ax_reward.set_xlabel("Step", color="#475569")
    ax_reward.set_ylabel("Total", color="#475569")
    ax_reward.tick_params(labelsize=11, colors="#475569")
    ax_reward.spines[["top", "right"]].set_visible(False)
    ax_reward.spines[["left", "bottom"]].set_color("#cbd5e1")
    ax_reward.grid(True, color="#e2e8f0", linewidth=0.9)
    reward_line, = ax_reward.plot([], [], color="#111827", linewidth=2.4)
    reward_dot, = ax_reward.plot([], [], marker="o", color="#111827", markersize=4.5)

    dynamic_artists: list[object] = []

    def scatter_passengers(
        passenger_ids: Sequence[int],
        positions: dict[int, tuple[float, float]],
        step: StepFrame,
        *,
        size: int,
        zorder: int,
    ) -> None:
        if not passenger_ids:
            return
        xs: list[float] = []
        ys: list[float] = []
        markers: list[str] = []
        for passenger_id in passenger_ids:
            passenger = step.passengers.get(passenger_id)
            position = positions.get(passenger_id)
            if passenger is None or position is None:
                continue
            xs.append(position[0])
            ys.append(position[1])
            markers.append(marker_for(passenger))

        for marker in ("^", "v"):
            marker_indexes = [index for index, item in enumerate(markers) if item == marker]
            if not marker_indexes:
                continue
            scatter = ax_building.scatter(
                [xs[index] for index in marker_indexes],
                [ys[index] for index in marker_indexes],
                c="#111827",
                marker=marker,
                s=size,
                edgecolors="#ffffff",
                linewidths=0.6,
                zorder=zorder,
            )
            dynamic_artists.append(scatter)

    def add_wait_labels(
        passenger_ids: Sequence[int],
        positions: dict[int, tuple[float, float]],
        step: StepFrame,
    ) -> None:
        for passenger_id in passenger_ids:
            passenger = step.passengers.get(passenger_id)
            position = positions.get(passenger_id)
            if passenger is None or position is None:
                continue
            wait_time = max(0, step.step - passenger.call_time)
            if wait_time <= 1:
                continue
            label = ax_building.text(
                position[0],
                position[1] + 0.22,
                str(wait_time),
                ha="center",
                va="center",
                fontsize=9,
                weight="bold",
                color="#0f172a",
                zorder=9,
            )
            dynamic_artists.append(label)

    def animate(visual_frame: int):
        nonlocal dynamic_artists
        for artist in dynamic_artists:
            artist.remove()
        dynamic_artists = []

        step_index = visual_frame // subframes
        subframe = visual_frame % subframes
        step = steps[step_index]

        if step.action in (ACTION_MOVE_UP, ACTION_MOVE_DOWN):
            move_progress = smoothstep(subframe / max(1, subframes - 1))
            elevator_y = step.prev_floor + (step.elevator_floor - step.prev_floor) * move_progress
        else:
            elevator_y = float(step.elevator_floor)

        doors_open = step.action == ACTION_OPEN_DOORS
        cabin.set_y(elevator_y - CABIN_HEIGHT / 2)
        cabin.set_facecolor("#22c55e" if doors_open else "#256d93")

        waiting_ids_by_floor: dict[int, list[int]] = {floor: [] for floor in range(num_floors)}
        for floor, ids in step.pending_up.items():
            waiting_ids_by_floor[floor].extend(ids)
        for floor, ids in step.pending_down.items():
            waiting_ids_by_floor[floor].extend(ids)

        waiting_positions: dict[int, tuple[float, float]] = {}
        waiting_ids: list[int] = []
        for floor, passenger_ids in waiting_ids_by_floor.items():
            ordered_ids = sorted(passenger_ids)
            floor_queue_positions = queue_positions(ordered_ids, floor=floor)
            for passenger_id in ordered_ids:
                passenger = step.passengers.get(passenger_id)
                if passenger is None:
                    continue
                waiting_ids.append(passenger_id)
                waiting_positions[passenger_id] = waiting_position(
                    passenger_id,
                    passenger,
                    floor_queue_positions[passenger_id],
                    step_index=step_index,
                    subframe=subframe,
                    subframes=subframes,
                )
        scatter_passengers(waiting_ids, waiting_positions, step, size=300, zorder=5)
        add_wait_labels(waiting_ids, waiting_positions, step)

        onboard_ids = [passenger_id for passenger_id in step.onboard if passenger_id not in step.boarded]
        onboard_position_map = onboard_positions(onboard_ids, elevator_y)
        scatter_passengers(onboard_ids, onboard_position_map, step, size=300, zorder=7)

        boarded_positions: dict[int, tuple[float, float]] = {}
        if doors_open and step.boarded:
            final_positions = onboard_positions(step.onboard, elevator_y)
            boarding_progress = smoothstep(subframe / max(1, subframes - 1))
            for passenger_id in step.boarded:
                passenger = step.passengers.get(passenger_id)
                final_position = final_positions.get(passenger_id)
                if passenger is None or final_position is None:
                    continue
                queue_x, queue_y = queue_positions([passenger_id], floor=passenger.origin)[passenger_id]
                boarded_positions[passenger_id] = (
                    queue_x + (final_position[0] - queue_x) * boarding_progress,
                    queue_y + (final_position[1] - queue_y) * boarding_progress,
                )
            scatter_passengers(step.boarded, boarded_positions, step, size=300, zorder=8)

        exiting_ids: list[int] = []
        exiting_positions: dict[int, tuple[float, float]] = {}
        for lookback_index in range(max(0, step_index - 3), step_index + 1):
            age_base = (step_index - lookback_index) * subframes + subframe
            for passenger_id in steps[lookback_index].dropped_off:
                if age_base > EXIT_WALK_FRAMES:
                    continue
                passenger = step.passengers.get(passenger_id) or steps[lookback_index].passengers.get(passenger_id)
                if passenger is None:
                    continue
                exiting_ids.append(passenger_id)
                exiting_positions[passenger_id] = exit_position(passenger, age=age_base)
        scatter_passengers(exiting_ids, exiting_positions, step, size=300, zorder=6)

        step_text.set_text(f"Step: {step.step} / {len(steps)}")
        action_text.set_text(ACTION_LABELS[step.action])
        passenger_text.set_text(
            f"Waiting  {step.waiting_passengers}\n"
            f"Onboard  {len(step.onboard)}\n"
            f"Served   {step.served_passengers}"
        )
        longest_wait = 0
        for prior_step in steps[: step_index + 1]:
            for passenger in prior_step.passengers.values():
                wait_end = passenger.pickup_time if passenger.pickup_time is not None else prior_step.step
                if wait_end > prior_step.step:
                    wait_end = prior_step.step
                longest_wait = max(longest_wait, wait_end - passenger.call_time)
        longest_wait_text.set_text(f"Longest Wait  {longest_wait}")

        x_data = np.arange(1, step_index + 2)
        y_data = np.array([item.total_reward for item in steps[: step_index + 1]])
        reward_line.set_data(x_data, y_data)
        reward_dot.set_data([x_data[-1]], [y_data[-1]])

        return [
            cabin,
            step_text,
            action_text,
            passenger_text,
            longest_wait_text,
            reward_line,
            reward_dot,
            *dynamic_artists,
        ]

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(steps) * subframes,
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".gif":
        writer = animation.PillowWriter(fps=fps)
        anim.save(output, writer=writer, dpi=130)
        saved_output = output
    elif animation.writers.is_available("ffmpeg"):
        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=2200,
            extra_args=["-pix_fmt", "yuv420p"],
        )
        anim.save(output, writer=writer, dpi=150)
        saved_output = output
    else:
        gif_output = output.with_suffix(".gif")
        writer = animation.PillowWriter(fps=fps)
        anim.save(gif_output, writer=writer, dpi=130)
        print(f"ffmpeg is not available; saved GIF instead: {gif_output}")
        saved_output = gif_output

    plt.close(fig)
    return saved_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a presentation-ready elevator animation.")
    parser.add_argument("--output", type=Path, default=Path("elevator_simulation.mp4"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--num-floors", type=int, default=5)
    parser.add_argument("--call-probability", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--subframes", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = collect_episode(
        steps=args.steps,
        num_floors=args.num_floors,
        call_probability=args.call_probability,
        seed=args.seed,
    )
    saved_output = build_animation(
        steps,
        num_floors=args.num_floors,
        fps=args.fps,
        subframes=args.subframes,
        output=args.output,
    )
    print(f"Saved animation: {saved_output}")


if __name__ == "__main__":
    main()
