from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


ACTION_STAY = 0
ACTION_MOVE_UP = 1
ACTION_MOVE_DOWN = 2
ACTION_OPEN_DOORS = 3

DIRECTION_IDLE = 0
DIRECTION_UP = 1
DIRECTION_DOWN = -1


@dataclass
class Passenger:
    id: int
    origin: int
    destination: int
    call_time: int
    pickup_time: Optional[int] = None
    dropoff_time: Optional[int] = None

    @property
    def direction(self) -> int:
        return DIRECTION_UP if self.destination > self.origin else DIRECTION_DOWN

    @property
    def is_waiting(self) -> bool:
        return self.pickup_time is None and self.dropoff_time is None

    @property
    def is_onboard(self) -> bool:
        return self.pickup_time is not None and self.dropoff_time is None

    @property
    def is_served(self) -> bool:
        return self.dropoff_time is not None


class SimpleElevatorEnv:
    """Small one-elevator environment with a Gymnasium-style API.

    The model is intentionally compact: each action is one discrete tick, movement
    changes the elevator by one floor, and opening doors boards/drops off every
    passenger at the current floor.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_floors: int = 5,
        episode_steps: int = 500,
        call_probability: float = 0.25,
        seed: Optional[int] = None,
    ) -> None:
        if num_floors < 2:
            raise ValueError("num_floors must be at least 2")
        if episode_steps <= 0:
            raise ValueError("episode_steps must be positive")
        if not 0.0 <= call_probability <= 1.0:
            raise ValueError("call_probability must be between 0 and 1")

        self.num_floors = num_floors
        self.num_elevators = 1
        self.episode_steps = episode_steps
        self.call_probability = call_probability
        self._initial_seed = seed
        self.rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    @property
    def observation_size(self) -> int:
        return 2 + (3 * self.num_floors) + 1

    @property
    def action_size(self) -> int:
        return 4

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self._initial_seed is not None:
            self.rng = np.random.default_rng(self._initial_seed)

        self.time = 0
        self.elevator_floor = 0
        self.elevator_direction = DIRECTION_IDLE
        self.pending_up: Dict[int, List[int]] = {floor: [] for floor in range(self.num_floors)}
        self.pending_down: Dict[int, List[int]] = {floor: [] for floor in range(self.num_floors)}
        self.onboard: List[int] = []
        self.passengers: Dict[int, Passenger] = {}
        self.next_passenger_id = 1
        self.total_reward = 0.0
        return self._observation(), self._info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if action not in (ACTION_STAY, ACTION_MOVE_UP, ACTION_MOVE_DOWN, ACTION_OPEN_DOORS):
            raise ValueError(f"unknown action: {action}")

        reward = -0.01
        served_now = 0
        opened_usefully = False

        prev_floor = self.elevator_floor
        if action == ACTION_MOVE_UP:
            if self.elevator_floor < self.num_floors - 1:
                self.elevator_floor += 1
            self.elevator_direction = DIRECTION_UP
        elif action == ACTION_MOVE_DOWN:
            if self.elevator_floor > 0:
                self.elevator_floor -= 1
            self.elevator_direction = DIRECTION_DOWN
        elif action == ACTION_STAY:
            self.elevator_direction = DIRECTION_IDLE
        else:
            self.elevator_direction = DIRECTION_IDLE
            dropped_off = self._drop_off_current_floor()
            boarded = self._board_current_floor()
            served_now = len(dropped_off)
            opened_usefully = bool(dropped_off or boarded)
            reward += float(served_now)
            reward += 0.3 * len(boarded)
            if not opened_usefully:
                reward -= 0.05

        if action in (ACTION_MOVE_UP, ACTION_MOVE_DOWN):
            nearest = self._nearest_target_floor()
            if nearest is not None:
                if abs(self.elevator_floor - nearest) < abs(prev_floor - nearest):
                    reward += 0.05

        self._maybe_generate_passenger()
        reward -= 0.02 * self.waiting_passenger_count

        self.time += 1
        truncated = self.time >= self.episode_steps
        terminated = False
        self.total_reward += reward

        info = self._info()
        info["served_now"] = served_now
        return self._observation(), float(reward), terminated, truncated, info

    @property
    def waiting_passenger_count(self) -> int:
        return sum(len(calls) for calls in self.pending_up.values()) + sum(
            len(calls) for calls in self.pending_down.values()
        )

    @property
    def served_passenger_count(self) -> int:
        return sum(passenger.is_served for passenger in self.passengers.values())

    def add_passenger(self, origin: int, destination: int) -> int:
        """Add a passenger immediately. Useful for deterministic tests and examples."""
        if origin == destination:
            raise ValueError("origin and destination must differ")
        if not 0 <= origin < self.num_floors or not 0 <= destination < self.num_floors:
            raise ValueError("origin and destination must be valid floors")

        passenger_id = self.next_passenger_id
        self.next_passenger_id += 1
        passenger = Passenger(
            id=passenger_id,
            origin=origin,
            destination=destination,
            call_time=self.time,
        )
        self.passengers[passenger_id] = passenger
        queue = self.pending_up if passenger.direction == DIRECTION_UP else self.pending_down
        queue[origin].append(passenger_id)
        return passenger_id

    def pending_call_floors(self) -> List[int]:
        floors = []
        for floor in range(self.num_floors):
            if self.pending_up[floor] or self.pending_down[floor]:
                floors.append(floor)
        return floors

    def onboard_destination_floors(self) -> List[int]:
        return [self.passengers[pid].destination for pid in self.onboard]

    def _nearest_target_floor(self) -> Optional[int]:
        targets = self.onboard_destination_floors() + self.pending_call_floors()
        if not targets:
            return None
        return min(targets, key=lambda f: abs(f - self.elevator_floor))

    def _maybe_generate_passenger(self) -> None:
        if self.rng.random() >= self.call_probability:
            return
        origin = int(self.rng.integers(0, self.num_floors))
        destination = int(self.rng.integers(0, self.num_floors - 1))
        if destination >= origin:
            destination += 1
        self.add_passenger(origin, destination)

    def _drop_off_current_floor(self) -> List[int]:
        dropped_off = []
        remaining_onboard = []
        for passenger_id in self.onboard:
            passenger = self.passengers[passenger_id]
            if passenger.destination == self.elevator_floor:
                passenger.dropoff_time = self.time
                dropped_off.append(passenger_id)
            else:
                remaining_onboard.append(passenger_id)
        self.onboard = remaining_onboard
        return dropped_off

    def _board_current_floor(self) -> List[int]:
        boarded = []
        floor = self.elevator_floor
        for queue in (self.pending_up, self.pending_down):
            while queue[floor]:
                passenger_id = queue[floor].pop(0)
                passenger = self.passengers[passenger_id]
                passenger.pickup_time = self.time
                self.onboard.append(passenger_id)
                boarded.append(passenger_id)
        return boarded

    def _observation(self) -> np.ndarray:
        floor = np.array([self.elevator_floor / (self.num_floors - 1)], dtype=np.float32)
        direction = np.array([self.elevator_direction], dtype=np.float32)
        up = np.array([1.0 if self.pending_up[floor] else 0.0 for floor in range(self.num_floors)], dtype=np.float32)
        down = np.array([1.0 if self.pending_down[floor] else 0.0 for floor in range(self.num_floors)], dtype=np.float32)
        onboard_destinations = set(self.onboard_destination_floors())
        onboard = np.array(
            [1.0 if floor in onboard_destinations else 0.0 for floor in range(self.num_floors)],
            dtype=np.float32,
        )
        time = np.array([self.time / self.episode_steps], dtype=np.float32)
        return np.concatenate([floor, direction, up, down, onboard, time])

    def _info(self) -> dict:
        return {
            "time": self.time,
            "elevator_floor": self.elevator_floor,
            "elevator_direction": self.elevator_direction,
            "pending_up": {floor: list(ids) for floor, ids in self.pending_up.items()},
            "pending_down": {floor: list(ids) for floor, ids in self.pending_down.items()},
            "onboard": list(self.onboard),
            "onboard_destinations": self.onboard_destination_floors(),
            "waiting_passengers": self.waiting_passenger_count,
            "served_passengers": self.served_passenger_count,
            "total_reward": self.total_reward,
        }
