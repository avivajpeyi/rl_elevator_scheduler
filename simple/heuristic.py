from __future__ import annotations

from typing import Optional

from .env import ACTION_MOVE_DOWN, ACTION_MOVE_UP, ACTION_OPEN_DOORS, ACTION_STAY, SimpleElevatorEnv


class NearestCallHeuristic:
    """Boring baseline that serves the nearest visible target."""

    def __init__(self) -> None:
        self.target_floor: Optional[int] = None

    def reset(self) -> None:
        self.target_floor = None

    def get_action(self, env: SimpleElevatorEnv) -> int:
        current_floor = env.elevator_floor

        if current_floor in env.onboard_destination_floors():
            self.target_floor = None
            return ACTION_OPEN_DOORS

        if current_floor in env.pending_call_floors():
            self.target_floor = None
            return ACTION_OPEN_DOORS

        destinations = env.onboard_destination_floors()
        if destinations:
            self.target_floor = min(destinations, key=lambda floor: abs(floor - current_floor))
        elif self.target_floor not in env.pending_call_floors():
            pending = env.pending_call_floors()
            self.target_floor = min(pending, key=lambda floor: abs(floor - current_floor)) if pending else None

        if self.target_floor is None:
            return ACTION_STAY
        if self.target_floor > current_floor:
            return ACTION_MOVE_UP
        if self.target_floor < current_floor:
            return ACTION_MOVE_DOWN
        self.target_floor = None
        return ACTION_OPEN_DOORS
