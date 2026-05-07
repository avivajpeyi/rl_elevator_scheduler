from __future__ import annotations

import sys
import time
from typing import Dict


class ProgressReporter:
    def __init__(self, every: int = 10, stream=None) -> None:
        self.every = max(1, every)
        self.stream = stream or sys.stderr
        self._started_at: dict[str, float] = {}

    def __call__(self, phase: str, current: int, total: int, metrics: Dict[str, float]) -> None:
        if phase not in self._started_at:
            self._started_at[phase] = time.monotonic()

        if current != 1 and current != total and current % self.every != 0:
            return

        elapsed = time.monotonic() - self._started_at[phase]
        rate = current / elapsed if elapsed > 0 else 0.0
        remaining = (total - current) / rate if rate > 0 else 0.0
        reward = metrics.get("total_reward", 0.0)
        served = metrics.get("served_passengers", 0.0)
        wait = metrics.get("mean_wait_time", 0.0)
        message = (
            f"[{phase}] {current}/{total} "
            f"reward={reward:.2f} served={served:.0f} wait={wait:.2f} "
            f"elapsed={elapsed:.1f}s eta={remaining:.1f}s"
        )
        print(message, file=self.stream, flush=True)
