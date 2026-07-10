from collections import deque
from statistics import median

class QueueSmoother:
    def __init__(self, window_size: int = 5, max_change_per_min: float = 1.5):
        self.readings = deque(maxlen=window_size)  # (timestamp, value)
        self.max_change_per_min = max_change_per_min

    def add_reading(self, timestamp, value: float) -> float:
        # Step 1: reject if implausible vs. current smoothed estimate
        if self.readings:
            last_ts, last_smoothed = self.readings[-1][0], self.current_estimate()
            elapsed_min = max((timestamp - last_ts).total_seconds() / 60, 0.01)
            max_delta = self.max_change_per_min * elapsed_min
            if abs(value - last_smoothed) > max_delta * 3:  # generous slack, just catches wild jumps
                # implausible spike — don't add it raw, but don't discard the info either;
                # log it for review, and feed in a capped version instead
                value = last_smoothed + max_delta * (1 if value > last_smoothed else -1)

        self.readings.append((timestamp, value))
        return self.current_estimate()

    def current_estimate(self) -> float:
        if not self.readings:
            return 0
        return median(v for _, v in self.readings)