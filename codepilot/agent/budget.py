from threading import Lock


class DecisionBudget:
    """Shared accounting; only the main agent uses the threshold to stop work."""

    def __init__(self, threshold: int):
        if threshold < 1:
            raise ValueError("max_model_calls must be positive")
        self.threshold = threshold
        self.lock = Lock()
        self.counts = {"main": 0, "subagent": 0, "final_analysis": 0}

    def record(self, role: str) -> int:
        with self.lock:
            self.counts[role] += 1
            return sum(self.counts.values())

    def snapshot(self) -> dict[str, int]:
        with self.lock:
            return {**self.counts, "total": sum(self.counts.values())}

    @property
    def exhausted(self) -> bool:
        return self.snapshot()["total"] >= self.threshold
