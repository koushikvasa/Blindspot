"""Run-level cost guard for the loop (H4): actual spend so far plus the next step's estimate vs a limit."""
import threading

from blindspot.generators.claude_client import CostLimitExceeded, Usage


class Budget:
    def __init__(self, limit_usd: float):
        self.limit = limit_usd
        self._usages: list[Usage] = []
        self._lock = threading.Lock()

    def register(self, usage: Usage) -> None:
        with self._lock:
            self._usages.append(usage)

    @property
    def spent(self) -> float:
        with self._lock:
            return sum(u.cost_usd for u in self._usages)

    def check(self, next_estimate_usd: float, label: str = "") -> None:
        """Raise before any paid call if spend so far + this step's estimate would pass the limit."""
        spent = self.spent
        if spent + next_estimate_usd > self.limit:
            raise CostLimitExceeded(
                f"{label}: spent ${spent:.2f} + next step ~${next_estimate_usd:.2f} exceeds the "
                f"${self.limit:.2f} limit; stopped before calling. Everything so far is cached; rerun resumes.")
