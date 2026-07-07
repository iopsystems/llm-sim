"""Sim-loop-owned virtual clock.

In scheduler-in-isolation the scheduler does not read the wall clock to make
decisions, so this is plain bookkeeping: the sim loop advances it by the
cost model's per-step latency and uses it to gate request arrivals.
"""


class VirtualClock:
    def __init__(self, start: float = 0.0):
        self._now = float(start)

    def now(self) -> float:
        return self._now

    def advance(self, dt: float) -> None:
        if dt < 0:
            raise ValueError(f"cannot advance clock by negative dt={dt}")
        self._now += dt

    def fast_forward_to(self, t: float) -> None:
        """Jump forward to ``t``. Monotonic: an earlier target is a no-op."""
        if t > self._now:
            self._now = float(t)
