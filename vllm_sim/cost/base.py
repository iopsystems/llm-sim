"""Cost model interface.

The per-step latency is the one hard input to the simulation (all the fidelity
risk lives here). The MVP ships a constant implementation; profiled/learned
models can be swapped in behind this same interface later.
"""

from abc import ABC, abstractmethod
from typing import Any


class CostModel(ABC):
    @abstractmethod
    def step_latency(self, scheduler_output: Any, state: Any) -> float:
        """Return the virtual-time cost (seconds) of executing one scheduled step."""
        raise NotImplementedError
