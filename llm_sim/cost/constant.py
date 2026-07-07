"""Constant per-step cost model (MVP)."""

from typing import Any

from llm_sim.cost.base import CostModel


class ConstantCostModel(CostModel):
    def __init__(self, latency_s: float):
        self.latency_s = float(latency_s)

    def step_latency(self, scheduler_output: Any, state: Any) -> float:
        return self.latency_s
