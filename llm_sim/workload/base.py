"""Workload interface shared by synthetic and trace sources.

A ``RequestSpec`` is the GPU-free, vLLM-free description of one request: when it
arrives and how long its prompt / output are. The factory turns it into a real
vLLM ``Request``; everything upstream of the factory deals only in specs.
"""

from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    arrival_time: float
    prompt_len: int
    output_len: int


@runtime_checkable
class WorkloadSource(Protocol):
    def generate(self) -> Iterable[RequestSpec]:
        """Return request specs in non-decreasing ``arrival_time`` order."""
        ...
