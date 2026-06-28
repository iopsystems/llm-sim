"""Trace-file workload loader (CSV or JSONL), behind the same WorkloadSource
interface as the synthetic generator.

Each row needs ``request_id``, ``arrival_time``, ``prompt_len``, ``output_len``.
Specs are returned sorted by arrival time (stable on ties).
"""

import csv
import json
from pathlib import Path
from typing import Iterable, Union

from vllm_sim.workload.base import RequestSpec


class TraceWorkload:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)

    def _row_to_spec(self, row: dict) -> RequestSpec:
        return RequestSpec(
            request_id=str(row["request_id"]),
            arrival_time=float(row["arrival_time"]),
            prompt_len=int(row["prompt_len"]),
            output_len=int(row["output_len"]),
        )

    def generate(self) -> Iterable[RequestSpec]:
        suffix = self.path.suffix.lower()
        if suffix == ".csv":
            with self.path.open(newline="") as f:
                specs = [self._row_to_spec(row) for row in csv.DictReader(f)]
        elif suffix in (".jsonl", ".ndjson"):
            with self.path.open() as f:
                specs = [
                    self._row_to_spec(json.loads(line))
                    for line in f
                    if line.strip()
                ]
        else:
            raise ValueError(
                f"unsupported trace extension {suffix!r}; use .csv or .jsonl"
            )
        return sorted(specs, key=lambda s: s.arrival_time)
