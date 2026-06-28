"""Per-step metrics recording for the simulation.

Each step produces a ``StepRecord``; the recorder dumps them as JSONL and
computes a small summary (peaks + totals). Cumulative fields (``preemptions``,
``finished``) are stored as running totals, so the summary just reads the last.
"""

import dataclasses
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union


@dataclass
class StepRecord:
    step: int
    vclock: float
    num_running: int
    num_waiting: int
    prefill_reqs: int
    decode_reqs: int
    tokens_scheduled: int
    blocks_used: int
    num_blocks: int
    preemptions: int  # cumulative
    finished: int  # cumulative


class MetricsRecorder:
    def __init__(self):
        self.records: list[StepRecord] = []

    def record(self, **kwargs) -> None:
        self.records.append(StepRecord(**kwargs))

    def summary(self) -> dict:
        if not self.records:
            return {
                "num_steps": 0,
                "virtual_time_s": 0.0,
                "peak_running": 0,
                "peak_waiting": 0,
                "peak_blocks_used": 0,
                "peak_batch_tokens": 0,
                "num_blocks": 0,
                "total_preemptions": 0,
                "total_finished": 0,
            }
        last = self.records[-1]
        return {
            "num_steps": len(self.records),
            "virtual_time_s": last.vclock,
            "peak_running": max(r.num_running for r in self.records),
            "peak_waiting": max(r.num_waiting for r in self.records),
            "peak_blocks_used": max(r.blocks_used for r in self.records),
            "peak_batch_tokens": max(r.tokens_scheduled for r in self.records),
            "num_blocks": last.num_blocks,
            "total_preemptions": last.preemptions,
            "total_finished": last.finished,
        }

    def write_jsonl(self, path: Union[str, Path]) -> None:
        path = Path(path)
        with path.open("w") as f:
            for r in self.records:
                f.write(json.dumps(asdict(r)) + "\n")
