"""Track 1 — lightweight live terminal visualization (stdlib only).

Renders the faithful scheduler-dynamics KPIs (batch composition, queue
occupancy, KV-block pressure, progress) as Unicode-block sparklines. No heavy
deps, no vLLM import: consumes plain per-step records (StepRecord dataclasses or
dicts), so it works both inline (`--viz`) and standalone on a saved JSONL.

Deliberately NOT latency/throughput — those are degenerate under the constant
cost model. See docs/plans/2026-07-02-metrics-visualization-design.md.
"""

import argparse
import json
import shutil
from dataclasses import asdict, is_dataclass
from typing import Iterable, List, Optional

BLOCKS = "▁▂▃▄▅▆▇█"


def _sparkline(values: List[float]) -> str:
    """One Unicode block per value, normalized to the series' own [min,max].

    Empty -> "". A constant series has no variation -> flat baseline (lowest
    block), which reads as "nothing changed" rather than a misleading ramp.
    """
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi == lo:
        return BLOCKS[0] * len(values)
    span = hi - lo
    top = len(BLOCKS) - 1
    out = []
    for v in values:
        idx = int((v - lo) / span * top + 0.5)  # nearest level
        out.append(BLOCKS[max(0, min(top, idx))])
    return "".join(out)


def _downsample(values, width: int, agg=max) -> list:
    """Bucket a series into at most ``width`` contiguous buckets.

    Peak-preserving by default (``agg=max``) so a transient spike (e.g. a
    preemption burst) survives the reduction instead of averaging away.
    """
    n = len(values)
    if n == 0:
        return []
    if width >= n:
        return list(values)
    out = []
    for i in range(width):
        start = i * n // width
        end = (i + 1) * n // width
        bucket = values[start:end] or [values[min(start, n - 1)]]
        out.append(agg(bucket))
    return out


def _as_dict(r):
    return asdict(r) if is_dataclass(r) else r


def render(records: Iterable, width: int = None) -> str:
    """Render a terminal dashboard for a run's per-step records."""
    recs = [_as_dict(r) for r in records]
    n = len(recs)
    vt = recs[-1]["vclock"] if recs else 0.0

    lines = [f"vLLM scheduler sim — {n} steps, virtual time {vt:.2f}s"]
    if not recs:
        return "\n".join(lines)

    if width is None:
        term = shutil.get_terminal_size((80, 24)).columns
        width = max(10, term - 34)

    def col(key):
        return [r[key] for r in recs]

    def spark(key):
        return _sparkline(_downsample(col(key), width))

    budget = recs[-1]["num_blocks"]
    peak_blocks = max(col("blocks_used"))
    occ = (peak_blocks / budget * 100) if budget else 0.0

    series = [
        ("batch tokens", spark("tokens_scheduled"), f"peak {max(col('tokens_scheduled'))}"),
        ("prefill reqs", spark("prefill_reqs"), f"peak {max(col('prefill_reqs'))}"),
        ("decode reqs", spark("decode_reqs"), f"peak {max(col('decode_reqs'))}"),
        ("running", spark("num_running"), f"peak {max(col('num_running'))}"),
        ("waiting", spark("num_waiting"), f"peak {max(col('num_waiting'))}"),
        ("KV blocks", spark("blocks_used"), f"peak {peak_blocks} / {budget}  ({occ:.0f}%)"),
        ("finished", spark("finished"), f"{col('finished')[-1]} done"),
        ("preemptions", spark("preemptions"), f"{col('preemptions')[-1]} total"),
    ]
    labelw = max(len(label) for label, _, _ in series)
    for label, sp, annot in series:
        lines.append(f"{label.ljust(labelw)}  {sp}  {annot}")
    return "\n".join(lines)


def _load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: Optional[List[str]] = None) -> str:
    """Standalone: render a saved run's per-step JSONL to stdout."""
    p = argparse.ArgumentParser(
        prog="vllm_sim.viz",
        description="Render a saved run's per-step JSONL as a terminal dashboard.",
    )
    p.add_argument("jsonl", help="path to per-step JSONL produced by a sim run")
    p.add_argument("--width", type=int, default=None, help="sparkline width (chars)")
    args = p.parse_args(argv)
    out = render(_load_jsonl(args.jsonl), width=args.width)
    print(out)
    return out


if __name__ == "__main__":
    main()
