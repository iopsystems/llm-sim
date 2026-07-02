"""Track 2 — Rezolus-compatible Parquet export (post-processing, opt-in deps).

Reads a run's per-step JSONL and emits a Parquet file the Rezolus viewer can
render as quantile/heatmap over (virtual) time. Strictly post-run; the only
heavy dep (pyarrow) is imported lazily inside `write_parquet`, so importing this
module — and the sim core — never requires it.

Model (verified against metriken-exposition / metriken-query sources):
  - `timestamp`: UInt64 nanoseconds. We map virtual time (vclock) onto it.
  - `<metric>:buckets`: Arrow List<UInt64>, one dense 496-bucket array per row,
    log-linear Config(grouping_power=3, max_value_power=64).
  - Buckets are CUMULATIVE (free-running); the viewer diffs consecutive rows to
    get each interval's distribution. So we accumulate a running histogram of the
    metric's per-step values and snapshot it once per fixed virtual-time interval.
  - Each histogram column MUST carry grouping_power/max_value_power field
    metadata or the viewer silently drops it. File-level `sampling_interval_ms`
    must be a clean integer.

The row-building (`build_histogram_rows`, `choose_interval_ns`) is pure and
unit-tested without pyarrow.
"""

import argparse
import json
from typing import Dict, List, Optional, Tuple

from vllm_sim.export.histogram import LogLinearHistogram

# The faithful scheduler-dynamics metrics worth a distribution-over-time view.
DEFAULT_METRICS = ["tokens_scheduled", "num_running", "blocks_used", "num_waiting"]

GROUPING_POWER = 3
MAX_VALUE_POWER = 64
_MS_NS = 1_000_000


def _t_ns(vclock: float) -> int:
    return int(round(vclock * 1e9))


def choose_interval_ns(records, target_rows: int = 150) -> int:
    """Pick a virtual-time interval (ns, whole ms) targeting ~target_rows rows."""
    recs = list(records)
    if not recs:
        return _MS_NS
    span_ns = max(_t_ns(r["vclock"]) for r in recs)
    if span_ns <= 0:
        return _MS_NS
    raw = span_ns // max(1, target_rows)
    return max(_MS_NS, (raw // _MS_NS) * _MS_NS)  # round down to whole ms, >= 1ms


def build_histogram_rows(
    records,
    metrics: List[str],
    interval_ns: int,
    base_ns: int = 0,
    grouping_power: int = GROUPING_POWER,
    max_value_power: int = MAX_VALUE_POWER,
) -> Tuple[List[int], Dict[str, List[List[int]]]]:
    """Resample per-step records into cumulative per-interval histogram snapshots.

    Returns (timestamps_ns, {metric: [bucket_array_per_row]}). One row per
    interval from 0..max; empty intervals repeat the previous cumulative snapshot.
    """
    recs = sorted(records, key=lambda r: r["vclock"])
    if not recs:
        return [], {m: [] for m in metrics}

    hists = {m: LogLinearHistogram(grouping_power, max_value_power) for m in metrics}
    by_interval: Dict[int, list] = {}
    for r in recs:
        by_interval.setdefault(_t_ns(r["vclock"]) // interval_ns, []).append(r)
    max_k = _t_ns(recs[-1]["vclock"]) // interval_ns

    timestamps: List[int] = []
    columns: Dict[str, List[List[int]]] = {m: [] for m in metrics}
    for k in range(max_k + 1):
        for r in by_interval.get(k, []):
            for m in metrics:
                hists[m].record(int(r[m]))
        timestamps.append(base_ns + k * interval_ns)
        for m in metrics:
            columns[m].append(hists[m].snapshot())
    return timestamps, columns


def write_parquet(
    records,
    path: str,
    metrics: List[str] = None,
    interval_ns: Optional[int] = None,
    target_rows: int = 150,
    base_epoch_ns: int = 0,
    source: str = "vllm-sim",
) -> dict:
    """Write a Rezolus-viewer-compatible Parquet file. Returns a small manifest."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - dependency guard
        raise ImportError(
            "Rezolus export needs pyarrow. Install the optional extra: "
            'pip install -e ".[rezolus]"'
        ) from e

    metrics = metrics or DEFAULT_METRICS
    recs = list(records)
    if interval_ns is None:
        interval_ns = choose_interval_ns(recs, target_rows)
    interval_ns = max(_MS_NS, (interval_ns // _MS_NS) * _MS_NS)  # whole ms
    sampling_interval_ms = interval_ns // _MS_NS

    ts_ns, cols = build_histogram_rows(recs, metrics, interval_ns, base_ns=base_epoch_ns)

    item = pa.field("item", pa.uint64(), nullable=True)
    list_type = pa.list_(item)
    hist_meta = {
        "grouping_power": str(GROUPING_POWER),
        "max_value_power": str(MAX_VALUE_POWER),
        "metric_type": "histogram",
        "source": source,
    }
    schema = pa.schema(
        [
            pa.field("timestamp", pa.uint64(), nullable=False,
                     metadata={"metric_type": "timestamp", "unit": "nanoseconds"}),
            *[
                pa.field(f"{m}:buckets", list_type, nullable=True, metadata=hist_meta)
                for m in metrics
            ],
        ],
        metadata={"sampling_interval_ms": str(sampling_interval_ms), "source": source},
    )
    arrays = [pa.array(ts_ns, pa.uint64())]
    arrays += [pa.array(cols[m], type=list_type) for m in metrics]
    table = pa.Table.from_arrays(arrays, schema=schema)
    pq.write_table(table, path, compression="zstd")

    return {
        "path": path,
        "rows": len(ts_ns),
        "metrics": list(metrics),
        "num_buckets": LogLinearHistogram(GROUPING_POWER, MAX_VALUE_POWER).num_buckets,
        "sampling_interval_ms": sampling_interval_ms,
    }


def _load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: Optional[List[str]] = None) -> dict:
    p = argparse.ArgumentParser(
        prog="vllm_sim.export.rezolus",
        description="Convert a run's per-step JSONL to a Rezolus-compatible Parquet.",
    )
    p.add_argument("jsonl", help="per-step JSONL from a sim run")
    p.add_argument("-o", "--out", required=True, help="output .parquet path")
    p.add_argument("--metrics", nargs="+", default=None,
                   help=f"metrics to histogram (default: {' '.join(DEFAULT_METRICS)})")
    p.add_argument("--target-rows", type=int, default=150,
                   help="approximate number of time buckets (heatmap resolution)")
    p.add_argument("--interval-ms", type=int, default=None,
                   help="explicit virtual-time bucket width (overrides --target-rows)")
    args = p.parse_args(argv)

    interval_ns = args.interval_ms * _MS_NS if args.interval_ms else None
    manifest = write_parquet(
        _load_jsonl(args.jsonl), args.out,
        metrics=args.metrics, interval_ns=interval_ns, target_rows=args.target_rows,
    )
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()
