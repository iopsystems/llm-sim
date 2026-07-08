# Metrics visualization — two-track (live ASCII + Rezolus export)

**Status:** SHIPPED — PR #2 (merge `21c7695`). Retrospective entry: bootstrapped
from the shipped code + the `2026-07-02-metrics-visualization-design.md` scratch
design (now absorbed here; that doc is local-only scratch, untracked).

**Effort dates:** opened 2026-07-02, landed via PR #2.

## Goal

Make the simulator's key outputs legible without external tooling, while keeping
heavy rendering dependencies out of the sim core. Two independent consumers of
the existing per-step metrics, separated by a hard dependency boundary:

```
sim core (llm_sim/*, no viz deps)
        │ writes
        ▼
   steps.jsonl  ── canonical ──┐
        │ reads                │ reads
        ▼                      ▼
 Track 1  llm_sim/viz.py   Track 2  llm_sim/export/rezolus.py
 (stdlib only)             (pyarrow + h2histogram, optional extra)
```

`metrics.StepRecord` — one flat scalar record per `schedule()` step — stays the
single source of truth, serialized as per-step JSONL. No histograms are stored in
the core; any distribution is derived by a consumer.

## Decisions & rationale

**Serialization: keep JSONL (measured).** If size ever bites, gzip it (~18× on
our data) rather than switching formats; gzip-JSONL beats even positional msgpack
while staying greppable. msgpack is not worth it here (keyed form barely beats
JSON). Go columnar (Parquet) only for cross-run analytics — which is exactly what
Track 2 is, downstream of the JSONL, never a replacement for it.

**KPI scope — scheduler dynamics only (NO-GO on latency/throughput for now).**
Under the current **constant** cost model, latency/throughput KPIs are degenerate:
every request's latency collapses to `steps × constant`, so a latency chart would
be a faithful picture of an unfaithful model. We deliberately do **not** visualize
latency/throughput or per-request distributions yet. We visualize only what comes
straight from the real scheduler's decisions and is therefore faithful regardless
of cost model: batch composition (`tokens_scheduled`, `prefill_reqs`,
`decode_reqs`), queue occupancy (`num_running`, `num_waiting`), KV-block pressure
(`blocks_used` vs `num_blocks`), and progress (cumulative `finished`,
`preemptions`).
**Reopen when** a profiled (non-constant) cost model lands — see the cost-model
seam in `llm_sim/cost/base.py`. Mirrored in `docs/backlog.md`.

## Track 1 — live terminal viz (`091524f`)

`llm_sim/viz.py`, stdlib only (`argparse`/`json`/`shutil`), no vLLM/pyarrow/rezolus
imports. Rendered at end of a run and re-runnable on any saved JSONL. NOT per-step
streaming — "live" = immediate, no external tooling (a fast batch sim would emit
thousands of frames).

- `_sparkline(values)` — 8-level Unicode blocks (`BLOCKS = "▁▂▃▄▅▆▇█"`); empty→"",
  constant series→flat, normalized to the series' own range.
- `_downsample(values, width, agg=max)` — peak-preserving bucketing to terminal
  width (`max` per bucket); `width >= len` → identity.
- `render(records, width=None)` — header (step count, virtual-time span) + one
  labeled, annotated sparkline per KPI series + a peaks/totals footer (reuses
  `metrics.summary()`).

Entry points share the renderer: `--viz` flag renders in-memory records to
**stderr** (stdout stays clean for piping); `python -m llm_sim.viz steps.jsonl`
re-renders a saved run to stdout. 15 unit tests in `tests/test_viz.py` (pure
primitives: empty/single/constant/full-range/width-reduction; `render` asserted
on structure).

## Track 2 — Rezolus-compatible Parquet export (`979a3f7`, `667e782`, `2c92d65`, `0cb2721`)

`llm_sim/export/rezolus.py`, behind optional extra
`pip install -e ".[rezolus]"` (`rezolus = ["pyarrow>=14", "h2histogram>=0.1"]` in
`pyproject.toml`). Reads JSONL only; never touches the sim core or Track 1.

Resolved schema (the original design's BLOCKED-ON item — settled by a source spike
against the rezolus + histogram repos):

- **`timestamp`**: `UInt64` nanoseconds; virtual time (`vclock`) mapped onto it.
- **`<metric>:buckets`**: Arrow `List<UInt64>`, one dense **496-bucket** array per
  row. Buckets are a log-linear histogram at **`grouping_power=3`,
  `max_value_power=64`** (→ 496 buckets), and are **CUMULATIVE** (free-running);
  the viewer diffs consecutive rows. Steps are bucketed by `vclock` into intervals;
  empty intervals repeat the previous cumulative snapshot.
- Bucketing uses the canonical **`h2histogram`** library (iopsystems, pure Python,
  zero deps) via `llm_sim/export/histogram.py`. This **replaced an earlier
  hand-port** of the Rust `value_to_index` (`667e782`); the old
  `vllm_sim/export/histogram.py` hand-port was verified byte-identical before
  removal.
- Metrics histogrammed (`DEFAULT_METRICS`): `tokens_scheduled`, `num_running`,
  `blocks_used`, `num_waiting`.
- **Descriptions**: carried as a single **file-level `descriptions` footer** in
  the Parquet schema metadata (`0cb2721`) — the fix corrected an earlier
  per-column placement (`2c92d65`) to match how Rezolus/metriken-exposition reads
  them.

Tests: `tests/test_export_histogram.py` (7 — bucketing parity), 
`tests/test_export_parquet.py` (10 — schema/round-trip), 
`tests/test_export_rezolus_rows.py` (6 — cumulative snapshot resampling).

## Deferred / reopen

- **Latency/throughput + per-request distribution KPIs** — reopen when a profiled
  cost model lands (`llm_sim/cost/base.py`). See `docs/backlog.md`.

## Notes for the next agent

- The dependency boundary is the load-bearing constraint: the sim core must stay
  free of `pyarrow`/`rezolus`/`h2histogram`. Keep those imports inside
  `llm_sim/export/`.
- The package was renamed `vllm_sim` → `llm_sim` (`40c736a`) during this arc; any
  older path references (`vllm_sim/…`) are stale.
