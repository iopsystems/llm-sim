# Metrics Visualization — Design

**Date:** 2026-07-02
**Status:** both tracks implemented. Track 1 (`--viz`) and Track 2 (Rezolus
Parquet export) shipped; the schema spike below resolved to
`grouping_power=3`/`max_value_power=64` → 496 buckets, cumulative, `List<UInt64>`
`<metric>:buckets` columns with mandatory grouping metadata (see
`llm_sim/export/`).

## Goal

Make the simulator's key outputs legible from the get-go. Two independent
consumers of the existing per-step metrics, with a hard dependency boundary.

## KPI scope — scheduler dynamics only (faithful today)

Under the **constant** cost model, latency/throughput KPIs are degenerate
(every request latency collapses to `steps × constant`), so we deliberately do
**not** visualize them yet. We visualize only what comes straight from the real
scheduler's decisions and is therefore faithful regardless of the cost model:

- Batch composition — `tokens_scheduled`, `prefill_reqs`, `decode_reqs`
- Queue occupancy — `num_running`, `num_waiting`
- KV-block pressure — `blocks_used` vs `num_blocks`
- Progress — cumulative `finished`, cumulative `preemptions`

Latency/throughput and per-request distributions are revisited when a profiled
cost model lands (see the cost-model seam in `cost/base.py`).

## Data capture (unchanged)

`metrics.StepRecord` — one flat, scalar record per `schedule()` step — remains
the **single source of truth**, serialized as per-step JSONL (canonical) plus a
scalar summary dict. No histograms are stored; any distribution is derived by a
consumer. Format decision (measured): keep JSONL; if size ever bites, gzip it
(~18× on our data, beats msgpack); go columnar (Parquet) only for cross-run
analytics. msgpack is not worth it here (keyed form barely beats JSON; gzip-JSONL
beats even positional msgpack while staying greppable).

## Track 1 — lightweight live viz (core, zero heavy deps)

Immediate in-terminal dashboard, stdlib only. Rendered at end of a run and
re-runnable on any saved JSONL. NOT per-step streaming (a fast batch sim would
emit thousands of frames); "live" = immediate, no external tooling.

- **Module:** `llm_sim/viz.py`. Pure functions, no vLLM/pyarrow/rezolus imports.
- **Primitives (TDD):**
  - `_sparkline(values) -> str` — 8-level Unicode blocks `▁▂▃▄▅▆▇█`; empty→"",
    constant series→flat, normalized to the series' own range.
  - `_downsample(values, width, agg=max) -> list` — bucket to terminal width,
    peak-preserving (`max` per bucket). `width >= len` → identity.
  - `render(records, width=None) -> str` — header (step count, virtual-time span)
    + one labeled, annotated sparkline per KPI series + a peaks/totals footer
    (reusing `metrics.summary()`).
- **Entry points (shared renderer):**
  - `--viz` flag on the run → render from in-memory records to stderr (stdout
    stays clean for piping).
  - `python -m llm_sim.viz steps.jsonl` → re-render a saved run to stdout.

Example (illustrative):

```
vLLM scheduler sim — 71 steps, virtual time 0.71s
batch tokens  ▁▂▅▇▆▃▂▁   peak 229
prefill reqs  ▁▃▂▁▁▁▁    peak 3
decode reqs   ▁▂▄▅▆▅▃▂   peak 14
running       ▁▃▅▇▇▅▃▂   peak 15
waiting       ▁▁▁▁▁▁▁    peak 0
KV blocks     ▁▂▄▅▄▂▁    peak 99 / 500  (20%)
finished      ▁▂▃▄▅▆▇█   30 / 30
preemptions   ─────────  0 total
```

## Track 2 — Rezolus export (post-processing only, opt-in deps)

Rezolus is the rich renderer (quantile heatmaps over time, A/B diff heatmaps).
Our job: emit a Rezolus-compatible **Parquet** artifact from the JSONL. ALL
rezolus/pyarrow/histogram deps confined here, strictly post-run.

- **Module:** `llm_sim/export/rezolus.py`, behind optional extra
  `pip install -e ".[rezolus]"`. Reads JSONL only; never touches the sim core or
  Track 1.
- **Model:**
  - `vclock → timestamp` column (virtual seconds on the heatmap x-axis).
  - **Histogram-per-interval:** bucket steps by `vclock`; per interval build a
    **log-linear** histogram (iopsystems/histogram layout) of each metric's
    values across that interval's steps. This is the shape Rezolus heatmaps render.
  - Metrics histogrammed: `tokens_scheduled`, `num_running`, `blocks_used`
    (occupancy), `num_waiting`.
- **BLOCKED ON:** exact Rezolus Parquet histogram schema — column layout per
  snapshot, timestamp unit, `grouping_power`/`max_value_power`, cumulative-vs-delta.
  Being resolved via a source spike against the rezolus + histogram repos.

## Dependency boundary (the key constraint)

```
sim core (llm_sim/*, no viz deps)
        │ writes
        ▼
   steps.jsonl  ── canonical ──┐
        │                      │
        ▼ reads                ▼ reads
 Track 1 viz.py           Track 2 export/rezolus.py
 (stdlib only)            (pyarrow + rezolus fmt, optional extra)
```

## Sequencing

1. **Track 1 now** — no external unknowns; pure and TDD-friendly. Ship first.
2. **Track 2 after schema** — implement once the spike returns the parquet schema.

## Testing

- Track 1: `_sparkline` / `_downsample` are pure → exhaustive unit tests
  (empty, single, constant, full-range monotonic, width reduction). `render`
  asserted on structure (header present, one line per series, peaks shown).
- Track 2: golden-file test of the emitted Parquet schema against the spec, plus
  a round-trip that the Rezolus viewer/loader accepts it (once schema known).
