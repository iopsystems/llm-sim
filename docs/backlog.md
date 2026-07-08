# Backlog

Consolidated index of deferred / reopen items, each tracing back to the journal
entry that owns its "why" and mechanism. Kept in step with `docs/journal/` (see
the engineering-journal skill).

## Open

- **Latency/throughput + per-request distribution KPIs.** Deliberately not
  visualized while the cost model is **constant** (latency collapses to
  `steps × constant`, so any chart would faithfully depict an unfaithful model).
  **Reopen when** a profiled (non-constant) cost model lands at the seam in
  `llm_sim/cost/base.py`.
  Source: [`docs/journal/2026-07-02-metrics-visualization.md`](journal/2026-07-02-metrics-visualization.md).
