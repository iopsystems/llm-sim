# Backlog

Consolidated index of deferred / reopen items, each tracing back to the journal
entry that owns its "why" and mechanism. Kept in step with `docs/journal/` (see
the engineering-journal skill).

## Open

- **v2 implementation: EngineCore becomes the engine.** Spike returned GO
  (2026-07-17); the decided design replaces the MVP SimLoop
  (`llm_sim/engine.py`, `llm_sim/harness/builder.py`, `llm_sim/sampler.py`)
  with the real `EngineCore.step()` loop on `llm_sim/mock/`, reusing
  `llm_sim/{cost,clock,metrics,workload}`. Includes: wire the virtual clock /
  cost model at the `_model_forward` seam, make `MOCK_KV_CACHE_BYTES` a sim
  config knob, retire the MVP harness only after the EngineCore path
  reproduces the MVP integration scenarios.
  Source: [`docs/journal/2026-07-09-gpu-mock-enginecore-boot.md`](journal/2026-07-09-gpu-mock-enginecore-boot.md).
- **Activation guard for the mock platform plugin.** `llm_sim.mock:register`
  activates unconditionally for every vLLM use in the venv; add an env-var
  gate **if** the environment ever needs to run real vLLM alongside the sim.
  Source: [`docs/journal/2026-07-09-gpu-mock-enginecore-boot.md`](journal/2026-07-09-gpu-mock-enginecore-boot.md).
- **Latency/throughput + per-request distribution KPIs.** Deliberately not
  visualized while the cost model is **constant** (latency collapses to
  `steps × constant`, so any chart would faithfully depict an unfaithful model).
  **Reopen when** a profiled (non-constant) cost model lands at the seam in
  `llm_sim/cost/base.py`.
  Source: [`docs/journal/2026-07-02-metrics-visualization.md`](journal/2026-07-02-metrics-visualization.md).
