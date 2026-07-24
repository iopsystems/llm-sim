# Backlog

Consolidated index of deferred / reopen items, each tracing back to the journal
entry that owns its "why" and mechanism. Kept in step with `docs/journal/` (see
the engineering-journal skill).

## Open

- **Effort B — engine abstraction** (`EngineAdapter` interface,
  shared-core-plus-extras StepRecord schema, `--engine {vllm,sglang}` CLI).
  **Blocked on** the SGLang spike's verdict — the interface is deliberately
  not designed until the second implementation's shape is known.
  Source: [`docs/journal/2026-07-24-sglang-engine-spike.md`](journal/2026-07-24-sglang-engine-spike.md).
- **Cross-engine comparison tooling** (side-by-side runs, diff reports).
  A possible third effort once two engines actually run; explicitly not part
  of Effort B.
  Source: [`docs/journal/2026-07-24-sglang-engine-spike.md`](journal/2026-07-24-sglang-engine-spike.md).
- **Prefix-cache fidelity under realistic token identities.** Constant
  dummy-token prompts defeat prefix caching in both engines; radix-cache
  studies need realistic prompts. Noted as a v2 limitation; becomes acute if
  SGLang (radix-centric scheduler) lands.
  Source: [`docs/journal/2026-07-24-sglang-engine-spike.md`](journal/2026-07-24-sglang-engine-spike.md).
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

## Done

- **v2 implementation: EngineCore becomes the engine.** Done (PR #4) — the
  real `EngineCore.step()` loop
  replaced the MVP SimLoop, the KV budget became a sim knob
  (`--kv-cache-bytes` / `--num-blocks`), the virtual clock / cost model is
  wired loop-side, and the MVP harness (`llm_sim/harness/builder.py`,
  `llm_sim/sampler.py`) is retired after the EngineCore path reproduced the
  MVP scenarios. Record (including deviations from the sketched design):
  [`docs/journal/2026-07-09-gpu-mock-enginecore-boot.md`](journal/2026-07-09-gpu-mock-enginecore-boot.md)
  §"Implementation (shipped 2026-07-17)".
