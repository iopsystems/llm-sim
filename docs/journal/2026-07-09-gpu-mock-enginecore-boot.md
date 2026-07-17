# v2: GPU-mock — boot the real vLLM stack in-process (EngineCore)

**Status:** CLOSED — **shipped** (2026-07-17). Spike returned **GO** (see
"Spike verdict" below); the EngineCore-backed engine replaced the MVP SimLoop
on branch `feat/v2-enginecore-engine`, landed via that branch's implementing
PR (see "Implementation" below).

**Opened:** 2026-07-09.

## Goal

Move llm-sim past scheduler-in-isolation: boot the **real** vLLM stack GPU-free,
in-process, so `EngineCore.step()` runs end-to-end — real Worker + `GPUModelRunner`
init and per-step machinery, not just the scheduler. The value is fidelity to
vLLM's actual init/execute code paths (metadata construction, the real step loop),
which the MVP's hand-rolled harness (`llm_sim/engine.py` SimLoop,
`llm_sim/harness/builder.py`, `llm_sim/sampler.py`) deliberately bypasses.

**Boundary (decided):** `EngineCore.step()` **in-process, single-process** — no
ZMQ/subprocess. EngineCore owns the scheduler and the step loop; the virtual clock
stays simple (one process, host-side time reads only — the GPU mock means no real
CUDA events to intercept). The full multiprocess/ZMQ path and `LLM.generate()`
offline surface are explicitly out of scope for v2.

Context and prior analysis: `vllm-sim-handoff.md` §"Mocking the GPU" and §"Suggested
first tasks". Pinned to `vllm==0.23.0` (platform/`GPUModelRunner` layers churn — see
that file's version caveat).

## Decision to make (deferred to the spike)

How v2 relates to the MVP is **not** pre-decided — it hangs on how invasive the
GPU mock turns out to be:

- **Small/clean injection surface → replace the loop.** EngineCore becomes THE
  engine; retire the hand-rolled SimLoop/builder/sampler; reuse `llm_sim/cost/`,
  `clock`, `metrics`, `workload` behind EngineCore's forward-pass + memory seams.
- **Large/churny surface → coexist as opt-in track 2**, keeping the low-churn
  scheduler-in-isolation path intact.
- **Can't cheaply stub → NO-GO**, recorded with mechanism + reopen condition.

Probe before building. Do not implement the design until the spike returns a
verdict.

## The spike

**One question:** Can a `MockPlatform` + stubbed forward pass make `EngineCore.step()`
run end-to-end with no GPU/CUDA and no subprocess on `vllm==0.23.0` — and *how
bounded* is the injection surface?

**Probes, cheapest-first (stop at first hard blocker):**

1. **Platform init.** Register a `MockPlatform` from vLLM's CPU-platform template;
   get `VllmConfig`/EngineCore to initialize with no CUDA present (device caps,
   `mem_get_info`, FP8/MX checks). *Does vLLM pick up an injected platform cleanly
   (plugin seam) or does it require monkeypatching?* — this is the load-bearing
   signal for replace-vs-coexist.
2. **EngineCore construction.** Does it build GPU-free? Known snags:
   `determine_num_available_blocks` (→ analytic memory model; `num_blocks` drives
   preemption, so it matters), attention-backend selection (**preserve** — that's
   the real "kernel selection", only kernels get stubbed), KV-cache init.
3. **Step loop.** Replace `GPUModelRunner.execute_model` with a synthetic-sampler +
   constant-cost stub; run `EngineCore.step()` for N steps on a trivial workload.
   Are scheduler decisions sane (a batch forms, `num_computed_tokens` advances, a
   request reaches finished)?

## GO / NO-GO (number-gated)

- **GO** — `EngineCore.step()` completes a full short run (one request admitted →
  decoded → finished) in-process, GPU-free, with the injection surface **bounded**:
  a clean platform plugin, or ≤ ~5 well-understood stub points.
- **NO-GO** — booting requires the ZMQ/subprocess path to even initialize;
  `GPUModelRunner` init carries CUDA dependencies that can't be cheaply stubbed; or
  the stub surface is unbounded / churns catastrophically across the pin. Land the
  negative result with its mechanism and a reopen condition (e.g. "revisit on a
  vLLM release that exposes a cleaner platform seam").

Either verdict closes out here honestly — a well-measured NO-GO on this surface is
exactly the kind of dead-end worth landing so it isn't re-paid later.

## Strategic check (keep in view)

Per `vllm-sim-handoff.md` §"Strategic check": if the end goal is scheduling-policy
*research* rather than fidelity to vLLM's exact code paths, scheduler-in-isolation
already gets most of the value at a fraction of the surface. The full platform-mock
earns its cost only if runner/backend-level behaviors (graph padding, token
budgeting, backend selection) become objects of study. The spike is priced
accordingly — cheap probe, honest verdict.

## Spike verdict — GO (2026-07-17)

All three probes passed on `vllm==0.23.0`, Linux, no GPU (`torch.cuda.
is_available() == False`, CUDA wheel `torch 2.11.0+cu130`). The spike code
landed as `llm_sim/mock/` with the gate as an integration test,
`tests/test_mock_enginecore.py`.

**Probe 1 — platform init: clean plugin seam, zero monkeypatching.** vLLM 0.23.0
resolves platforms lazily through the `vllm.platform_plugins` entry-point group
(`vllm/platforms/__init__.py`, `resolve_current_platform_cls_qualname`), and an
out-of-tree plugin takes priority over builtins. On a GPU-less box with the CUDA
wheel, *no* builtin platform activates (CUDA needs NVML devices; CPU needs a
`+cpu` build or macOS), so our plugin is the sole winner. This was the
load-bearing signal for replace-vs-coexist, and it came back as clean as it
gets: `llm_sim.mock:register` in `pyproject.toml` is the whole injection.

**Probe 2 — EngineCore constructs GPU-free, in-process.** Key discovery: in
0.23.0 the CPU lineage *is* the real GPU machinery — `CPUWorker` subclasses
`gpu_worker.Worker` (`vllm/v1/worker/cpu_worker.py:33`) and `CPUModelRunner`
subclasses `GPUModelRunner` (`vllm/v1/worker/cpu_model_runner.py:22`) — so
inheriting `CpuPlatform` keeps vLLM's real init and per-step code paths.
Construction: real `UniProcExecutor` → `MockWorker(Worker)` →
`MockModelRunner(CPUModelRunner)` → dummy-loaded `facebook/opt-125m`, real
attention-backend selection (`CPU_ATTN`), real `Scheduler`. KV sizing comes
from the analytic budget (1 GiB → `num_gpu_blocks=113` at `block_size=128`,
fp32), not host RAM.

**Probe 3 — `EngineCore.step()` end-to-end.** Two requests (prompt/output
100/8 and 48/4) admitted → chunked-prefill → decode → finished, with
hand-computable step counts: first token samples at step 0 (both prompts fit
one 2048-token prefill budget), `output_len=N` finishes at step `N-1` —
observed `{r1: 3, r0: 7}`. Real metadata construction, real logits projection,
real `Sampler`; only the transformer forward is synthetic. Letting the *real*
forward run was also probed and is a NO-GO on the CUDA wheel: `CPU_ATTN`'s
kernels live in CPU-build-only `csrc/cpu` ops that the CUDA wheel lacks.

**Injection surface (the GO criterion asked ≤ ~5 well-understood points):**

1. Entry-point platform plugin `MockPlatform(CpuPlatform)` — sanctioned seam
   (`llm_sim/mock/platform.py`).
2. `check_and_update_config` override: keep the uniproc executor (CpuPlatform
   forces multiprocess for OMP thread binding) and point `worker_cls` at
   MockWorker — both settings platforms are *designed* to own.
3. `MockWorker(Worker)` (`llm_sim/mock/worker.py`): drops `CPUWorker.__init__`'s
   NUMA binding (needs CPU-build-only `torch.ops._C.init_cpu_memory_env`),
   replaces `determine_available_memory` with the analytic KV budget, skips
   model warmup.
4. `MockModelRunner._model_forward` — GPUModelRunner's *documented* override
   seam ("This method can be overridden by subclasses for model execution",
   `vllm/v1/worker/gpu_model_runner.py:3718`).
5. Two pure-torch `torch.library` FRAGMENT fallbacks for CPU-build-only `_C`
   ops the decode path hits (`llm_sim/mock/ops.py`):
   `compute_slot_mapping_kernel_impl` (real slot-mapping math) and
   `get_scheduler_metadata` (opaque, consumed only by the never-run attention
   kernel). Supported registration API, not monkeypatching.

**Decision: replace the loop** — per this entry's pre-committed rule
("small/clean injection surface → replace"). EngineCore becomes THE engine;
`llm_sim/engine.py` SimLoop, `llm_sim/harness/builder.py`, and
`llm_sim/sampler.py` retire once the EngineCore-backed path reproduces the
MVP integration scenarios; `llm_sim/{cost,clock,metrics,workload}` are reused
behind the `_model_forward` + `determine_available_memory` seams.

**Honest limitations (carried into implementation):**

- The entry-point plugin activates for *every* vLLM use in this venv (the
  `register()` gate is unconditional). MVP tests are unaffected (95/95 pass
  with the plugin installed), but any future real-vLLM use in the same env
  would need an activation guard (e.g. env-var gate in `register()`).
- Sampled token identities come from the real Sampler over dummy-weight logits
  of a zeros hidden state — arbitrary, so prefix-cache hit fidelity remains
  deferred (as scoped at open).
- Spec-decode/eagle `_C` fallbacks are not provided — speculative decoding
  stays out of v2 scope.
- KV budget is a constant (`MOCK_KV_CACHE_BYTES = 1 GiB`); making it a sim
  config knob is part of the implementation (it drives `num_blocks`, hence
  preemption). — **Resolved in the implementation:**
  `llm_sim.mock.worker.set_kv_cache_bytes()` (default
  `DEFAULT_KV_CACHE_BYTES` = 1 GiB) plus CLI flags `--kv-cache-bytes` /
  `--num-blocks`.
- The virtual clock is not yet wired: EngineCore's step loop still runs on
  wall time. Charging `cost_model` virtual time at the `_model_forward` seam
  is the core of the implementation work. — **Resolved in the
  implementation**, though loop-side rather than at `_model_forward` — see
  deviation 1 below.

## Implementation (shipped 2026-07-17)

Landed on `feat/v2-enginecore-engine` (commits `03b17cf`..`d42b3e6` off
`6dd96f3`, via the branch's implementing PR). The spike's replace decision
executed as decided; what shipped:

- `llm_sim/harness/enginecore.py` — `build_engine_core()`: boots a live
  EngineCore (`load_format="dummy"`, `enforce_eager=True`, `dtype="float32"`,
  `seed=42`); KV sizing is sim-owned via `num_gpu_blocks_override` (exact
  block counts) or the analytic-budget knob
  `llm_sim.mock.worker.set_kv_cache_bytes()` (`DEFAULT_KV_CACHE_BYTES` =
  1 GiB); `SimScheduler` is injected by setting
  `scheduler_config.scheduler_cls` after `create_engine_config()` — safe
  because EngineCore resolves it at construction
  (`vllm/v1/engine/core.py:136`) and nothing reads it earlier.
- `llm_sim/harness/scheduler.py` — `SimScheduler(Scheduler)`: the real
  `schedule()`, observed not modified; captures each step's `SchedulerOutput`
  for the cost model and metrics. The loop nulls the capture before every
  `core.step()` so a stale capture trips an assert instead of being silently
  reused.
- `llm_sim/engine.py` — SimLoop rewritten to drive `core.step()`: admits
  arrivals gated by the virtual clock, charges the cost model per step,
  records metrics. Zero-token steps (the trailing cleanup step after the last
  finish, or a wedged workload) are never recorded. Prefill/decode
  classification recovers the pre-step computed count as
  `num_computed_tokens - n_sched` — valid because `update_from_output` leaves
  `num_computed_tokens` alone outside the spec-decode/KV-connector paths
  (verified against the pinned 0.23.0; comment at the classification site).
- `llm_sim/cli.py` — boots EngineCore. `--num-blocks` is now optional (unset →
  derive from the budget; the default 1 GiB gives 910 blocks for opt-125m
  fp32 at block 16); `--kv-cache-bytes` added. KV-capacity `ValueError`s are
  translated into advice naming `--num-blocks` / `--kv-cache-bytes` /
  `--max-model-len`, gated on the two verbatim raise-site messages of
  `_check_enough_kv_cache_memory` (`vllm/v1/core/kv_cache_utils.py:720` and
  `:740`) — note the `:720` message ("No available memory for the cache
  blocks") does not contain the substring "KV cache", so a naive gate would
  miss it. Any other `ValueError` surfaces vLLM's message without the KV
  advice.
- Retired (per the spike's replace rule): `llm_sim/harness/builder.py`,
  `llm_sim/sampler.py`, `tests/test_builder.py`, `tests/test_sampler.py`
  (11 tests) — the scheduler-in-isolation harness is gone.

### Deviations from the sketch above

1. **Virtual time is charged loop-side, not inside `_model_forward`.** The
   plan was to charge cost at the `_model_forward` seam; the shipped loop
   charges it after `core.step()` returns, from the captured
   `SchedulerOutput`. Equivalent for the current model: in-process with
   `log_stats=False`, nothing in EngineCore reads wall time to make a
   scheduling decision (FCFS orders on `Request.arrival_time`, which our
   factory sets explicitly), so the clock is pure loop-side bookkeeping.
   Reopen if a future cost model needs runner-internal signals (e.g.
   per-microbatch timing) — that would move cost charging back inside the
   seam.
2. **`CostModel.step_latency()`'s second argument is now the
   post-`update_from_output` scheduler state** (the MVP passed pre-update
   state). `ConstantCostModel` is unaffected; the contract is documented in
   the `llm_sim/cost/base.py` docstring. Matters only for future stateful
   cost models.
3. **Tiny-blocks/huge-`max_model_len` configs are unrepresentable.** vLLM
   0.23.0 validates KV capacity against the *overridden* block count
   (`kv_cache_utils.py:2021-2052` scales available memory to
   `override × bytes_per_block` before calling
   `_check_enough_kv_cache_memory`), so the MVP preemption scenario
   (`num_blocks=8` / `max_model_len=4096`) cannot boot. Re-derived as
   `num_blocks=12` / `max_model_len=128` / 4 requests of 16 prompt + 48
   output tokens — observed 2 preemptions, all 4 requests finish,
   `peak_blocks_used` 10 (re-run 2026-07-17).
4. **`max_model_len` passes through as `None`** instead of defaulting to
   `max_num_batched_tokens`: opt-125m's ModelConfig validation caps
   `max_model_len` at 2048 (`max_position_embeddings`), so a 8192 default
   would be rejected; `None` lets vLLM derive the model's own limit.
5. **`blocks_used` counts vLLM's always-allocated null block** (`BlockPool`
   pops it from the free queue at init, `vllm/v1/core/block_pool.py:176`): an
   idle engine reports `blocks_used == 1`, and tests assert exact peaks that
   include it (e.g. 11 = 5 requests × 2 blocks + null).
6. **KV blocks are really allocated as CPU tensors** (~1.125 MiB/block for
   opt-125m fp32 at block 16). The MVP-era default of 10 000 blocks would
   cost ~11 GiB of host RAM — hence `--num-blocks` became optional with the
   budget-derived default.

### Fidelity evidence

`./simulate.sh demo --viz` reproduces the MVP's exact headline numbers through
the full real stack: 71 steps, peak batch 229 tokens, peak running 15, peak KV
99/500 blocks (re-run 2026-07-17). The ported MVP scenarios — lockstep step
counts, staggered-arrival fast-forward (`num_steps == 4`, virtual time
≥ 100 s), forced preemption — pass with their assertions unweakened
(`tests/test_engine.py`). Two sequential EngineCore lifecycles in one process
work
(`tests/test_harness_enginecore.py::test_second_lifecycle_in_same_process`);
the spike had only ever built one.

### Suite state

90 tests, all passing, ~44 s wall (the spike-era suite was 95; the 11 MVP
harness tests retired, EngineCore harness / loop / CLI coverage added).

Still open: the plugin activation guard (tracked in `docs/backlog.md`), and —
scoped out at open, unchanged by this landing — prefix-cache /
sampled-token-identity fidelity and spec-decode (see "Honest limitations"
above).
