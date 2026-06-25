# vLLM GPU-Mock + Time-Emulator — Handoff Brief

## Goal

Run vLLM (V1 engine) without a GPU, faithfully reproducing its control-plane
decisions — scheduling, kernel/backend selection, KV-block allocation, batch
composition — plus a time emulator so the simulated timeline drives those
decisions the way real hardware would.

## Core architecture insight

**Don't mock the scheduler — run it for real and mock everything beneath it,
feeding it a clock.** The V1 scheduler is deterministic given its inputs, so
supplying (arrivals, prompt/output lengths, block budget, token budget, prefix
structure, per-step latency) reproduces real batch composition, chunked-prefill
splits, preemptions, and prefix reuse — because it's the real scheduler code.

The surface decomposes into three buckets:

- **Control plane → run unmodified.** V1 `Scheduler`, `KVCacheManager`,
  `BlockPool`, prefix-cache hashing, `EngineCore` loop. Python/numpy and
  hardware-agnostic. Paged block allocation simulates *exactly*.
- **Per-step latency → needs a cost model.** The only hard input; ~all fidelity
  risk lives here.
- **Token identities → synthetic.** Drive output length from a trace or
  stop-model. Token *identities* don't affect scheduling — one exception:
  prefix caching (see Fidelity traps).

## The two halves, by difficulty

### Mocking the GPU — moderate, feasible

Natural seam is the platform abstraction (`current_platform`); the existing CPU
platform is a usable template. Components to stub:

- `MockPlatform` — device caps, FP8/MX checks, `mem_get_info`. Small.
- Worker / `GPUModelRunner` — **the bulk.** Run all the metadata-construction
  paths; replace `model.forward()` with a cost-model call + synthetic sampler
  output. This is the part that churns most across vLLM releases.
- Attention **backend selection** (`get_attn_backend_cls`) — preserve as-is.
  This *is* the "kernel selection" of interest; only the kernels get stubbed.
- KV-cache memory profiling (`determine_num_available_blocks`) — replace with an
  analytic memory model. Matters because `num_blocks` drives preemption/recompute.
- CUDA graphs / torch.compile capture-replay — stub.
- Custom ops / quant kernels — stub.
- Collectives / NCCL — only if simulating TP/PP; skippable for single-replica.

### Time emulator — hard, fidelity-capped

This is the Vidur problem. The cost model basically must be profiled/learned,
not purely analytic:

- Analytic (roofline-ish): ~20–40% error. Fine for *relative* policy comparison.
- Profiled/ML: <10% achievable (Vidur reports <9%).

Errors cascade under load because the loop is closed: latency → completions →
freed memory → next batch composition → next latency. High arrival rates need
accurate per-iteration prediction to stay faithful.

## Prior art to reuse (don't rebuild)

- **Vidur** (MLSys'24) — reference for the cost-model approach and its fidelity
  limits.
- **NVIDIA Dynamo AIConfigurator** — pre-measured per-op/per-layer perf database
  (attention / FFN / comm / memory) feeding simulation-based recommendations.
  Ready-made seed data for the cost model. Caveat: TRT-LLM-on-Hopper-first, not
  vLLM-centric.
- **Dynamo Mocker engine** (+ trace replay, AIConfigurator perf shim) — a working
  mock-the-forward-pass + injected-clock implementation. Read it as a reference
  for the clock/replay mechanics.
- **Key differentiator:** both NVIDIA tools sit at the Dynamo *orchestration*
  layer (routing, planner, disaggregation), NOT vLLM's internal V1 scheduler.
  Reproducing vLLM-internal behavior faithfully is exactly the gap this project
  fills — and the reason to wrap the real scheduler rather than reimplement.

## Clock injection — decision

**Rejected: fluxcapacitor / syscall-level faking.** Its primitive (collapse a
sleep into an instant virtual-time jump) is conceptually elegant for "this kernel
took 8 ms," but it only leaps when the *whole process tree* is blocked on
recognized waits (epoll/select/poll/nanosleep — **not futex**). That breaks
against vLLM's futex-heavy threaded, multi-process, ZMQ/asyncio runtime. Plus:
ptrace overhead on a syscall-heavy stack, background timers firing in virtual
time, python2-era and unmaintained. Underlying mismatch: fluxcapacitor couples
time-advancement to the process *actually blocking*; a discrete-event sim wants
to couple it to events *it controls*.

**Chosen: code-level virtual clock.** Own the clock in the sim loop, advance it
explicitly at the kernel seam (`vclock += cost_model(batch)`), route reads
through it:

- Monkeypatch `time.monotonic`, `time.time`, `time.perf_counter`, and the asyncio
  loop's `time()`.
- Replace `time.sleep` / `asyncio.sleep` with advance-and-yield.

Gives deterministic, reproducible event ordering. Main work item: the EngineCore
subprocess — either run single-process for sim, or share the clock via shared
memory. Helpful narrowing: because the GPU is mocked, there are **no real CUDA
events to intercept** — only host-side Python time reads.

Fallback: **libfaketime** if don't-touch-source is required (LD_PRELOAD, low
overhead) — but it can't collapse real sleeps, so kernels must be no-ops and you
advance time by bumping its offset; weaker event-ordering control.

## Fidelity traps to budget for

- **CUDA-graph batch padding** — vLLM pads batches to captured sizes, changing
  effective batch and latency, which feeds back into scheduling. Must model.
- **Prefix caching** — hit rate depends on real token hashes. Synthetic token IDs
  won't reproduce real sharing unless the trace carries realistic prefix structure.
- **Chunked prefill** — token budget interacts with the cost model inside a loop;
  cost-model error there is doubly load-bearing.
- **Speculative decoding** — workload/model-dependent acceptance rate complicates
  the step model.
- **Host compute** — model as zero virtual time (or a separately-modeled cost);
  virtual time advances only at the kernel seam.

## Suggested first tasks (Claude Code)

1. Stand up `MockPlatform` from the CPU-platform template; get vLLM to initialize
   with no CUDA present.
2. Map the concrete vLLM call sites / modules that read time — defines the
   virtual-clock injection surface.
3. Replace `GPUModelRunner.execute_model` to return synthetic sampler output + a
   stub cost (constant first), running all metadata paths.
4. Replace `determine_num_available_blocks` with an analytic memory model.
5. Wire the sim loop + virtual clock; validate the scheduler produces sane batches
   under a constant-latency model before adding a real cost model.
6. Swap the constant cost for a profiled model (seed from AIConfigurator per-op
   data if usable).

## Strategic check to keep in view

If the end goal is scheduling-policy *research* rather than fidelity to vLLM's
exact code paths, running the real V1 scheduler in isolation with a cost model
may get most of the value at a fraction of the surface area. Reach for the full
platform-mock only if runner/backend-level behaviors (graph padding, token
budgeting quirks, backend selection) are themselves objects of study.

## Version caveat

vLLM internals (especially `GPUModelRunner` and the platform layer) move fast.
Pin to a specific vLLM commit/tag at the start and verify these seams against
that revision before building — names and call sites drift.
