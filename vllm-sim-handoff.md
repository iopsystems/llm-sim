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

---

# MVP Implementation Handoff (decided)

> Pick this up on a **Linux** box. The brief above is the rationale; this section is the
> decided MVP plan plus the verified V1 interface map, so the other end need not re-explore.

## Decisions

- **Scope:** scheduler-in-isolation. Import the real `Scheduler` / `KVCacheManager` / `BlockPool`
  unmodified (decisions are bit-identical to production); bypass everything below the scheduler
  (GPU, Worker, `GPUModelRunner`, EngineCore). Feed inputs directly; synthesize forward-pass output.
- **MVP success:** *sane batches under constant latency* — feed a trace, run the real scheduler
  with a constant per-step cost, show batch composition / chunked-prefill splits / preemptions
  evolving over virtual time (brief task #5).
- **Env:** Linux (`pip install vllm==<tag>` pulls CPU torch; we only *import* V1 modules, no CUDA).
  vLLM-on-macOS is a source-build trap — do not develop there.
- **vLLM version:** pin a recent **stable** `vX.Y.Z` tag (NOT `main`).
- **Workload:** synthetic generator now, **plus** a trace-file loader behind the same interface.

**Key simplification:** in isolation the scheduler does **not** read the wall clock to make
decisions — FCFS uses `Request.arrival_time` (we set it explicitly); the only `time.*` calls are
KV-event timestamps. So the virtual clock is a **sim-loop-owned bookkeeping variable** that gates
arrivals — **no monkeypatching of vLLM internals** for the constant-latency MVP.

### Deferred
Full platform-mock / `MockPlatform` / EngineCore boot / Worker / `GPUModelRunner`;
analytic/profiled cost models (interface only, constant impl now); monkeypatching vLLM `time.*`;
CUDA-graph batch padding; speculative decoding; TP/PP; realistic prefix-cache hashing fidelity.

## Gate 0 — do FIRST, nothing proceeds until green

```bash
pip install "vllm==<pinned stable tag>"   # pick latest: git tag | sort -V | grep -v rc | tail
python -c "from vllm.v1.core.sched.scheduler import Scheduler; \
from vllm.v1.outputs import ModelRunnerOutput; \
from vllm.v1.request import Request; print('ok')"
```
Then re-confirm the signatures below against the pinned tag (line numbers drift; names matter).

## Verified V1 interface map

> Source: `main@dc55936f` (2026-06-24). **Gold construction references:**
> `tests/v1/core/utils.py` (`create_scheduler`, `create_requests`) and
> `tests/v1/core/test_scheduler.py` (the `update_from_output` loop). Copy these, pinned to the tag.

```python
# vllm/v1/core/sched/scheduler.py
class Scheduler(SchedulerInterface):
    def __init__(self, vllm_config, kv_cache_config, structured_output_manager,
                 block_size, hash_block_size=None, mm_registry=MULTIMODAL_REGISTRY,
                 include_finished_set=False, log_stats=False): ...
    def schedule(self, throttle_prefills=False) -> SchedulerOutput: ...
    def update_from_output(self, scheduler_output, model_runner_output) -> dict[int, EngineCoreOutputs]: ...
    def add_request(self, request) -> None: ...   # -> self.waiting, status=WAITING
```

- `SchedulerOutput` key fields: `scheduled_new_reqs`, `scheduled_cached_reqs`,
  `num_scheduled_tokens: dict[str,int]`, `total_num_scheduled_tokens: int`,
  `finished_req_ids: set[str]`, `num_common_prefix_blocks`.
- `ModelRunnerOutput` (we synthesize): `req_ids`, `req_id_to_index`,
  `sampled_token_ids: list[list[int]]`, `logprobs=None`, `prompt_logprobs_dict={}`, `pooler_output=[]`.
- `Request(request_id, prompt_token_ids, sampling_params, pooling_params=None,
  arrival_time=..., priority=0, ...)` — leave mm/embeds/lora None. Scheduler mutates
  `num_computed_tokens`, `status`, `num_preemptions`, `spec_token_ids`.

### GPU-free KVCacheConfig (no profiling)
```python
cache_config.num_gpu_blocks = N                       # direct, no GPU call
kv_cache_config = KVCacheConfig(
    num_blocks=N, kv_cache_tensors=[],
    kv_cache_groups=[KVCacheGroupSpec(["layer"],
        FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float32))])
register_all_kvcache_specs(vllm_config)               # vllm.v1.core.single_type_kv_cache_manager
structured_output_manager = StructuredOutputManager(vllm_config)
```
Gotchas: `skip_tokenizer_init=True` + explicit `max_model_len` to dodge HF config fetches; leave
prompt_embeds/multimodal/KV-connector/MoE off.

### Deterministic output length + prefill/decode rule (synthetic sampler)
- Per request `SamplingParams(max_tokens=output_len, ignore_eos=True)` → finishes exactly after
  `output_len` decode steps. Token identity irrelevant → emit a fixed dummy id.
- Emit a token only once prompt fully scheduled:
  `new_computed = req.num_computed_tokens + out.num_scheduled_tokens[id]`; if
  `new_computed >= len(prompt_token_ids)` → `[dummy]`, else `[]`. Mirrors
  `test_scheduler.py` `sampled_token_ids=[[0], [], []]`. **Verify `num_computed_tokens`
  read-timing against the pinned tag in the integration test.**

## Simulation loop

```
admit arrivals where arrival_time <= clock.now()
out = scheduler.schedule()
if out.total_num_scheduled_tokens == 0:
    if running empty and arrivals pending: clock.fast_forward_to(next arrival); continue
    if nothing pending and nothing running:  break
dt = cost_model.step_latency(out, state)              # MVP: constant
runner_out = sampler.build_runner_output(out, requests)
scheduler.update_from_output(out, runner_out)
clock.advance(dt); metrics.record(...)
```

## Package layout — `vllm_sim/`

vLLM-coupling quarantined in `harness/builder.py` and `sampler.py` (the two files that churn).

- `harness/builder.py` — `build_scheduler(...) -> Scheduler` (replicate `create_scheduler`). **Highest drift risk.**
- `workload/base.py` — `RequestSpec(request_id, arrival_time, prompt_len, output_len)` + `WorkloadSource` protocol.
- `workload/synthetic.py` — seeded generator (Poisson/fixed arrivals, length dists).
- `workload/trace.py` — CSV/JSONL loader, same interface.
- `workload/factory.py` — `RequestSpec -> vllm Request` (reuse `create_requests`).
- `cost/base.py` — `CostModel.step_latency(scheduler_output, state) -> float`.
- `cost/constant.py` — `ConstantCostModel(latency_s)`.
- `clock.py` — `VirtualClock`: `now()`, `advance(dt)`, `fast_forward_to(t)`.
- `sampler.py` — `build_runner_output(scheduler_output, requests_by_id) -> ModelRunnerOutput`. **Second drift risk.**
- `metrics.py` — per-step record (vclock, step, num_running/waiting, prefill/decode reqs,
  tokens_scheduled, blocks_used/num_blocks, preemptions, finished) → JSONL + summary.
- `engine.py` — `SimLoop`; terminate when all finished and no arrivals.
- `cli.py` / `__main__.py` — run scenario, dump JSONL + summary.
- `pyproject.toml` (`vllm==<tag>`, dev `pytest`), `README.md` (env setup).

## TDD order (failing test first, then implement)

1. `clock` — advance / fast_forward.
2. `cost/constant` — fixed latency.
3. `workload/synthetic` — deterministic under seed; arrivals monotonic; lengths in bounds.
4. `workload/trace` — parses fixture into identical `RequestSpec`s.
5. `workload/factory` — correct `max_tokens`, prompt length, `ignore_eos`, `arrival_time`.
6. `harness/builder` — live `Scheduler`; one `schedule()` on one request → non-empty output.
7. `sampler` — fabricated `SchedulerOutput` + states → correct token/empty pattern.
8. **`engine` integration (marquee):** hand-built deterministic trace → assert all finish, total
   step count matches hand-computed expectation, per-step batch composition matches. Second
   scenario with tight `num_blocks` → assert ≥1 preemption. This *is* the success demo.

## Verification

- Gate 0 import. `pytest tests/` green. `python -m vllm_sim --workload synthetic ...` → summary
  shows batch sizes evolving, all requests finishing, blocks ≤ `num_blocks`, virtual time =
  `constant_latency * num_steps`. Tight-budget scenario → JSONL shows preemptions; summary reports them.
