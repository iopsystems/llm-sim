# v2: GPU-mock — boot the real vLLM stack in-process (EngineCore)

**Status:** OPEN — intent landed, spike pending. This is the coordination entry;
the GO/NO-GO verdict and any design land in follow-up updates to this entry.

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
