# SGLang engine: mockability spike (Effort A of engine generality)

**Status:** CLOSED — spike returned **GO (full-engine)** (2026-07-24, PR #5,
see "Spike verdict" below). Effort B (engine abstraction) is unblocked and
opens as its own entry.

**Opened:** 2026-07-24.

## Goal

Make llm-sim engine-agnostic, with SGLang as the second engine that proves the
abstraction. This entry is **Effort A**: a GO/NO-GO spike on whether SGLang's
scheduling control plane can run GPU-free, in-process, under the sim's virtual
clock — the same question the v2 spike answered for vLLM's EngineCore
(`docs/journal/2026-07-09-gpu-mock-enginecore-boot.md`), on a very different
architecture. **Effort B** (extracting an `EngineAdapter` interface from the
two concrete implementations, shared-core-plus-extras metrics, `--engine`
CLI) opens only after this spike returns a verdict: the abstraction is only
as good as its second implementation, so the interface is not designed until
SGLang's shape is known.

Decided at open (with the user, 2026-07-23/24):

- **Primary goal is framework generality** — not cross-engine comparison
  tooling (a possible third effort) and not an SGLang-specific study.
- **The spike decides the fidelity level**: full-engine GPU-free (v2-style),
  scheduler-in-isolation (retired-MVP-style), or NO-GO.
- **Metrics direction for Effort B**: engine-neutral StepRecord core
  (running, waiting, tokens scheduled, KV utilization as a fraction,
  cumulative finished, cumulative preemptions/evictions) plus engine-specific
  extra columns — comparable where honest, faithful where engines differ.
  This is a deliberate breaking JSONL schema change, made once, in B.

## Why this is not a rerun of the v2 spike

The three properties that made v2 cheap are absent or unknown here:

1. **No platform-plugin seam.** vLLM resolved `MockPlatform` through the
   sanctioned `vllm.platform_plugins` entry point — that was the entire
   injection. SGLang has no equivalent; injection likely means constructing
   internals directly or subclassing, and the spike measures how invasive
   that is.
2. **Multi-process by default.** SGLang's `Engine` spawns
   TokenizerManager / Scheduler / Detokenizer processes over ZMQ. The sim
   needs the scheduling loop in-process and single-process; whether SGLang's
   `Scheduler` can exist outside its process harness is the load-bearing
   unknown.
3. **Tighter scheduler↔runner coupling.** SGLang's Scheduler owns the
   TpModelWorker, which owns the ModelRunner and its KV memory pools (torch
   tensors — possibly constructible on CPU device). Less separable than
   vLLM's scheduler.

A fourth risk is epistemic: the above is training-data knowledge of a
fast-moving project. Probe 0 verifies current reality before anything is
built on it, and this entry's spike record must cite the installed source.

## The spike

**One question:** can SGLang's real scheduling step loop run end-to-end,
GPU-free, in-process, on a synthetic forward result — and how bounded is the
injection surface?

**Probes, cheapest-first (stop at the first hard blocker):**

0. **Environment.** Fresh isolated venv (`.venv-sglang` — SGLang may pin a
   torch incompatible with the vllm venv's `torch 2.11.0+cu130`); install;
   import GPU-free. A module-level import failure that cannot be worked
   around is an immediate NO-GO. Output either way: the exact version pin,
   recorded here.
1. **In-process Scheduler construction.** Construct SGLang's `Scheduler`
   directly (real `ServerArgs`, dummy/mock model load, CPU device for the KV
   pools) with no ZMQ/subprocess. This is the load-bearing signal for
   full-engine vs scheduler-level.
2. **Step loop.** Drive the real scheduling iteration (per the pinned
   version's actual loop — e.g. `get_next_batch_to_run` → synthetic batch
   result → `process_batch_result`) for a small workload: requests admitted →
   prefill → decode → finished, with radix-cache state advancing sanely.
3. **Determinism.** The same two-request workload twice produces identical
   step traces. The sim's value depends on this.

## GO / NO-GO

- **GO (full-engine)** — probe 2 completes with the injection surface
  bounded: ≤ ~6 named, well-understood stub points (counted and inventoried
  like the v2 entry's five).
- **GO (scheduler-level)** — probe 1 fails (the Scheduler cannot exist
  outside its process harness) but the batching/radix-cache logic is
  separable with bounded stubs.
- **NO-GO** — unbounded stub surface either way. Land the negative result
  with its mechanism and a reopen condition (e.g. "revisit on an SGLang
  release that decouples the scheduler from its process harness").

Either verdict closes this entry honestly; B's scope is set by the verdict.

## Spike verdict — GO (full-engine) (2026-07-24)

All four probes passed on the first or second attempt, Linux x86_64, no GPU
(`torch.cuda.is_available() == False`). **Pin: `sglang==0.5.15.post1`**
(bare `pip install sglang`; the base wheel now includes the full srt runtime —
122 pinned deps, an 11 GB venv including driverless CUDA wheel payloads;
`torch==2.11.0+cu130`, same pin as the vllm venv, though the venvs stay
separate). The gate landed as `tests/test_sglang_scheduler.py` — skips
cleanly in the repo's main venv (no sglang), gates under `.venv-sglang`.

**Probe 0 — imports clean; this entry's risk #1 was FALSE.** SGLang 0.5.15
has a platform seam: the `sglang.srt.platforms` entry-point group resolves
`current_platform` lazily (built-ins include `CpuSRTPlatform`;
`SGLANG_USE_CPU_ENGINE=1` or `SGLANG_PLATFORM` select it) — the vLLM-style
plugin architecture the entry assumed absent. `sgl_kernel` genuinely cannot
import without CUDA, but it is not on the scheduler path: tree-wide only 12
files import it unguarded at module level, all runtime-selected backends
(e.g. `srt/layers/attention/flashattention_backend.py:39`).

**Probe 1 — Scheduler constructs in-process, GPU-free, ~1.9 s.** Real
`ServerArgs(model_path="facebook/opt-125m", device="cpu",
load_format="dummy")` + `PortArgs.init_new` + direct `Scheduler(...)`: zero
child processes, one benign watchdog daemon thread. The constructor's ZMQ
sockets (`init_ipc_channels`, scheduler.py:615) bind peer-less `ipc://`
endpoints and never block; `torch.distributed` initializes in-process (gloo,
world_size=1); attention backend auto-falls-back to `TorchNativeAttnBackend`
(no Intel AMX on this host); NUMA/affinity binding lives outside `__init__`
(`configure_scheduler_process`, scheduler.py:4270, env-gated) and is bypassed
entirely. Real `RadixCache`, `ReqToTokenPool`, `MHATokenToKVPool` on CPU fp16
tensors (k_buffer shape (4097, 12, 64) for opt-125m).

**Probe 2 — real step loop end-to-end, and the real forward RUNS.** Mirroring
`event_loop_normal` (scheduler.py:1542) minus ZMQ recv:
`handle_generate_request` (scheduler.py:2043) →
`get_next_batch_to_run` (scheduler.py:2607) → `run_batch` (scheduler.py:3200)
→ `process_batch_result` (scheduler.py:3461). Two requests: prefill → decode
→ both `FINISH_LENGTH` at exactly `max_new_tokens`; batch shrinks 2→1 after
the first finish; on finish, tokens move to radix-**evictable** rather than
freeing (proper retention semantics). Unlike v2 (where CPU kernels were
absent from the CUDA wheel), the **real transformer forward executes** via
`TorchNativeAttnBackend` on dummy fp16 weights: prefill (28 tok, bs=2)
~80 ms, decode ~16–21 ms/step, `process_batch_result` &lt;1 ms. ZMQ egress
(output streaming toward the absent detokenizer) is a silent in-memory
buffered send — no error, no block.

**Probe 3 — deterministic.** Fresh processes, same seed → bit-identical
per-step traces (batch composition, occupancy, finish steps, output ids),
verified for both greedy and temperature-1.0 sampling.

**Injection surface: 3 units** (GO criterion allowed ≤ ~6), all sanctioned
APIs — zero monkeypatches, zero source edits, zero stubs:

1. CPU device signal — `ServerArgs(device="cpu")` or `SGLANG_USE_CPU_ENGINE=1`
   (either alone suffices; without one, `get_device()` raises at
   `srt/utils/common.py:899`).
2. `load_format="dummy"` — real `OPTForCausalLM` with random weights, no
   weight files (config/tokenizer from HF cache, `HF_HUB_OFFLINE=1`).
3. `SamplingParams.normalize(None)` before injection — we inject past
   TokenizerManager, which normally normalizes params
   (tokenizer_manager.py:1133); skipping it crashes
   `_check_str_based_finish` (schedule_batch.py:1360).

**Synthetic-forward seam for Effort B (named, not built):**
`TpModelWorker.forward_batch_generation` (tp_worker.py:489), reached via the
plain `self.model_worker` attribute (set in `init_model_worker`,
scheduler.py:877) — a duck-typed wrapper drops in with no plumbing;
fabricate `GenerationBatchResult` (managers/utils.py:39). Inner alternative:
`ModelRunner.forward` (model_runner.py:3001). Since the real forward works,
the seam is for cost control and speed at scale, not for unblocking.

**Honest limitations (carried into Effort B):**

- Greedy over dummy weights samples token id 0 every step — token identities
  are degenerate, so radix/prefix-cache *hit* fidelity remains deferred
  (same limitation as v2; acute for SGLang, whose scheduler is radix-centric).
- The real CPU forward at ~20 ms/decode-step is fine for tests, too slow for
  large sims — Effort B should charge virtual time at the
  `forward_batch_generation` seam (or accept wall cost for small runs).
- Peer-less ZMQ sends buffer in memory indefinitely; long sims should read
  results from `req.output_ids` directly and avoid unbounded egress queues.
- The pin is heavy (11 GB venv) and SGLang's release cadence is fast;
  re-verify the three injection units and the four loop call sites before
  any pin bump.

## Deferred (mirrored in `docs/backlog.md`)

- **Effort B — engine abstraction** (`EngineAdapter`, shared-core+extras
  metrics, `--engine` flag): blocked on this spike's verdict.
- **Cross-engine comparison tooling** (side-by-side runs, diff reports):
  a possible third effort once two engines actually run; not part of B.
- **Prefix-cache fidelity under realistic token identities**: today's
  constant-dummy-token prompts defeat prefix caching in both engines; radix
  cache studies need realistic prompts. Already noted as a v2 limitation;
  becomes acute if SGLang (whose scheduler is radix-centric) lands.
