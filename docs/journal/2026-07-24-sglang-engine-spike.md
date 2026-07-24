# SGLang engine: mockability spike (Effort A of engine generality)

**Status:** OPEN — intent landed, spike pending.

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

## Deferred (mirrored in `docs/backlog.md`)

- **Effort B — engine abstraction** (`EngineAdapter`, shared-core+extras
  metrics, `--engine` flag): blocked on this spike's verdict.
- **Cross-engine comparison tooling** (side-by-side runs, diff reports):
  a possible third effort once two engines actually run; not part of B.
- **Prefix-cache fidelity under realistic token identities**: today's
  constant-dummy-token prompts defeat prefix caching in both engines; radix
  cache studies need realistic prompts. Already noted as a v2 limitation;
  becomes acute if SGLang (whose scheduler is radix-centric) lands.
