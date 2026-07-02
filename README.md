# vllm-sim — vLLM V1 scheduler-in-isolation simulator

Run vLLM's **real** V1 `Scheduler` / `KVCacheManager` / `BlockPool` without a
GPU, over a **virtual clock**, to reproduce its control-plane decisions —
batch composition, chunked-prefill splits, KV-block allocation, preemptions —
driven by a synthetic or trace workload and a pluggable per-step cost model.

The scheduler is not mocked: it is imported unmodified and fed real inputs.
Everything *below* it (GPU, Worker, `GPUModelRunner`, EngineCore) is bypassed —
the forward pass is replaced by a synthetic sampler, and per-step latency comes
from a cost model. See `vllm-sim-handoff.md` for the full rationale.

## Status (MVP)

Scope is **scheduler-in-isolation under a constant cost model** ("sane batches
under constant latency"). Deferred: `MockPlatform` / full GPU mock, profiled/ML
cost models, `time.*` monkeypatching, CUDA-graph padding, speculative decoding,
TP/PP, realistic prefix-cache hashing. The cost model is pluggable
(`cost/base.py`) so a profiled model can drop in behind the same interface.

## Environment

Linux only. vLLM-on-macOS is a source-build trap — do not develop there. The
standard CUDA wheel imports and runs fine GPU-free (the scheduler is pure
Python/numpy); only a benign `libcuda.so.1` warning appears.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"   # pins vllm==0.23.0 + pytest
```

First run fetches `facebook/opt-125m`'s tiny `config.json` from HuggingFace
(needs network once; cached after). No model weights are downloaded.

> **Pinned to `vllm==0.23.0`.** vLLM internals move fast. The two files that
> couple to vLLM — `harness/builder.py` and `sampler.py` — are quarantined and
> replicate vLLM's gold references (`tests/v1/core/utils.py::create_scheduler`
> and `test_scheduler.py`'s update loop). Re-verify their signatures before
> bumping the pin.

## Usage

Simplest path — `./simulate.sh` sets up the venv on first run, then forwards
args to the CLI (run it from anywhere):

```bash
./simulate.sh demo       # synthetic demo: batch composition evolving over time
./simulate.sh preempt    # tight block budget -> forces a preemption
./simulate.sh --help     # full flag list
# any flags after a shortcut override its defaults, e.g.:
./simulate.sh preempt --latency 0.5 --jsonl preempt.jsonl
# or drive it directly:
./simulate.sh --workload trace --trace mytrace.csv --num-blocks 500
```

### Visualization

`--viz` renders a lightweight terminal dashboard (Unicode sparklines, zero extra
deps) of the faithful scheduler-dynamics KPIs after a run:

```bash
./simulate.sh demo --viz
```
```
vLLM scheduler sim — 71 steps, virtual time 0.71s
batch tokens  ▃▂▁▂▁█▁▁▁▅▁▂▄▄▄▄▄▁▇▁▃▄▁▁▆▃█▁▇▄▁▁ …  peak 229
decode reqs   ▁▁▂▂▂▄▄▄▃▃▄▄▅▅▆▆▇▇▇█▇▇▇▆▆▇██▇██▇ …  peak 15
running       ▁▂▂▂▂▄▄▄▃▄▄▅▅▅▆▆▇▇█▇▇▇▇▆▇██▇██▇▇ …  peak 15
KV blocks     ▁▁▁▁▁▃▃▃▃▃▃▄▄▄▅▅▅▅▆▆▆▆▆▆▇▇▇▆▇█▇▇ …  peak 99 / 500  (20%)
finished      ▁▁▁▁▁▁▁▁▁▁▁▁▂▂▂▂▂▂▂▃▃▃▄▄▄▄▄▅▅▅▅▆ …  30 done
preemptions   ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ …  0 total
```

Re-render any saved run's JSONL later (no re-run needed):

```bash
python -m vllm_sim.viz steps.jsonl
```

KPIs are scheduler-dynamics only (batch composition, queue occupancy, KV-block
pressure, progress) — faithful under any cost model. Latency/throughput are
intentionally omitted (degenerate under the constant cost model). Rich offline
visualization via Rezolus is planned as an opt-in post-processing export; see
`docs/plans/2026-07-02-metrics-visualization-design.md`.

Or invoke the CLI yourself once the venv is set up:

```bash
# Synthetic workload (Poisson arrivals), dump per-step JSONL + summary:
.venv/bin/python -m vllm_sim --workload synthetic \
    --num-requests 30 --arrival-rate 50 \
    --prompt-len 32 128 --output-len 8 32 --seed 7 \
    --num-blocks 500 --latency 0.01 \
    --jsonl steps.jsonl --summary summary.json

# Tight block budget -> preemptions:
.venv/bin/python -m vllm_sim --workload synthetic \
    --num-requests 4 --interval 0.0 \
    --prompt-len 16 16 --output-len 16 16 \
    --num-blocks 8 --max-num-seqs 64 --max-model-len 4096 \
    --jsonl preempt.jsonl

# Replay a trace (CSV or JSONL with request_id,arrival_time,prompt_len,output_len):
.venv/bin/python -m vllm_sim --workload trace --trace mytrace.csv --num-blocks 500
```

Each step is recorded as JSONL (`vclock`, `num_running`, `num_waiting`,
`prefill_reqs`, `decode_reqs`, `tokens_scheduled`, `blocks_used`/`num_blocks`,
cumulative `preemptions`, cumulative `finished`); the summary reports peaks and
totals. With a constant cost model, `virtual_time_s == latency * num_steps`
when there are no idle arrival gaps.

## Architecture

```
workload/  RequestSpec generators (synthetic, trace) behind one interface
   factory  RequestSpec -> real vllm.Request  (vLLM-coupled, churns)
harness/
   builder  build a live GPU-free vllm Scheduler  (vLLM-coupled, highest churn)
sampler    synthesize ModelRunnerOutput (token iff prompt fully computed)  (vLLM-coupled)
cost/      CostModel.step_latency(...)  -> ConstantCostModel (MVP)
clock      VirtualClock: now / advance / fast_forward_to  (sim-loop-owned)
metrics    per-step records -> JSONL + summary
engine     SimLoop: admit -> schedule -> sample -> update -> advance -> record
cli        argparse entry point (python -m vllm_sim)
```

### The simulation loop

```
admit arrivals where arrival_time <= clock.now()
out = scheduler.schedule()
if out.total_num_scheduled_tokens == 0:
    if arrivals pending: clock.fast_forward_to(next arrival); continue
    else: break                      # nothing running, nothing arriving
dt = cost_model.step_latency(out, state)        # MVP: constant
scheduler.update_from_output(out, sampler.build_runner_output(out, requests))
clock.advance(dt); metrics.record(...)
```

In isolation the scheduler never reads the wall clock (FCFS uses
`Request.arrival_time`, set explicitly by the factory), so the virtual clock is
pure bookkeeping — no monkeypatching of vLLM internals is needed for the
constant-latency MVP.

## Key vLLM-0.23.0 facts (verified, drift-prone)

- `VllmConfig` must be built with `DeviceConfig(device="cpu")` — on a CUDA wheel
  with no driver, device inference otherwise raises "Failed to infer device type".
- `Scheduler.schedule()` takes no args (the handoff's `throttle_prefills` is gone).
- `Request(...)` has `pooling_params` as a **required** positional and a
  `client_index` param; `SchedulerConfig` requires `is_encoder_decoder`.
- `schedule()` advances `request.num_computed_tokens` to its post-step value, so
  the synthetic sampler emits a token iff `num_computed_tokens >= num_prompt_tokens`
  (adding `num_scheduled_tokens` double-counts — see `sampler.py`).

## Tests

```bash
.venv/bin/pytest tests/ -q
```

Pure-Python modules (clock, cost, workload, metrics) need no vLLM; the rest
import the live scheduler. The marquee test (`tests/test_engine.py`) runs the
real scheduler under the loop and asserts batch composition, step counts, and a
forced preemption.
```
