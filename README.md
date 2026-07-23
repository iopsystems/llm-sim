# llm-sim — GPU-free vLLM EngineCore simulator

Run vLLM's **real** V1 engine — `EngineCore.step()` end-to-end — without a
GPU, over a **virtual clock**, to reproduce its control-plane decisions —
batch composition, chunked-prefill splits, KV-block allocation, preemptions —
driven by a synthetic or trace workload and a pluggable per-step cost model.

The engine is not mocked: a `MockPlatform` plugin (`llm_sim/mock/`) boots the
real stack in-process — real `UniProcExecutor` → real `Worker` init → real
attention-backend selection (`CPU_ATTN`) → real `Scheduler` → real `Sampler`,
with a `GPUModelRunner`-lineage model runner and dummy-loaded weights. **Only
the transformer forward pass is synthetic**; per-step latency comes from a
cost model over the virtual clock. See `vllm-sim-handoff.md` for the full
rationale.

## Status (v2 — EngineCore-backed)

The MVP's scheduler-in-isolation loop (hand-built scheduler, synthetic
sampler) was retired in v2; the sim now runs vLLM's own step loop. Still
deferred: profiled/ML cost models, speculative decoding, TP/PP, prefix-cache
fidelity (sampled token identities are arbitrary), the multiprocess/ZMQ path,
and the `LLM.generate()` offline surface. The cost model is pluggable
(`cost/base.py`) so a profiled model can drop in behind the same interface.

## Environment

Linux only. vLLM-on-macOS is a source-build trap — do not develop there. The
standard CUDA wheel imports and runs fine GPU-free (the mock platform keeps
everything on CPU tensors); only a benign `libcuda.so.1` warning appears.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"   # pins vllm==0.23.0 + pytest
```

First run fetches `facebook/opt-125m`'s tiny `config.json` from HuggingFace
(needs network once; cached after). No model weights are downloaded — the
engine dummy-loads them (`load_format="dummy"`).

> **Pinned to `vllm==0.23.0`.** vLLM internals move fast. The vLLM-coupled
> surface is `llm_sim/mock/` (the platform plugin: `MockPlatform`,
> `MockWorker` with an analytic KV budget, `MockModelRunner` overriding only
> `_model_forward`, pure-torch `_C` op fallbacks) plus `llm_sim/harness/`
> (EngineCore builder + `SimScheduler` capture). The plugin is registered
> under the `vllm.platform_plugins` entry point in `pyproject.toml`, so it
> activates `MockPlatform` for **every** vLLM use in this venv. Also
> pin-coupled: the verbatim KV-capacity error signatures in `llm_sim/cli.py`
> (`_KV_CAPACITY_SIGNATURES`; test-enforced, degrades to a generic message on
> drift). Re-verify the injection surface (inventoried in
> `docs/journal/2026-07-09-gpu-mock-enginecore-boot.md`) before bumping the pin.

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

### KV-cache sizing

`--num-blocks` is optional: by default the block count derives from a 1 GiB
**analytic KV budget** (`--kv-cache-bytes`) — analytic in that the per-block
cost is computed from the model's shape, replacing vLLM's GPU memory
profiling — which comes out to ~910 blocks for opt-125m fp32 at
`--block-size 16`. Unlike the MVP, KV blocks are **really allocated** as CPU
tensors (~1.1 MiB/block for opt-125m fp32 at block 16) — small budgets are
cheap, huge ones cost real host RAM.

The config must hold one maximum-length request — `num_blocks × block_size ≥
max_model_len` (derived 2048 for opt-125m) — even when `--num-blocks`
overrides the count (mechanism under Key facts below). Tight-budget
experiments therefore need a lowered `--max-model-len` (see the preempt
example below). When a config doesn't fit, the CLI translates vLLM's
rejection into an actionable error naming `--num-blocks` /
`--kv-cache-bytes` / `--max-model-len`.

### Visualization

`--viz` renders a lightweight terminal dashboard (Unicode sparklines, zero extra
deps) of the faithful scheduler-dynamics KPIs after a run:

```bash
./simulate.sh demo --viz
```
```
vLLM scheduler sim — 71 steps, virtual time 0.71s
batch tokens  ▃▂▁▂▁█▁▁▁▅▁▂▄▄▄▄▄▁▇▁▃▄▁▁▆▃█▁▇▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁  peak 229
prefill reqs  ▃▃▁▃▁█▁▁▁▃▁▃▃▃▃▃▆▁▆▁▃▃▁▁▆▃▆▁▆▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁  peak 3
decode reqs   ▁▁▂▂▂▄▄▄▃▃▄▄▅▅▆▆▇▇▇█▇▇▇▆▆▇██▇██▇▆▅▅▅▄▄▄▃▂▂▂▂▂▂  peak 15
running       ▁▂▂▂▂▄▄▄▃▄▄▅▅▅▆▆▇▇█▇▇▇▇▆▇██▇██▇▇▅▅▅▄▄▄▃▂▂▂▂▂▂▁  peak 15
waiting       ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁  peak 0
KV blocks     ▁▁▁▁▁▃▃▃▃▃▃▄▄▄▅▅▅▅▆▆▆▆▆▆▇▇▇▆▇█▇▇▅▅▅▄▄▄▃▁▁▁▂▂▂▁  peak 99 / 500  (20%)
finished      ▁▁▁▁▁▁▁▁▁▁▁▁▂▂▂▂▂▂▂▃▃▃▄▄▄▄▄▅▅▅▅▆▆▆▆▇▇▇▇███████  30 done
preemptions   ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁  0 total
```

Re-render any saved run's JSONL later (no re-run needed):

```bash
python -m llm_sim.viz steps.jsonl
```

KPIs are scheduler-dynamics only (batch composition, queue occupancy, KV-block
pressure, progress) — faithful under any cost model. Latency/throughput are
intentionally omitted (degenerate under the constant cost model).

### Rezolus export (optional, offline heatmaps)

For rich distribution-over-time views, a post-processing step converts a run's
JSONL into a [Rezolus](https://github.com/iopsystems/rezolus)-compatible Parquet
(log-linear histograms, `grouping_power=3`/`max_value_power=64` → 496 buckets,
cumulative per virtual-time interval; `vclock` maps to the timestamp axis). All
heavy deps are confined to this opt-in extra — the sim core and `--viz` never
import them.

```bash
pip install -e ".[rezolus]"                       # pulls pyarrow
python -m llm_sim --workload synthetic ... --jsonl steps.jsonl
python -m llm_sim.export.rezolus steps.jsonl -o run.parquet --target-rows 120
# then open run.parquet in the Rezolus viewer (quantile / heatmap over time)
```

Metrics histogrammed by default: `tokens_scheduled`, `num_running`,
`blocks_used`, `num_waiting` (override with `--metrics`). Design +
schema-compatibility notes:
`docs/journal/2026-07-02-metrics-visualization.md`.

Or invoke the CLI yourself once the venv is set up:

```bash
# Synthetic workload (Poisson arrivals), dump per-step JSONL + summary.
# No --num-blocks: KV blocks derive from the default 1 GiB budget (~910).
.venv/bin/python -m llm_sim --workload synthetic \
    --num-requests 30 --arrival-rate 50 \
    --prompt-len 32 128 --output-len 8 32 --seed 7 \
    --latency 0.01 \
    --jsonl steps.jsonl --summary summary.json

# Tight block budget -> preemptions. max_model_len must fit the budget
# (12 blocks x 16 = 192 >= 128) or the engine rejects the config:
.venv/bin/python -m llm_sim --workload synthetic \
    --num-requests 4 --interval 0.0 \
    --prompt-len 16 16 --output-len 48 48 \
    --num-blocks 12 --max-num-seqs 64 --max-model-len 128 --latency 0.01 \
    --jsonl preempt.jsonl

# Replay a trace (CSV or JSONL with request_id,arrival_time,prompt_len,output_len):
.venv/bin/python -m llm_sim --workload trace --trace mytrace.csv --num-blocks 500
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
mock/      vLLM platform plugin (entry point vllm.platform_plugins):
           MockPlatform(CpuPlatform), MockWorker (analytic KV budget),
           MockModelRunner (only _model_forward overridden), pure-torch
           _C op fallbacks  (vLLM-coupled, highest churn)
harness/
   enginecore  build a live GPU-free vllm EngineCore  (vLLM-coupled)
   scheduler   SimScheduler(Scheduler): captures each step's SchedulerOutput
cost/      CostModel.step_latency(...)  -> ConstantCostModel
clock      VirtualClock: now / advance / fast_forward_to  (sim-loop-owned)
metrics    per-step records -> JSONL + summary
engine     SimLoop: admit -> core.step() -> cost -> advance -> record
cli        argparse entry point (python -m llm_sim)
```

### The simulation loop

```
admit arrivals where arrival_time <= clock.now()
core.step()                        # real schedule -> execute -> update_from_output
out = scheduler.last_scheduler_output    # SimScheduler capture seam
if out.total_num_scheduled_tokens == 0:
    if arrivals pending: clock.fast_forward_to(next arrival); continue
    else: break                    # trailing cleanup step, or wedged
dt = cost_model.step_latency(out, scheduler)     # constant, for now
clock.advance(dt); metrics.record(...)
```

In-process with `log_stats` off, EngineCore makes no wall-clock-dependent
scheduling decisions (FCFS uses `Request.arrival_time`, set explicitly by the
factory), so the virtual clock is pure bookkeeping — no monkeypatching of vLLM
internals is needed for the constant-latency cost model.

## Key vLLM-0.23.0 facts (verified, drift-prone)

- **The platform-plugin seam is clean.** vLLM resolves platforms lazily via
  the `vllm.platform_plugins` entry-point group; on a GPU-less box with the
  CUDA wheel, *no* builtin platform activates, so `MockPlatform` is the sole
  winner — zero monkeypatching.
- **The CPU lineage *is* the GPU machinery.** `CPUWorker` subclasses
  `gpu_worker.Worker` (`vllm/v1/worker/cpu_worker.py:33`) and `CPUModelRunner`
  subclasses `GPUModelRunner` (`vllm/v1/worker/cpu_model_runner.py:22`) — so
  inheriting `CpuPlatform` keeps vLLM's real init and per-step code paths.
- **`scheduler_config.scheduler_cls` is a sanctioned injection point.**
  EngineCore resolves it at construction (`vllm/v1/engine/core.py:136`),
  nothing reads it earlier, so setting it post-`create_engine_config` swaps in
  `SimScheduler` (real `schedule()`, observed not modified).
- **`num_gpu_blocks_override` doesn't bypass the capacity check.** The
  override sets the exact block count, but `check_enough_kv_cache_memory`
  (`vllm/v1/core/kv_cache_utils.py`) still validates against it:
  `num_blocks × block_size` must cover `max_model_len` or engine
  construction raises.
- **Token timing:** the first output token samples in the same step the
  prompt finishes computing, so a request with `output_len=N` finishes `N-1`
  steps after its prompt completes.
- **`blocks_used` includes the null block.** `BlockPool` pops its
  always-allocated null block from the free queue at init
  (`vllm/v1/core/block_pool.py:176`), so an idle engine reports
  `blocks_used == 1` and only `num_gpu_blocks - 1` blocks are usable.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Pure-Python modules (clock, cost, workload, metrics, viz, export) need no
vLLM; the EngineCore-backed tests boot the real engine, which dummy-loads
opt-125m (a few seconds per boot) and needs its `config.json` in the local HF cache
— they skip cleanly when it's absent (the suite runs with `HF_HUB_OFFLINE=1`).
The marquee test (`tests/test_engine.py`) runs real `EngineCore.step()` under
the loop and asserts batch composition, step counts, deterministic peak
KV-block usage, and a forced preemption. Full suite: 90 tests, ~45 s.
