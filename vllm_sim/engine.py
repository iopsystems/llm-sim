"""The simulation loop.

Runs the real vLLM V1 ``Scheduler`` over a virtual timeline: admit arrivals
gated by the clock, call ``schedule()``, synthesize the model-runner output,
``update_from_output``, advance the clock by the cost model, record metrics.
Terminates when nothing is running and no arrivals remain.

The virtual clock is sim-loop-owned bookkeeping: in isolation the scheduler does
not read the wall clock to make decisions (FCFS uses ``Request.arrival_time``,
set explicitly by the factory), so the clock only gates arrivals and accumulates
per-step latency.
"""

from typing import Iterable, Optional

from vllm.v1.request import RequestStatus

from vllm_sim.clock import VirtualClock
from vllm_sim.cost.base import CostModel
from vllm_sim.metrics import MetricsRecorder
from vllm_sim.sampler import build_runner_output
from vllm_sim.workload.base import RequestSpec


class SimLoop:
    def __init__(
        self,
        scheduler,
        specs: Iterable[RequestSpec],
        factory,
        cost_model: CostModel,
        clock: Optional[VirtualClock] = None,
        metrics: Optional[MetricsRecorder] = None,
        max_steps: int = 1_000_000,
    ):
        self.scheduler = scheduler
        self.specs = sorted(specs, key=lambda s: s.arrival_time)
        self.factory = factory
        self.cost_model = cost_model
        self.clock = clock if clock is not None else VirtualClock()
        self.metrics = metrics if metrics is not None else MetricsRecorder()
        self.max_steps = max_steps

    def _num_waiting(self) -> int:
        return sum(1 for _ in self.scheduler.waiting)

    def run(self) -> MetricsRecorder:
        idx = 0
        n = len(self.specs)
        requests_by_id = {}
        step = 0

        while True:
            # Admit all arrivals whose time has come.
            while idx < n and self.specs[idx].arrival_time <= self.clock.now():
                spec = self.specs[idx]
                idx += 1
                req = self.factory.to_request(spec)
                self.scheduler.add_request(req)
                requests_by_id[req.request_id] = req

            scheduler_output = self.scheduler.schedule()

            if scheduler_output.total_num_scheduled_tokens == 0:
                if idx < n:
                    # Nothing to do now, but more arrivals are coming: jump the
                    # clock forward to the next arrival instead of busy-looping.
                    self.clock.fast_forward_to(self.specs[idx].arrival_time)
                    continue
                # No future arrivals. If the scheduler is also empty we're done;
                # otherwise the workload is wedged (no progress possible) -> stop.
                break

            # Classify each scheduled request as prefill or decode this step.
            # schedule() has already advanced num_computed_tokens to the post-step
            # value, so subtract this step's scheduled tokens to recover the
            # pre-step count: a request whose prompt wasn't yet complete at the
            # start of the step is prefilling.
            prefill_reqs = decode_reqs = 0
            for rid, n_sched in scheduler_output.num_scheduled_tokens.items():
                req = requests_by_id[rid]
                pre_computed = req.num_computed_tokens - n_sched
                if pre_computed < req.num_prompt_tokens:
                    prefill_reqs += 1
                else:
                    decode_reqs += 1

            dt = self.cost_model.step_latency(scheduler_output, self.scheduler)
            runner_output = build_runner_output(scheduler_output, requests_by_id)
            self.scheduler.update_from_output(scheduler_output, runner_output)
            self.clock.advance(dt)

            block_pool = self.scheduler.kv_cache_manager.block_pool
            blocks_used = block_pool.num_gpu_blocks - block_pool.get_num_free_blocks()
            finished = sum(
                1 for r in requests_by_id.values() if RequestStatus.is_finished(r.status)
            )
            preemptions = sum(r.num_preemptions for r in requests_by_id.values())

            self.metrics.record(
                step=step,
                vclock=self.clock.now(),
                num_running=len(self.scheduler.running),
                num_waiting=self._num_waiting(),
                prefill_reqs=prefill_reqs,
                decode_reqs=decode_reqs,
                tokens_scheduled=scheduler_output.total_num_scheduled_tokens,
                blocks_used=blocks_used,
                num_blocks=block_pool.num_gpu_blocks,
                preemptions=preemptions,
                finished=finished,
            )

            step += 1
            if step >= self.max_steps:
                raise RuntimeError(
                    f"sim exceeded max_steps={self.max_steps}; possible non-termination"
                )

        return self.metrics
