"""The simulation loop over the real EngineCore.

Admits arrivals gated by the virtual clock, calls the real
``EngineCore.step()`` (schedule -> execute -> update_from_output, all vLLM
code; the transformer forward is synthetic — see llm_sim/mock/), then advances
the clock by the cost model and records metrics. The step's ``SchedulerOutput``
comes from the SimScheduler capture seam (llm_sim/harness/scheduler.py).

The virtual clock is loop-owned bookkeeping: in-process with log_stats off,
EngineCore makes no wall-clock-dependent scheduling decisions (FCFS uses
``Request.arrival_time``, set by the factory), so the clock only gates arrivals
and accumulates per-step latency. Terminates when nothing is pending and no
arrivals remain.
"""

from typing import Iterable, Optional

from vllm.v1.engine.core import EngineCore
from vllm.v1.request import RequestStatus

from llm_sim.clock import VirtualClock
from llm_sim.cost.base import CostModel
from llm_sim.metrics import MetricsRecorder
from llm_sim.workload.base import RequestSpec
from llm_sim.workload.factory import RequestFactory


class SimLoop:
    def __init__(
        self,
        core: EngineCore,
        specs: Iterable[RequestSpec],
        factory: RequestFactory,
        cost_model: CostModel,
        clock: Optional[VirtualClock] = None,
        metrics: Optional[MetricsRecorder] = None,
        max_steps: int = 1_000_000,
    ):
        self.core = core
        self.specs = sorted(specs, key=lambda s: s.arrival_time)
        self.factory = factory
        self.cost_model = cost_model
        self.clock = clock if clock is not None else VirtualClock()
        self.metrics = metrics if metrics is not None else MetricsRecorder()
        self.max_steps = max_steps

    def run(self) -> MetricsRecorder:
        scheduler = self.core.scheduler
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
                self.core.add_request(req)
                requests_by_id[req.request_id] = req

            if not scheduler.has_requests():
                if idx < n:
                    # Idle but more arrivals are coming: jump the clock forward
                    # to the next arrival instead of busy-looping.
                    self.clock.fast_forward_to(self.specs[idx].arrival_time)
                    continue
                break

            # Clear the capture so the assert below catches a stale value from a
            # previous iteration, not just a never-set one on the first step.
            scheduler.last_scheduler_output = None
            self.core.step()
            scheduler_output = scheduler.last_scheduler_output
            # Invariant: the has_requests() guard above matches EngineCore.step()'s
            # internal short-circuit, so schedule() always ran this iteration. If a
            # future vLLM bump breaks that pairing, fail loudly here instead of
            # silently reusing a stale capture.
            assert scheduler_output is not None

            if scheduler_output.total_num_scheduled_tokens == 0:
                # Nothing scheduled: either the trailing cleanup step after the
                # last finish (finished requests leave the persistent batch one
                # step late), or a wedged workload. Never recorded as a step.
                if idx < n:
                    self.clock.fast_forward_to(self.specs[idx].arrival_time)
                    continue
                # Either everything is finished (cleanup) or no progress is
                # possible (wedged) -> stop.
                break

            # Classify each scheduled request as prefill or decode this step.
            # schedule() advanced num_computed_tokens to the post-step value
            # (and update_from_output leaves it alone on the non-spec-decode
            # path), so subtract this step's scheduled tokens to recover the
            # pre-step count.
            prefill_reqs = decode_reqs = 0
            for rid, n_sched in scheduler_output.num_scheduled_tokens.items():
                req = requests_by_id[rid]
                pre_computed = req.num_computed_tokens - n_sched
                if pre_computed < req.num_prompt_tokens:
                    prefill_reqs += 1
                else:
                    decode_reqs += 1

            dt = self.cost_model.step_latency(scheduler_output, scheduler)
            self.clock.advance(dt)

            block_pool = scheduler.kv_cache_manager.block_pool
            blocks_used = block_pool.num_gpu_blocks - block_pool.get_num_free_blocks()
            finished = sum(
                1 for r in requests_by_id.values() if RequestStatus.is_finished(r.status)
            )
            preemptions = sum(r.num_preemptions for r in requests_by_id.values())

            self.metrics.record(
                step=step,
                vclock=self.clock.now(),
                num_running=len(scheduler.running),
                num_waiting=sum(1 for _ in scheduler.waiting),
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
