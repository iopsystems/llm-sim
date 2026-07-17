"""SimScheduler: the real v1 Scheduler plus a capture of the last SchedulerOutput.

EngineCore.step() runs schedule -> execute -> update_from_output internally and
returns only per-request outputs; the sim needs the step's SchedulerOutput to
feed CostModel.step_latency() and the per-step metrics. Injected via the
sanctioned ``scheduler_config.scheduler_cls`` seam (resolved in
vllm/v1/engine/core.py by ``get_scheduler_cls()``); ``schedule()`` is the real
method, observed, not modified.
"""

from typing import TYPE_CHECKING

from vllm.v1.core.sched.scheduler import Scheduler

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class SimScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_scheduler_output: "SchedulerOutput | None" = None

    def schedule(self) -> "SchedulerOutput":
        output = super().schedule()
        self.last_scheduler_output = output
        return output
