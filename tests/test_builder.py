from vllm.v1.core.sched.scheduler import Scheduler

from llm_sim.harness.builder import build_scheduler
from llm_sim.workload.base import RequestSpec
from llm_sim.workload.factory import RequestFactory


def test_builds_a_live_scheduler():
    sched = build_scheduler()
    assert isinstance(sched, Scheduler)


def test_schedules_one_request_nonempty():
    block_size = 16
    sched = build_scheduler(block_size=block_size, num_blocks=100)
    factory = RequestFactory(block_size=block_size)
    req = factory.to_request(
        RequestSpec(request_id="r0", arrival_time=0.0, prompt_len=20, output_len=5)
    )
    sched.add_request(req)
    out = sched.schedule()
    assert out.total_num_scheduled_tokens == 20
    assert out.num_scheduled_tokens["r0"] == 20
    assert len(out.scheduled_new_reqs) == 1


def test_empty_scheduler_schedules_nothing():
    sched = build_scheduler()
    out = sched.schedule()
    assert out.total_num_scheduled_tokens == 0


def test_num_blocks_is_respected():
    sched = build_scheduler(num_blocks=42, block_size=16)
    # The KV cache manager's block pool should reflect our budget.
    assert sched.kv_cache_manager.block_pool.num_gpu_blocks == 42


def test_chunked_prefill_splits_long_prompt():
    # token budget smaller than prompt -> prompt scheduled across steps
    block_size = 16
    sched = build_scheduler(
        block_size=block_size,
        num_blocks=1000,
        max_num_batched_tokens=32,
        max_model_len=1024,
    )
    factory = RequestFactory(block_size=block_size)
    req = factory.to_request(
        RequestSpec(request_id="r0", arrival_time=0.0, prompt_len=100, output_len=1)
    )
    sched.add_request(req)
    out = sched.schedule()
    assert out.num_scheduled_tokens["r0"] == 32  # capped by token budget
