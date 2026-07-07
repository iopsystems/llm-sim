"""Marquee integration test: run the real V1 scheduler under the sim loop with a
constant cost model and assert the emergent control-plane behavior.
"""

from llm_sim.cost.constant import ConstantCostModel
from llm_sim.engine import SimLoop
from llm_sim.harness.builder import build_scheduler
from llm_sim.workload.base import RequestSpec
from llm_sim.workload.factory import RequestFactory

BLOCK = 16


def _loop(specs, num_blocks=100, latency=0.01, **build_kw):
    sched = build_scheduler(block_size=BLOCK, num_blocks=num_blocks, **build_kw)
    factory = RequestFactory(block_size=BLOCK)
    return SimLoop(
        scheduler=sched,
        specs=specs,
        factory=factory,
        cost_model=ConstantCostModel(latency_s=latency),
    )


def test_two_concurrent_requests_finish_in_lockstep():
    specs = [
        RequestSpec("r0", 0.0, 16, 2),
        RequestSpec("r1", 0.0, 16, 2),
    ]
    metrics = _loop(specs, latency=0.01).run()
    recs = metrics.records

    # output_len=2, prompt fits in one prefill step -> exactly 2 steps.
    assert len(recs) == 2

    # Step 0: both prefill the 16-token prompt (16+16 tokens).
    assert recs[0].prefill_reqs == 2
    assert recs[0].decode_reqs == 0
    assert recs[0].tokens_scheduled == 32

    # Step 1: both decode one token.
    assert recs[1].prefill_reqs == 0
    assert recs[1].decode_reqs == 2
    assert recs[1].tokens_scheduled == 2

    s = metrics.summary()
    assert s["total_finished"] == 2
    # virtual time = constant_latency * num_steps
    assert abs(s["virtual_time_s"] - 0.02) < 1e-9
    assert s["num_steps"] == 2


def test_all_requests_finish_and_blocks_never_exceed_budget():
    specs = [RequestSpec(f"r{i}", 0.0, 16, 3) for i in range(5)]
    loop = _loop(specs, num_blocks=100, latency=0.005)
    metrics = loop.run()
    s = metrics.summary()
    assert s["total_finished"] == 5
    assert all(r.blocks_used <= r.num_blocks for r in metrics.records)


def test_staggered_arrival_fast_forwards_idle_gap():
    # r1 arrives long after r0 has finished -> loop must jump virtual time.
    specs = [
        RequestSpec("r0", 0.0, 16, 2),
        RequestSpec("r1", 100.0, 16, 2),
    ]
    metrics = _loop(specs, latency=1.0).run()
    s = metrics.summary()
    assert s["total_finished"] == 2
    assert s["num_steps"] == 4  # 2 steps each, no overlap
    # clock fast-forwarded across the idle gap to r1's arrival (100) + its 2 steps
    assert s["virtual_time_s"] >= 100.0


def test_tight_block_budget_forces_preemption():
    # 4 reqs each needing 2 blocks at peak, only ~7 usable blocks -> contention.
    specs = [RequestSpec(f"r{i}", 0.0, 16, 16) for i in range(4)]
    loop = _loop(specs, num_blocks=8, latency=0.01, max_num_seqs=64, max_model_len=4096)
    metrics = loop.run()
    s = metrics.summary()
    assert s["total_preemptions"] >= 1  # this is the preemption demo
    assert s["total_finished"] == 4  # ...and they still all complete
    assert all(r.blocks_used <= r.num_blocks for r in metrics.records)
