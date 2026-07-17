"""Marquee integration test: the real EngineCore.step() under the sim loop with
a constant cost model — same emergent control-plane scenarios the MVP asserted,
now through the full stack (real Worker/ModelRunner/Sampler, synthetic forward).
"""

import pytest

from tests.conftest import MODEL, model_cached

pytestmark = pytest.mark.skipif(
    not model_cached(), reason=f"{MODEL} not in local HF cache"
)

from llm_sim.cost.constant import ConstantCostModel
from llm_sim.engine import SimLoop
from llm_sim.harness.enginecore import build_engine_core
from llm_sim.workload.base import RequestSpec
from llm_sim.workload.factory import RequestFactory

BLOCK = 16


def _run(specs, num_blocks=100, latency=0.01, **build_kw):
    # max_model_len must fit the block budget: 100 blocks at block 16 = 1600
    # tokens < the derived 2048 default.
    build_kw.setdefault("max_model_len", 512)
    core = build_engine_core(block_size=BLOCK, num_blocks=num_blocks, **build_kw)
    try:
        loop = SimLoop(
            core=core,
            specs=specs,
            factory=RequestFactory(block_size=BLOCK),
            cost_model=ConstantCostModel(latency_s=latency),
        )
        return loop.run()
    finally:
        core.shutdown()


def test_two_concurrent_requests_finish_in_lockstep():
    specs = [
        RequestSpec("r0", 0.0, 16, 2),
        RequestSpec("r1", 0.0, 16, 2),
    ]
    metrics = _run(specs, latency=0.01)
    recs = metrics.records

    # output_len=2, prompt fits one prefill step -> exactly 2 recorded steps
    # (the trailing zero-token cleanup step is not recorded).
    assert len(recs) == 2

    # Step 0: both prefill the 16-token prompt (16+16 tokens); the first output
    # token samples in this same step (prompt fully computed).
    assert recs[0].prefill_reqs == 2
    assert recs[0].decode_reqs == 0
    assert recs[0].tokens_scheduled == 32

    # Step 1: both decode their second (final) token.
    assert recs[1].prefill_reqs == 0
    assert recs[1].decode_reqs == 2
    assert recs[1].tokens_scheduled == 2

    s = metrics.summary()
    assert s["total_finished"] == 2
    assert abs(s["virtual_time_s"] - 0.02) < 1e-9
    assert s["num_steps"] == 2
