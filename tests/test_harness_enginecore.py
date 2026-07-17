"""build_engine_core: a live in-process EngineCore on MockPlatform, with the
SimScheduler capture seam and sim-controlled KV sizing."""

import pytest

from tests.conftest import MODEL, model_cached

pytestmark = pytest.mark.skipif(
    not model_cached(), reason=f"{MODEL} not in local HF cache"
)


def test_build_and_capture_and_shutdown():
    from llm_sim.harness.enginecore import build_engine_core
    from llm_sim.harness.scheduler import SimScheduler
    from llm_sim.workload.base import RequestSpec
    from llm_sim.workload.factory import RequestFactory

    # max_model_len must fit the pinned block budget: vLLM 0.23.0 checks
    # check_enough_kv_cache_memory against the overridden capacity
    # (100 blocks = 1600 tokens), and the model caps it at 2048 anyway.
    core = build_engine_core(block_size=16, num_blocks=100, max_model_len=512)
    try:
        from vllm.platforms import current_platform

        assert type(current_platform).__name__ == "MockPlatform"
        # Exact sim-pinned block budget (num_gpu_blocks_override).
        assert core.vllm_config.cache_config.num_gpu_blocks == 100
        # The capture seam is active.
        assert isinstance(core.scheduler, SimScheduler)
        assert core.scheduler.last_scheduler_output is None

        factory = RequestFactory(block_size=16)
        core.add_request(
            factory.to_request(RequestSpec("r0", 0.0, prompt_len=16, output_len=2))
        )
        core.step()
        so = core.scheduler.last_scheduler_output
        assert so is not None
        assert so.total_num_scheduled_tokens == 16  # the whole prompt, one chunk
    finally:
        core.shutdown()


def test_second_lifecycle_in_same_process():
    # The sim (tests, CLI scenarios) must be able to build/tear down more than
    # one EngineCore per process; the spike only ever built one.
    from llm_sim.harness.enginecore import build_engine_core

    core = build_engine_core(block_size=16, num_blocks=64, max_model_len=512)
    core.shutdown()
    core2 = build_engine_core(block_size=16, num_blocks=64, max_model_len=512)
    try:
        assert core2.vllm_config.cache_config.num_gpu_blocks == 64
    finally:
        core2.shutdown()


def test_kv_cache_bytes_budget_drives_num_blocks():
    from llm_sim.harness.enginecore import build_engine_core

    # max_model_len must fit the 64 MiB budget or vLLM's
    # check_enough_kv_cache_memory raises (512 tokens fp32 = ~36 MiB).
    core = build_engine_core(
        block_size=16, kv_cache_bytes=64 << 20, max_model_len=512
    )  # 64 MiB
    try:
        small = core.vllm_config.cache_config.num_gpu_blocks
    finally:
        core.shutdown()
    # opt-125m fp32 @ block_size=16 is ~1.125 MiB/block -> ~56 blocks from 64 MiB.
    assert 0 < small < 100
