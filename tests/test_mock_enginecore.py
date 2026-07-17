"""Integration: real EngineCore.step() runs end-to-end on MockPlatform.

This is the v2 spike gate (docs/journal/2026-07-09-gpu-mock-enginecore-boot.md):
GPU-free, single-process, no ZMQ/subprocess — real UniProcExecutor, real
Worker/ModelRunner init, real Scheduler, real Sampler; only the transformer
forward is synthetic.

Needs facebook/opt-125m in the local HF cache (config only; weights are
dummy-loaded). Skips cleanly when it isn't available.
"""

import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from llm_sim.workload.base import RequestSpec
from llm_sim.workload.factory import RequestFactory

MODEL = "facebook/opt-125m"


def _model_cached() -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(MODEL, local_files_only=True, allow_patterns=["config.json"])
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _model_cached(), reason=f"{MODEL} not in local HF cache"
)


@pytest.fixture(scope="module")
def engine_core():
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.abstract import Executor

    engine_args = EngineArgs(
        model=MODEL,
        load_format="dummy",
        enforce_eager=True,
        max_model_len=512,
        dtype="float32",
    )
    vllm_config = engine_args.create_engine_config()
    core = EngineCore(vllm_config, Executor.get_class(vllm_config), log_stats=False)
    yield core, vllm_config
    core.shutdown()


def test_platform_is_mock():
    from vllm.platforms import current_platform

    assert type(current_platform).__name__ == "MockPlatform"


def test_config_gpu_free(engine_core):
    _, vllm_config = engine_core
    # num_blocks derives from MockWorker's analytic budget, not host memory.
    from llm_sim.mock.worker import MOCK_KV_CACHE_BYTES

    assert vllm_config.parallel_config.worker_cls == "llm_sim.mock.worker.MockWorker"
    assert vllm_config.cache_config.num_gpu_blocks is not None
    assert vllm_config.cache_config.num_gpu_blocks > 0
    per_block = (
        vllm_config.cache_config.block_size  # tokens/block
    )
    assert per_block > 0
    assert MOCK_KV_CACHE_BYTES == 1 << 30

    # Scheduling semantics the sim relies on.
    assert vllm_config.scheduler_config.async_scheduling is False
    assert vllm_config.scheduler_config.enable_chunked_prefill is True


def test_step_loop_finishes_requests(engine_core):
    core, vllm_config = engine_core
    factory = RequestFactory(block_size=vllm_config.cache_config.block_size)
    specs = [
        RequestSpec(request_id="r0", arrival_time=0.0, prompt_len=100, output_len=8),
        RequestSpec(request_id="r1", arrival_time=0.0, prompt_len=48, output_len=4),
    ]
    for spec in specs:
        core.add_request(factory.to_request(spec))

    finished_at: dict[str, int] = {}
    for step_i in range(64):
        outputs_by_client, executed = core.step()
        assert executed
        for outs in outputs_by_client.values():
            for out in outs.outputs:
                if out.finished:
                    finished_at[out.request_id] = step_i
        if len(finished_at) == len(specs):
            break

    # Both prompts fit one chunked-prefill budget, so the first token samples
    # at step 0 and a request with output_len N finishes at step N-1.
    assert finished_at == {"r1": 3, "r0": 7}
    assert len(core.scheduler.running) == 0
    # Finished requests are removed from the persistent batch on the next
    # step ("finished and not yet removed" — see EngineCore.step()).
    core.step()
    assert not core.scheduler.has_requests()
