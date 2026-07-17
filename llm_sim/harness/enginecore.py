"""Build a live, GPU-free, in-process vLLM EngineCore on MockPlatform.

Replaces the retired scheduler-in-isolation builder: instead of hand-building a
Scheduler from configs, this boots the real stack — UniProcExecutor ->
MockWorker -> MockModelRunner (dummy-loaded weights, real attention-backend
selection, real Sampler; only the transformer forward is synthetic — see
llm_sim/mock/). Pinned to vllm==0.23.0.

KV sizing is sim-owned, two knobs:
- ``kv_cache_bytes``: the analytic budget MockWorker reports (default 1 GiB);
  num_blocks derives from it. Must hold one max_model_len request or vLLM's
  check_enough_kv_cache_memory raises — pass a small ``max_model_len`` along
  with a small budget. (max_model_len defaults to the model config's derived
  limit — 2048 for opt-125m, ~144 MiB fp32 — because ModelConfig validation
  rejects anything larger than max_position_embeddings.)
- ``num_blocks``: exact block count via vLLM's own num_gpu_blocks_override.
  NOTE: KV blocks are really allocated as CPU tensors (~1.125 MiB/block for
  opt-125m fp32 at block_size=16) — large counts cost real host RAM.
"""

from vllm.engine.arg_utils import EngineArgs
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor

import llm_sim.mock.worker as mock_worker

SIM_SCHEDULER_CLS = "llm_sim.harness.scheduler.SimScheduler"


def build_engine_core(
    model: str = "facebook/opt-125m",
    block_size: int = 16,
    num_blocks: int | None = None,
    kv_cache_bytes: int | None = None,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    max_model_len: int | None = None,
    enable_chunked_prefill: bool = True,
) -> EngineCore:
    # max_model_len=None lets vLLM derive it from the model config (2048 for
    # opt-125m); ModelConfig validation rejects values above the model's
    # max_position_embeddings, so max_num_batched_tokens (8192) is not a
    # usable default here.
    mock_worker.set_kv_cache_bytes(kv_cache_bytes)

    engine_args = EngineArgs(
        model=model,
        load_format="dummy",
        enforce_eager=True,
        dtype="float32",
        seed=42,
        block_size=block_size,
        num_gpu_blocks_override=num_blocks,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_prefix_caching=False,
    )
    vllm_config = engine_args.create_engine_config()
    # Safe post-config injection: EngineCore resolves scheduler_cls at
    # construction (vllm/v1/engine/core.py:136); nothing reads it earlier.
    vllm_config.scheduler_config.scheduler_cls = SIM_SCHEDULER_CLS
    return EngineCore(vllm_config, Executor.get_class(vllm_config), log_stats=False)
