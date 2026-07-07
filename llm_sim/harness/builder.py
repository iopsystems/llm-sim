"""Build a live, GPU-free vLLM V1 ``Scheduler``.

HIGHEST DRIFT RISK in this package. Replicates vLLM's gold reference
``tests/v1/core/utils.py::create_scheduler`` (pinned to v0.23.0), with one
addition the reference doesn't need: on a standard CUDA wheel running without a
GPU driver, ``current_platform`` resolves to ``UnspecifiedPlatform`` (empty
``device_type``), so ``VllmConfig`` construction would raise "Failed to infer
device type". We pass an explicit ``DeviceConfig(device="cpu")`` to bypass
device inference while keeping the neutral platform (no CPU-specific config
rewrites). We never touch a GPU — only the scheduler's Python control plane.

KV-cache config is built directly (no memory profiling): ``num_blocks`` is the
block budget that drives preemption/recompute, set explicitly here.
"""

import torch
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.structured_output import StructuredOutputManager


def build_scheduler(
    model: str = "facebook/opt-125m",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    enable_chunked_prefill: bool = True,
    enable_prefix_caching: bool = False,
    long_prefill_token_threshold: int = 0,
    num_blocks: int = 10000,
    block_size: int = 16,
    max_model_len: int | None = None,
) -> Scheduler:
    model_config = ModelConfig(
        model=model,
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    if max_model_len is None:
        max_model_len = max_num_batched_tokens
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        long_prefill_token_threshold=long_prefill_token_threshold,
        enable_chunked_prefill=enable_chunked_prefill,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )
