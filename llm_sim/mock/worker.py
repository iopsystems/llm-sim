"""MockWorker / MockModelRunner: real vLLM v1 machinery, GPU-free.

MockWorker subclasses the real ``gpu_worker.Worker`` (the same base CPUWorker
uses) and replaces:

- ``CPUWorker.__init__``'s NUMA binding — needs CPU-build-only torch.ops._C
  ops, and couples worker startup to host memory topology;
- ``determine_available_memory`` — an analytic KV budget so the sim controls
  ``num_blocks``; host RAM must not leak into scheduling decisions;
- ``compile_or_warm_up_model`` — no real forward exists to warm up.

``init_device`` mirrors ``CPUWorker.init_device`` minus thread binding.

MockModelRunner overrides exactly one method: ``_model_forward``,
GPUModelRunner's documented override seam ("This method can be overridden by
subclasses for model execution"). Input/metadata prep, logits computation, and
the real Sampler still run; only the transformer forward is synthetic.
"""

import torch

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.cpu_model_runner import CPUModelRunner
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.worker_base import CompilationTimes

import llm_sim.mock.ops  # noqa: F401  -- registers _C CPU op fallbacks

# Sim-owned analytic KV budget (bytes). num_blocks derives from it and drives
# preemption; the sim sets it per run via set_kv_cache_bytes().
DEFAULT_KV_CACHE_BYTES = 1 << 30
_kv_cache_bytes = DEFAULT_KV_CACHE_BYTES


def set_kv_cache_bytes(n: int | None) -> None:
    """Set the KV budget for subsequently built engines (None = default 1 GiB)."""
    global _kv_cache_bytes
    _kv_cache_bytes = DEFAULT_KV_CACHE_BYTES if n is None else int(n)


class MockModelRunner(CPUModelRunner):
    def _model_forward(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        **model_kwargs,
    ):
        num_tokens = (
            input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        )
        hidden_size = self.model_config.get_hidden_size()
        return torch.zeros(
            (num_tokens, hidden_size), dtype=self.dtype, device=self.device
        )


class MockWorker(Worker):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.parallel_config.disable_custom_all_reduce = True
        self.profiler = None

    def init_device(self):
        self.device = torch.device("cpu")
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )
        set_random_seed(self.model_config.seed)
        self.model_runner = MockModelRunner(self.vllm_config, self.device)

    def determine_available_memory(self) -> int:
        return _kv_cache_bytes

    def compile_or_warm_up_model(self) -> CompilationTimes:
        set_random_seed(self.model_config.seed)
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )

    def sleep(self, level: int = 1) -> None:
        pass

    def wake_up(self, tags=None) -> None:
        pass
