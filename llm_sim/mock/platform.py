"""MockPlatform: the CPU platform, minus its host-machine coupling.

CpuPlatform already routes to the real v1 Worker/GPUModelRunner machinery
(CPUWorker subclasses gpu_worker.Worker; CPUModelRunner subclasses
GPUModelRunner), so inheriting it keeps vLLM's real init and per-step code
paths. Each override below is one unit of injection surface — keep this list
short and every entry justified.
"""

from vllm.platforms.cpu import CpuPlatform


class MockPlatform(CpuPlatform):
    device_name: str = "mock"

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        super().check_and_update_config(vllm_config)
        # Override #1: CpuPlatform forces the multiprocess executor (OMP thread
        # binding); the sim needs everything in one process.
        vllm_config.parallel_config.distributed_executor_backend = "uni"
        # Override #2: CPUWorker's NUMA binding needs CPU-build-only _C ops and
        # couples KV sizing to host RAM; MockWorker is the same real Worker
        # base minus that coupling.
        vllm_config.parallel_config.worker_cls = "llm_sim.mock.worker.MockWorker"
