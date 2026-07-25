"""Integration: real SGLang Scheduler runs its step loop in-process, GPU-free.

This is the SGLang spike gate (docs/journal/2026-07-24-sglang-engine-spike.md),
pinned to sglang==0.5.15.post1: the real ``sglang.srt.managers.scheduler.
Scheduler`` constructed directly — real TpModelWorker/ModelRunner (gloo,
world_size=1), real RadixCache and CPU KV pools, no ZMQ peers, no subprocess —
and the real scheduling iteration (``get_next_batch_to_run`` → ``run_batch`` →
``process_batch_result``) driven end-to-end: admit → prefill → decode →
FINISH_LENGTH. The forward pass is the real model forward (TorchNativeAttn
backend, dummy-loaded weights); nothing is stubbed.

The full injection surface is three sanctioned units:
  1. CPU device signal (``device="cpu"`` / ``SGLANG_USE_CPU_ENGINE=1``),
  2. ``load_format="dummy"``,
  3. ``SamplingParams.normalize(None)`` when injecting past TokenizerManager.

Greedy sampling over dummy weights yields degenerate token ids (all 0), so
assertions cover lengths, schedule shape, and pool/radix accounting — never
token identities.

Runs under .venv-sglang; skips cleanly where sglang is absent (the main .venv)
or facebook/opt-125m's config is not in the local HF cache (weights are
dummy-loaded, never downloaded).
"""

import os

import pytest

from tests.conftest import MODEL, model_cached  # sets HF_HUB_OFFLINE before sglang

os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")

sglang = pytest.importorskip("sglang")

pytestmark = pytest.mark.skipif(
    not model_cached(), reason=f"{MODEL} not in local HF cache"
)

MAX_TOTAL_TOKENS = 4096
MAX_RUNNING_REQUESTS = 8


@pytest.fixture(scope="module")
def scheduler():
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.server_args import PortArgs, ServerArgs

    server_args = ServerArgs(
        model_path=MODEL,
        device="cpu",
        load_format="dummy",
        tp_size=1,
        skip_tokenizer_init=True,
        disable_overlap_schedule=True,
        max_running_requests=MAX_RUNNING_REQUESTS,
        max_total_tokens=MAX_TOTAL_TOKENS,
        random_seed=42,
    )
    # ipc:// endpoints on throwaway tempfiles; sockets buffer in-process,
    # no peer processes exist or are needed.
    port_args = PortArgs.init_new(server_args)
    return Scheduler(
        server_args,
        port_args,
        gpu_id=0,
        tp_rank=0,
        moe_ep_rank=0,
        pp_rank=0,
        attn_cp_rank=0,
        moe_dp_rank=0,
        dp_rank=None,
    )


def _make_request(rid: str, prompt_ids: list[int], max_new_tokens: int):
    from array import array

    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
    from sglang.srt.sampling.sampling_params import SamplingParams

    params = SamplingParams(
        max_new_tokens=max_new_tokens,
        ignore_eos=True,
        temperature=0.0,  # greedy
    )
    # TokenizerManager normalizes before sending (tokenizer_manager.py:1133);
    # direct injection bypasses it. Injection unit 3 of 3.
    params.normalize(None)
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=array("q", prompt_ids),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=params,
        return_logprob=False,
        logprob_start_len=0,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )


def test_construction_gpu_free(scheduler):
    import psutil
    import torch
    import torch.distributed as dist

    assert scheduler.device == "cpu"
    assert type(scheduler.tree_cache).__name__ == "RadixCache"
    assert type(scheduler.req_to_token_pool).__name__ == "ReqToTokenPool"
    assert scheduler.req_to_token_pool.size == MAX_RUNNING_REQUESTS
    assert scheduler.max_total_num_tokens == MAX_TOTAL_TOKENS
    assert scheduler.token_to_kv_pool_allocator.available_size() == MAX_TOTAL_TOKENS
    assert scheduler.tree_cache.evictable_size() == 0

    # Real CPU tensors in the KV pool, sized for opt-125m (12 heads x 64 dim).
    kv_cache = scheduler.token_to_kv_pool_allocator.get_kvcache()
    k0 = kv_cache.k_buffer[0]
    assert k0.device.type == "cpu"
    assert k0.shape[1:] == (12, 64)
    assert not torch.cuda.is_available()

    # Single process, in-process gloo "distributed" group.
    assert psutil.Process().children(recursive=True) == []
    assert dist.is_initialized()
    assert dist.get_backend() == "gloo"
    assert dist.get_world_size() == 1


def test_step_loop_finishes_requests(scheduler):
    # 16-token prompt -> 6 new tokens; 12-token prompt -> 4 new tokens.
    specs = [("req-a", list(range(10, 26)), 6), ("req-b", list(range(100, 112)), 4)]
    for rid, prompt_ids, max_new in specs:
        scheduler.handle_generate_request(_make_request(rid, prompt_ids, max_new))
    assert len(scheduler.waiting_queue) == 2
    reqs = {req.rid: req for req in scheduler.waiting_queue}

    def occupancy() -> int:
        return (
            scheduler.max_total_num_tokens
            - scheduler.token_to_kv_pool_allocator.available_size()
        )

    # The event_loop_normal iteration (scheduler.py:1542), minus ZMQ recv.
    trace = []
    idle = 0
    for _ in range(50):
        batch = scheduler.get_next_batch_to_run()
        scheduler.cur_batch = batch
        if batch is None:
            idle += 1
            scheduler.last_batch = batch
            if idle >= 2:
                break
            continue
        idle = 0
        result = scheduler.run_batch(batch)
        scheduler.process_batch_result(batch, result)
        scheduler.last_batch = batch
        trace.append(
            (
                batch.forward_mode.name,
                tuple(
                    f"{r.rid}:{len(r.output_ids)}:{int(r.finished())}"
                    for r in sorted(batch.reqs, key=lambda r: r.rid)
                ),
                occupancy(),
                scheduler.tree_cache.evictable_size(),
            )
        )

    # Deterministic expected trace (seed 42; greedy; identical across fresh
    # processes — verified during the spike). Occupancy is prompt(28) plus one
    # slot per decoded token; a finished request's tokens move to *evictable*
    # radix-cache space (retained for prefix reuse), they are not freed.
    assert trace == [
        ("EXTEND", ("req-a:1:0", "req-b:1:0"), 28, 0),
        ("DECODE", ("req-a:2:0", "req-b:2:0"), 30, 0),
        ("DECODE", ("req-a:3:0", "req-b:3:0"), 32, 0),
        ("DECODE", ("req-a:4:0", "req-b:4:1"), 34, 15),  # req-b finishes, 2->1
        ("DECODE", ("req-a:5:0",), 35, 15),
        ("DECODE", ("req-a:6:1",), 36, 36),  # req-a finishes
    ]

    # Finish is length-determined, at exactly max_new_tokens.
    for rid, _, max_new in specs:
        req = reqs[rid]
        assert req.finished()
        assert type(req.finished_reason).__name__ == "FINISH_LENGTH"
        assert req.finished_reason.length == max_new
        assert len(req.output_ids) == max_new

    # Queues drain; nothing is left scheduled.
    assert scheduler.waiting_queue == []
    assert len(scheduler.running_batch.reqs) == 0
    assert scheduler.tree_cache.protected_size() == 0
