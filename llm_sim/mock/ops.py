"""Pure-torch CPU fallbacks for CPU-build-only vllm._C ops.

The CUDA wheel registers the ``_C`` torch library but not the CPU-backend
kernels (csrc/cpu). ``torch.library`` FRAGMENT registration is the supported
way to add the missing ops — no monkeypatching. Only ops the non-speculative
decode path needs are provided; eagle/spec-decode ops are out of v2 scope.
"""

import torch

_lib = torch.library.Library("_C", "FRAGMENT")

_lib.define(
    "compute_slot_mapping_kernel_impl("
    "Tensor query_start_loc, Tensor positions, Tensor block_table, "
    "Tensor(a!) slot_mapping, int block_size) -> ()"
)


def _compute_slot_mapping(
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    qsl = query_start_loc.to(torch.int64)
    num_reqs = qsl.numel() - 1
    counts = qsl[1:] - qsl[:-1]
    num_tokens = int(qsl[-1].item())
    req_idx = torch.repeat_interleave(torch.arange(num_reqs), counts)
    pos = positions[:num_tokens]
    blocks = block_table[req_idx, pos // block_size].to(torch.int64)
    slot_mapping[:num_tokens] = blocks * block_size + pos % block_size


_lib.impl("compute_slot_mapping_kernel_impl", _compute_slot_mapping, "CPU")


# CPU_ATTN's metadata builder calls this per step; its output is opaque and
# consumed only by the CPU attention kernel, which never runs under the
# stubbed forward. A shape-valid empty tensor is sufficient.
_lib.define(
    "get_scheduler_metadata("
    "int num_reqs, int num_heads, int num_kv_heads, int head_dim, "
    "Tensor seq_lens, ScalarType dtype, Tensor query_start_loc, bool causal, "
    "int sliding_window_size, str isa, bool enable_kv_split) -> Tensor"
)


def _get_scheduler_metadata(
    num_reqs,
    num_heads,
    num_kv_heads,
    head_dim,
    seq_lens,
    dtype,
    query_start_loc,
    causal,
    sliding_window_size,
    isa,
    enable_kv_split,
) -> torch.Tensor:
    return torch.empty(0, dtype=torch.int64)


_lib.impl("get_scheduler_metadata", _get_scheduler_metadata, "CPU")
