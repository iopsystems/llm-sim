"""Turn a GPU-free ``RequestSpec`` into a real vLLM ``Request``.

Replicates the construction path of vLLM's gold reference
``tests/v1/core/utils.py::create_requests`` (pinned to v0.23.0), quarantined
here because the ``Request`` / ``SamplingParams`` surface churns across releases.

Determinism: ``SamplingParams(max_tokens=output_len, ignore_eos=True)`` makes a
request finish exactly ``output_len`` decode steps after its prompt is fully
scheduled. Token identities don't affect scheduling (prefix caching is off in
the MVP), so prompts are filled with a constant dummy token.
"""

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request

from vllm_sim.workload.base import RequestSpec

# EOS id used by the gold reference; irrelevant under ignore_eos but kept for
# faithful SamplingParams construction.
EOS_TOKEN_ID = 50256
DUMMY_TOKEN_ID = 0


class RequestFactory:
    def __init__(self, block_size: int = 16, eos_token_id: int = EOS_TOKEN_ID):
        self.block_size = block_size
        self.eos_token_id = eos_token_id
        # init_none_hash is idempotent-safe to call repeatedly; needed before
        # building a block hasher.
        init_none_hash(sha256)
        self._block_hasher = get_request_block_hasher(block_size, sha256)

    def to_request(self, spec: RequestSpec) -> Request:
        sampling_params = SamplingParams(
            ignore_eos=True,
            max_tokens=spec.output_len,
        )
        sampling_params.update_from_generation_config({}, self.eos_token_id)
        return Request(
            request_id=spec.request_id,
            prompt_token_ids=[DUMMY_TOKEN_ID] * spec.prompt_len,
            sampling_params=sampling_params,
            pooling_params=None,
            arrival_time=spec.arrival_time,
            block_hasher=self._block_hasher,
        )
