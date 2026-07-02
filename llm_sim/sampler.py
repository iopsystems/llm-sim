"""Synthesize the model-runner output the scheduler expects.

SECOND DRIFT RISK. Replaces ``GPUModelRunner.execute_model`` with a synthetic
sampler: token *identities* don't affect scheduling, so we emit a fixed dummy
id — the only thing that matters is *when* a token appears.

Rule (mirrors vLLM's gold ``tests/v1/core/test_scheduler.py``): a request emits
one sampled token in a step iff its prompt is fully computed by the end of that
step, i.e. ``num_computed_tokens >= num_prompt_tokens``. Otherwise it is still
prefilling and emits nothing. Decode steps (prompt already computed) always
emit. With ``SamplingParams(max_tokens=output_len, ignore_eos=True)`` a request
therefore finishes exactly ``output_len`` steps after its prompt completes.

Read-timing note (verified live in ``tests/test_sampler.py``): this is called
AFTER ``schedule()`` and BEFORE ``update_from_output``. ``schedule()`` already
advances ``num_computed_tokens`` to include this step's scheduled tokens, so the
condition is checked directly against ``num_computed_tokens`` — adding
``num_scheduled_tokens`` would double-count and spuriously emit a token mid-way
through a chunked prefill.
"""

from typing import Mapping

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

DUMMY_TOKEN_ID = 0


def build_runner_output(
    scheduler_output, requests_by_id: Mapping[str, Request]
) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}
    sampled_token_ids = []
    for rid in req_ids:
        req = requests_by_id[rid]
        # num_computed_tokens is already post-schedule here (see module docstring).
        if req.num_computed_tokens >= req.num_prompt_tokens:
            sampled_token_ids.append([DUMMY_TOKEN_ID])
        else:
            sampled_token_ids.append([])
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
