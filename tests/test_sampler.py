from vllm.v1.outputs import ModelRunnerOutput

from vllm_sim.harness.builder import build_scheduler
from vllm_sim.sampler import build_runner_output
from vllm_sim.workload.base import RequestSpec
from vllm_sim.workload.factory import RequestFactory

BLOCK = 16


def _setup(max_num_batched_tokens=8192, max_model_len=None):
    sched = build_scheduler(
        block_size=BLOCK,
        num_blocks=1000,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
    )
    factory = RequestFactory(block_size=BLOCK)
    return sched, factory


def test_returns_model_runner_output():
    sched, factory = _setup()
    req = factory.to_request(RequestSpec("r0", 0.0, 20, 3))
    sched.add_request(req)
    out = sched.schedule()
    runner_out = build_runner_output(out, {"r0": req})
    assert isinstance(runner_out, ModelRunnerOutput)
    assert runner_out.req_ids == ["r0"]
    assert runner_out.req_id_to_index == {"r0": 0}


def test_emits_token_when_prompt_fully_scheduled():
    sched, factory = _setup()
    req = factory.to_request(RequestSpec("r0", 0.0, 20, 3))
    sched.add_request(req)
    out = sched.schedule()  # 20-token prompt fits the budget -> fully prefilled
    runner_out = build_runner_output(out, {"r0": req})
    assert runner_out.sampled_token_ids == [[0]]


def test_emits_empty_during_chunked_prefill():
    sched, factory = _setup(max_num_batched_tokens=32, max_model_len=1024)
    req = factory.to_request(RequestSpec("r0", 0.0, 100, 3))
    sched.add_request(req)
    out = sched.schedule()  # only 32 of 100 prompt tokens scheduled
    assert out.num_scheduled_tokens["r0"] == 32
    runner_out = build_runner_output(out, {"r0": req})
    assert runner_out.sampled_token_ids == [[]]


def test_decode_step_emits_token():
    sched, factory = _setup()
    req = factory.to_request(RequestSpec("r0", 0.0, 20, 3))
    sched.add_request(req)
    # Step 1: prefill -> emit, then feed back so the request enters decode.
    out1 = sched.schedule()
    sched.update_from_output(out1, build_runner_output(out1, {"r0": req}))
    # Step 2: decode -> 1 token scheduled, prompt already computed -> emit.
    out2 = sched.schedule()
    assert out2.num_scheduled_tokens["r0"] == 1
    runner_out = build_runner_output(out2, {"r0": req})
    assert runner_out.sampled_token_ids == [[0]]


def test_no_token_until_prompt_actually_completes():
    # prompt=32, token budget=16: the first chunk schedules 16/32 prompt tokens
    # and does NOT complete the prompt, so NO token may be emitted yet. (Regression:
    # a num_computed + num_scheduled formula double-counts and emits spuriously.)
    sched, factory = _setup(max_num_batched_tokens=16, max_model_len=1024)
    req = factory.to_request(RequestSpec("r0", 0.0, 32, 2))
    sched.add_request(req)
    out1 = sched.schedule()
    assert out1.num_scheduled_tokens["r0"] == 16  # partial prefill
    assert build_runner_output(out1, {"r0": req}).sampled_token_ids == [[]]
    sched.update_from_output(out1, build_runner_output(out1, {"r0": req}))
    # Second chunk completes the prompt (16 more -> 32) -> first real token now.
    out2 = sched.schedule()
    assert out2.num_scheduled_tokens["r0"] == 16
    assert build_runner_output(out2, {"r0": req}).sampled_token_ids == [[0]]


def test_mixed_batch_prefill_and_chunk():
    # r0 short (fully prefills), r1 long (chunked) in the same step.
    sched, factory = _setup(max_num_batched_tokens=64, max_model_len=1024)
    r0 = factory.to_request(RequestSpec("r0", 0.0, 16, 3))
    r1 = factory.to_request(RequestSpec("r1", 0.0, 200, 3))
    sched.add_request(r0)
    sched.add_request(r1)
    out = sched.schedule()
    runner_out = build_runner_output(out, {"r0": r0, "r1": r1})
    by_id = dict(zip(runner_out.req_ids, runner_out.sampled_token_ids))
    # r0 fully scheduled -> token; r1 only partially -> empty.
    assert by_id["r0"] == [0]
    assert by_id["r1"] == []
