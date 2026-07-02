from vllm.v1.request import Request

from llm_sim.workload.base import RequestSpec
from llm_sim.workload.factory import RequestFactory


def _spec(**kw):
    base = dict(request_id="r0", arrival_time=1.5, prompt_len=20, output_len=7)
    base.update(kw)
    return RequestSpec(**base)


def test_builds_a_vllm_request():
    req = RequestFactory().to_request(_spec())
    assert isinstance(req, Request)


def test_prompt_length_matches_spec():
    req = RequestFactory().to_request(_spec(prompt_len=23))
    assert len(req.prompt_token_ids) == 23


def test_max_tokens_matches_output_len():
    req = RequestFactory().to_request(_spec(output_len=9))
    assert req.sampling_params.max_tokens == 9


def test_ignore_eos_set():
    req = RequestFactory().to_request(_spec())
    assert req.sampling_params.ignore_eos is True


def test_arrival_time_preserved():
    req = RequestFactory().to_request(_spec(arrival_time=3.25))
    assert req.arrival_time == 3.25


def test_request_id_preserved():
    req = RequestFactory().to_request(_spec(request_id="abc"))
    assert req.request_id == "abc"


def test_no_pooling_or_lora():
    req = RequestFactory().to_request(_spec())
    assert req.pooling_params is None
    assert req.lora_request is None


def test_distinct_specs_yield_distinct_requests():
    f = RequestFactory()
    r0 = f.to_request(_spec(request_id="r0"))
    r1 = f.to_request(_spec(request_id="r1"))
    assert r0.request_id != r1.request_id
