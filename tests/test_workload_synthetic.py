from llm_sim.workload.base import RequestSpec, WorkloadSource
from llm_sim.workload.synthetic import SyntheticWorkload


def _cfg(**kw):
    base = dict(
        num_requests=50,
        arrival_rate=10.0,
        prompt_len=(8, 32),
        output_len=(4, 16),
        seed=123,
    )
    base.update(kw)
    return base


def test_is_a_workload_source():
    wl = SyntheticWorkload(**_cfg())
    assert isinstance(wl, WorkloadSource)


def test_generates_requested_count_of_specs():
    specs = list(SyntheticWorkload(**_cfg(num_requests=37)).generate())
    assert len(specs) == 37
    assert all(isinstance(s, RequestSpec) for s in specs)


def test_deterministic_under_same_seed():
    a = list(SyntheticWorkload(**_cfg(seed=7)).generate())
    b = list(SyntheticWorkload(**_cfg(seed=7)).generate())
    assert a == b


def test_different_seed_differs():
    a = list(SyntheticWorkload(**_cfg(seed=1)).generate())
    b = list(SyntheticWorkload(**_cfg(seed=2)).generate())
    assert a != b


def test_arrivals_are_monotonic_nondecreasing():
    specs = list(SyntheticWorkload(**_cfg()).generate())
    times = [s.arrival_time for s in specs]
    assert times == sorted(times)
    assert times[0] >= 0.0


def test_lengths_within_bounds():
    specs = list(SyntheticWorkload(**_cfg(prompt_len=(8, 32), output_len=(4, 16))).generate())
    assert all(8 <= s.prompt_len <= 32 for s in specs)
    assert all(4 <= s.output_len <= 16 for s in specs)
    assert all(s.prompt_len >= 1 and s.output_len >= 1 for s in specs)


def test_fixed_lengths_when_bounds_equal():
    specs = list(SyntheticWorkload(**_cfg(prompt_len=(10, 10), output_len=(5, 5))).generate())
    assert all(s.prompt_len == 10 and s.output_len == 5 for s in specs)


def test_request_ids_unique():
    specs = list(SyntheticWorkload(**_cfg()).generate())
    ids = [s.request_id for s in specs]
    assert len(set(ids)) == len(ids)


def test_fixed_interval_arrivals_when_rate_none():
    specs = list(SyntheticWorkload(**_cfg(arrival_rate=None, interval=0.5, num_requests=4)).generate())
    times = [s.arrival_time for s in specs]
    assert times == [0.0, 0.5, 1.0, 1.5]
