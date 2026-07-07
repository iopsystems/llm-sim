import json

import pytest

from llm_sim.workload.base import RequestSpec, WorkloadSource
from llm_sim.workload.trace import TraceWorkload

EXPECTED = [
    RequestSpec(request_id="a", arrival_time=0.0, prompt_len=10, output_len=5),
    RequestSpec(request_id="b", arrival_time=0.5, prompt_len=20, output_len=8),
    RequestSpec(request_id="c", arrival_time=1.25, prompt_len=4, output_len=16),
]


@pytest.fixture
def csv_file(tmp_path):
    p = tmp_path / "trace.csv"
    p.write_text(
        "request_id,arrival_time,prompt_len,output_len\n"
        "a,0.0,10,5\n"
        "b,0.5,20,8\n"
        "c,1.25,4,16\n"
    )
    return p


@pytest.fixture
def jsonl_file(tmp_path):
    p = tmp_path / "trace.jsonl"
    rows = [
        {"request_id": "a", "arrival_time": 0.0, "prompt_len": 10, "output_len": 5},
        {"request_id": "b", "arrival_time": 0.5, "prompt_len": 20, "output_len": 8},
        {"request_id": "c", "arrival_time": 1.25, "prompt_len": 4, "output_len": 16},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_is_a_workload_source(csv_file):
    assert isinstance(TraceWorkload(csv_file), WorkloadSource)


def test_csv_parses_into_specs(csv_file):
    assert list(TraceWorkload(csv_file).generate()) == EXPECTED


def test_jsonl_parses_into_specs(jsonl_file):
    assert list(TraceWorkload(jsonl_file).generate()) == EXPECTED


def test_csv_and_jsonl_identical(csv_file, jsonl_file):
    assert list(TraceWorkload(csv_file).generate()) == list(TraceWorkload(jsonl_file).generate())


def test_sorts_by_arrival_time(tmp_path):
    p = tmp_path / "unsorted.csv"
    p.write_text(
        "request_id,arrival_time,prompt_len,output_len\n"
        "late,2.0,1,1\n"
        "early,0.1,1,1\n"
    )
    specs = list(TraceWorkload(p).generate())
    assert [s.request_id for s in specs] == ["early", "late"]


def test_unknown_extension_raises(tmp_path):
    p = tmp_path / "trace.txt"
    p.write_text("nope")
    with pytest.raises(ValueError):
        list(TraceWorkload(p).generate())
