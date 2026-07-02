import json

from vllm_sim.metrics import MetricsRecorder


def _rec(m, **kw):
    base = dict(
        step=0, vclock=0.0, num_running=0, num_waiting=0, prefill_reqs=0,
        decode_reqs=0, tokens_scheduled=0, blocks_used=0, num_blocks=100,
        preemptions=0, finished=0,
    )
    base.update(kw)
    m.record(**base)


def test_records_accumulate():
    m = MetricsRecorder()
    _rec(m, step=0)
    _rec(m, step=1)
    assert len(m.records) == 2


def test_summary_reports_step_count_and_virtual_time():
    m = MetricsRecorder()
    _rec(m, step=0, vclock=0.01)
    _rec(m, step=1, vclock=0.02)
    _rec(m, step=2, vclock=0.03)
    s = m.summary()
    assert s["num_steps"] == 3
    assert s["virtual_time_s"] == 0.03


def test_summary_reports_peaks_and_totals():
    m = MetricsRecorder()
    _rec(m, step=0, num_running=2, tokens_scheduled=20, blocks_used=3, preemptions=0, finished=0)
    _rec(m, step=1, num_running=5, tokens_scheduled=5, blocks_used=9, preemptions=2, finished=1)
    s = m.summary()
    assert s["peak_running"] == 5
    assert s["peak_blocks_used"] == 9
    assert s["num_blocks"] == 100
    assert s["total_preemptions"] == 2  # cumulative -> last value
    assert s["total_finished"] == 1


def test_write_jsonl_roundtrips(tmp_path):
    m = MetricsRecorder()
    _rec(m, step=0, vclock=0.01, tokens_scheduled=20)
    _rec(m, step=1, vclock=0.02, tokens_scheduled=1)
    path = tmp_path / "out.jsonl"
    m.write_jsonl(path)
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["step"] == 0
    assert lines[0]["tokens_scheduled"] == 20
    assert lines[1]["vclock"] == 0.02


def test_empty_summary_is_safe():
    s = MetricsRecorder().summary()
    assert s["num_steps"] == 0
    assert s["virtual_time_s"] == 0.0
