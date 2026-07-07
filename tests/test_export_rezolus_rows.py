import pytest

pytest.importorskip("h2histogram")  # export stage dep ([rezolus] extra)

from llm_sim.export.histogram import LogLinearHistogram
from llm_sim.export.rezolus import build_histogram_rows, choose_interval_ns

SEC = 1_000_000_000
IDX = LogLinearHistogram().value_to_index  # value -> bucket index


def _recs(vclocks, running):
    return [{"vclock": v, "num_running": r} for v, r in zip(vclocks, running)]


def test_single_interval_accumulates_all_steps():
    recs = _recs([0.0, 0.01, 0.02, 0.03], [1, 2, 3, 4])
    ts, cols = build_histogram_rows(recs, ["num_running"], interval_ns=SEC)
    assert ts == [0]                      # one interval, base 0
    row = cols["num_running"][0]
    assert len(row) == 496
    assert row[1] == 1 and row[2] == 1 and row[3] == 1 and row[4] == 1
    assert sum(row) == 4


def test_multiple_intervals_are_cumulative():
    # interval 15ms; steps at 0,10,20,30 ms -> buckets 0,0,1,2
    recs = _recs([0.0, 0.010, 0.020, 0.030], [1, 1, 2, 3])
    ts, cols = build_histogram_rows(recs, ["num_running"], interval_ns=15_000_000)
    assert ts == [0, 15_000_000, 30_000_000]
    rows = cols["num_running"]
    assert sum(rows[0]) == 2            # two steps in interval 0
    assert sum(rows[1]) == 3            # + one step
    assert sum(rows[2]) == 4            # + one step
    # cumulative: every bucket non-decreasing across rows
    for b in range(496):
        assert rows[0][b] <= rows[1][b] <= rows[2][b]


def test_empty_interval_repeats_previous_cumulative():
    # steps at 0ms and 40ms, interval 15ms -> buckets 0 and 2; interval 1 empty
    recs = _recs([0.0, 0.040], [5, 6])
    ts, cols = build_histogram_rows(recs, ["num_running"], interval_ns=15_000_000)
    rows = cols["num_running"]
    assert ts == [0, 15_000_000, 30_000_000]
    assert rows[1] == rows[0]           # empty interval -> unchanged cumulative
    assert sum(rows[2]) == 2


def test_multiple_metrics_tracked_independently():
    recs = [
        {"vclock": 0.0, "num_running": 1, "tokens_scheduled": 20},
        {"vclock": 0.01, "num_running": 2, "tokens_scheduled": 1},
    ]
    ts, cols = build_histogram_rows(recs, ["num_running", "tokens_scheduled"], interval_ns=SEC)
    assert set(cols) == {"num_running", "tokens_scheduled"}
    assert cols["num_running"][0][1] == 1 and cols["num_running"][0][2] == 1
    # value 20 is in the log-linear region -> bucket index 18, not 20.
    assert cols["tokens_scheduled"][0][IDX(20)] == 1
    assert sum(cols["tokens_scheduled"][0]) == 2


def test_empty_records():
    ts, cols = build_histogram_rows([], ["num_running"], interval_ns=SEC)
    assert ts == []
    assert cols == {"num_running": []}


def test_choose_interval_targets_row_count():
    recs = _recs([i * 0.01 for i in range(100)], [1] * 100)  # spans ~1s
    interval = choose_interval_ns(recs, target_rows=10)
    assert interval > 0
    ts, _ = build_histogram_rows(recs, ["num_running"], interval_ns=interval)
    # roughly the requested resolution, never zero rows
    assert 5 <= len(ts) <= 20
