from llm_sim.export.histogram import LogLinearHistogram


def test_default_config_has_496_buckets():
    # Config::new(3, 64): lower=16, upper=(64-4)*8=480 -> 496
    h = LogLinearHistogram()
    assert h.grouping_power == 3
    assert h.max_value_power == 64
    assert h.num_buckets == 496


def test_linear_region_index_equals_value():
    h = LogLinearHistogram()
    for v in [0, 1, 5, 15]:
        assert h.value_to_index(v) == v


def test_log_region_indices_match_rust_formula():
    h = LogLinearHistogram()
    # hand-computed from config.rs value_to_index (grouping_power=3):
    expected = {16: 16, 17: 16, 18: 17, 24: 20, 31: 23, 32: 24}
    for value, idx in expected.items():
        assert h.value_to_index(value) == idx, f"value {value}"


def test_index_is_monotonic_nondecreasing():
    h = LogLinearHistogram()
    prev = -1
    for v in range(0, 5000):
        idx = h.value_to_index(v)
        assert idx >= prev
        assert 0 <= idx < h.num_buckets
        prev = idx


def test_record_increments_the_right_bucket():
    h = LogLinearHistogram()
    h.record(5)
    h.record(5)
    h.record(32)
    snap = h.snapshot()
    assert snap[5] == 2
    assert snap[24] == 1
    assert len(snap) == 496
    assert sum(snap) == 3


def test_snapshot_is_a_copy():
    h = LogLinearHistogram()
    h.record(1)
    snap = h.snapshot()
    h.record(1)
    assert snap[1] == 1  # earlier snapshot not mutated by later record


def test_record_with_count():
    h = LogLinearHistogram()
    h.record(10, count=4)
    assert h.snapshot()[10] == 4
