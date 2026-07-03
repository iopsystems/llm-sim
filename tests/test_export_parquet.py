"""Validates the emitted Parquet against the exact schema contract the Rezolus
viewer (metriken-query) enforces. Needs pyarrow (the [rezolus] extra)."""

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("h2histogram")

from llm_sim.export.rezolus import write_parquet


def _records(n=40):
    recs = []
    for i in range(n):
        recs.append(dict(
            step=i, vclock=round(0.01 * (i + 1), 2),
            num_running=(i % 8) + 1, num_waiting=0,
            prefill_reqs=1, decode_reqs=i % 4,
            tokens_scheduled=20 if i % 5 == 0 else 1,
            blocks_used=i + 1, num_blocks=500,
            preemptions=0, finished=i,
        ))
    return recs


@pytest.fixture
def written(tmp_path):
    path = tmp_path / "run.parquet"
    manifest = write_parquet(_records(), str(path), interval_ns=50_000_000)
    return path, manifest


def test_timestamp_column_is_uint64_nanoseconds(written):
    path, _ = written
    schema = pq.read_schema(path)
    f = schema.field("timestamp")
    assert f.type == pa.uint64()
    assert not f.nullable


def test_histogram_columns_are_list_uint64(written):
    path, _ = written
    schema = pq.read_schema(path)
    for m in ["tokens_scheduled", "num_running", "blocks_used", "num_waiting"]:
        f = schema.field(f"{m}:buckets")
        assert pa.types.is_list(f.type)
        assert f.type.value_type == pa.uint64()


def test_histogram_columns_carry_grouping_metadata(written):
    # The viewer SILENTLY DROPS a histogram column lacking these -> mandatory.
    path, _ = written
    schema = pq.read_schema(path)
    meta = schema.field("num_running:buckets").metadata
    assert meta[b"grouping_power"] == b"3"
    assert meta[b"max_value_power"] == b"64"


def test_file_metadata_has_integer_sampling_interval(written):
    path, _ = written
    meta = pq.read_schema(path).metadata
    assert meta[b"sampling_interval_ms"] == b"50"  # 50_000_000 ns
    assert int(meta[b"sampling_interval_ms"])       # parses as int (else viewer panics)


def test_each_bucket_row_has_496_entries(written):
    path, manifest = written
    assert manifest["num_buckets"] == 496
    t = pq.read_table(path)
    col = t.column("num_running:buckets").to_pylist()
    assert all(len(row) == 496 for row in col)


def test_buckets_are_cumulative_monotonic(written):
    path, _ = written
    t = pq.read_table(path)
    rows = t.column("num_running:buckets").to_pylist()
    for a, b in zip(rows, rows[1:]):
        assert all(x <= y for x, y in zip(a, b))  # free-running cumulative


def test_timestamps_strictly_increasing_by_interval(written):
    path, _ = written
    t = pq.read_table(path)
    ts = t.column("timestamp").to_pylist()
    assert ts == sorted(ts)
    diffs = {b - a for a, b in zip(ts, ts[1:])}
    assert diffs == {50_000_000}  # uniform spacing == sampling interval
