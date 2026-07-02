import json

from vllm_sim.viz import _downsample, _sparkline, main, render

BLOCKS = "▁▂▃▄▅▆▇█"


# ---- _sparkline -----------------------------------------------------------

def test_sparkline_empty_is_empty_string():
    assert _sparkline([]) == ""


def test_sparkline_full_range_monotonic_maps_all_levels():
    assert _sparkline([0, 1, 2, 3, 4, 5, 6, 7]) == BLOCKS


def test_sparkline_constant_series_is_flat_lowest():
    # A constant series has no variation -> render as the lowest level (baseline).
    assert _sparkline([5, 5, 5]) == BLOCKS[0] * 3


def test_sparkline_single_value_is_one_lowest_block():
    assert _sparkline([42]) == BLOCKS[0]


def test_sparkline_min_maps_low_max_maps_high():
    s = _sparkline([10, 20])
    assert s[0] == BLOCKS[0]
    assert s[-1] == BLOCKS[-1]


def test_sparkline_length_matches_input():
    assert len(_sparkline([0, 1, 2, 3, 4])) == 5


# ---- _downsample ----------------------------------------------------------

def test_downsample_identity_when_width_ge_length():
    assert _downsample([1, 2, 3], 5) == [1, 2, 3]
    assert _downsample([1, 2, 3], 3) == [1, 2, 3]


def test_downsample_reduces_to_width():
    assert len(_downsample(list(range(100)), 10)) == 10


def test_downsample_is_peak_preserving():
    # Default agg=max: a spike must survive downsampling.
    values = [0] * 50 + [99] + [0] * 49
    out = _downsample(values, 10)
    assert max(out) == 99


def test_downsample_empty():
    assert _downsample([], 10) == []


# ---- render ---------------------------------------------------------------

def _records(n=5):
    recs = []
    for i in range(n):
        recs.append(
            dict(
                step=i, vclock=round(0.01 * (i + 1), 2),
                num_running=i, num_waiting=0,
                prefill_reqs=1 if i == 0 else 0,
                decode_reqs=0 if i == 0 else i,
                tokens_scheduled=20 if i == 0 else 1,
                blocks_used=i + 1, num_blocks=100,
                preemptions=0, finished=i,
            )
        )
    return recs


def test_render_returns_string_with_header_and_series():
    out = render(_records(), width=20)
    assert isinstance(out, str)
    # Header mentions step count and virtual time.
    assert "5 steps" in out
    # One labeled line per faithful KPI series.
    for label in ["batch tokens", "prefill", "decode", "running", "waiting",
                  "KV blocks", "finished", "preemptions"]:
        assert label in out


def test_render_shows_block_budget_and_finished_totals():
    out = render(_records(6), width=20)
    assert "100" in out          # num_blocks budget shown
    assert "finished" in out


def test_render_empty_records_is_graceful():
    out = render([], width=20)
    assert isinstance(out, str)
    assert "0 steps" in out


def test_render_accepts_steprecord_objects():
    # render must accept both dicts and StepRecord dataclass instances.
    from vllm_sim.metrics import MetricsRecorder
    m = MetricsRecorder()
    for r in _records(4):
        m.record(**r)
    out = render(m.records, width=20)
    assert "4 steps" in out


# ---- standalone CLI (python -m vllm_sim.viz steps.jsonl) -------------------

def test_main_reads_jsonl_and_prints_dashboard(tmp_path, capsys):
    path = tmp_path / "steps.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _records(5)) + "\n")
    out = main([str(path), "--width", "20"])
    printed = capsys.readouterr().out
    assert "5 steps" in printed
    assert "batch tokens" in printed
    assert out == printed.rstrip("\n")
