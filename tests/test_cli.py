import json

from llm_sim.cli import main


def test_synthetic_run_writes_jsonl_and_summary(tmp_path):
    jsonl = tmp_path / "steps.jsonl"
    summary_path = tmp_path / "summary.json"
    summary = main(
        [
            "--workload", "synthetic",
            "--num-requests", "6",
            "--interval", "0.0",
            "--prompt-len", "16", "16",
            "--output-len", "3", "3",
            "--seed", "1",
            "--num-blocks", "200",
            "--latency", "0.01",
            "--jsonl", str(jsonl),
            "--summary", str(summary_path),
        ]
    )
    # All requests finish.
    assert summary["total_finished"] == 6
    # Batch sizes evolve: at least one step has more than one token scheduled.
    assert summary["peak_batch_tokens"] >= 16
    # JSONL has one line per step.
    lines = jsonl.read_text().splitlines()
    assert len(lines) == summary["num_steps"]
    assert json.loads(lines[0])["step"] == 0
    # Summary file written and matches returned summary.
    assert json.loads(summary_path.read_text())["total_finished"] == 6
    # Blocks never exceed the budget.
    assert all(json.loads(l)["blocks_used"] <= json.loads(l)["num_blocks"] for l in lines)


def test_viz_flag_renders_dashboard_to_stderr(tmp_path, capsys):
    summary = main(
        [
            "--workload", "trace", "--trace", str(_two_req_trace(tmp_path)),
            "--num-blocks", "100", "--latency", "0.01", "--viz",
        ]
    )
    err = capsys.readouterr().err
    assert summary["total_finished"] == 2
    # The dashboard is rendered to stderr (stdout stays clean).
    assert "steps, virtual time" in err
    assert "batch tokens" in err


def _two_req_trace(tmp_path):
    trace = tmp_path / "trace.csv"
    trace.write_text(
        "request_id,arrival_time,prompt_len,output_len\n"
        "a,0.0,16,2\n"
        "b,0.0,16,2\n"
    )
    return trace


def test_trace_run_from_csv(tmp_path):
    trace = tmp_path / "trace.csv"
    trace.write_text(
        "request_id,arrival_time,prompt_len,output_len\n"
        "a,0.0,16,2\n"
        "b,0.0,16,2\n"
    )
    summary = main(
        [
            "--workload", "trace",
            "--trace", str(trace),
            "--num-blocks", "100",
            "--latency", "0.01",
            "--jsonl", str(tmp_path / "s.jsonl"),
        ]
    )
    assert summary["total_finished"] == 2
    assert summary["num_steps"] == 2
