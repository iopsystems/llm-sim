"""Command-line entry point: run a scenario, dump per-step JSONL + a summary.

    python -m vllm_sim --workload synthetic --num-requests 50 --arrival-rate 20 ...
    python -m vllm_sim --workload trace --trace path/to/trace.csv ...
"""

import argparse
import json
import sys
from typing import List, Optional

from vllm_sim.cost.constant import ConstantCostModel
from vllm_sim.engine import SimLoop
from vllm_sim.harness.builder import build_scheduler
from vllm_sim.workload.factory import RequestFactory
from vllm_sim.workload.synthetic import SyntheticWorkload
from vllm_sim.workload.trace import TraceWorkload


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vllm_sim", description=__doc__)

    p.add_argument("--workload", choices=["synthetic", "trace"], default="synthetic")

    # Synthetic workload
    p.add_argument("--num-requests", type=int, default=50)
    p.add_argument("--arrival-rate", type=float, default=None,
                   help="Poisson arrival rate (reqs/sec); mutually exclusive with --interval")
    p.add_argument("--interval", type=float, default=None,
                   help="fixed inter-arrival seconds; mutually exclusive with --arrival-rate")
    p.add_argument("--prompt-len", type=int, nargs=2, metavar=("LO", "HI"), default=[8, 256])
    p.add_argument("--output-len", type=int, nargs=2, metavar=("LO", "HI"), default=[4, 64])
    p.add_argument("--seed", type=int, default=0)

    # Trace workload
    p.add_argument("--trace", type=str, default=None, help="path to CSV/JSONL trace")

    # Scheduler config
    p.add_argument("--model", type=str, default="facebook/opt-125m")
    p.add_argument("--num-blocks", type=int, default=10000)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--no-chunked-prefill", action="store_true")

    # Cost model
    p.add_argument("--latency", type=float, default=0.01, help="constant per-step latency (s)")

    # Output
    p.add_argument("--jsonl", type=str, default=None, help="write per-step records here")
    p.add_argument("--summary", type=str, default=None, help="write summary JSON here")
    p.add_argument("--viz", action="store_true",
                   help="render a terminal dashboard of the run to stderr")

    return p


def _build_workload(args):
    if args.workload == "trace":
        if not args.trace:
            raise SystemExit("--trace is required for --workload trace")
        return TraceWorkload(args.trace)
    # synthetic
    arrival_rate, interval = args.arrival_rate, args.interval
    if arrival_rate is None and interval is None:
        interval = 0.0  # default: all arrive at t=0 (saturated)
    return SyntheticWorkload(
        num_requests=args.num_requests,
        prompt_len=tuple(args.prompt_len),
        output_len=tuple(args.output_len),
        arrival_rate=arrival_rate,
        interval=interval,
        seed=args.seed,
    )


def main(argv: Optional[List[str]] = None) -> dict:
    args = _build_parser().parse_args(argv)

    specs = list(_build_workload(args).generate())
    scheduler = build_scheduler(
        model=args.model,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        enable_chunked_prefill=not args.no_chunked_prefill,
    )
    factory = RequestFactory(block_size=args.block_size)
    loop = SimLoop(
        scheduler=scheduler,
        specs=specs,
        factory=factory,
        cost_model=ConstantCostModel(latency_s=args.latency),
    )
    metrics = loop.run()
    summary = metrics.summary()
    summary["num_requests"] = len(specs)

    if args.jsonl:
        metrics.write_jsonl(args.jsonl)
    if args.summary:
        with open(args.summary, "w") as f:
            json.dump(summary, f, indent=2)

    if args.viz:
        from vllm_sim.viz import render
        print(render(metrics.records), file=sys.stderr)

    # Human-readable summary to stderr so stdout stays clean for piping.
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return summary


if __name__ == "__main__":
    main()
