#!/usr/bin/env bash
#
# simulate.sh — convenience wrapper for the vLLM scheduler-in-isolation simulator.
#
# Ensures the project virtualenv exists (creating + installing on first run),
# then forwards all arguments to the `vllm_sim` CLI. Run it from anywhere.
#
#   ./simulate.sh --workload synthetic --num-requests 30 --arrival-rate 50 \
#       --prompt-len 32 128 --output-len 8 32 --num-blocks 500 --jsonl steps.jsonl
#   ./simulate.sh --workload trace --trace mytrace.csv --num-blocks 500
#   ./simulate.sh --help
#
# Named shortcuts (forward extra flags after the name to override defaults):
#   ./simulate.sh demo       # synthetic, batches evolving over virtual time
#   ./simulate.sh preempt    # tight block budget -> forces a preemption
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"

log() { printf '[simulate] %s\n' "$*" >&2; }

# --- ensure the venv exists and the package is importable -------------------
if [[ ! -x "$PY" ]]; then
  log "creating virtualenv at .venv ..."
  python3 -m venv "$VENV"
fi

if ! "$PY" -c "import vllm_sim" >/dev/null 2>&1; then
  log "installing vllm-sim (pulls vllm==0.23.0; first run is slow) ..."
  "$PY" -m pip install -q --upgrade pip
  ( cd "$ROOT" && "$PY" -m pip install -e ".[dev]" )
fi

# --- no args: print a quickstart instead of launching a large default run ---
if [[ $# -eq 0 ]]; then
  cat >&2 <<'USAGE'
Usage: ./simulate.sh [vllm_sim args...]
       ./simulate.sh demo       # synthetic demo: batches evolving over time
       ./simulate.sh preempt    # tight-budget demo: forces a preemption
       ./simulate.sh --help     # full flag list

Example:
  ./simulate.sh --workload synthetic --num-requests 30 --arrival-rate 50 \
      --prompt-len 32 128 --output-len 8 32 --num-blocks 500 --jsonl steps.jsonl
USAGE
  exit 0
fi

# --- named shortcuts --------------------------------------------------------
case "${1:-}" in
  demo)
    shift
    exec "$PY" -m vllm_sim --workload synthetic \
      --num-requests 30 --arrival-rate 50 \
      --prompt-len 32 128 --output-len 8 32 --seed 7 \
      --num-blocks 500 --latency 0.01 "$@"
    ;;
  preempt)
    shift
    exec "$PY" -m vllm_sim --workload synthetic \
      --num-requests 4 --interval 0.0 \
      --prompt-len 16 16 --output-len 16 16 \
      --num-blocks 8 --max-num-seqs 64 --max-model-len 4096 --latency 0.01 "$@"
    ;;
  *)
    exec "$PY" -m vllm_sim "$@"
    ;;
esac
