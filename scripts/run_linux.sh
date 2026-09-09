#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
# Each sweep already parallelizes at the process level.  Keep numerical
# libraries from creating a second full thread pool inside every worker.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
python_bin="${PYTHON_BIN:-python3}"

if (( $# == 0 )); then
    echo "Usage: scripts/run_linux.sh --experiments Exp1_J [Exp2_L ...] [run_sweeps options]" >&2
    exit 2
fi

exec "$python_bin" -X utf8 run_sweeps.py "$@"
