#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
python_bin="${PYTHON_BIN:-python3}"

if (( $# == 0 )); then
    echo "Usage: scripts/run_linux.sh --experiments Exp1_J [Exp2_L ...] [run_sweeps options]" >&2
    exit 2
fi

exec "$python_bin" -X utf8 run_sweeps.py "$@"
