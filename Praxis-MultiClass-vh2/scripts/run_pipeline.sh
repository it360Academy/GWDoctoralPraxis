#!/usr/bin/env bash
# Run the H2 pipeline in its own memory-capped systemd scope.
#
# Why: launched from a VS Code terminal, the pipeline lives in VS Code's
# cgroup. If it exhausts RAM the kernel OOM-killer picks victims across the
# whole session and VS Code dies with it (and, being in VS Code's scope,
# a VS Code exit kills the pipeline too). In its own scope the pipeline
# survives VS Code closing, and if it ever does blow its budget only the
# pipeline is killed.
#
# Usage: scripts/run_pipeline.sh [main.py args...]   e.g. --stage negative_controls
# Detached run:  nohup scripts/run_pipeline.sh > detached_run_stdout.log 2>&1 &
# Other entry point (same memory cap): scripts/run_pipeline.sh -m <module> [args...]
#   e.g. -m src.negative_controls.investigate_2021_artifact --n-permutations 2000
set -euo pipefail
cd "$(dirname "$0")/.."

total_kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
high_gb=$(( total_kb * 70 / 100 / 1024 / 1024 ))   # throttle/reclaim above 70%
max_gb=$((  total_kb * 80 / 100 / 1024 / 1024 ))   # hard kill above 80%

entry=(main.py)
if [[ "${1:-}" == "-m" ]]; then
    entry=(-m "$2")
    shift 2
fi

echo "pipeline memory cap: high=${high_gb}G max=${max_gb}G (of $((total_kb / 1024 / 1024))G)"
exec systemd-run --user --scope --quiet \
    -p MemoryHigh="${high_gb}G" -p MemoryMax="${max_gb}G" -p MemorySwapMax=0 \
    .venv/bin/python -u "${entry[@]}" "$@"
