"""Memory-aware worker sizing for the permutation controls.

Every permutation task ships the year's full secondary-diagnosis block
(millions of rows x 25 object-dtype string columns, ~2 GB pickled at
real-NIS volume) to a worker process, which then holds its own copy plus
a shuffled copy plus detect() temporaries (~6-8 GB peak). Sizing the pool
by CPU count alone (n_jobs=-1 on a 20-core box = 20 workers) therefore
exhausts RAM: the kernel OOM-killer takes the pipeline down and, because
it shares a user session, can take VS Code with it. The worker count is
instead capped by the memory actually available.
"""
from __future__ import annotations

import logging
import os
from typing import Sequence

import pandas as pd

log = logging.getLogger("h2.parallel")

# Peak per-worker footprint as a multiple of the deep in-memory size of the
# data shipped to it: its unpickled copy, the shuffled copy built by
# pd.concat, detect()'s normalised-string temporaries, and the pickled copy
# joblib keeps queued in the parent. Measured ~3.3x on NIS 2023; rounded up.
WORKER_MEMORY_FACTOR = 3.5
# Share of currently-available RAM the worker pool may plan to use. The
# rest is headroom for the parent process, the desktop and VS Code.
AVAILABLE_MEMORY_BUDGET = 0.5


def _available_memory_bytes() -> int | None:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def resolve_n_jobs(requested: int, df: pd.DataFrame, cols: Sequence[str]) -> int:
    """Turn a joblib-style `requested` (-1 = all cores) into a worker count
    that fits in memory. Never returns less than 1."""
    cpus = os.cpu_count() or 1
    requested = int(requested)
    if requested > 0:
        wanted = min(requested, cpus)
    else:  # joblib convention: -1 = all cores, -2 = all but one, ...
        wanted = max(1, cpus + 1 + requested)
    if wanted == 1:
        return 1

    available = _available_memory_bytes()
    if available is None:
        log.warning("Cannot read MemAvailable; not memory-capping n_jobs=%d", wanted)
        return wanted

    per_worker = WORKER_MEMORY_FACTOR * float(df[list(cols)].memory_usage(deep=True).sum())
    fits = max(1, int(available * AVAILABLE_MEMORY_BUDGET // per_worker))
    n_jobs = min(wanted, fits)
    log.info("n_jobs: requested=%s cpus=%d memory-safe=%d -> using %d "
             "(~%.1f GB/worker, %.1f GB available)",
             requested, cpus, fits, n_jobs, per_worker / 1e9, available / 1e9)
    return n_jobs
