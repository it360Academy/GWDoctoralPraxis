"""Structured logging + stage runtime tracking."""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List

_RUNTIMES: List[Dict] = []


def setup_logging(logs_dir: Path, run_name: str) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = logs_dir / f"{run_name}_{stamp}.log"

    logger = logging.getLogger()  # root logger, so any get_logger(name) in
                                   # any subsystem (h1.*, h2.*, ...) propagates
                                   # up to these handlers regardless of prefix
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in (logging.FileHandler(logfile, encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    logger.info("Logging to %s", logfile)
    return logger


def get_logger(name: str = "h1") -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def stage_timer(stage: str, logger: logging.Logger | None = None):
    """Context manager that logs and records the wall time of a pipeline stage."""
    logger = logger or get_logger()
    logger.info("=== STAGE START: %s ===", stage)
    t0 = time.perf_counter()
    status = "ok"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        elapsed = time.perf_counter() - t0
        _RUNTIMES.append({"stage": stage, "seconds": round(elapsed, 2),
                          "status": status,
                          "finished_at": datetime.now().isoformat(timespec="seconds")})
        logger.info("=== STAGE END: %s (%.1fs, %s) ===", stage, elapsed, status)


def dump_runtimes(outputs_tables_dir: Path) -> None:
    if not _RUNTIMES:
        return
    outputs_tables_dir.mkdir(parents=True, exist_ok=True)
    with open(outputs_tables_dir / "runtime_tracking.json", "w", encoding="utf-8") as fh:
        json.dump(_RUNTIMES, fh, indent=2)
