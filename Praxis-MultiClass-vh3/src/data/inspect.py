"""Dataset structure inspection for the H3 replication. Only I10_DX1 and
I10_DX2 are used for phenotype flags (see ingestion.py), but detection
scans the full DX window so a schema change is caught rather than silently
truncating, and the remaining positions feed the coding-intensity EDA."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..config import Config
from ..logging_utils import get_logger
from .csv_io import read_nis_csv


def _find(candidates: List[str], columns: List[str]) -> str | None:
    upper = {c.upper(): c for c in columns}
    for cand in candidates:
        if cand.upper() in upper:
            return upper[cand.upper()]
    return None


def detect_columns(cfg: Config, columns: List[str]) -> Dict:
    ing = cfg["ingestion"]
    dx_re = re.compile(ing["dx_regex"], re.IGNORECASE)
    dx_cols = sorted(
        (c for c in columns if dx_re.match(c)),
        key=lambda c: int(re.findall(r"\d+", c)[-1]),
    )
    return {
        "key": _find(ing["key_candidates"], columns),
        "weight": _find(ing["weight_candidates"], columns),
        "dx_cols": dx_cols,
    }


def inspect_datasets(cfg: Config) -> pd.DataFrame:
    log = get_logger("h3.inspect")
    rows = []
    for role, year in cfg.years.items():
        path = cfg.nis_file(year, "core")
        if not path.exists():
            log.warning("Missing file: %s", path)
            rows.append({"role": role, "year": year, "path": str(path), "exists": False})
            continue
        head = read_nis_csv(path, nrows=5, low_memory=False)
        cols = list(head.columns)
        det = detect_columns(cfg, cols)
        size_gb = path.stat().st_size / 1e9
        log.info("%s (%s) core: %d columns, %.2f GB, %d DX columns detected "
                  "(DX1=%s, DX2=%s), key=%s, wt=%s",
                  role, year, len(cols), size_gb, len(det["dx_cols"]),
                  det["dx_cols"][0] if det["dx_cols"] else None,
                  det["dx_cols"][1] if len(det["dx_cols"]) > 1 else None,
                  det["key"], det["weight"])
        rows.append({
            "role": role, "year": year, "path": str(path), "exists": True,
            "size_gb": round(size_gb, 3), "n_columns": len(cols),
            "n_dx_columns": len(det["dx_cols"]),
            "key_column": det["key"], "weight_column": det["weight"],
        })
    report = pd.DataFrame(rows)
    out = cfg.outputs("tables") / "dataset_inspection.csv"
    report.to_csv(out, index=False)
    log.info("Inspection report -> %s", out)

    missing = report[~report["exists"]]
    if len(missing):
        raise FileNotFoundError(
            f"Core NIS files missing:\n{missing[['role', 'year', 'path']].to_string(index=False)}"
        )
    short = report[report["n_dx_columns"] < 2]
    if len(short):
        raise ValueError(
            f"Fewer than 2 DX columns detected for role(s) {list(short['role'])}. "
            f"H3 requires at least I10_DX1 and I10_DX2."
        )
    return report
