"""Dataset structure inspection for NIS 2021-2023 Core files.

Only the Core file is needed for this study: I10_DX1 supplies the label
(principal diagnosis) and I10_DX2..I10_DX25 supply the predictor set
(secondary-diagnosis profile). Run first (``python main.py --stage
inspect``) so all downstream ingestion is driven by the actual file
structure rather than assumptions.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..config import Config
from ..logging_utils import get_logger
from .csv_io import NIS_CSV_ENCODING, detect_delimiter, read_nis_csv


def _find(candidates: List[str], columns: List[str]) -> str | None:
    upper = {c.upper(): c for c in columns}
    for cand in candidates:
        if cand.upper() in upper:
            return upper[cand.upper()]
    return None


def detect_columns(cfg: Config, columns: List[str]) -> Dict:
    ing = cfg["ingestion"]
    dx_re = re.compile(ing["dx_regex"], re.IGNORECASE)
    all_dx_cols = sorted(
        (c for c in columns if dx_re.match(c)),
        key=lambda c: int(re.findall(r"\d+", c)[-1]),
    )
    max_sec = int(ing.get("max_secondary_dx", 25))
    principal_col = all_dx_cols[0] if all_dx_cols else None
    # positions 2..(max_sec+1) are the secondary-diagnosis predictor window
    secondary_cols = all_dx_cols[1: 1 + max_sec]
    return {
        "key": _find(ing["key_candidates"], columns),
        "hosp": _find(ing["hosp_candidates"], columns),
        "weight": _find(ing["weight_candidates"], columns),
        "all_dx_cols": all_dx_cols,
        "principal_col": principal_col,
        "secondary_cols": secondary_cols,
        "age": _find(["AGE"], columns),
        "female": _find(["FEMALE"], columns),
        "elective": _find(["ELECTIVE"], columns),
        "aweekend": _find(["AWEEKEND"], columns),
        "pay1": _find(["PAY1"], columns),
    }


def inspect_datasets(cfg: Config) -> pd.DataFrame:
    log = get_logger("h1.inspect")
    rows = []
    for year in cfg.years.values():
        path = cfg.nis_file(year, "core")
        if not path.exists():
            log.warning("Missing file: %s", path)
            rows.append({"year": year, "kind": "core", "path": str(path), "exists": False})
            continue
        head = read_nis_csv(path, nrows=5, low_memory=False)
        cols = list(head.columns)
        det = detect_columns(cfg, cols)
        size_gb = path.stat().st_size / 1e9
        detected_delim = detect_delimiter(path, encoding=NIS_CSV_ENCODING)
        log.info(
            "%s core: %d columns, %.2f GB, principal=%s, %d secondary DX cols "
            "(window<=%s), key=%s, wt=%s, delimiter=%r",
            year, len(cols), size_gb, det["principal_col"], len(det["secondary_cols"]),
            cfg["ingestion"].get("max_secondary_dx", 25), det["key"], det["weight"], detected_delim,
        )
        rows.append({
            "year": year, "kind": "core", "path": str(path), "exists": True,
            "size_gb": round(size_gb, 3), "n_columns": len(cols),
            "n_secondary_dx_columns": len(det["secondary_cols"]),
            "detected_delimiter": detected_delim,
            "key_column": det["key"], "weight_column": det["weight"],
            "principal_dx_column": det["principal_col"],
            "columns": ";".join(cols),
        })
    report = pd.DataFrame(rows)
    out = cfg.outputs("tables") / "dataset_inspection.csv"
    report.to_csv(out, index=False)
    log.info("Inspection report -> %s", out)

    missing = report[~report["exists"]]
    if len(missing):
        raise FileNotFoundError(
            f"Core NIS files missing:\n{missing[['year', 'path']].to_string(index=False)}"
        )
    zero_sec = report[report["n_secondary_dx_columns"] == 0]
    if len(zero_sec):
        raise ValueError(
            f"No secondary DX columns detected for year(s) {list(zero_sec['year'])}. "
            f"Check config.ingestion.dx_regex matches your file's actual column "
            f"names (expected I10_DX1..I10_DX40 for 2021-2023 NIS)."
        )
    return report
