"""Loader for the Hospital file - source of hospital-level characteristics
(region, teaching status, bed size) used ONLY by the sensitivity feature set
(config: feature_sets.hospital). The official H1 feature set does not use
these (see README methodology section).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger
from .csv_io import read_nis_csv
from .inspect import detect_columns

HOSP_FEATURE_COLS = ["HOSP_REGION", "HOSP_TEACH", "HOSP_BEDSIZE"]


def load_hospital_features(cfg: Config, year: str) -> Optional[pd.DataFrame]:
    log = get_logger("h1.hospital")
    path = cfg.nis_file(year, "hospital")
    if not path.exists():
        log.warning("[%s] Hospital file not found at %s", year, path)
        return None

    header = read_nis_csv(path, nrows=0, low_memory=False)
    det = detect_columns(cfg, list(header.columns))
    hosp_key = det["hosp"]
    if hosp_key is None:
        log.warning("[%s] Hospital file has no detectable HOSP_NIS column; "
                    "cannot merge hospital characteristics", year)
        return None

    region_col = det["hosp_region"]
    teach_col = det["hosp_teach"]
    bedsize_col = det["hosp_bedsize"]
    usecols = [c for c in [hosp_key, region_col, teach_col, bedsize_col] if c]
    df = read_nis_csv(path, usecols=usecols, low_memory=False)
    rename = {hosp_key: "HOSP_NIS"}
    if region_col:
        rename[region_col] = "HOSP_REGION"
    if teach_col:
        rename[teach_col] = "HOSP_TEACH"
    if bedsize_col:
        rename[bedsize_col] = "HOSP_BEDSIZE"
    df = df.rename(columns=rename)
    df["HOSP_NIS"] = pd.to_numeric(df["HOSP_NIS"], errors="coerce")

    for c in HOSP_FEATURE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    keep = ["HOSP_NIS"] + HOSP_FEATURE_COLS
    df = df[keep].drop_duplicates("HOSP_NIS")
    log.info("[%s] loaded hospital characteristics for %s hospitals",
             year, f"{len(df):,}")
    return df
