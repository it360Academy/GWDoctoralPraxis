"""Stream NIS Core CSVs into a per-year parquet checkpoint for H2.

Extends H1's ingestion pattern (same column-detection, same chunked
streaming) with one addition H2 needs and H1 didn't: the exact,
normalized I10_DX1 CODE is preserved (not just collapsed to a phenotype
label), because DEV detection needs to test DX1's actual code against the
Excludes1 ontology, not just "which of the 15 phenotypes it belongs to."

Columns written:
  KEY, HOSP_NIS (if present)
  DX1_RAW: normalized I10_DX1 code (e.g. "I214")
  LABEL: phenotype name matched by DX1_RAW, or NaN if DX1 isn't one of
    the 15 phenotypes (H2's cohort, per the DX1-scope decision, is
    restricted to these 15 phenotypes as DX1 - same restriction as H1)
  NO_PHENOTYPE_DX1, AMBIGUOUS_DX1: same audit flags as H1
  SECDX_1..SECDX_<max_secondary_dx>: normalized DX2-25 codes
  AGE, FEMALE, ELECTIVE, AWEEKEND, PAY1, DISCWT, HOSP_REGION/TEACH/BEDSIZE (optional)
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..config import Config
from ..features.definitions import DISEASE_ORDER, DISEASES
from ..features.icd_classifier import CodeClassifier, normalize_dx
from ..logging_utils import get_logger
from .csv_io import read_nis_csv
from .hospital import HOSP_FEATURE_COLS, load_hospital_features
from .inspect import detect_columns


def checkpoint_path(cfg: Config, year: str) -> Path:
    return cfg.data_dir / f"raw_{year}.parquet"


def _phenotype_classifier() -> CodeClassifier:
    return CodeClassifier(
        {d: s["icd10_prefixes"] for d, s in DISEASES.items()},
        exclude={d: s.get("icd10_exclude", []) for d, s in DISEASES.items()},
    )


def _derive_label(principal: pd.DataFrame, classifier: CodeClassifier) -> pd.DataFrame:
    flags = classifier.flags_for_dx_frame(principal)
    n_hits = flags.sum(axis=1)
    label_idx = flags.argmax(axis=1)
    names = np.array(DISEASE_ORDER, dtype=object)
    label = np.where(n_hits == 1, names[label_idx], None)
    # flags_for_dx_frame returns a bare ndarray, so n_hits/label carry no
    # index - build this frame on principal's original (chunk-offset) index
    # rather than a fresh 0-based one, or the axis=1 concat in the caller
    # unions mismatched indices instead of aligning rows.
    return pd.DataFrame({
        "LABEL": label,
        "NO_PHENOTYPE_DX1": (n_hits == 0).astype(np.uint8),
        "AMBIGUOUS_DX1": (n_hits > 1).astype(np.uint8),
    }, index=principal.index)


def build_year_checkpoint(cfg: Config, year: str, force: bool = False) -> Path:
    log = get_logger("h2.ingest")
    out_path = checkpoint_path(cfg, year)
    if out_path.exists() and not force and not cfg.get("checkpointing", "overwrite", default=False):
        log.info("[%s] checkpoint exists, skipping (%s)", year, out_path.name)
        return out_path

    core = cfg.nis_file(year, "core")
    header = read_nis_csv(core, nrows=0, low_memory=False)
    det = detect_columns(cfg, list(header.columns))
    if det["principal_col"] is None:
        raise ValueError(f"No I10_DX1 (principal diagnosis) column detected in {core}.")
    if not det["secondary_cols"]:
        raise ValueError(f"No secondary DX columns detected in {core}.")

    max_sec = int(cfg["ingestion"].get("max_secondary_dx", 25))
    sec_cols = det["secondary_cols"]
    all_dx_cols = det["all_dx_cols"]
    usecols = [c for c in [det["key"], det["hosp"], det["principal_col"],
                            det["age"], det["female"], det["elective"],
                            det["aweekend"], det["pay1"], det["weight"]] if c] + all_dx_cols
    log.info("[%s] streaming %s | principal=%s | %d secondary DX cols | chunksize=%d",
              year, core.name, det["principal_col"], len(sec_cols), cfg["ingestion"]["chunksize"])

    classifier = _phenotype_classifier()
    parts: List[pd.DataFrame] = []
    total = 0
    reader = read_nis_csv(core, usecols=usecols, chunksize=cfg["ingestion"]["chunksize"],
                           dtype={c: "object" for c in [det["principal_col"]] + all_dx_cols},
                           low_memory=False)
    for i, chunk in enumerate(reader):
        principal = normalize_dx(chunk[det["principal_col"]])
        label_df = _derive_label(principal.to_frame("DX1"), classifier)

        secondary = chunk[sec_cols].apply(normalize_dx)
        secondary.columns = [f"SECDX_{j+1}" for j in range(len(sec_cols))]
        for j in range(len(sec_cols), max_sec):
            secondary[f"SECDX_{j+1}"] = pd.NA

        out = pd.DataFrame(index=chunk.index)
        out["DX1_RAW"] = principal.astype(object).where(principal.notna(), None)
        if det["key"]:
            out["KEY"] = pd.to_numeric(chunk[det["key"]], errors="coerce")
        if det["hosp"]:
            out["HOSP_NIS"] = pd.to_numeric(chunk[det["hosp"]], errors="coerce")
        for name, col in [("AGE", det["age"]), ("FEMALE", det["female"]),
                           ("ELECTIVE", det["elective"]), ("AWEEKEND", det["aweekend"]),
                           ("PAY1", det["pay1"])]:
            out[name] = pd.to_numeric(chunk[col], errors="coerce") if col else np.nan
        for c in ("AGE", "FEMALE", "ELECTIVE", "AWEEKEND", "PAY1"):
            out.loc[out[c] < 0, c] = np.nan
        out["DISCWT"] = (pd.to_numeric(chunk[det["weight"]], errors="coerce")
                          if det["weight"] else 1.0)
        out["DISCWT"] = out["DISCWT"].fillna(1.0).clip(lower=0)

        # coding-intensity covariate: how many of the (up to 40) DX slots
        # are actually filled - used both as a sensitivity variable and
        # as a DEV-risk predictor (never as part of DEV detection itself)
        all_dx_cols = det["all_dx_cols"]
        out["N_DX_FILLED"] = chunk[all_dx_cols].apply(normalize_dx).notna().sum(axis=1).astype(np.int16)

        part = pd.concat([out, label_df, secondary], axis=1)
        parts.append(part)
        total += len(chunk)
        if (i + 1) % 4 == 0:
            log.info("[%s] processed %s rows...", year, f"{total:,}")

    feats = pd.concat(parts, ignore_index=True)
    del parts
    feats["LABEL"] = feats["LABEL"].astype("category")

    # Hospital characteristics (optional; used for hospital-pattern
    # subgroup analysis - never a DEV-detection input)
    hosp = load_hospital_features(cfg, year)
    if hosp is not None and "HOSP_NIS" in feats.columns:
        before = len(feats)
        feats = feats.merge(hosp, on="HOSP_NIS", how="left")
        if len(feats) != before:
            raise RuntimeError(
                f"[{year}] HOSP_NIS merge with Hospital file changed row "
                f"count ({before:,} -> {len(feats):,}) - expected a clean "
                f"many-to-one join. Check for duplicate HOSP_NIS values."
            )
    else:
        for c in HOSP_FEATURE_COLS:
            feats[c] = np.nan

    feats.to_parquet(out_path, index=False)

    n_cohort = feats["LABEL"].notna().sum()
    log.info("[%s] checkpoint written: %s rows (%s in 15-phenotype DX1 cohort, "
              "%s no-phenotype, %s ambiguous) -> %s",
              year, f"{len(feats):,}", f"{n_cohort:,}",
              f"{int(feats['NO_PHENOTYPE_DX1'].sum()):,}",
              f"{int(feats['AMBIGUOUS_DX1'].sum()):,}", out_path)
    return out_path


def load_year(cfg: Config, year: str, columns: List[str] | None = None) -> pd.DataFrame:
    path = checkpoint_path(cfg, year)
    if not path.exists():
        build_year_checkpoint(cfg, year)
    return pd.read_parquet(path, columns=columns)
