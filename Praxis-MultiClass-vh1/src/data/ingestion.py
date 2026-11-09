"""Stream NIS Core CSVs into a per-year parquet checkpoint holding:

  * KEY / HOSP_NIS identifiers (if present)
  * LABEL: the single phenotype name matched by I10_DX1 (or NaN if I10_DX1
    doesn't match any of the 15 phenotypes / matches more than one - see
    `_derive_label`), plus AMBIGUOUS_DX1 / NO_PHENOTYPE_DX1 flags so cohort
    exclusions are auditable rather than silent.
  * SECDX_1..SECDX_<max_secondary_dx>: normalized ICD-10-CM codes from
    I10_DX2..I10_DX(1+max_secondary_dx) (the secondary-diagnosis profile
    that is H1's predictor set), one column per position, raw (not yet
    masked or encoded - that happens in src/features/dx_features.py, which
    needs the training year's full code distribution before it can decide
    what to mask and what vocabulary to fit).
  * AGE, FEMALE, ELECTIVE, AWEEKEND, PAY1, DISCWT: optional covariates /
    survey weight.

This checkpoint intentionally keeps raw secondary-DX codes (not yet
one-hot encoded) because the feature vocabulary must be fit on the
TRAINING year only and then applied unchanged to validation/test - that
step needs to see the actual code strings, not a pre-encoded matrix sized
for a different year's vocabulary.
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
from .inspect import detect_columns


def checkpoint_path(cfg: Config, year: str) -> Path:
    return cfg.data_dir / f"raw_{year}.parquet"


def _phenotype_classifier() -> CodeClassifier:
    return CodeClassifier(
        {d: s["icd10_prefixes"] for d, s in DISEASES.items()},
        exclude={d: s.get("icd10_exclude", []) for d, s in DISEASES.items()},
    )


def _derive_label(principal: pd.DataFrame, classifier: CodeClassifier) -> pd.DataFrame:
    """One-hot phenotype flags for I10_DX1, collapsed into a single LABEL
    column. A discharge whose principal diagnosis matches zero phenotypes
    is out of the multiclass cohort (LABEL=NaN, NO_PHENOTYPE_DX1=1). A
    discharge matching more than one phenotype is dropped as ambiguous
    (LABEL=NaN, AMBIGUOUS_DX1=1) rather than silently assigned one - this
    should be rare/zero given the disease definitions are meant to be
    mutually exclusive code ranges, and the count is reported so that
    assumption is checked, not just believed.
    """
    flags = classifier.flags_for_dx_frame(principal)  # (n, 15) uint8
    n_hits = flags.sum(axis=1)
    label_idx = flags.argmax(axis=1)
    names = np.array(DISEASE_ORDER, dtype=object)
    label = np.where(n_hits == 1, names[label_idx], None)
    return pd.DataFrame({
        "LABEL": label,
        "NO_PHENOTYPE_DX1": (n_hits == 0).astype(np.uint8),
        "AMBIGUOUS_DX1": (n_hits > 1).astype(np.uint8),
    }, index=principal.index)


def build_year_checkpoint(cfg: Config, year: str, force: bool = False) -> Path:
    log = get_logger("h1.ingest")
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
    usecols = [c for c in [det["key"], det["hosp"], det["principal_col"],
                            det["age"], det["female"], det["elective"],
                            det["aweekend"], det["pay1"], det["weight"]] if c] + sec_cols
    log.info("[%s] streaming %s | principal=%s | %d secondary DX cols | chunksize=%d",
              year, core.name, det["principal_col"], len(sec_cols), cfg["ingestion"]["chunksize"])

    classifier = _phenotype_classifier()
    parts: List[pd.DataFrame] = []
    total = 0
    reader = read_nis_csv(core, usecols=usecols, chunksize=cfg["ingestion"]["chunksize"],
                           dtype={c: "object" for c in [det["principal_col"]] + sec_cols},
                           low_memory=False)
    for i, chunk in enumerate(reader):
        principal = normalize_dx(chunk[det["principal_col"]]).to_frame("DX1")
        label_df = _derive_label(principal, classifier)

        secondary = chunk[sec_cols].apply(normalize_dx)
        secondary.columns = [f"SECDX_{j+1}" for j in range(len(sec_cols))]
        # pad up to max_sec columns if the source file has fewer than expected
        for j in range(len(sec_cols), max_sec):
            secondary[f"SECDX_{j+1}"] = pd.NA

        out = pd.DataFrame(index=chunk.index)
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

        part = pd.concat([out, label_df, secondary], axis=1)
        parts.append(part)
        total += len(chunk)
        if (i + 1) % 4 == 0:
            log.info("[%s] processed %s rows...", year, f"{total:,}")

    feats = pd.concat(parts, ignore_index=True)
    del parts
    feats["LABEL"] = feats["LABEL"].astype("category")
    feats.to_parquet(out_path, index=False)

    n_cohort = feats["LABEL"].notna().sum()
    log.info("[%s] checkpoint written: %s rows (%s in 15-phenotype cohort, "
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
