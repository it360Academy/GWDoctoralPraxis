"""NIS Core-file ingestion matching Newman-Toker et al. (2024, BMJ Qual
Saf) Table 1's disease-incidence definition: a discharge counts toward a
disease's incidence if that disease appears in DX1 (principal) OR DX2
(first-listed secondary) - NOT any-position (as in H2) and NOT
principal-only. Paper p.112: "counting discharge or in-hospital death
diagnoses coded in either the principal or first-listed secondary
diagnosis positions, as these diagnoses are often of equal, competing
weight" - chosen because second-position codes "can increase sensitivity
without sacrificing specificity" and secondary diseases are also incident
cases that can be misdiagnosed. This module also computes the DX1-only
variant, used only for H3's criterion 8 sensitivity analysis (Chapter 3,
Section 3.6.3), as an actual second computation rather than a cited
approximation. The remaining DX columns are read only to count filled
diagnosis positions (coding-intensity EDA), never for phenotype flags.

Only the ten vascular/infectious phenotypes are computed here. The five
oncologic phenotypes use published cancer-registry incidence
(literature_inputs.py), since most cancer care is outpatient and NIS
inpatient discharges would badly undercount true cancer incidence - the
same limitation the source paper itself works around by using registry
data for its five cancers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..config import Config
from ..features.definitions import DISEASES
from ..features.icd_classifier import CodeClassifier, normalize_dx
from ..logging_utils import get_logger
from .csv_io import read_nis_csv
from .inspect import detect_columns

VASC_INFECT_DISEASES: List[str] = [
    "stroke", "vte", "arterial_te", "aortic_aneurysm", "mi",
    "sepsis", "pneumonia", "meningitis_encephalitis", "spinal_abscess",
    "endocarditis",
]
ANY_COLS: List[str] = [f"PRIN_OR_SEC1_{d}" for d in VASC_INFECT_DISEASES]
PRIN_ONLY_COLS: List[str] = [f"PRIN_ONLY_{d}" for d in VASC_INFECT_DISEASES]


def checkpoint_path(cfg: Config, year: str) -> Path:
    return cfg.data_dir / f"bigthree_features_{year}.parquet"


def definitions_fingerprint() -> str:
    """Hash of the code lists that build the phenotype flags. Stored next to
    each checkpoint so a checkpoint built from older definitions is rebuilt
    rather than silently reused."""
    spec = {d: [DISEASES[d]["icd10_prefixes"], DISEASES[d].get("icd10_exclude", [])]
            for d in VASC_INFECT_DISEASES}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def _fingerprint_path(cfg: Config, year: str) -> Path:
    return cfg.data_dir / f"bigthree_features_{year}.definitions.json"


def checkpoint_is_current(cfg: Config, year: str) -> bool:
    fp = _fingerprint_path(cfg, year)
    if not (checkpoint_path(cfg, year).exists() and fp.exists()):
        return False
    return json.loads(fp.read_text()).get("fingerprint") == definitions_fingerprint()


def _classifier() -> CodeClassifier:
    return CodeClassifier(
        {d: DISEASES[d]["icd10_prefixes"] for d in VASC_INFECT_DISEASES},
        exclude={d: DISEASES[d].get("icd10_exclude", []) for d in VASC_INFECT_DISEASES},
    )


def build_year_checkpoint(cfg: Config, year: str, force: bool = False) -> Path:
    log = get_logger("h3.ingest")
    out_path = checkpoint_path(cfg, year)
    overwrite = force or cfg.get("checkpointing", "overwrite", default=False)
    if out_path.exists() and not overwrite and checkpoint_is_current(cfg, year):
        log.info("[%s] checkpoint exists, skipping (%s)", year, out_path.name)
        return out_path

    core = cfg.nis_file(year, "core")
    header = read_nis_csv(core, nrows=0, low_memory=False)
    det = detect_columns(cfg, list(header.columns))
    dx_cols = det["dx_cols"]
    if len(dx_cols) < 2:
        raise ValueError(f"{core.name}: need at least DX1 and DX2 columns")
    dx1_2 = dx_cols[:2]  # principal + first-listed secondary only (incidence definition)

    header_upper = {c.upper(): c for c in header.columns}
    covariate_map = {name: header_upper.get(name) for name in
                      ("AGE", "FEMALE", "ELECTIVE", "AWEEKEND", "PAY1")}
    covariate_cols = [c for c in covariate_map.values() if c]

    usecols = ([c for c in [det["key"], det["weight"]] if c] + dx1_2
               + dx_cols + covariate_cols)
    usecols = list(dict.fromkeys(usecols))  # de-dupe, preserve order
    log.info("[%s] streaming %s | DX1+DX2 for incidence, %d DX cols for coding "
              "intensity, %d covariates | chunksize=%d",
              year, core.name, len(dx_cols), len(covariate_cols), cfg["ingestion"]["chunksize"])

    classifier = _classifier()
    parts: List[pd.DataFrame] = []
    total = 0
    reader = read_nis_csv(core, usecols=usecols, chunksize=cfg["ingestion"]["chunksize"],
                           dtype={c: "object" for c in dx_cols}, low_memory=False)
    for i, chunk in enumerate(reader):
        dx_all_norm = chunk[dx_cols].apply(normalize_dx)
        dx = dx_all_norm[dx1_2]
        flags_any = classifier.flags_for_dx_frame(dx)                # DX1 or DX2 (primary)
        flags_prin = classifier.flags_for_dx_frame(dx.iloc[:, [0]])  # DX1 only (criterion 8)

        part = pd.DataFrame({
            "DISCWT": (pd.to_numeric(chunk[det["weight"]], errors="coerce").fillna(1.0)
                       if det["weight"] else 1.0),
        })
        for j, col in enumerate(ANY_COLS):
            part[col] = flags_any[:, j]
        for j, col in enumerate(PRIN_ONLY_COLS):
            part[col] = flags_prin[:, j]

        part["N_DX_FILLED"] = dx_all_norm.notna().sum(axis=1).astype(np.int16)
        for name, col in covariate_map.items():
            part[name] = pd.to_numeric(chunk[col], errors="coerce") if col else np.nan
        for name in ("AGE", "FEMALE", "ELECTIVE", "AWEEKEND", "PAY1"):
            part.loc[part[name] < 0, name] = np.nan

        parts.append(part)
        total += len(chunk)
        if (i + 1) % 4 == 0:
            log.info("[%s] processed %s rows...", year, f"{total:,}")

    feats = pd.concat(parts, ignore_index=True)
    del parts
    for c in ANY_COLS + PRIN_ONLY_COLS:
        feats[c] = feats[c].astype(np.uint8)

    feats.to_parquet(out_path, index=False)
    _fingerprint_path(cfg, year).write_text(json.dumps(
        {"fingerprint": definitions_fingerprint(),
         "definitions": {d: DISEASES[d]["icd10_prefixes"] for d in VASC_INFECT_DISEASES}},
        indent=2))
    log.info("[%s] checkpoint written: %s rows -> %s", year, f"{len(feats):,}", out_path)
    return out_path


def load_year(cfg: Config, year: str, columns: List[str] | None = None) -> pd.DataFrame:
    path = checkpoint_path(cfg, year)
    if not checkpoint_is_current(cfg, year):
        get_logger("h3.ingest").warning(
            "[%s] checkpoint missing or built from older disease definitions; rebuilding", year)
        build_year_checkpoint(cfg, year, force=True)
    return pd.read_parquet(path, columns=columns)
