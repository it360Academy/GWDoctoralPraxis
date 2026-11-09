"""Build the DX2-DX25 -> 3-character-ICD-10-category feature representation.

Pipeline:
  1. Melt every year's SECDX_1..SECDX_<max_secondary_dx> columns into long
     form (one row per non-missing secondary code).
  2. MASK: drop any code that matches a phenotype definition (leakage
     control - see src/leakage/controls.py). This mask is computed once,
     globally, from the training year's code distribution and re-applied
     unchanged to validation/test (it is a property of the code list, not
     of any particular year's data).
  3. VOCABULARY: truncate surviving codes to their 3-character ICD-10-CM
     category, keep categories occurring >= min_category_frequency times
     in the TRAINING year only. This is fit once on train and applied
     unchanged to validation/test - a category never seen in training
     contributes nothing to any year's feature matrix (not an error).
  4. ENCODE: for any year, build a sparse (n_rows x n_categories) 0/1
     matrix (presence of category among that discharge's DX2-25), plus
     optional dense demographic/admission covariate columns, and return a
     single combined sparse matrix so it is uniform across
     logistic regression / random forest / LightGBM / XGBoost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import sparse

from ..config import Config
from ..leakage.controls import leaky_codes
from ..logging_utils import get_logger

log = get_logger("h1.features")


def _sec_cols(cfg: Config) -> List[str]:
    max_sec = int(cfg["ingestion"].get("max_secondary_dx", 25))
    return [f"SECDX_{j+1}" for j in range(max_sec)]


def _melt_codes(df: pd.DataFrame, sec_cols: List[str]) -> pd.Series:
    """Return a Series indexed by original row position, one entry per
    non-missing secondary code (a row may contribute 0..len(sec_cols)
    entries)."""
    long = df[sec_cols].stack(future_stack=True)
    long = long.dropna()
    long = long.astype(str)
    long = long[(long.str.len() > 0) & (long.str.lower() != "nan") & (long.str.lower() != "none")]
    long.index = long.index.get_level_values(0)
    return long


@dataclass
class DxVocabulary:
    categories: List[str]                     # retained 3-char categories, fixed column order
    masked_codes: List[str]                    # full codes dropped for leakage
    code_counts_train: Dict[str, int]           # secondary-position frequency in training year (pre-mask)
    category_counts_train: Dict[str, int]        # frequency of retained categories in training year
    covariate_cols: List[str] = field(default_factory=list)
    covariate_medians: Dict[str, float] = field(default_factory=dict)
    min_category_frequency: int = 50

    @property
    def n_features(self) -> int:
        return len(self.categories) + len(self.covariate_cols)


def fit_vocabulary(cfg: Config, train_df: pd.DataFrame) -> DxVocabulary:
    """Fit the masked category vocabulary from the TRAINING year only."""
    sec_cols = _sec_cols(cfg)
    train_df = train_df.reset_index(drop=True)
    long = _melt_codes(train_df, sec_cols)

    code_counts_train = long.value_counts().to_dict()
    unique_codes = list(code_counts_train.keys())
    masked = leaky_codes(unique_codes)
    log.info("Leakage masking: %d/%d unique training-year secondary codes "
             "match a phenotype definition and are dropped from the "
             "feature vocabulary.", len(masked), len(unique_codes))

    kept = long[~long.isin(masked)]
    categories = kept.str[:3]
    cat_counts = categories.value_counts()
    min_freq = int(cfg["features"].get("min_category_frequency", 50))
    retained = cat_counts[cat_counts >= min_freq].index.tolist()
    retained = sorted(retained)
    log.info("Feature vocabulary: %d ICD-10 3-char categories retained "
             "(>= %d occurrences in training year), out of %d observed "
             "post-mask.", len(retained), min_freq, cat_counts.shape[0])

    covariate_cols: List[str] = []
    covariate_medians: Dict[str, float] = {}
    if cfg["features"].get("include_demographic_covariates", True):
        covariate_cols = list(cfg["features"].get("demographic_cols", [])) + \
                          list(cfg["features"].get("admission_cols", []))
        covariate_cols = [c for c in covariate_cols if c in train_df.columns]
        for c in covariate_cols:
            covariate_medians[c] = float(pd.to_numeric(train_df[c], errors="coerce").median())

    return DxVocabulary(
        categories=retained,
        masked_codes=sorted(masked),
        code_counts_train=code_counts_train,
        category_counts_train=cat_counts.to_dict(),
        covariate_cols=covariate_cols,
        covariate_medians=covariate_medians,
        min_category_frequency=min_freq,
    )


def encode(cfg: Config, df: pd.DataFrame, vocab: DxVocabulary) -> sparse.csr_matrix:
    """Encode ANY year's dataframe using an ALREADY-FIT vocabulary (no
    refitting - this is what keeps validation/test genuinely out-of-sample
    with respect to feature selection, not just model fitting)."""
    sec_cols = _sec_cols(cfg)
    df = df.reset_index(drop=True)
    n = len(df)
    cat_index = {c: i for i, c in enumerate(vocab.categories)}
    n_cat = len(vocab.categories)

    long = _melt_codes(df, sec_cols)
    if len(long):
        categories = long.str[:3]
        col_pos = categories.map(cat_index)
        valid = col_pos.notna()
        row_pos = long.index[valid].to_numpy()
        col_pos = col_pos[valid].to_numpy(dtype=np.int64)
        data = np.ones(len(row_pos), dtype=np.uint8)
        dx_mat = sparse.coo_matrix((data, (row_pos, col_pos)), shape=(n, n_cat)).tocsr()
        dx_mat.data[:] = 1  # duplicate (row, category) entries -> presence, not count
    else:
        dx_mat = sparse.csr_matrix((n, n_cat), dtype=np.uint8)

    blocks = [dx_mat]
    for c in vocab.covariate_cols:
        vals = pd.to_numeric(df[c], errors="coerce").fillna(vocab.covariate_medians.get(c, 0.0))
        blocks.append(sparse.csr_matrix(vals.to_numpy().reshape(-1, 1).astype(np.float64)))
    return sparse.hstack(blocks, format="csr")


def feature_names(vocab: DxVocabulary) -> List[str]:
    return [f"DX_CAT_{c}" for c in vocab.categories] + list(vocab.covariate_cols)


def build_xy(cfg: Config, df: pd.DataFrame, vocab: DxVocabulary):
    """Restrict to the 15-phenotype cohort (LABEL not null), encode X,
    return (X, y, sample_weight, row_mask_used)."""
    df = df.reset_index(drop=True)
    cohort_mask = df["LABEL"].notna().to_numpy()
    cohort = df.loc[cohort_mask].reset_index(drop=True)
    X = encode(cfg, cohort, vocab)
    y = cohort["LABEL"].astype(str).to_numpy()
    w = cohort["DISCWT"].to_numpy(dtype=np.float64)
    return X, y, w, cohort_mask
