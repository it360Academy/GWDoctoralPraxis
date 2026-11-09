"""Aggregates a year's DEV detections into raw and DISCWT-weighted rates:
overall, per-phenotype, per-pair, per-secondary-position, and
per-coding-intensity strata. All estimates report both raw (sample) and
weighted (national) figures side by side, plus a stratified-bootstrap CI
on the weighted rate.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from ..dev.detector import DevResult


def _weighted_rate_ci(dev: np.ndarray, weights: np.ndarray, n_resamples: int = 500,
                       seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    n = len(dev)
    point = float(np.average(dev, weights=weights)) if weights.sum() > 0 else float("nan")
    if n == 0:
        return point, float("nan"), float("nan")
    boots = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        w = weights[idx]
        boots[i] = np.average(dev[idx], weights=w) if w.sum() > 0 else np.nan
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


def overall_rate(df: pd.DataFrame, dev: np.ndarray, weight_col: str = "DISCWT") -> Dict:
    weights = df[weight_col].to_numpy(dtype=np.float64)
    raw_rate = float(dev.mean()) if len(dev) else float("nan")
    w_point, w_lo, w_hi = _weighted_rate_ci(dev, weights)
    n_weighted_discharges = float(weights.sum())
    n_weighted_devs = float((dev * weights).sum())
    return {
        "n_hospitalizations": int(len(dev)),
        "n_dev_raw": int(dev.sum()),
        "raw_dev_rate": raw_rate,
        "weighted_dev_rate": w_point,
        "weighted_dev_rate_ci_lo": w_lo,
        "weighted_dev_rate_ci_hi": w_hi,
        "national_weighted_discharges_est": n_weighted_discharges,
        "national_weighted_dev_est": n_weighted_devs,
    }


def _clustered_weighted_rate_ci(dev: np.ndarray, weights: np.ndarray, clusters: pd.Series,
                                 n_resamples: int = 1000, seed: int = 42) -> tuple:
    """Weighted-rate CI from a CLUSTER bootstrap: whole hospitals are
    resampled with replacement, so the interval reflects that discharges
    within a hospital are correlated (an i.i.d. bootstrap over discharges
    understates the uncertainty of a hospital-clustered survey like NIS).
    Per-hospital numerator/denominator are aggregated once, so each
    resample costs O(n_hospitals), not O(n_discharges)."""
    codes, _ = pd.factorize(clusters)
    missing = codes < 0                     # discharges with no hospital ID
    codes[missing] = codes.max() + 1 + np.arange(int(missing.sum()))  # -> singleton clusters
    k = int(codes.max()) + 1
    num = np.bincount(codes, weights=dev * weights, minlength=k)
    den = np.bincount(codes, weights=weights, minlength=k)
    point = float(num.sum() / den.sum()) if den.sum() > 0 else float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, k, size=k)
        d = den[idx].sum()
        boots[i] = num[idx].sum() / d if d > 0 else np.nan
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0:
        return point, float("nan"), float("nan"), k
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi), k


def cohort_rate(df: pd.DataFrame, dev: np.ndarray, weight_col: str = "DISCWT",
                hosp_col: str = "HOSP_NIS", seed: int = 42) -> Dict:
    """PRIMARY H2 statistic: the weighted DEV rate among ELIGIBLE discharges,
    i.e. those whose principal diagnosis is one of the studied phenotypes
    (LABEL not null). Only these can ever be flagged - detect() never
    evaluates any other discharge - so dividing by all discharges (as
    overall_rate does) dilutes the rate by the ~86% of discharges that
    were never checked. The CI is a hospital-clustered bootstrap when
    `hosp_col` is available, else an i.i.d. bootstrap (flagged in
    `ci_method`)."""
    eligible = df["LABEL"].notna().to_numpy()
    d = dev[eligible]
    w = df[weight_col].to_numpy(dtype=np.float64)[eligible]
    out = {"n_eligible": int(eligible.sum()), "n_all_discharges": int(len(df)),
           "eligible_share_of_discharges": float(eligible.mean()) if len(df) else float("nan"),
           "n_dev_raw": int(d.sum()),
           "raw_dev_rate": float(d.mean()) if len(d) else float("nan")}
    if hosp_col in df.columns:
        point, lo, hi, k = _clustered_weighted_rate_ci(d, w, df.loc[eligible, hosp_col],
                                                       seed=seed)
        out.update(ci_method="hospital-clustered bootstrap (1000 resamples)", n_clusters=k)
    else:
        point, lo, hi = _weighted_rate_ci(d, w, n_resamples=1000, seed=seed)
        out.update(ci_method="i.i.d. bootstrap (HOSP_NIS unavailable - CI is too narrow)",
                   n_clusters=None)
    out.update(weighted_dev_rate=point, weighted_dev_rate_ci_lo=lo, weighted_dev_rate_ci_hi=hi)
    return out


def phenotype_rates(df: pd.DataFrame, dev: np.ndarray, weight_col: str = "DISCWT") -> pd.DataFrame:
    tmp = df[["LABEL", weight_col]].copy()
    tmp["dev"] = dev
    rows = []
    for phenotype, g in tmp.groupby("LABEL", observed=True):
        d = g["dev"].to_numpy()
        w = g[weight_col].to_numpy(dtype=np.float64)
        w_point, w_lo, w_hi = _weighted_rate_ci(d, w)
        rows.append({
            "phenotype": phenotype, "n": len(g), "n_dev_raw": int(d.sum()),
            "raw_dev_rate": float(d.mean()) if len(d) else float("nan"),
            "weighted_dev_rate": w_point, "weighted_dev_rate_ci_lo": w_lo,
            "weighted_dev_rate_ci_hi": w_hi,
        })
    return pd.DataFrame(rows).sort_values("weighted_dev_rate", ascending=False).reset_index(drop=True)


def pair_rates(hit_log: pd.DataFrame, df: pd.DataFrame, weight_col: str = "DISCWT",
               n_total: int | None = None) -> pd.DataFrame:
    """Rate is per-pair prevalence among the phenotype's cohort (not among
    all discharges), since a pair can only occur when DX1 is that
    phenotype."""
    if hit_log.empty:
        return pd.DataFrame(columns=["phenotype", "home_code", "target_raw", "note_text",
                                      "n_hospitalizations_hit", "weighted_n_hit",
                                      "rank_by_weighted_n"])
    weights = df[weight_col].to_numpy(dtype=np.float64)
    # one row per (row, pair) - dedupe multiple secondary positions hitting the
    # same pair on the same hospitalization before counting hospitalizations
    dedup = hit_log.drop_duplicates(subset=["row", "home_code", "target_raw"])
    dedup = dedup.assign(weight=lambda d: weights[d["row"].to_numpy()])
    grouped = dedup.groupby(["phenotype", "home_code", "target_raw", "note_text"], as_index=False).agg(
        n_hospitalizations_hit=("row", "count"), weighted_n_hit=("weight", "sum"))
    grouped = grouped.sort_values("weighted_n_hit", ascending=False).reset_index(drop=True)
    grouped["rank_by_weighted_n"] = np.arange(1, len(grouped) + 1)
    return grouped


def position_rates(hit_log: pd.DataFrame) -> pd.DataFrame:
    if hit_log.empty:
        return pd.DataFrame(columns=["secondary_position", "n_hits"])
    dedup = hit_log.drop_duplicates(subset=["row", "secondary_position"])
    out = dedup.groupby("secondary_position", as_index=False).size().rename(columns={"size": "n_hits"})
    return out.sort_values("secondary_position").reset_index(drop=True)


def violations_per_hospitalization(dev_result: DevResult) -> pd.DataFrame:
    vc = pd.Series(dev_result.n_violations).value_counts().sort_index()
    return pd.DataFrame({"n_violations": vc.index, "n_hospitalizations": vc.values})


def coding_intensity_stratified_rate(df: pd.DataFrame, dev: np.ndarray,
                                      n_dx_col: str = "N_DX_FILLED",
                                      weight_col: str = "DISCWT",
                                      bins=(0, 5, 10, 15, 20, 25, 40)) -> pd.DataFrame:
    tmp = df[[n_dx_col, weight_col]].copy()
    tmp["dev"] = dev
    tmp["intensity_bin"] = pd.cut(tmp[n_dx_col], bins=bins, include_lowest=True)
    rows = []
    for b, g in tmp.groupby("intensity_bin", observed=True):
        d = g["dev"].to_numpy()
        w = g[weight_col].to_numpy(dtype=np.float64)
        w_point, _, _ = _weighted_rate_ci(d, w, n_resamples=100)
        rows.append({"n_dx_filled_bin": str(b), "n": len(g),
                      "raw_dev_rate": float(d.mean()) if len(d) else float("nan"),
                      "weighted_dev_rate": w_point})
    return pd.DataFrame(rows)
