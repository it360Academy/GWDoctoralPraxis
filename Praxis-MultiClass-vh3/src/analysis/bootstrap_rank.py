"""Bootstrap rank stability (Chapter 3, Section 3.6.4): repeatedly
resamples the underlying incidence and harm estimates and recomputes,
for each resample, each disease's rank, its probability of appearing
within the top-five ranked diseases, and the resulting top-five and
top-fifteen harm-concentration shares.

Rather than running a second, separate resampling loop, this module
reuses the Monte Carlo simulation draws already produced by
montecarlo.run_simulation (src/analysis/montecarlo.py): each Monte Carlo
draw is itself one plausible joint sample from the incidence and
error/harm-rate uncertainty distributions, so ranking within each draw
IS the bootstrap resample this analysis calls for, without duplicating
computation or introducing a second, inconsistent resampling scheme.

Eighteen categories are ranked per draw: the fifteen named phenotypes
plus the three "other" (non-top-15) vascular/infection/cancer buckets,
since the "other" buckets are large enough (thousands of cases) to
plausibly outrank individual named phenotypes - top-15 membership is
therefore a real, non-trivial event, not a foregone conclusion.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

ALL_CATEGORY_NAMES: List[str] = [
    "stroke", "vte", "arterial_te", "aortic_aneurysm", "mi",
    "sepsis", "pneumonia", "meningitis_encephalitis", "spinal_abscess", "endocarditis",
    "lung_cancer", "breast_cancer", "colorectal_cancer", "melanoma", "prostate_cancer",
    "other_vascular", "other_infection", "other_cancer",
]


def rank_stability(draws: pd.DataFrame) -> pd.DataFrame:
    harm_cols = [f"harms_{name}_thousands" for name in ALL_CATEGORY_NAMES]
    missing = [c for c in harm_cols if c not in draws.columns]
    if missing:
        raise KeyError(f"Monte Carlo draws missing expected columns: {missing}")

    harms = draws[harm_cols].to_numpy()
    # rank 1 = highest harm; ranks computed independently per draw (row)
    ranks = (-harms).argsort(axis=1).argsort(axis=1) + 1

    rows = []
    for j, name in enumerate(ALL_CATEGORY_NAMES):
        disease_ranks = ranks[:, j]
        rows.append({
            "disease": name,
            "mean_rank": float(disease_ranks.mean()),
            "median_rank": float(np.median(disease_ranks)),
            "rank_p5": float(np.percentile(disease_ranks, 5)),
            "rank_p95": float(np.percentile(disease_ranks, 95)),
            "p_top5": float((disease_ranks <= 5).mean()),
            "p_top15": float((disease_ranks <= 15).mean()),
        })
    return pd.DataFrame(rows).sort_values("mean_rank").reset_index(drop=True)


def concentration_share_stability(draws: pd.DataFrame) -> dict:
    """Point estimate and bootstrap distribution summary for top-5/top-15
    harm-concentration shares, drawn directly from the existing Monte
    Carlo columns (already computed in montecarlo.py)."""
    out = {}
    for col in ("top5_share_of_grand_total", "top5_fixed_share_of_grand_total",
                "top15_share_of_grand_total"):
        v = draws[col]
        out[col] = {
            "median": float(v.median()),
            "p2_5": float(v.quantile(0.025)),
            "p97_5": float(v.quantile(0.975)),
            "std": float(v.std()),
        }
    return out
