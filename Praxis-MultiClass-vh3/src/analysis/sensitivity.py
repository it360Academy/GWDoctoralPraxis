"""Threshold-sensitivity analysis for the H3 decision rule (Chapter 3,
Section 3.6.3).

Every numeric value in config.yaml: criteria_thresholds is stated
explicitly, both in this project's README and in the H3 validation
report's own caveat banner, as an unverified placeholder pending
literature-derived or committee-approved benchmarks. A single
seven-of-seven "H3: SUPPORTED" headline does not by itself say whether
that result is robust to that uncertainty or is an artifact of exactly
where the placeholders happened to be set. This module answers that
question directly, rather than leaving it as a qualitative caveat in
prose: it re-scores the same underlying computed statistics (nothing is
re-simulated or re-extracted) against a grid of alternative, equally
defensible threshold values, and reports what share of that grid still
yields overall support.

Two complementary views are produced:
  1. A full-factorial grid across five perturbable threshold parameters
     (criteria 1, 2, 3/4, 6, 8; criterion 7 has no configurable
     threshold and criterion 5 is descriptive-only per Chapter 3) -
     the rigorous, if less intuitive, summary statistic.
  2. A one-dimensional "strictness sweep," in which all five thresholds
     move simultaneously from their most lenient to their most strict
     grid endpoint - a single interpretable curve for the report.
"""
from __future__ import annotations

import itertools
from typing import Dict

import numpy as np
import pandas as pd

DEFAULT_SPAN = {
    "max_incidence_relative_difference": 0.05,   # +/- 5 percentage points
    "min_rank_correlation": 0.10,                 # +/- 0.10
    "max_concentration_diff_pp": 3.0,             # +/- 3 percentage points, shared by criteria 3 & 4
    "max_total_harm_relative_difference": 0.05,   # +/- 5 percentage points
    "max_sensitivity_reduction_diff": 0.05,       # +/- 5 percentage points
}


def _raw_inputs(criterion_1_df: pd.DataFrame, criterion_2: Dict, criterion_3_4: Dict,
                 criterion_6: Dict, criterion_7: Dict, criterion_8: Dict) -> Dict:
    """The underlying computed statistics, frozen once. Every grid point
    re-scores against these same numbers; none of them are recomputed."""
    return {
        "c1_rel_diffs": criterion_1_df["relative_difference"].abs().to_numpy(),
        "c2_spearman_r": criterion_2["spearman_r"],
        "c3_abs_diff_pp": criterion_3_4["top5_abs_diff_pp"],
        "c4_abs_diff_pp": criterion_3_4["top15_abs_diff_pp"],
        "c6_relative_difference": abs(criterion_6["relative_difference"]),
        "c7_all_overlap": all(v["has_overlap"] for v in criterion_7.values() if isinstance(v, dict)),
        "c8_reduction_pct_abs_diff": criterion_8["reduction_pct_abs_diff"],
    }


def build_grid(base_thresholds: Dict, n_steps: int = 3, span: Dict = None) -> Dict[str, np.ndarray]:
    span = span or DEFAULT_SPAN
    grid = {}
    for param, delta in span.items():
        base = base_thresholds[param]
        lo = max(0.0, base - delta)
        hi = base + delta
        grid[param] = np.linspace(lo, hi, n_steps)
    return grid


def _score(raw: Dict, thresholds: Dict, min_share: float, min_criteria_met: int) -> Dict:
    c1_share = float((raw["c1_rel_diffs"] <= thresholds["max_incidence_relative_difference"]).mean())
    m1 = c1_share >= min_share
    m2 = raw["c2_spearman_r"] >= thresholds["min_rank_correlation"]
    m3 = raw["c3_abs_diff_pp"] <= thresholds["max_concentration_diff_pp"]
    m4 = raw["c4_abs_diff_pp"] <= thresholds["max_concentration_diff_pp"]
    m6 = raw["c6_relative_difference"] <= thresholds["max_total_harm_relative_difference"]
    m7 = raw["c7_all_overlap"]
    m8 = raw["c8_reduction_pct_abs_diff"] <= thresholds["max_sensitivity_reduction_diff"]
    n_met = sum([m1, m2, m3, m4, m6, m7, m8])
    return {"criterion_1_met": m1, "criterion_2_met": m2, "criterion_3_met": m3,
            "criterion_4_met": m4, "criterion_6_met": m6, "criterion_7_met": m7,
            "criterion_8_met": m8, "n_met": n_met, "H3_SUPPORTED": n_met >= min_criteria_met,
            "criterion_1_share_within": c1_share}


def run_grid_sensitivity(cfg, criterion_1_df, criterion_2, criterion_3_4,
                          criterion_6, criterion_7, criterion_8, n_steps: int = 3) -> pd.DataFrame:
    """Full-factorial grid. n_steps=3 across 5 parameters = 243 grid
    points; each is a handful of arithmetic comparisons, not a
    re-simulation, so this runs in well under a second."""
    base = cfg["criteria_thresholds"]
    grid = build_grid(base, n_steps=n_steps)
    raw = _raw_inputs(criterion_1_df, criterion_2, criterion_3_4, criterion_6, criterion_7, criterion_8)
    min_share = base.get("min_share_diseases_within_threshold", 0.5)
    min_criteria_met = base.get("min_criteria_met", 5)

    keys = list(grid.keys())
    rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        thresholds = dict(zip(keys, combo))
        result = _score(raw, thresholds, min_share, min_criteria_met)
        row = dict(zip(keys, combo))
        row.update(result)
        rows.append(row)
    return pd.DataFrame(rows)


def run_strictness_sweep(cfg, criterion_1_df, criterion_2, criterion_3_4,
                          criterion_6, criterion_7, criterion_8, n_points: int = 21) -> pd.DataFrame:
    """One-dimensional sweep: strictness in [-1, 1], where -1 moves every
    threshold to its most lenient grid endpoint simultaneously, +1 to its
    most strict endpoint, and 0 is the baseline (placeholder) config.
    Produces one row per strictness level with the resulting n_met and
    H3_SUPPORTED - the basis for a single interpretable robustness curve."""
    base = cfg["criteria_thresholds"]
    span = DEFAULT_SPAN
    raw = _raw_inputs(criterion_1_df, criterion_2, criterion_3_4, criterion_6, criterion_7, criterion_8)
    min_share = base.get("min_share_diseases_within_threshold", 0.5)
    min_criteria_met = base.get("min_criteria_met", 5)

    # "Strict" means harder to pass: lower max_* thresholds, higher min_rank_correlation.
    rows = []
    for s in np.linspace(-1, 1, n_points):
        thresholds = {
            "max_incidence_relative_difference": max(0.0, base["max_incidence_relative_difference"] - s * span["max_incidence_relative_difference"]),
            "min_rank_correlation": min(1.0, max(0.0, base["min_rank_correlation"] + s * span["min_rank_correlation"])),
            "max_concentration_diff_pp": max(0.0, base["max_concentration_diff_pp"] - s * span["max_concentration_diff_pp"]),
            "max_total_harm_relative_difference": max(0.0, base["max_total_harm_relative_difference"] - s * span["max_total_harm_relative_difference"]),
            "max_sensitivity_reduction_diff": max(0.0, base["max_sensitivity_reduction_diff"] - s * span["max_sensitivity_reduction_diff"]),
        }
        result = _score(raw, thresholds, min_share, min_criteria_met)
        rows.append({"strictness": float(s), **thresholds, **result})
    return pd.DataFrame(rows)


def summarize(grid_df: pd.DataFrame) -> Dict:
    n = len(grid_df)
    crit_cols = [c for c in grid_df.columns if c.startswith("criterion_") and c.endswith("_met")]
    return {
        "n_grid_points": n,
        "support_rate": float(grid_df["H3_SUPPORTED"].mean()),
        "per_criterion_met_rate": {c: float(grid_df[c].mean()) for c in crit_cols},
        "min_n_met": int(grid_df["n_met"].min()),
        "max_n_met": int(grid_df["n_met"].max()),
        "baseline_note": (
            "baseline (config.yaml placeholder) thresholds are the grid's center point; "
            "support_rate is the share of the full grid, not just the baseline, for which "
            "H3_SUPPORTED is true"
        ),
    }


def min_criteria_met_sensitivity(grid_df: pd.DataFrame, thresholds_to_test=(4, 5, 6, 7)) -> pd.DataFrame:
    """Secondary check: holding the grid's n_met distribution fixed, how
    does the support rate change if the aggregation rule itself (minimum
    number of the seven scored criteria required) were set differently?"""
    rows = []
    for t in thresholds_to_test:
        rows.append({"min_criteria_met": t, "support_rate": float((grid_df["n_met"] >= t).mean())})
    return pd.DataFrame(rows)
