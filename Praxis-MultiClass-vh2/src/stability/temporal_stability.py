"""Temporal reproducibility testing: 2021 (development) vs 2022 (temporal
validation) vs 2023 (independent replication). Tests DEV-rate stability,
enemy-pair rank-ordering stability, dominant-pair consistency, diagnosis-
position patterns, and AI/ML performance consistency across all three
years, against PRESPECIFIED thresholds (config: stability_criteria - see
README for why these are placeholders pending your decision, same
situation as H1's h1_criteria floors).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def rate_stability(overall_by_year: Dict[str, Dict], rate_ratio_threshold: float) -> Dict:
    """Pairwise rate ratios between consecutive years, each flagged
    against the prespecified threshold (e.g. "within +/-20%")."""
    years = sorted(overall_by_year.keys())
    rows = []
    for a, b in zip(years[:-1], years[1:]):
        ra = overall_by_year[a]["weighted_dev_rate"]
        rb = overall_by_year[b]["weighted_dev_rate"]
        ratio = rb / ra if ra else float("nan")
        within_threshold = bool(ratio == ratio and (1 / rate_ratio_threshold) <= ratio <= rate_ratio_threshold)
        rows.append({"year_a": a, "year_b": b, "rate_a": ra, "rate_b": rb,
                      "rate_ratio_b_over_a": ratio, "within_stability_threshold": within_threshold})
    return {"pairwise": pd.DataFrame(rows),
            "all_pairs_stable": bool(pd.DataFrame(rows)["within_stability_threshold"].all()) if rows else False}


def pair_rank_stability(pair_rates_by_year: Dict[str, pd.DataFrame],
                         rank_correlation_threshold: float, top_n: int = 20) -> Dict:
    """Spearman rank correlation of top-N pair rankings (by weighted_n_hit)
    between each pair of years. A pair present in one year's top-N but
    absent from another's ranking table is assigned the WORST possible
    rank (len+1) rather than dropped, so absence is penalized rather than
    ignored."""
    years = sorted(pair_rates_by_year.keys())
    key_cols = ["phenotype", "home_code", "target_raw"]

    def pair_key(row):
        return tuple(row[c] for c in key_cols)

    all_keys = set()
    ranks_by_year = {}
    for y in years:
        df = pair_rates_by_year[y].copy()
        if df.empty:
            ranks_by_year[y] = {}
            continue
        df = df.sort_values("weighted_n_hit", ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        ranks = {pair_key(row): row["rank"] for _, row in df.iterrows()}
        ranks_by_year[y] = ranks
        all_keys |= set(ranks.keys())

    rows = []
    for a, b in zip(years[:-1], years[1:]):
        ra, rb = ranks_by_year[a], ranks_by_year[b]
        worst_a = max(ra.values(), default=0) + 1
        worst_b = max(rb.values(), default=0) + 1
        keys = sorted(set(ra) | set(rb))[:max(top_n, len(all_keys))]
        if not keys:
            rows.append({"year_a": a, "year_b": b, "spearman_r": float("nan"),
                         "p_value": float("nan"), "meets_threshold": False})
            continue
        xa = [ra.get(k, worst_a) for k in keys]
        xb = [rb.get(k, worst_b) for k in keys]
        if len(set(xa)) <= 1 or len(set(xb)) <= 1:
            r, p = float("nan"), float("nan")
        else:
            r, p = scipy_stats.spearmanr(xa, xb)
        rows.append({"year_a": a, "year_b": b, "spearman_r": r, "p_value": p,
                      "meets_threshold": bool(r == r and r >= rank_correlation_threshold)})
    df_out = pd.DataFrame(rows)
    return {"pairwise": df_out,
            "all_pairs_stable": bool(df_out["meets_threshold"].all()) if len(df_out) else False}


def dominant_pair_consistency(pair_rates_by_year: Dict[str, pd.DataFrame], top_n: int = 5) -> pd.DataFrame:
    """Which pairs are in the top-N by weighted hit count in EVERY year -
    a simple, interpretable consistency check alongside the rank
    correlation."""
    years = sorted(pair_rates_by_year.keys())
    top_sets = {}
    for y in years:
        df = pair_rates_by_year[y]
        if df.empty:
            top_sets[y] = set()
            continue
        top = df.sort_values("weighted_n_hit", ascending=False).head(top_n)
        top_sets[y] = set(zip(top["phenotype"], top["home_code"], top["target_raw"]))
    consistent = set.intersection(*top_sets.values()) if top_sets else set()
    rows = [{"phenotype": p, "home_code": h, "target_raw": t,
             "in_top_n_every_year": True} for (p, h, t) in consistent]
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["phenotype", "home_code", "target_raw", "in_top_n_every_year"])


def position_pattern_stability(position_rates_by_year: Dict[str, pd.DataFrame],
                                rank_correlation_threshold: float) -> Dict:
    years = sorted(position_rates_by_year.keys())
    rows = []
    for a, b in zip(years[:-1], years[1:]):
        da = position_rates_by_year[a].set_index("secondary_position")["n_hits"]
        db = position_rates_by_year[b].set_index("secondary_position")["n_hits"]
        idx = sorted(set(da.index) | set(db.index))
        xa = [da.get(i, 0) for i in idx]
        xb = [db.get(i, 0) for i in idx]
        if len(set(xa)) <= 1 or len(set(xb)) <= 1:
            r, p = float("nan"), float("nan")
        else:
            r, p = scipy_stats.spearmanr(xa, xb)
        rows.append({"year_a": a, "year_b": b, "spearman_r": r,
                      "meets_threshold": bool(r == r and r >= rank_correlation_threshold)})
    return {"pairwise": pd.DataFrame(rows)}


def aiml_performance_stability(perf_by_year: Dict[str, Dict], metric: str = "auroc",
                                max_abs_diff: float = 0.10) -> Dict:
    years = sorted(perf_by_year.keys())
    rows = []
    for a, b in zip(years[:-1], years[1:]):
        va, vb = perf_by_year[a][metric], perf_by_year[b][metric]
        diff = abs(va - vb) if va == va and vb == vb else float("nan")
        rows.append({"year_a": a, "year_b": b, f"{metric}_a": va, f"{metric}_b": vb,
                      "abs_diff": diff, "within_threshold": bool(diff == diff and diff <= max_abs_diff)})
    df = pd.DataFrame(rows)
    return {"pairwise": df, "all_pairs_stable": bool(df["within_threshold"].all()) if len(df) else False}
