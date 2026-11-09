"""The eight prespecified H3 replication criteria (Chapter 3, Section
3.6.3 / Table 3.3). Per the explicit design requirement, replication is
never determined from the top-fifteen harm-concentration share, or any
other single criterion, in isolation - all eight are computed and
reported jointly, each against a numerical stability threshold fixed in
config.yaml before this module is run against the final data.

All thresholds in config.yaml: criteria_thresholds are unverified
placeholders pending literature-derived or committee-approved benchmarks,
the same status as the h1_criteria/h2 stability thresholds in the
companion H1 and H2 projects. This module applies whatever is configured;
it does not choose the thresholds.

Re-derived quantities use the DX1-or-DX2 incidence definition (see
ingestion.py), matching the published Table 1 definition. The DX1-only
variant enters only through criterion 8.

Criterion 2 is computed over the ten NIS-derived vascular/infectious
phenotypes only: the five cancers use published registry incidence in both
structures, so their harms are identical by design and would inflate the
rank correlation without testing anything.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from ..features.literature_inputs import (BIG_THREE_HARM_FRACTION, OTHER_BUCKETS,
                                          PUBLISHED_RESULTS, TABLE1_CANCERS,
                                          TABLE1_TOP5)

ALL_SPECS = {**TABLE1_TOP5, **TABLE1_CANCERS}
INDEPENDENT_DISEASES = list(TABLE1_TOP5)   # incidence re-derived from NIS
CATEGORY_MEMBERS = {
    "vascular": [d for d, s in TABLE1_TOP5.items() if s.category == "vascular"] + ["other_vascular"],
    "infection": [d for d, s in TABLE1_TOP5.items() if s.category == "infection"] + ["other_infection"],
    "cancer": list(TABLE1_CANCERS) + ["other_cancer"],
}
CRITERION_NAMES = {
    1: "Disease-specific incidence relative difference",
    2: "Rank-correlation of harm rankings",
    3: "Top-5 harm-concentration difference",
    4: "Top-15 harm-concentration difference",
    5: "Category-level burden-share stability",
    6: "Total serious-harm relative difference",
    7: "Monte Carlo uncertainty overlap",
    8: "DX1-only sensitivity comparison",
}


def _interval(v: pd.Series) -> tuple:
    return float(v.median()), float(v.quantile(0.025)), float(v.quantile(0.975))


def criterion_1_incidence_relative_difference(avg_incidence: pd.DataFrame) -> pd.DataFrame:
    """Published vs. re-derived (DX1 or DX2) mean annual incidence by
    phenotype. ``avg_incidence`` is incidence.three_year_average output."""
    inc = avg_incidence.set_index("disease")
    rows = []
    for name, spec in TABLE1_TOP5.items():
        pub = spec.published_incidence_thousands
        red = float(inc.loc[name, "mean_annual_incidence_thousands"]) if name in inc.index else np.nan
        lo = float(inc.loc[name, "mean_ci_low_thousands"]) if name in inc.index else np.nan
        hi = float(inc.loc[name, "mean_ci_high_thousands"]) if name in inc.index else np.nan
        rel = (red - pub) / pub if pub and red == red else np.nan
        rows.append({"disease": name, "category": spec.category,
                     "published_thousands": pub, "rederived_thousands": red,
                     "rederived_ci_low_thousands": lo, "rederived_ci_high_thousands": hi,
                     "absolute_difference_thousands": red - pub, "relative_difference": rel})
    return pd.DataFrame(rows)


def rederived_incidence_map(c1: pd.DataFrame) -> Dict[str, float]:
    """All fifteen phenotypes' incidence for the re-derived structure:
    NIS values for the ten computed phenotypes, published registry
    values for the five cancers."""
    out = {n: s.published_incidence_thousands for n, s in ALL_SPECS.items()}
    out.update(c1.dropna(subset=["rederived_thousands"])
                 .set_index("disease")["rederived_thousands"].astype(float).to_dict())
    return out


def harm_comparison(c1: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    """Published vs. re-derived serious harm and harm rank by phenotype.

    Published harm is reconstructed as published incidence x published
    harm rate (point estimates); re-derived harm is the same product on
    re-derived incidence, with its Monte Carlo median and 95% interval.

    Ranks are given two ways: among all fifteen phenotypes, and among the
    ten NIS-derived phenotypes only (the basis of criterion 2)."""
    red_inc = rederived_incidence_map(c1)
    rows = []
    for name, spec in ALL_SPECS.items():
        pub_inc = spec.published_incidence_thousands
        med, lo, hi = _interval(draws[f"harms_{name}_thousands"])
        pub_harm = pub_inc * spec.harm_rate
        rows.append({"disease": name, "category": spec.category,
                     "published_incidence_thousands": pub_inc,
                     "rederived_incidence_thousands": red_inc[name],
                     "harm_rate": spec.harm_rate,
                     "published_harm_thousands": pub_harm,
                     "rederived_harm_thousands": red_inc[name] * spec.harm_rate,
                     "rederived_harm_mc_median": med, "rederived_harm_mc_p2_5": lo,
                     "rederived_harm_mc_p97_5": hi,
                     "published_harm_in_mc_interval": bool(lo <= pub_harm <= hi)})
    df = pd.DataFrame(rows)
    df["published_rank"] = df["published_harm_thousands"].rank(ascending=False, method="min").astype(int)
    df["rederived_rank"] = df["rederived_harm_thousands"].rank(ascending=False, method="min").astype(int)
    df["rank_change"] = df["published_rank"] - df["rederived_rank"]
    df["independent"] = df["disease"].isin(INDEPENDENT_DISEASES)
    ind = df["independent"]
    df.loc[ind, "published_rank_10"] = df.loc[ind, "published_harm_thousands"].rank(
        ascending=False, method="min")
    df.loc[ind, "rederived_rank_10"] = df.loc[ind, "rederived_harm_thousands"].rank(
        ascending=False, method="min")
    df["rank_change_10"] = df["published_rank_10"] - df["rederived_rank_10"]
    for col in ("published_rank_10", "rederived_rank_10", "rank_change_10"):
        df[col] = df[col].astype("Int64")
    return df.sort_values("published_rank").reset_index(drop=True)


def criterion_2_rank_correlation(harm_df: pd.DataFrame) -> Dict:
    """Spearman (scored) and Kendall (reported) rank correlation of harm,
    over the ten NIS-derived phenotypes only."""
    ind = harm_df[harm_df["disease"].isin(INDEPENDENT_DISEASES)]
    pub, red = ind["published_harm_thousands"], ind["rederived_harm_thousands"]
    r, p = scipy_stats.spearmanr(pub, red)
    tau, tau_p = scipy_stats.kendalltau(pub, red)
    pub_top5 = set(harm_df.nsmallest(5, "published_rank")["disease"])
    red_top5 = set(harm_df.nsmallest(5, "rederived_rank")["disease"])
    return {"spearman_r": float(r), "p_value": float(p), "kendall_tau": float(tau),
            "kendall_p_value": float(tau_p), "n_diseases": int(len(ind)),
            "diseases": "10 NIS-derived (cancers excluded: identical by design)",
            "top5_overlap_all15": len(pub_top5 & red_top5),
            "max_abs_rank_change_10": int((ind["published_rank_10"]
                                           - ind["rederived_rank_10"]).abs().max())}


def criterion_3_4_concentration_difference(sim_summary: Dict) -> Dict:
    pub5 = PUBLISHED_RESULTS["top5_harms_share"]
    pub15 = PUBLISHED_RESULTS["top15_harms_share"]
    my5, lo5, hi5 = sim_summary["top5_share_of_grand_total"]
    my15, lo15, hi15 = sim_summary["top15_share_of_grand_total"]
    out = {
        "top5_published": pub5, "top5_rederived": my5,
        "top5_rederived_p2_5": lo5, "top5_rederived_p97_5": hi5,
        "top5_abs_diff_pp": abs(my5 - pub5) * 100,
        "top5_published_in_interval": bool(lo5 <= pub5 <= hi5),
        "top15_published": pub15, "top15_rederived": my15,
        "top15_rederived_p2_5": lo15, "top15_rederived_p97_5": hi15,
        "top15_abs_diff_pp": abs(my15 - pub15) * 100,
        "top15_published_in_interval": bool(lo15 <= pub15 <= hi15),
    }
    if "top5_fixed_share_of_grand_total" in sim_summary:
        out["top5_fixed_published_set_rederived"] = sim_summary["top5_fixed_share_of_grand_total"][0]
    return out


def _published_category_shares() -> Dict[str, float]:
    """Point-estimate category shares on the published inputs: category
    harm (incidence x harm rate) over the extrapolated grand total."""
    harms = {n: s.published_incidence_thousands * s.harm_rate for n, s in ALL_SPECS.items()}
    harms.update({b: spec["published_incidence_thousands"] * spec["harm_rate"]
                  for b, spec in OTHER_BUCKETS.items()})
    grand_total = sum(harms.values()) / BIG_THREE_HARM_FRACTION
    return {cat: sum(harms[m] for m in members) / grand_total
            for cat, members in CATEGORY_MEMBERS.items()}


def criterion_5_category_shares(draws: pd.DataFrame) -> pd.DataFrame:
    total = draws["grand_total_harms_thousands"]
    published = _published_category_shares()
    rows = []
    for cat, members in CATEGORY_MEMBERS.items():
        share = sum(draws[f"harms_{m}_thousands"] for m in members) / total
        med, lo, hi = _interval(share)
        rows.append({"category": cat, "published_share": published[cat],
                     "share_median": med, "share_p2_5": lo, "share_p97_5": hi,
                     "diff_pp": (med - published[cat]) * 100,
                     "published_in_interval": bool(lo <= published[cat] <= hi)})
    return pd.DataFrame(rows)


def criterion_6_total_harm_difference(sim_summary: Dict) -> Dict:
    pub_med, pub_lo, pub_hi = PUBLISHED_RESULTS["grand_total_harms_thousands"]
    my_med, my_lo, my_hi = sim_summary["grand_total_harms_thousands"]
    rel_diff = (my_med - pub_med) / pub_med
    return {"published_median": pub_med, "published_range": (pub_lo, pub_hi),
            "rederived_median": my_med, "rederived_range": (my_lo, my_hi),
            "relative_difference": rel_diff,
            "published_median_in_rederived_range": bool(my_lo <= pub_med <= my_hi)}


def criterion_7_monte_carlo_overlap(sim_summary: Dict) -> Dict:
    """Overlap between the reconstructed 95% simulation interval and the
    published range, for each headline metric."""
    rows = {}
    for metric in ("big3_errors_thousands", "big3_harms_thousands",
                   "grand_total_errors_thousands", "grand_total_harms_thousands"):
        my_med, my_lo, my_hi = sim_summary[metric]
        pub_med, pub_lo, pub_hi = PUBLISHED_RESULTS[metric]
        overlap_lo = max(my_lo, pub_lo)
        overlap_hi = min(my_hi, pub_hi)
        has_overlap = overlap_hi > overlap_lo
        union_lo = min(my_lo, pub_lo)
        union_hi = max(my_hi, pub_hi)
        overlap_fraction = ((overlap_hi - overlap_lo) / (union_hi - union_lo)
                             if has_overlap and union_hi > union_lo else 0.0)
        rows[metric] = {"my_interval": (my_lo, my_hi), "published_interval": (pub_lo, pub_hi),
                         "has_overlap": bool(has_overlap), "overlap_fraction": float(overlap_fraction),
                         "published_point_in_my_interval": bool(my_lo <= pub_med <= my_hi),
                         "my_point_in_published_interval": bool(pub_lo <= my_med <= pub_hi)}
    return rows


def criterion_8_dx1_only_sensitivity(sim_summary_primary: Dict, sim_summary_dx1_only: Dict) -> Dict:
    """How much total serious harm falls when incidence is restricted from
    DX1-or-DX2 to DX1 only, re-derived vs. published. Scored on the
    absolute difference between the two percentage reductions.

    Benchmark confirmed against the paper (p.113): "Using only principal
    diagnosis NIS codes ... gave lower estimates by about 30% (grand totals
    1.78 M missed diagnoses and 549000 serious harms (online supplemental
    file 1-B2))". The re-derived DX1-only run scales the "other"
    vascular/infection buckets to match (montecarlo.dx1_only_other_scale)."""
    pub_primary = PUBLISHED_RESULTS["grand_total_harms_thousands"][0]
    pub_dx1_only = PUBLISHED_RESULTS["sensitivity_conservative_harms_thousands"]
    pub_reduction_pct = (pub_primary - pub_dx1_only) / pub_primary

    my_primary = sim_summary_primary["grand_total_harms_thousands"][0]
    my_dx1_only, my_dx1_lo, my_dx1_hi = sim_summary_dx1_only["grand_total_harms_thousands"]
    my_reduction_pct = (my_primary - my_dx1_only) / my_primary

    return {
        "published_primary": pub_primary, "published_dx1_only": pub_dx1_only,
        "published_reduction_pct": pub_reduction_pct,
        "rederived_primary": my_primary, "rederived_dx1_only": my_dx1_only,
        "rederived_dx1_only_range": (my_dx1_lo, my_dx1_hi),
        "rederived_reduction_pct": my_reduction_pct,
        "reduction_pct_abs_diff": abs(my_reduction_pct - pub_reduction_pct),
        "dx1_only_relative_difference": (my_dx1_only - pub_dx1_only) / pub_dx1_only,
    }


def apply_thresholds(cfg, results: Dict) -> Dict:
    t = cfg["criteria_thresholds"]
    verdict = {}

    c1 = results["criterion_1"]
    within = c1["relative_difference"].abs() <= t["max_incidence_relative_difference"]
    verdict["criterion_1"] = {
        "share_diseases_within_threshold": float(within.mean()),
        "threshold": t["max_incidence_relative_difference"],
        "min_share": t.get("min_share_diseases_within_threshold", 0.5),
        "met": bool(within.mean() >= t.get("min_share_diseases_within_threshold", 0.5)),
    }

    c2 = results["criterion_2"]
    verdict["criterion_2"] = {
        "spearman_r": c2["spearman_r"], "threshold": t["min_rank_correlation"],
        "met": bool(c2["spearman_r"] >= t["min_rank_correlation"]),
    }

    c34 = results["criterion_3_4"]
    verdict["criterion_3"] = {
        "abs_diff_pp": c34["top5_abs_diff_pp"], "threshold_pp": t["max_concentration_diff_pp"],
        "met": bool(c34["top5_abs_diff_pp"] <= t["max_concentration_diff_pp"]),
    }
    verdict["criterion_4"] = {
        "abs_diff_pp": c34["top15_abs_diff_pp"], "threshold_pp": t["max_concentration_diff_pp"],
        "met": bool(c34["top15_abs_diff_pp"] <= t["max_concentration_diff_pp"]),
    }

    c5 = results["criterion_5"]
    verdict["criterion_5"] = {"per_category": c5.to_dict("records"),
                               "note": "descriptive; category-share stability is reported, "
                                       "not scored against a pass/fail threshold; the published "
                                       "comparison share is a point-estimate reconstruction from "
                                       "Table 1 inputs, not a share the paper reports."}

    c6 = results["criterion_6"]
    verdict["criterion_6"] = {
        "relative_difference": c6["relative_difference"], "threshold": t["max_total_harm_relative_difference"],
        "met": bool(abs(c6["relative_difference"]) <= t["max_total_harm_relative_difference"]),
    }

    c7 = results["criterion_7"]
    verdict["criterion_7"] = {
        metric: {"has_overlap": v["has_overlap"], "overlap_fraction": v["overlap_fraction"]}
        for metric, v in c7.items()
    }
    verdict["criterion_7"]["met"] = bool(all(v["has_overlap"] for v in c7.values()))

    c8 = results["criterion_8"]
    verdict["criterion_8"] = {
        "reduction_pct_abs_diff": c8["reduction_pct_abs_diff"],
        "threshold": t["max_sensitivity_reduction_diff"],
        "met": bool(c8["reduction_pct_abs_diff"] <= t["max_sensitivity_reduction_diff"]),
    }

    n_met = sum(1 for k in ("criterion_1", "criterion_2", "criterion_3", "criterion_4",
                            "criterion_6", "criterion_7", "criterion_8")
                if verdict[k]["met"])
    verdict["n_scored_criteria_met"] = n_met
    verdict["n_scored_criteria_total"] = 7  # criterion 5 is descriptive, not pass/fail
    verdict["min_criteria_required"] = t.get("min_criteria_met", 5)
    verdict["H3_SUPPORTED"] = bool(n_met >= t.get("min_criteria_met", 5))
    verdict["note"] = ("All thresholds in criteria_thresholds are unverified placeholders "
                        "pending literature-derived or committee-approved benchmarks; this "
                        "verdict is only as trustworthy as those thresholds. Per the "
                        "prespecified design, no single criterion (including top-15 "
                        "concentration) determines replication in isolation.")
    return verdict


def criteria_summary(verdict: Dict, results: Dict) -> pd.DataFrame:
    """One row per criterion: the statistic, its threshold, and the result."""
    c34, c5, c7 = results["criterion_3_4"], results["criterion_5"], results["criterion_7"]
    v = verdict
    rows = [
        (1, f"{v['criterion_1']['share_diseases_within_threshold']:.0%} of phenotypes within threshold",
         f"|rel diff| <= {v['criterion_1']['threshold']:.0%} for >= "
         f"{v['criterion_1']['min_share']:.0%} of phenotypes", v["criterion_1"]["met"]),
        (2, f"Spearman r = {v['criterion_2']['spearman_r']:.3f} "
            f"({results['criterion_2']['n_diseases']} NIS-derived phenotypes)",
         f"r >= {v['criterion_2']['threshold']:.2f}", v["criterion_2"]["met"]),
        (3, f"{c34['top5_rederived']:.1%} vs {c34['top5_published']:.1%} "
            f"({c34['top5_abs_diff_pp']:.1f} pp)",
         f"<= {v['criterion_3']['threshold_pp']:.1f} pp", v["criterion_3"]["met"]),
        (4, f"{c34['top15_rederived']:.1%} vs {c34['top15_published']:.1%} "
            f"({c34['top15_abs_diff_pp']:.1f} pp)",
         f"<= {v['criterion_4']['threshold_pp']:.1f} pp", v["criterion_4"]["met"]),
        (5, f"max category shift {c5['diff_pp'].abs().max():.1f} pp; published share in "
            f"interval for {int(c5['published_in_interval'].sum())}/{len(c5)} categories",
         "descriptive", None),
        (6, f"{v['criterion_6']['relative_difference']:+.1%}",
         f"|rel diff| <= {v['criterion_6']['threshold']:.0%}", v["criterion_6"]["met"]),
        (7, f"{sum(1 for x in c7.values() if isinstance(x, dict) and x['has_overlap'])}"
            f"/{sum(1 for x in c7.values() if isinstance(x, dict))} headline intervals overlap",
         "all overlap", v["criterion_7"]["met"]),
        (8, f"DX1-only reduction {results['criterion_8']['rederived_reduction_pct']:.1%} vs "
            f"published {results['criterion_8']['published_reduction_pct']:.1%} "
            f"({v['criterion_8']['reduction_pct_abs_diff'] * 100:.1f} pp)",
         f"<= {v['criterion_8']['threshold'] * 100:.0f} pp", v["criterion_8"]["met"]),
    ]
    return pd.DataFrame([{"criterion": n, "name": CRITERION_NAMES[n], "result": stat,
                          "threshold": thr,
                          "met": "descriptive" if met is None else ("yes" if met else "no")}
                         for n, stat, thr, met in rows])
