"""Negative control 2: coding-artifact null.

Tests a DIFFERENT null than chance_control: not "is DEV more common than
pure random pairing," but "is DEV more common than what systematic
coding artifacts (copy-forward, template inheritance) alone would
produce." Two complementary, deliberately different operationalizations
(per project decision - both are implemented, not just one):

(a) HOSPITAL-PRESERVING SHUFFLE: permute the DX1<->secondary-profile
    pairing only WITHIN each hospital-year, so each hospital's own
    coding "style" (which secondary codes it tends to template/copy
    across patients) is preserved in the null, isolating true clinical
    DX1-secondary pairing from hospital-level artifact tendency. If
    DEVs occur at a similar rate under this within-hospital shuffle as
    they do observed, that's consistent with the DEVs being a hospital-
    level coding-pattern artifact rather than patient-specific.

(b) DUPLICATE-PROFILE HEURISTIC: directly flags hospitalizations whose
    full DX2-25 secondary-code SET is a near-exact duplicate of another
    hospitalization's at the same hospital (a real, observable
    copy-forward/template signature - not simulated), and reports the
    DEV rate specifically within that flagged "artifact-suspect"
    subpopulation versus the rest. If DEV concentrates heavily inside
    the duplicate-profile subpopulation, that is direct evidence for
    (rather than against) a coding-artifact explanation for at least
    part of the observed rate.
*** PARALLELIZED (design a) *** Same reasoning as chance_control.py:
permutations are independent, dispatched across worker processes via
joblib with pre-spawned SeedSequence child seeds for reproducibility
independent of n_jobs/scheduling.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..dev.detector import detect
from ..ontology.excludes1_loader import Excludes1Pair
from .parallel_utils import resolve_n_jobs


def _one_hospital_permutation(seed: int, fixed: pd.DataFrame, sec_block: pd.DataFrame,
                               groups: dict, n: int, pairs_map: Dict[str, List[Excludes1Pair]],
                               sec_cols: List[str], weights: np.ndarray) -> float:
    rng = np.random.default_rng(seed)
    perm_idx = np.arange(n)
    for _, idxs in groups.items():
        if len(idxs) > 1:
            perm_idx[idxs] = rng.permutation(idxs)
    shuffled = pd.concat([fixed, sec_block.iloc[perm_idx].reset_index(drop=True)], axis=1)
    res = detect(shuffled, pairs_map, sec_cols)
    return float(np.average(res.dev, weights=weights)) if weights.sum() else float("nan")


def hospital_preserving_shuffle(df: pd.DataFrame, pairs_map: Dict[str, List[Excludes1Pair]],
                                 sec_cols: List[str], weight_col: str = "DISCWT",
                                 hosp_col: str = "HOSP_NIS", n_permutations: int = 200,
                                 seed: int = 42, n_jobs: int = 1) -> Dict:
    n = len(df)
    weights = df[weight_col].to_numpy(dtype=np.float64)

    if hosp_col not in df.columns or df[hosp_col].isna().all():
        return {"error": f"'{hosp_col}' not available - hospital-preserving "
                          f"shuffle requires HOSP_NIS in the checkpoint.",
                "n_permutations": 0}

    observed = detect(df, pairs_map, sec_cols)
    observed_rate = float(np.average(observed.dev, weights=weights)) if weights.sum() else float("nan")

    fixed = df[["DX1_RAW", "LABEL"]].reset_index(drop=True)
    sec_block = df[sec_cols].reset_index(drop=True)
    hosp = df[hosp_col].reset_index(drop=True)

    # precompute row indices per hospital once
    groups = hosp.groupby(hosp).indices  # {hosp_id: array of positions}

    child_seeds = np.random.SeedSequence(seed).spawn(n_permutations)
    n_jobs = resolve_n_jobs(n_jobs, df, ["DX1_RAW", "LABEL", *sec_cols])
    if n_jobs == 1:
        null_rates = np.array([
            _one_hospital_permutation(cs, fixed, sec_block, groups, n, pairs_map, sec_cols, weights)
            for cs in child_seeds
        ])
    else:
        # pre_dispatch=n_jobs: see chance_control.run()
        null_rates = np.array(Parallel(n_jobs=n_jobs, backend="loky", pre_dispatch="n_jobs")(
            delayed(_one_hospital_permutation)(cs, fixed, sec_block, groups, n, pairs_map, sec_cols, weights)
            for cs in child_seeds
        ))

    valid_null = null_rates[~np.isnan(null_rates)]
    n_at_or_above = int((valid_null >= observed_rate).sum())
    p_value = (n_at_or_above + 1) / (len(valid_null) + 1) if len(valid_null) else float("nan")
    # Lower-tail counterpart - see chance_control.py's identical addition
    # for why this (not the "higher" test above) is what H2 as currently
    # worded actually requires evidence from.
    n_at_or_below = int((valid_null <= observed_rate).sum())
    p_value_lower = (n_at_or_below + 1) / (len(valid_null) + 1) if len(valid_null) else float("nan")
    min_achievable_p = 1 / (n_permutations + 1)

    # Two-sided reading + direction, added alongside the original
    # one-sided-higher test (kept for continuity with the prespecified
    # decision rule) so this control can support EITHER an "elevated
    # above artifact null" reading or a "suppressed below artifact null"
    # (compliance) reading - mirrors chance_control.run()'s reporting,
    # see that module's docstring for the full rationale.
    null_mean = float(np.mean(valid_null)) if len(valid_null) else float("nan")
    share_above = (valid_null >= observed_rate).mean() if len(valid_null) else float("nan")
    share_below = (valid_null <= observed_rate).mean() if len(valid_null) else float("nan")
    p_value_two_sided = (float(min(1.0, 2 * min(share_above, share_below)))
                          if len(valid_null) else float("nan"))
    direction = ("elevated_vs_artifact_null" if observed_rate == observed_rate and null_mean == null_mean
                 and observed_rate > null_mean
                 else "suppressed_vs_artifact_null" if observed_rate == observed_rate and null_mean == null_mean
                 and observed_rate < null_mean
                 else "indeterminate")

    return {
        "observed_weighted_dev_rate": observed_rate,
        "null_mean_weighted_dev_rate": null_mean,
        "null_std_weighted_dev_rate": float(np.std(valid_null)) if len(valid_null) else float("nan"),
        "n_permutations": n_permutations,
        "p_value": p_value,
        "p_value_one_sided_lower_than_artifact_null": p_value_lower,
        "p_value_two_sided": p_value_two_sided,
        "direction": direction,
        "min_achievable_p": min_achievable_p,
        "underpowered": min_achievable_p > 0.05,
        "distinguishable_from_artifact_null_at_0.05": bool(
            p_value == p_value and p_value < 0.05 and min_achievable_p <= 0.05),
        "significantly_below_artifact_null_at_0.05": bool(
            p_value_lower == p_value_lower and p_value_lower < 0.05 and min_achievable_p <= 0.05),
        "distinguishable_from_artifact_null_two_sided_at_0.05": bool(
            p_value_two_sided == p_value_two_sided and p_value_two_sided < 0.05
            and min_achievable_p <= 0.05),
    }


def duplicate_profile_heuristic(df: pd.DataFrame, sec_cols: List[str],
                                 dev: np.ndarray, weight_col: str = "DISCWT",
                                 hosp_col: str = "HOSP_NIS") -> Dict:
    if hosp_col not in df.columns or df[hosp_col].isna().all():
        return {"error": f"'{hosp_col}' not available - duplicate-profile "
                          f"heuristic requires HOSP_NIS in the checkpoint."}

    profile = df[sec_cols].fillna("").astype(str).agg("|".join, axis=1)
    key = df[hosp_col].astype(str) + "::" + profile
    dup_counts = key.value_counts()
    is_duplicate = key.map(dup_counts).to_numpy() > 1
    # a profile of all-blank secondary codes is not a meaningful "template" -
    # exclude it from the duplicate-suspect flag
    non_blank = (profile.str.replace("|", "", regex=False).str.len() > 0).to_numpy()
    suspect = is_duplicate & non_blank

    weights = df[weight_col].to_numpy(dtype=np.float64)

    def _rate(mask):
        if mask.sum() == 0:
            return float("nan"), 0
        w = weights[mask]
        d = dev[mask]
        return (float(np.average(d, weights=w)) if w.sum() else float("nan")), int(mask.sum())

    suspect_rate, n_suspect = _rate(suspect)
    clean_rate, n_clean = _rate(~suspect)

    return {
        "n_duplicate_profile_suspect": n_suspect,
        "share_duplicate_profile_suspect": n_suspect / len(df) if len(df) else float("nan"),
        "dev_rate_in_suspect_subpopulation": suspect_rate,
        "dev_rate_in_non_suspect_subpopulation": clean_rate,
        "rate_ratio_suspect_vs_clean": (suspect_rate / clean_rate
                                          if clean_rate and clean_rate == clean_rate and clean_rate > 0
                                          else float("nan")),
        "interpretation": (
            "If dev_rate_in_suspect_subpopulation is similar to or higher than "
            "dev_rate_in_non_suspect_subpopulation, that is consistent with (does "
            "not rule out) coding-artifact explanations for at least part of the "
            "observed DEV rate. If DEV concentrates almost entirely OUTSIDE the "
            "duplicate-profile subpopulation, that argues against a copy-forward "
            "artifact explanation."
        ),
    }
