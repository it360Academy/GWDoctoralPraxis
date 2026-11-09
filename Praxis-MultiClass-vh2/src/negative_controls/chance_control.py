"""Negative control 1: chance-co-occurrence null (stratified).

Null hypothesis: DX1 and the DX2-25 secondary-diagnosis profile are
paired at random WITHIN clinically-plausible strata - i.e. the observed
DEV rate is no different from what random diagnosis-code pairing would
produce among patients who could plausibly have received that profile.

*** STRATIFIED, NOT A FULL POOL SHUFFLE ***
An earlier version of this control shuffled DX2-25 profiles across the
ENTIRE dataset, with no regard for whether the resulting pairing was
even physiologically possible (e.g. a neonatal-sepsis secondary code
landing on an elderly stroke patient's chart). That inflates the null:
it estimates "how often would this pairing occur if we ignored basic
patient plausibility," not "how often would this pairing occur by
genuine chance among comparable patients." This version shuffles
secondary-diagnosis profiles only WITHIN age-band x sex strata (the same
block-permutation pattern already used for the hospital-preserving
artifact control), so the null reflects a fairer, patient-plausible
notion of "chance."

*** RATE-RATIO + CI REPORTING, NOT JUST A p<0.05 VERDICT ***
A single p-value collapses three genuinely different outcomes (DEVs
occur more than chance / no different from chance / less than chance)
into a binary pass/fail, which under-communicates what was actually
found. This module now reports:
  - the observed weighted DEV rate with its own stratified-bootstrap CI
    (uncertainty in the observed statistic itself, not just the null)
  - the null distribution's mean, std, and percentile CI
  - rate_ratio = observed / null_mean, with its own bootstrap CI
    (built by pairing bootstrap draws of the observed rate against
    resampled draws from the null distribution - propagates uncertainty
    from BOTH sides rather than treating the null as fixed)
  - a two-sided p-value (in addition to the original one-sided
    "higher than chance" p-value, kept for continuity with the
    prespecified H2 decision rule)
*** PARALLELIZED ***
Each permutation is an independent draw with no dependency on any other
permutation, making this an embarrassingly-parallel workload - and
empirically the dominant cost in the whole pipeline (repeated full
detect() calls). Permutations are dispatched across worker PROCESSES
(joblib, loky backend - real parallelism, not threads, since detect() is
pandas/numpy CPU-bound and would be GIL-limited under threading).
Reproducibility is preserved under parallel execution by pre-splitting
the master seed into independent child seeds with numpy's
SeedSequence.spawn() BEFORE dispatch, so the set of permutations drawn
is identical regardless of n_jobs or worker completion order - only
their execution order (never their values) depends on scheduling.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..dev.detector import detect
from ..ontology.excludes1_loader import Excludes1Pair
from .parallel_utils import resolve_n_jobs


def _age_band(age: pd.Series, width: int = 10) -> pd.Series:
    """Coarse age bands (0-9, 10-19, ...) - fine enough to block out
    obviously implausible cross-life-stage pairings (neonatal codes
    landing on elderly patients) without fragmenting strata so finely
    that few permutation partners remain within each one."""
    a = pd.to_numeric(age, errors="coerce")
    band = (a // width * width).clip(lower=0)
    return band.fillna(-1).astype(int)  # -1 = missing age, its own stratum


def _stratum_key(df: pd.DataFrame, strata_cols: Tuple[str, ...], age_band_width: int) -> pd.Series:
    parts = []
    for c in strata_cols:
        if c == "AGE":
            parts.append(_age_band(df["AGE"], age_band_width).astype(str))
        else:
            parts.append(df[c].fillna(-1).astype(str))
    key = parts[0]
    for p in parts[1:]:
        key = key + "|" + p
    return key


def _stratified_permutation_indices(strata: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Permute row positions WITHIN each stratum only - a stratum of size
    1 has nothing to shuffle against and is left in place (contributes no
    information to the null for that row, same as it would for any
    within-stratum permutation test)."""
    n = len(strata)
    perm_idx = np.arange(n)
    for _, idxs in strata.groupby(strata).indices.items():
        if len(idxs) > 1:
            perm_idx[idxs] = rng.permutation(idxs)
    return perm_idx


def _weighted_rate(dev: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(dev, weights=weights)) if weights.sum() > 0 else float("nan")


def _observed_rate_bootstrap(dev: np.ndarray, weights: np.ndarray, n_bootstrap: int,
                              rng: np.random.Generator) -> np.ndarray:
    n = len(dev)
    boots = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots[i] = _weighted_rate(dev[idx], weights[idx])
    return boots


def _one_chance_permutation(seed: int, fixed: pd.DataFrame, sec_block: pd.DataFrame,
                             strata: pd.Series, pairs_map: Dict[str, List[Excludes1Pair]],
                             sec_cols: List[str], weights: np.ndarray) -> float:
    """One permutation's worth of work, run in a worker process. Takes
    plain-ish inputs (not closures over the caller's rng) so it pickles
    cleanly for joblib's loky backend."""
    rng = np.random.default_rng(seed)
    perm_idx = _stratified_permutation_indices(strata, rng)
    shuffled = pd.concat([fixed, sec_block.iloc[perm_idx].reset_index(drop=True)], axis=1)
    res = detect(shuffled, pairs_map, sec_cols)
    return _weighted_rate(res.dev, weights)


def run(df: pd.DataFrame, pairs_map: Dict[str, List[Excludes1Pair]], sec_cols: List[str],
        weight_col: str = "DISCWT", n_permutations: int = 200, n_bootstrap: int = 200,
        strata_cols: Tuple[str, ...] = ("AGE", "FEMALE"), age_band_width: int = 10,
        seed: int = 42, n_jobs: int = 1) -> Dict:
    rng = np.random.default_rng(seed)
    n = len(df)
    weights = df[weight_col].to_numpy(dtype=np.float64)

    observed_result = detect(df, pairs_map, sec_cols)
    observed_dev = observed_result.dev
    observed_rate = _weighted_rate(observed_dev, weights)

    missing_strata_cols = [c for c in strata_cols if c not in df.columns]
    if missing_strata_cols:
        return {"error": f"Stratification columns not available: {missing_strata_cols} - "
                          f"cannot run the stratified chance-null.", "n_permutations": 0}

    strata = _stratum_key(df, strata_cols, age_band_width)
    n_strata = strata.nunique()
    singleton_strata = int((strata.value_counts() == 1).sum())

    sec_block = df[sec_cols].reset_index(drop=True)
    fixed = df[["DX1_RAW", "LABEL"]].reset_index(drop=True)

    # Independent child seeds spawned up front, so the SET of permutations
    # drawn is fixed and reproducible regardless of n_jobs/worker scheduling.
    child_seeds = np.random.SeedSequence(seed).spawn(n_permutations)
    n_jobs = resolve_n_jobs(n_jobs, df, ["DX1_RAW", "LABEL", *sec_cols])
    if n_jobs == 1:
        null_rates = np.array([
            _one_chance_permutation(cs, fixed, sec_block, strata, pairs_map, sec_cols, weights)
            for cs in child_seeds
        ])
    else:
        # pre_dispatch=n_jobs: the default (2*n_jobs) pickles that many
        # multi-GB task payloads into the parent's memory up front.
        null_rates = np.array(Parallel(n_jobs=n_jobs, backend="loky", pre_dispatch="n_jobs")(
            delayed(_one_chance_permutation)(cs, fixed, sec_block, strata, pairs_map, sec_cols, weights)
            for cs in child_seeds
        ))

    valid_null = null_rates[~np.isnan(null_rates)]

    # ONE-SIDED HIGHER p-value: probability, under the null, of a rate at
    # or above what was observed. Small => evidence DEVs are MORE common
    # than chance (a violation/safety-surveillance finding).
    n_at_or_above = int((valid_null >= observed_rate).sum())
    p_value_one_sided = (n_at_or_above + 1) / (len(valid_null) + 1) if len(valid_null) else float("nan")

    # ONE-SIDED LOWER p-value: probability, under the null, of a rate at
    # or BELOW what was observed. Small => evidence DEVs are LESS common
    # than chance - this is the test H2 as currently worded
    # ("...occur less often than expected by chance...") actually needs.
    # A small p_value_one_sided_higher_than_chance is NOT evidence for
    # H2; only a small p_value_one_sided_lower_than_chance is. Failing to
    # reject the "higher" null is a DIFFERENT, weaker claim (merely "not
    # significantly elevated") and must never be read as support for H2 -
    # see h2_criteria.py's per_year_compliance_verdict for where this
    # distinction is enforced.
    n_at_or_below = int((valid_null <= observed_rate).sum())
    p_value_one_sided_lower = (n_at_or_below + 1) / (len(valid_null) + 1) if len(valid_null) else float("nan")

    # two-sided p-value: informative in EITHER direction (elevated or
    # suppressed relative to chance), which is what a rate-ratio framing
    # actually calls for
    share_above = (valid_null >= observed_rate).mean() if len(valid_null) else float("nan")
    share_below = (valid_null <= observed_rate).mean() if len(valid_null) else float("nan")
    p_value_two_sided = float(min(1.0, 2 * min(share_above, share_below))) if len(valid_null) else float("nan")

    min_achievable_p = 1 / (n_permutations + 1)

    # --- effect size: rate ratio with a CI that propagates uncertainty
    # from BOTH the observed rate and the null distribution, rather than
    # treating either as a fixed point estimate
    obs_boot = _observed_rate_bootstrap(observed_dev, weights, n_bootstrap, rng)
    obs_boot = obs_boot[~np.isnan(obs_boot)]
    null_mean = float(np.mean(valid_null)) if len(valid_null) else float("nan")
    null_std = float(np.std(valid_null)) if len(valid_null) else float("nan")

    if len(obs_boot) and len(valid_null):
        null_draws_for_ratio = rng.choice(valid_null, size=len(obs_boot), replace=True)
        # guard against division by (near-)zero null draws
        safe = null_draws_for_ratio > 0
        ratio_draws = obs_boot[safe] / null_draws_for_ratio[safe]
        rate_ratio = observed_rate / null_mean if null_mean else float("nan")
        if len(ratio_draws):
            ratio_ci_lo, ratio_ci_hi = np.quantile(ratio_draws, [0.025, 0.975])
        else:
            ratio_ci_lo = ratio_ci_hi = float("nan")
    else:
        rate_ratio, ratio_ci_lo, ratio_ci_hi = float("nan"), float("nan"), float("nan")

    obs_ci_lo, obs_ci_hi = (np.quantile(obs_boot, [0.025, 0.975]) if len(obs_boot)
                              else (float("nan"), float("nan")))
    null_ci_lo, null_ci_hi = (np.quantile(valid_null, [0.025, 0.975]) if len(valid_null)
                                else (float("nan"), float("nan")))

    # Absolute effect size: observed-minus-expected weighted rate, with a
    # CI built the same paired-bootstrap way as the rate ratio above
    # (propagates uncertainty from both the observed rate AND the null).
    if len(obs_boot) and len(valid_null):
        null_draws_for_diff = rng.choice(valid_null, size=len(obs_boot), replace=True)
        diff_draws = obs_boot - null_draws_for_diff
        rate_diff = observed_rate - null_mean
        diff_ci_lo, diff_ci_hi = np.quantile(diff_draws, [0.025, 0.975])
    else:
        rate_diff, diff_ci_lo, diff_ci_hi = float("nan"), float("nan"), float("nan")

    # Same effect sizes expressed as national-weighted DEV-occurrence
    # counts (not just rates), since a rate ratio/difference alone can be
    # hard to interpret without knowing the population it applies to.
    n_weighted_discharges = float(weights.sum())
    expected_weighted_devs = null_mean * n_weighted_discharges if null_mean == null_mean else float("nan")
    observed_weighted_devs = observed_rate * n_weighted_discharges

    return {
        "stratification": {"strata_cols": list(strata_cols), "age_band_width": age_band_width,
                            "n_strata": int(n_strata), "n_singleton_strata": singleton_strata,
                            "singleton_share": singleton_strata / n_strata if n_strata else float("nan")},
        "observed_weighted_dev_rate": observed_rate,
        "observed_weighted_dev_rate_ci_lo": float(obs_ci_lo),
        "observed_weighted_dev_rate_ci_hi": float(obs_ci_hi),
        "null_mean_weighted_dev_rate": null_mean,
        "null_std_weighted_dev_rate": null_std,
        "null_weighted_dev_rate_ci_lo": float(null_ci_lo),
        "null_weighted_dev_rate_ci_hi": float(null_ci_hi),
        "rate_ratio_observed_over_null": rate_ratio,
        "rate_ratio_ci_lo": float(ratio_ci_lo),
        "rate_ratio_ci_hi": float(ratio_ci_hi),
        "rate_diff_observed_minus_null": float(rate_diff),
        "rate_diff_ci_lo": float(diff_ci_lo),
        "rate_diff_ci_hi": float(diff_ci_hi),
        "expected_weighted_dev_count": float(expected_weighted_devs),
        "observed_weighted_dev_count": float(observed_weighted_devs),
        "direction": ("elevated_vs_chance" if rate_ratio == rate_ratio and rate_ratio > 1
                      else "suppressed_vs_chance" if rate_ratio == rate_ratio and rate_ratio < 1
                      else "indeterminate"),
        "n_permutations": n_permutations,
        "n_bootstrap": n_bootstrap,
        "p_value_one_sided_higher_than_chance": p_value_one_sided,
        "p_value_one_sided_lower_than_chance": p_value_one_sided_lower,
        "p_value_two_sided": p_value_two_sided,
        "min_achievable_p": min_achievable_p,
        "underpowered": min_achievable_p > 0.05,
        "distinguishable_from_chance_at_0.05": bool(
            p_value_one_sided == p_value_one_sided and p_value_one_sided < 0.05
            and min_achievable_p <= 0.05),
        "significantly_below_chance_at_0.05": bool(
            p_value_one_sided_lower == p_value_one_sided_lower and p_value_one_sided_lower < 0.05
            and min_achievable_p <= 0.05),
        "distinguishable_from_chance_two_sided_at_0.05": bool(
            p_value_two_sided == p_value_two_sided and p_value_two_sided < 0.05
            and min_achievable_p <= 0.05),
    }
