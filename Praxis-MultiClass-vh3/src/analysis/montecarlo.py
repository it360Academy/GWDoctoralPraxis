"""Monte Carlo simulation reproducing Newman-Toker et al.'s extrapolation
methodology, substituting this project's own NIS-computed incidence for
the 10 vascular/infectious top-5 diseases in place of the paper's
published incidence, while using the paper's cited error rates, harm
rates, "other"-bucket incidence, and extrapolation factors unchanged.

**Sampling-distribution approximation, stated explicitly**: the paper
does not publish the exact parametric family used per parameter in its R
Monte Carlo code (only that CIs/PRs were used as inputs to "probabilistic
sampling"). This project samples each rate and CI-bearing incidence value
from a triangular distribution (low, point-estimate-as-mode, high) built
directly from the reported range -- a standard, simple, and defensibly
transparent choice for expert-elicited or CI-bounded ranges without a
stated distributional family, but not guaranteed to exactly match the
paper's own (unpublished-in-detail) sampling scheme. This is the single
largest methodological approximation in this replication and is reported
as such.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..features.literature_inputs import (BIG_THREE_ERROR_FRACTION,
                                          BIG_THREE_HARM_FRACTION,
                                          OTHER_BUCKETS,
                                          PUBLISHED_TOP5_HARM_DISEASES,
                                          TABLE1_CANCERS, TABLE1_TOP5)


def _triangular(rng: np.random.Generator, low: float, mode: float, high: float,
                n: int) -> np.ndarray:
    lo, hi = min(low, mode), max(high, mode)
    if lo == hi:
        return np.full(n, mode)
    return rng.triangular(lo, mode, hi, size=n)


@dataclass
class SimulationResult:
    draws: pd.DataFrame          # one row per Monte Carlo draw, all totals
    summary: Dict[str, tuple]    # name -> (median, p2.5, p97.5) in thousands


def save_simulation(sim: "SimulationResult", tables_dir, name: str) -> None:
    """Persist both the draws (already a natural CSV) and the summary
    (a dict of tuples, not naturally tabular) so a later stage run in a
    separate process can reload a complete SimulationResult from disk."""
    import json
    sim.draws.to_csv(tables_dir / f"montecarlo_draws_{name}.csv", index=False)
    with open(tables_dir / f"montecarlo_summary_{name}.json", "w") as fh:
        json.dump(sim.summary, fh, indent=2)


def load_simulation(tables_dir, name: str) -> "SimulationResult":
    import json
    draws = pd.read_csv(tables_dir / f"montecarlo_draws_{name}.csv")
    with open(tables_dir / f"montecarlo_summary_{name}.json") as fh:
        summary = {k: tuple(v) for k, v in json.load(fh).items()}
    return SimulationResult(draws=draws, summary=summary)


def dx1_only_other_scale(avg_primary: pd.DataFrame, avg_prin: pd.DataFrame) -> Dict[str, float]:
    """Scale factors for the "other" vascular/infection buckets under the
    DX1-only definition.

    The paper's principal-only sensitivity analysis (549k; p.112-113,
    supplemental 1-B2) re-measured incidence "for vascular events and
    infections" from principal NIS diagnoses only - which includes the
    large "other" buckets, not just the ten named phenotypes. This project
    does not compute the "other" buckets from NIS, so their DX1-only
    incidence is approximated by applying each category's re-derived
    DX1-only / DX1-or-DX2 incidence ratio across its five named phenotypes.
    Cancers are registry-based and unaffected."""
    p = avg_primary.set_index("disease")["mean_annual_incidence_thousands"]
    q = avg_prin.set_index("disease")["mean_annual_incidence_thousands"]
    out = {}
    for bucket, spec in OTHER_BUCKETS.items():
        if spec["category"] == "cancer":
            continue
        members = [d for d, s in TABLE1_TOP5.items() if s.category == spec["category"]]
        out[bucket] = float(q[members].sum() / p[members].sum())
    return out


def run_simulation(my_incidence: pd.DataFrame, n_draws: int, seed: int,
                   other_incidence_scale: Optional[Dict[str, float]] = None) -> SimulationResult:
    """``my_incidence``: output of incidence.three_year_average (this
    project's own computed 3-year-average
    incidence, in thousands, with CI) for the 10 vascular/infectious
    diseases.

    ``other_incidence_scale``: optional {bucket: factor} multiplying the
    published "other"-bucket incidence (used for the DX1-only run; see
    dx1_only_other_scale)."""
    rng = np.random.default_rng(seed)
    scale = other_incidence_scale or {}
    inc = my_incidence.set_index("disease")

    # incidence draws (thousands) for the 10 computed diseases
    incidence_draws: Dict[str, np.ndarray] = {}
    for disease in TABLE1_TOP5:
        row = inc.loc[disease]
        incidence_draws[disease] = _triangular(
            rng, row["mean_ci_low_thousands"], row["mean_annual_incidence_thousands"],
            row["mean_ci_high_thousands"], n_draws)

    # cancers: fixed incidence (registry census, no sampling uncertainty),
    # per the paper's own treatment
    for disease, spec in TABLE1_CANCERS.items():
        incidence_draws[disease] = np.full(n_draws, spec.published_incidence_thousands)

    # "other" buckets: fixed incidence unless a CI was reported
    for bucket, spec in OTHER_BUCKETS.items():
        f = scale.get(bucket, 1.0)
        if spec.get("incidence_low") is not None:
            incidence_draws[bucket] = f * _triangular(
                rng, spec["incidence_low"], spec["published_incidence_thousands"],
                spec["incidence_high"], n_draws)
        else:
            incidence_draws[bucket] = np.full(n_draws, f * spec["published_incidence_thousands"])

    # error/harm rate draws for the 15 named diseases
    error_draws: Dict[str, np.ndarray] = {}
    harm_draws: Dict[str, np.ndarray] = {}
    for name, spec in {**TABLE1_TOP5, **TABLE1_CANCERS}.items():
        error_draws[name] = _triangular(rng, spec.error_rate_low, spec.error_rate,
                                        spec.error_rate_high, n_draws)
        harm_draws[name] = _triangular(rng, spec.harm_rate_low, spec.harm_rate,
                                       spec.harm_rate_high, n_draws)

    # "other" buckets: harm rate sampled directly; error rate = incidence-
    # weighted mean of that category's top-5 error draws (paper's stated
    # substitution method for diseases with no literature-derivable rate)
    category_diseases = {
        "vascular": [d for d, s in TABLE1_TOP5.items() if s.category == "vascular"],
        "infection": [d for d, s in TABLE1_TOP5.items() if s.category == "infection"],
        "cancer": list(TABLE1_CANCERS),
    }
    for bucket, spec in OTHER_BUCKETS.items():
        cat_diseases = category_diseases[spec["category"]]
        weights = np.array([incidence_draws[d] for d in cat_diseases])   # (k, n_draws)
        errs = np.array([error_draws[d] for d in cat_diseases])
        weighted_error = (weights * errs).sum(axis=0) / weights.sum(axis=0)
        error_draws[bucket] = weighted_error
        harm_draws[bucket] = _triangular(rng, spec["harm_rate_low"], spec["harm_rate"],
                                         spec["harm_rate_high"], n_draws)

    all_names = list(TABLE1_TOP5) + list(TABLE1_CANCERS) + list(OTHER_BUCKETS)
    big3_errors = np.zeros(n_draws)
    big3_harms = np.zeros(n_draws)
    per_disease_harms: Dict[str, np.ndarray] = {}
    for name in all_names:
        disease_errors = incidence_draws[name] * error_draws[name]
        disease_harms = incidence_draws[name] * harm_draws[name]
        big3_errors += disease_errors
        big3_harms += disease_harms
        per_disease_harms[name] = disease_harms

    grand_total_errors = big3_errors / BIG_THREE_ERROR_FRACTION
    grand_total_harms = big3_harms / BIG_THREE_HARM_FRACTION

    named = list(TABLE1_TOP5) + list(TABLE1_CANCERS)
    top15_harms = sum(per_disease_harms[d] for d in named)
    # Top-5 = the five highest-harm named phenotypes *within each draw*, so
    # the concentration share reflects the re-derived ranking rather than
    # assuming the published top five keep their places.
    named_matrix = np.column_stack([per_disease_harms[d] for d in named])
    top5_harms = np.sort(named_matrix, axis=1)[:, -5:].sum(axis=1)
    # The published top five (stroke, sepsis, pneumonia, VTE, lung cancer),
    # held fixed - reported alongside for comparison.
    top5_fixed_harms = sum(per_disease_harms[d] for d in PUBLISHED_TOP5_HARM_DISEASES)

    draws = pd.DataFrame({
        "big3_errors_thousands": big3_errors, "big3_harms_thousands": big3_harms,
        "grand_total_errors_thousands": grand_total_errors,
        "grand_total_harms_thousands": grand_total_harms,
        "top15_harms_thousands": top15_harms, "top5_harms_thousands": top5_harms,
        "top15_share_of_grand_total": top15_harms / grand_total_harms,
        "top5_share_of_grand_total": top5_harms / grand_total_harms,
        "top5_fixed_share_of_grand_total": top5_fixed_harms / grand_total_harms,
    })
    for name in all_names:
        draws[f"harms_{name}_thousands"] = per_disease_harms[name]

    def _pct(col: str) -> tuple:
        v = draws[col]
        return (float(v.median()), float(v.quantile(0.025)), float(v.quantile(0.975)))

    summary = {col: _pct(col) for col in [
        "big3_errors_thousands", "big3_harms_thousands", "grand_total_errors_thousands",
        "grand_total_harms_thousands", "top15_harms_thousands", "top5_harms_thousands",
        "top15_share_of_grand_total", "top5_share_of_grand_total",
        "top5_fixed_share_of_grand_total"]}

    return SimulationResult(draws=draws, summary=summary)


def published_incidence_frame() -> pd.DataFrame:
    """The published Table 1 incidence for the ten vascular/infectious
    phenotypes, in run_simulation's input format. No incidence CI is
    transcribed, so incidence is held fixed and only the error/harm-rate
    uncertainty is sampled - the published structure's counterpart to the
    re-derived simulation, used for consensus clustering."""
    rows = [{"disease": d, "mean_annual_incidence_thousands": s.published_incidence_thousands,
             "mean_ci_low_thousands": s.published_incidence_thousands,
             "mean_ci_high_thousands": s.published_incidence_thousands}
            for d, s in TABLE1_TOP5.items()]
    return pd.DataFrame(rows)
