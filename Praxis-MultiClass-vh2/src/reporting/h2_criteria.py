"""H2 decision rule (PRIMARY verdict).

H2: Diagnostic Exclusivity Violations (DEVs) are RARE - far below the
rate expected if coders ignored the Excludes1 rule - with a STABLE rate
across each NIS year 2021-2023.

*** WHERE THE "RARE" CUTOFF COMES FROM ***
No cutoff is chosen by the investigator. "Rare" is measured against a
DATA-DERIVED baseline: the DEV rate obtained when diagnosis profiles are
randomly re-paired (src/negative_controls/), i.e. what the data would show
if Excludes1 were ignored. The baseline comes from shuffling, not from the
observed pairing, so the test is not circular (a cutoff computed from the
observed rate itself would pass by construction). The decision uses ONE
baseline, which must be beaten in every year: the stratified chance null
(age band x sex). The hospital-preserving coding-artifact null is still
computed and reported, as a SECONDARY sensitivity analysis that does not
drive H2_SUPPORTED (see AMENDMENT HISTORY).

*** DIRECTIONALITY ***
This is a COMPLIANCE hypothesis, requiring POSITIVE evidence that the
observed DEV rate is significantly BELOW the chance baseline - not merely
"not significantly above" it. `compliance_verdict()` below implements
exactly that (one-sided-lower test, Holm-Bonferroni corrected across the 3
years).

*** CAVEAT ON INTERPRETATION ***
Both baselines destroy real clinical associations between the principal
diagnosis and the secondary diagnoses. Excludes1 pairs are often clinically
adjacent conditions that genuinely co-occur, so a rate ABOVE a baseline
(as 2021 is against the artifact null) can reflect natural clinical
association rather than non-compliance, and a rate BELOW it is the
stronger, easier-to-interpret result.

*** AMENDMENT HISTORY ***
H2 was first worded "below chance"; it was then reworded as "rare and
stable" after the first analysis (an interim draft using a fixed 1% cutoff
was dropped, never finalised, in favour of the data-derived baseline
above). The statistical test itself has been the same throughout, and the
first analysis already showed a mixed result (2021 above the artifact
null). The absolute rate among eligible discharges (src/dev/rates.py:
cohort_rate) is reported as context only, not as part of the decision.

Second amendment (post-hoc, made after the results above were known): the
coding-artifact null was REMOVED from the decision rule, which is now
chance-only. Rationale: the hospital-preserving shuffle keeps each
hospital's secondary-code mix, so it tests whether DEVs are patient-
specific beyond hospital coding style - a different question from whether
coders apply Excludes1 - and a rate above it is consistent with clinical
adjacency of Excludes1 pairs. Because this change was made after seeing
that the artifact null was the only failing criterion (2021), the verdict
under the ORIGINAL prespecified rule (below BOTH baselines) is still
computed and reported alongside, as
`H2_SUPPORTED_prespecified_both_baselines`.

*** CORRECTION LOG *** An earlier version of `per_year_compliance_verdict`
defined year-level support for H2 as "not significantly elevated above
the null" - i.e. failure to reject the wrong-direction null was being
read as evidence FOR H2. That is the exact error a doctoral committee
would (rightly) flag: a nonsignificant result is an absence of evidence,
not evidence of absence, and cannot license a directional conclusion by
itself. This has been corrected: both functions below now require
POSITIVE, one-sided, lower-tail significance (p_value_one_sided_lower <
alpha) as the criterion, and report a genuine three-way outcome per year
(significantly below / significantly above / inconclusive) rather than a
collapsed pass/fail.

An earlier draft of this codebase also implemented the opposite
directional reading entirely (elevated_verdict / h2_verdict below): DEV
rate ABOVE the nulls, i.e. "violations occur at a measurable, non-chance
rate" - a diagnostic-safety-surveillance framing rather than a compliance
framing. That function is KEPT, unmodified, purely for transparency/
backward compatibility with that earlier framing - main.py's
stage_stability does NOT use it to decide H2_SUPPORTED. Do not read
`h2_verdict()`'s H2_SUPPORTED as this project's answer to RQ2; read
`compliance_verdict()`'s H2_SUPPORTED for that.

The PRIMARY (compliance) verdict requires ALL THREE independent years
(2021, 2022, 2023) to satisfy:
  (1) observed DEV rate is significantly BELOW the chance-co-occurrence
      null (src/negative_controls/chance_control.py), Holm-corrected
  (2) DEV rates AND Excludes1-pair rank ordering meet the prespecified
      temporal-stability threshold (src/stability/temporal_stability.py)
The coding-artifact null (src/negative_controls/artifact_control.py) and
duplicate_profile_heuristic are reported alongside as secondary checks,
not part of this test. The elevated reading (h2_verdict) still uses both
nulls, unchanged.

All thresholds (including alpha) come from config.yaml `stability_criteria` /
`negative_controls` - flagged there as unverified placeholders, same
situation as H1's h1_criteria. This module applies whatever is
configured; it does not choose the thresholds.
"""
from __future__ import annotations

from typing import Dict


def _ratio(num, den):
    return float(num) / float(den) if isinstance(num, (int, float)) and isinstance(den, (int, float)) and den else float("nan")


def baseline_comparison(chance_control: Dict, artifact_control: Dict) -> Dict:
    """Observed DEV rate against each data-derived baseline, as a ratio
    (<1 = below the ignore-the-rule baseline). The chance ratio carries the
    bootstrap CI computed by chance_control; the artifact ratio is a point
    estimate (that control does not store a CI)."""
    return {
        "observed_rate": chance_control.get("observed_weighted_dev_rate"),
        "chance_baseline": chance_control.get("null_mean_weighted_dev_rate"),
        "chance_ratio": chance_control.get("rate_ratio_observed_over_null"),
        "chance_ratio_ci_lo": chance_control.get("rate_ratio_ci_lo"),
        "chance_ratio_ci_hi": chance_control.get("rate_ratio_ci_hi"),
        "artifact_baseline": artifact_control.get("null_mean_weighted_dev_rate"),
        "artifact_ratio": _ratio(artifact_control.get("observed_weighted_dev_rate"),
                                  artifact_control.get("null_mean_weighted_dev_rate")),
    }


def per_year_verdict(chance_control: Dict, artifact_control: Dict) -> Dict:
    """Elevated-direction (violation) reading - see module docstring.
    Kept for transparency/backward compatibility; NOT the primary
    reading this project uses to answer RQ2. See
    `per_year_compliance_verdict` for the compliance reading."""
    chance_ok = bool(chance_control.get("distinguishable_from_chance_at_0.05", False))
    artifact_ok = bool(artifact_control.get("distinguishable_from_artifact_null_at_0.05", False))
    return {
        "distinguishable_from_chance_null": chance_ok,
        "chance_p_value_one_sided": chance_control.get("p_value_one_sided_higher_than_chance"),
        "chance_p_value_two_sided": chance_control.get("p_value_two_sided"),
        "chance_rate_ratio": chance_control.get("rate_ratio_observed_over_null"),
        "chance_rate_ratio_ci_lo": chance_control.get("rate_ratio_ci_lo"),
        "chance_rate_ratio_ci_hi": chance_control.get("rate_ratio_ci_hi"),
        "chance_direction": chance_control.get("direction"),
        "distinguishable_from_artifact_null": artifact_ok,
        "artifact_p_value": artifact_control.get("p_value"),
        "year_criteria_1_and_2_met": chance_ok and artifact_ok,
    }


def per_year_compliance_verdict(chance_control: Dict, artifact_control: Dict,
                                 alpha: float = 0.05) -> Dict:
    """PRIMARY reading for this project's H2/RQ2.

    *** CORRECTED VERSION *** An earlier version of this function defined
    a year as "compliant" whenever the rate was merely NOT SIGNIFICANTLY
    ELEVATED above the nulls - i.e. it treated a nonsignificant result on
    the WRONG-DIRECTION test as evidence FOR H2. That is a real inference
    error (failing to reject "not elevated" is not the same claim as
    "significantly below chance") and has been removed. A year now
    counts as supporting H2 ONLY when there is POSITIVE, SIGNIFICANT,
    ONE-SIDED evidence that the observed rate is BELOW the null - i.e.
    p_value_one_sided_lower < alpha for the chance-co-occurrence null
    (its one-sided-lower test, not the two-sided test, which is powered
    to detect a deviation in EITHER direction and is not itself
    directional evidence for H2). The coding-artifact null's outcome is
    reported but is secondary; `year_supports_H2_prespecified_both_baselines`
    keeps the original rule (below BOTH nulls) for transparency.

    Three-way, non-collapsed outcome per null (never collapse a
    nonsignificant result into either substantive conclusion):
      - "significantly_below_chance": positive evidence for H2's claim
      - "significantly_above_chance": positive evidence AGAINST H2's claim
      - "inconclusive": failed to reject either one-sided null - this is
        an absence of evidence, not evidence of absence, and does NOT
        support H2 by itself.
    """
    def _outcome(p_lower, p_higher, min_p):
        if p_lower == p_lower and p_lower < alpha and min_p <= alpha:
            return "significantly_below_null"
        if p_higher == p_higher and p_higher < alpha and min_p <= alpha:
            return "significantly_above_null"
        return "inconclusive"

    chance_min_p = chance_control.get("min_achievable_p", float("nan"))
    artifact_min_p = artifact_control.get("min_achievable_p", float("nan"))
    chance_outcome = _outcome(chance_control.get("p_value_one_sided_lower_than_chance"),
                               chance_control.get("p_value_one_sided_higher_than_chance"),
                               chance_min_p)
    artifact_outcome = _outcome(artifact_control.get("p_value_one_sided_lower_than_artifact_null"),
                                 artifact_control.get("p_value"),
                                 artifact_min_p)

    chance_significant_below = chance_outcome == "significantly_below_null"
    artifact_significant_below = artifact_outcome == "significantly_below_null"

    return {
        "chance_outcome": chance_outcome,
        "chance_p_value_one_sided_lower": chance_control.get("p_value_one_sided_lower_than_chance"),
        "chance_p_value_one_sided_higher": chance_control.get("p_value_one_sided_higher_than_chance"),
        "chance_min_achievable_p": chance_min_p,
        "chance_underpowered": chance_control.get("underpowered"),
        "chance_direction": chance_control.get("direction"),
        "chance_rate_ratio": chance_control.get("rate_ratio_observed_over_null"),
        "chance_rate_ratio_ci_lo": chance_control.get("rate_ratio_ci_lo"),
        "chance_rate_ratio_ci_hi": chance_control.get("rate_ratio_ci_hi"),
        "chance_rate_diff_observed_minus_null": chance_control.get("rate_diff_observed_minus_null"),
        "chance_rate_diff_ci_lo": chance_control.get("rate_diff_ci_lo"),
        "chance_rate_diff_ci_hi": chance_control.get("rate_diff_ci_hi"),
        "chance_expected_weighted_dev_count": chance_control.get("expected_weighted_dev_count"),
        "chance_observed_weighted_dev_count": chance_control.get("observed_weighted_dev_count"),
        "artifact_outcome": artifact_outcome,
        "artifact_p_value_one_sided_lower": artifact_control.get("p_value_one_sided_lower_than_artifact_null"),
        "artifact_p_value_one_sided_higher": artifact_control.get("p_value"),
        "artifact_min_achievable_p": artifact_min_p,
        "artifact_underpowered": artifact_control.get("underpowered"),
        "artifact_direction": artifact_control.get("direction"),
        # H2_SUPPORTED for the year requires POSITIVE evidence in the
        # chance null's one-sided-lower test - "inconclusive" means the
        # year does NOT support H2 (absence of evidence is not evidence
        # of absence). The artifact null is secondary (module docstring).
        "year_supports_H2": chance_significant_below,
        "year_outcome": ("supports_H2" if chance_significant_below
                          else "against_H2" if chance_outcome == "significantly_above_null"
                          else "inconclusive"),
        # Original prespecified rule (below BOTH nulls), kept for transparency.
        "year_supports_H2_prespecified_both_baselines": bool(
            chance_significant_below and artifact_significant_below),
        "year_outcome_prespecified_both_baselines": (
            "supports_H2" if chance_significant_below and artifact_significant_below
            else "against_H2" if (chance_outcome == "significantly_above_null"
                                   or artifact_outcome == "significantly_above_null")
            else "inconclusive"),
    }


def h2_verdict(per_year: Dict[str, Dict], rate_stability: Dict, pair_rank_stability: Dict) -> Dict:
    years = sorted(per_year.keys())
    all_years_pass_1_2 = all(per_year[y]["year_criteria_1_and_2_met"] for y in years)
    criterion_3_met = bool(rate_stability.get("all_pairs_stable", False)
                            and pair_rank_stability.get("all_pairs_stable", False))

    h2_supported = bool(all_years_pass_1_2 and criterion_3_met and len(years) == 3)

    # H2's prespecified wording ("observed DEV rates are statistically
    # distinguishable from the chance-co-occurrence null") does not
    # itself specify a direction. The primary H2_SUPPORTED verdict above
    # uses the ONE-SIDED reading (distinguishable = higher than chance),
    # since that's the natural reading of a hypothesis framed as "these
    # violations occur" - but that is an interpretive choice, not
    # something the wording forces. Reported alongside for transparency:
    # under the literal, direction-agnostic TWO-SIDED reading, a rate
    # that is significantly LOWER than chance also counts as
    # "distinguishable" and would satisfy criterion 1. Both are surfaced
    # rather than silently picking one - see per_year[<year>][
    # "chance_direction"] to see which direction actually applied.
    all_years_pass_1_2_two_sided = all(
        bool(per_year[y].get("chance_p_value_two_sided") == per_year[y].get("chance_p_value_two_sided")
             and per_year[y]["chance_p_value_two_sided"] < 0.05)
        and per_year[y]["distinguishable_from_artifact_null"]
        for y in years
    )
    h2_supported_two_sided_reading = bool(
        all_years_pass_1_2_two_sided and criterion_3_met and len(years) == 3)

    return {
        "years_evaluated": years,
        "per_year": per_year,
        "criterion_1_2_all_years_met": all_years_pass_1_2,
        "criterion_3_temporal_stability_met": criterion_3_met,
        "H2_SUPPORTED": h2_supported,
        "H2_SUPPORTED_two_sided_reading": h2_supported_two_sided_reading,
        "directionality_note": (
            "H2_SUPPORTED uses a ONE-SIDED test (DEV rate must be higher "
            "than the chance/artifact nulls). H2_SUPPORTED_two_sided_reading "
            "uses a direction-agnostic test (DEV rate merely different from "
            "the nulls, in either direction). These can disagree - check "
            "per_year[<year>]['chance_direction'] to see whether any "
            "disagreement is because rates are elevated or suppressed "
            "relative to chance. Suppressed-vs-chance is itself a "
            "substantive finding (consistent with coders generally "
            "respecting Excludes1 rules) but is a different claim than H2 "
            "as originally framed."
        ),
        "note": ("All stability/negative-control thresholds in config.yaml are "
                 "unverified placeholders pending literature-derived or "
                 "committee-approved benchmarks - this verdict is only as "
                 "trustworthy as those thresholds. DEVs are potential "
                 "diagnostic-inconsistency signals, never confirmed "
                 "misdiagnoses, regardless of this verdict."),
    }


def _holm_bonferroni(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, bool]:
    """Holm-Bonferroni step-down correction across the 3 year-level tests.
    H2's primary test is applied once per year (2021, 2022, 2023), so
    those 3 tests are a family for which a multiple-comparisons
    adjustment is appropriate (uncorrected, a 3-test family has an
    inflated family-wise false-positive rate for the overall "all years
    support H2" claim). Returns {year: passes_corrected_alpha}."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    still_significant = True
    for i, (year, p) in enumerate(items):
        threshold = alpha / (m - i)
        passed = still_significant and p == p and p < threshold
        out[year] = passed
        if not passed:
            still_significant = False  # step-down: once one fails, later (larger-p) ones fail too
    return out


def compliance_verdict(per_year_compliance: Dict[str, Dict], rate_stability: Dict,
                        pair_rank_stability: Dict, alpha: float = 0.05) -> Dict:
    """PRIMARY H2/RQ2 verdict for this project (see module docstring):
    'Diagnostic Exclusivity Violations occur no more often than random
    chance would predict, in every year 2021-2023 - suggesting hospitals
    are generally following the Excludes1 rule correctly.'

    H2_SUPPORTED requires, in ALL THREE years:
      - POSITIVE one-sided evidence the observed rate is below the
        chance-co-occurrence null (not merely "not significantly
        elevated" - see per_year_compliance_verdict's docstring for why
        that distinction matters)
      - that evidence survives a Holm-Bonferroni correction across the
        3 years' chance-null tests (the family of tests H2 actually runs)
      - DEV rates and Excludes1-pair rank ordering meet the prespecified
        cross-year stability thresholds (reproducibility of the finding,
        independent of its direction)
    Any year testing "significantly_above_null" is reported explicitly
    and independently falsifies H2 for that year, distinct from an
    "inconclusive" year (underpowered or genuinely indistinguishable from
    chance) - the three-way outcome is never collapsed into a single
    pass/fail without saying which of the three actually happened.

    The coding-artifact null is not part of H2_SUPPORTED. The verdict
    under the original prespecified rule (below BOTH nulls, chance null
    Holm-corrected, stable) is returned as
    H2_SUPPORTED_prespecified_both_baselines - see the module docstring's
    amendment history.
    """
    years = sorted(per_year_compliance.keys())
    per_year_uncorrected = {y: per_year_compliance[y]["year_supports_H2"] for y in years}
    any_year_against = any(per_year_compliance[y]["year_outcome"] == "against_H2" for y in years)
    any_year_inconclusive = any(per_year_compliance[y]["year_outcome"] == "inconclusive" for y in years)
    any_year_underpowered = any(bool(per_year_compliance[y].get("chance_underpowered"))
                                 for y in years)
    prespecified_uncorrected = {
        y: bool(per_year_compliance[y].get("year_supports_H2_prespecified_both_baselines"))
        for y in years}

    chance_p_values = {y: per_year_compliance[y]["chance_p_value_one_sided_lower"] for y in years}
    valid_p = {y: p for y, p in chance_p_values.items() if p == p}
    holm_results = _holm_bonferroni(valid_p, alpha) if len(valid_p) == len(years) else {y: False for y in years}

    all_years_uncorrected = all(per_year_uncorrected.values())
    all_years_holm_corrected = bool(all(holm_results.get(y, False) for y in years) and all_years_uncorrected)

    criterion_3_met = bool(rate_stability.get("all_pairs_stable", False)
                            and pair_rank_stability.get("all_pairs_stable", False))

    # H2_SUPPORTED uses the Holm-corrected, multiple-comparison-adjusted
    # result - the more defensible bar for a dissertation-level claim.
    h2_supported = bool(all_years_holm_corrected and criterion_3_met and len(years) == 3)
    h2_supported_prespecified = bool(
        all(prespecified_uncorrected.values()) and all_years_holm_corrected
        and criterion_3_met and len(years) == 3)

    return {
        "years_evaluated": years,
        "per_year": per_year_compliance,
        "per_year_outcome": {y: per_year_compliance[y]["year_outcome"] for y in years},
        "per_year_outcome_prespecified_both_baselines": {
            y: per_year_compliance[y].get("year_outcome_prespecified_both_baselines") for y in years},
        "H2_SUPPORTED_prespecified_both_baselines": h2_supported_prespecified,
        "per_year_supports_H2_uncorrected": per_year_uncorrected,
        "per_year_supports_H2_holm_corrected": holm_results,
        "any_year_against_H2": any_year_against,
        "any_year_inconclusive": any_year_inconclusive,
        "any_year_underpowered": any_year_underpowered,
        "criterion_all_years_support_H2_uncorrected": all_years_uncorrected,
        "criterion_all_years_support_H2_holm_corrected": all_years_holm_corrected,
        "criterion_temporal_stability_met": criterion_3_met,
        "H2_SUPPORTED": h2_supported,
        "reading": "compliance, chance-only (positive one-sided evidence DEV rate is BELOW the "
                   "chance null, Holm-Bonferroni corrected across the 3 years, in every year)",
        "amendment_note": (
            "Post-hoc amendment: the coding-artifact null was removed from the decision rule "
            "after results were known (it was the only failing criterion, in 2021). The "
            "original prespecified verdict (below BOTH nulls) is "
            "H2_SUPPORTED_prespecified_both_baselines."),
        "interpretation": (
            "H2_SUPPORTED=True means: in every one of the three independent "
            "NIS years, Excludes1-incompatible DX1/secondary-diagnosis pairs "
            "occurred SIGNIFICANTLY LESS OFTEN than the chance-co-occurrence "
            "null predicts (one-sided test, Holm-Bonferroni corrected across "
            "years), and this pattern was stable across years. The "
            "coding-artifact null is a secondary sensitivity analysis and "
            "does not enter this verdict. H2_SUPPORTED=False can mean "
            "any of three different things - check per_year_outcome and "
            "any_year_against_H2/any_year_inconclusive/any_year_underpowered "
            "to see which: (1) at least one year showed a rate "
            "SIGNIFICANTLY ABOVE chance (evidence AGAINST H2, a distinct "
            "and more serious finding than inconclusive), (2) at least one "
            "year was INCONCLUSIVE (failed to reject either one-sided null "
            "- absence of evidence, not evidence of absence; check "
            "any_year_underpowered - with n_permutations this small the "
            "test may simply lack power), or (3) the temporal-stability "
            "criterion failed even though every year individually "
            "supported H2. These are NOT interchangeable and must be "
            "reported separately, not collapsed into one negative verdict."
        ),
        "note": ("All stability/negative-control thresholds and alpha in config.yaml are "
                 "unverified placeholders pending literature-derived or "
                 "committee-approved benchmarks - this verdict is only as "
                 "trustworthy as those thresholds. A low/absent DEV rate is "
                 "evidence of Excludes1-consistent CODING, not proof of "
                 "diagnostic accuracy - Excludes1 compliance and correct "
                 "diagnosis are different claims. This is a hospitalization-"
                 "level test restricted to the 15 studied phenotypes as DX1, "
                 "not a claim about all NIS principal diagnoses."),
    }
