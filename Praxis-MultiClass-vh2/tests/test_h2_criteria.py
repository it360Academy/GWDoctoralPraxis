from src.reporting import h2_criteria as h2c


def _chance(p_lower, p_higher, min_p=0.01, direction="suppressed_vs_chance"):
    return {
        "p_value_one_sided_lower_than_chance": p_lower,
        "p_value_one_sided_higher_than_chance": p_higher,
        "min_achievable_p": min_p,
        "underpowered": min_p > 0.05,
        "direction": direction,
        "rate_ratio_observed_over_null": 0.5,
        "rate_ratio_ci_lo": 0.1, "rate_ratio_ci_hi": 0.9,
        "rate_diff_observed_minus_null": -0.01,
        "rate_diff_ci_lo": -0.02, "rate_diff_ci_hi": -0.001,
        "expected_weighted_dev_count": 100.0, "observed_weighted_dev_count": 50.0,
    }


def _artifact(p_lower, p_higher, min_p=0.01, direction="suppressed_vs_artifact_null"):
    return {
        "p_value_one_sided_lower_than_artifact_null": p_lower,
        "p_value": p_higher,
        "min_achievable_p": min_p,
        "underpowered": min_p > 0.05,
        "direction": direction,
    }


def test_per_year_verdict_supports_h2_only_with_positive_lower_evidence():
    """The core fix: a year supports H2 ONLY when BOTH nulls show
    significant one-sided-LOWER evidence - not merely 'not significantly
    elevated'."""
    chance = _chance(p_lower=0.01, p_higher=0.99)
    artifact = _artifact(p_lower=0.01, p_higher=0.99)
    out = h2c.per_year_compliance_verdict(chance, artifact)
    assert out["year_supports_H2"] is True
    assert out["year_outcome"] == "supports_H2"
    assert out["chance_outcome"] == "significantly_below_null"


def test_nonsignificant_result_does_NOT_count_as_support():
    """Regression test for the exact bug that was found and fixed: a
    nonsignificant (inconclusive) result must NEVER be read as evidence
    FOR H2, even though it is also not evidence against it."""
    chance = _chance(p_lower=0.40, p_higher=0.60)  # neither tail significant
    artifact = _artifact(p_lower=0.40, p_higher=0.60)
    out = h2c.per_year_compliance_verdict(chance, artifact)
    assert out["year_supports_H2"] is False
    assert out["year_outcome"] == "inconclusive"
    assert out["chance_outcome"] == "inconclusive"


def test_elevated_result_is_reported_as_against_h2_not_merely_unsupported():
    chance = _chance(p_lower=0.99, p_higher=0.001, direction="elevated_vs_chance")
    artifact = _artifact(p_lower=0.99, p_higher=0.001, direction="elevated_vs_artifact_null")
    out = h2c.per_year_compliance_verdict(chance, artifact)
    assert out["year_supports_H2"] is False
    assert out["year_outcome"] == "against_H2"
    assert out["chance_outcome"] == "significantly_above_null"


def test_underpowered_test_is_flagged_and_cannot_pass():
    """min_achievable_p above alpha means even the most extreme possible
    permutation result couldn't reach significance - such a test must be
    flagged underpowered and can never itself support H2."""
    chance = _chance(p_lower=0.06, p_higher=0.94, min_p=0.06)  # too few permutations
    artifact = _artifact(p_lower=0.01, p_higher=0.99, min_p=0.01)
    out = h2c.per_year_compliance_verdict(chance, artifact)
    assert out["chance_underpowered"] is True
    assert out["year_supports_H2"] is False


def test_compliance_verdict_requires_all_years_and_holm_correction():
    per_year = {
        y: h2c.per_year_compliance_verdict(_chance(0.01, 0.99), _artifact(0.01, 0.99))
        for y in ("2021", "2022", "2023")
    }
    stable = {"all_pairs_stable": True}
    out = h2c.compliance_verdict(per_year, stable, stable)
    assert out["H2_SUPPORTED"] is True
    assert out["criterion_all_years_support_H2_holm_corrected"] is True

    unstable = {"all_pairs_stable": False}
    out2 = h2c.compliance_verdict(per_year, unstable, stable)
    assert out2["H2_SUPPORTED"] is False


def test_compliance_verdict_false_and_flagged_if_any_year_elevated():
    per_year = {
        "2021": h2c.per_year_compliance_verdict(_chance(0.01, 0.99), _artifact(0.01, 0.99)),
        "2022": h2c.per_year_compliance_verdict(
            _chance(0.99, 0.001, direction="elevated_vs_chance"),
            _artifact(0.99, 0.001, direction="elevated_vs_artifact_null")),
        "2023": h2c.per_year_compliance_verdict(_chance(0.01, 0.99), _artifact(0.01, 0.99)),
    }
    stable = {"all_pairs_stable": True}
    out = h2c.compliance_verdict(per_year, stable, stable)
    assert out["H2_SUPPORTED"] is False
    assert out["any_year_against_H2"] is True


def test_holm_bonferroni_is_stricter_than_uncorrected():
    """Borderline p-values (e.g. all just under 0.05 uncorrected) should
    be caught by the Holm step-down and NOT produce H2_SUPPORTED=True on
    the corrected criterion, illustrating why the correction matters."""
    per_year = {
        y: h2c.per_year_compliance_verdict(_chance(0.04, 0.96), _artifact(0.04, 0.96))
        for y in ("2021", "2022", "2023")
    }
    stable = {"all_pairs_stable": True}
    out = h2c.compliance_verdict(per_year, stable, stable)
    # uncorrected all pass (0.04 < 0.05) but Holm (alpha/3, alpha/2, alpha)
    # fails the smallest two p-values at 0.04
    assert out["criterion_all_years_support_H2_uncorrected"] is True
    assert out["criterion_all_years_support_H2_holm_corrected"] is False
    assert out["H2_SUPPORTED"] is False


def test_baseline_comparison_reports_ratios_against_both_baselines():
    chance = {"observed_weighted_dev_rate": 0.0010, "null_mean_weighted_dev_rate": 0.0012,
              "rate_ratio_observed_over_null": 0.8333, "rate_ratio_ci_lo": 0.80, "rate_ratio_ci_hi": 0.87}
    artifact = {"observed_weighted_dev_rate": 0.0010, "null_mean_weighted_dev_rate": 0.0008}
    out = h2c.baseline_comparison(chance, artifact)
    assert out["chance_ratio"] == 0.8333 and out["chance_baseline"] == 0.0012
    assert abs(out["artifact_ratio"] - 1.25) < 1e-9          # ABOVE the artifact baseline


def test_baseline_comparison_survives_missing_artifact_control():
    out = h2c.baseline_comparison({"observed_weighted_dev_rate": 0.001},
                                  {"error": "HOSP_NIS not available", "n_permutations": 0})
    assert out["artifact_ratio"] != out["artifact_ratio"]     # NaN, not a crash


def test_artifact_null_does_not_enter_the_decision():
    """Chance-only rule: a year below chance but ABOVE the artifact null
    supports H2; the original both-baselines rule is still reported."""
    per_year = {
        "2021": h2c.per_year_compliance_verdict(
            _chance(0.001, 0.999),
            _artifact(0.999, 0.001, direction="elevated_vs_artifact_null")),
        "2022": h2c.per_year_compliance_verdict(_chance(0.001, 0.999), _artifact(0.001, 0.999)),
        "2023": h2c.per_year_compliance_verdict(_chance(0.001, 0.999), _artifact(0.001, 0.999)),
    }
    assert per_year["2021"]["year_outcome"] == "supports_H2"
    assert per_year["2021"]["year_outcome_prespecified_both_baselines"] == "against_H2"
    stable = {"all_pairs_stable": True}
    out = h2c.compliance_verdict(per_year, stable, stable)
    assert out["H2_SUPPORTED"] is True
    assert out["any_year_against_H2"] is False
    assert out["H2_SUPPORTED_prespecified_both_baselines"] is False
    assert out["per_year_outcome_prespecified_both_baselines"]["2021"] == "against_H2"


def test_underpowered_artifact_null_does_not_flag_the_decision():
    per_year = {y: h2c.per_year_compliance_verdict(_chance(0.001, 0.999),
                                                   _artifact(0.06, 0.94, min_p=0.06))
                for y in ("2021", "2022", "2023")}
    stable = {"all_pairs_stable": True}
    out = h2c.compliance_verdict(per_year, stable, stable)
    assert out["any_year_underpowered"] is False
    assert out["H2_SUPPORTED"] is True
    assert out["H2_SUPPORTED_prespecified_both_baselines"] is False
