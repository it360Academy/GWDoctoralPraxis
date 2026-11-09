from src.reporting.report import findings_summary


def _verdict(chance, artifact, stable=True, supported=False):
    return {"H2_SUPPORTED": supported, "criterion_temporal_stability_met": stable,
            "per_year": {y: {"chance_outcome": chance[y], "artifact_outcome": artifact[y]} for y in chance}}


BELOW, ABOVE = "significantly_below_null", "significantly_above_null"
COHORT = {y: {"weighted_dev_rate": 0.007, "weighted_dev_rate_ci_hi": 0.0075} for y in ("2021", "2022", "2023")}


def _baseline(a2021=1.05):
    return {"2021": {"chance_ratio": 0.84, "artifact_ratio": a2021},
            "2022": {"chance_ratio": 0.76, "artifact_ratio": 0.95},
            "2023": {"chance_ratio": 0.65, "artifact_ratio": 0.82}}


def test_artifact_exception_is_secondary_and_prespecified_verdict_is_disclosed():
    """Chance-only decision: 2021 above the artifact baseline no longer
    blocks H2, but it is still named, as is the original-rule verdict."""
    v = _verdict({"2021": BELOW, "2022": BELOW, "2023": BELOW},
                 {"2021": ABOVE, "2022": BELOW, "2023": BELOW}, supported=True)
    v["H2_SUPPORTED_prespecified_both_baselines"] = False
    text = findings_summary(v, _baseline(), COHORT)
    assert "below the chance baseline in every year" in text and "H2 is supported" in text
    assert "Secondary analysis (not part of the decision)" in text
    assert "2021 vs the hospital-preserving" in text and "significantly ABOVE it" in text and "1.05" in text
    assert "original prespecified rule" in text and "H2 is not supported." in text


def test_chance_exception_blocks_h2():
    v = _verdict({"2021": ABOVE, "2022": BELOW, "2023": BELOW},
                 {"2021": BELOW, "2022": BELOW, "2023": BELOW})
    text = findings_summary(v, _baseline(), COHORT)
    assert "The one exception is 2021 vs the chance baseline" in text
    assert "H2 (below the chance baseline in every year) is not supported" in text


def test_no_exception_and_supported_says_so():
    ok = {"2021": BELOW, "2022": BELOW, "2023": BELOW}
    text = findings_summary(_verdict(ok, ok, supported=True), _baseline(0.9), COHORT)
    assert "H2 is supported" in text
    assert "exception is" not in text and "Exceptions:" not in text


def test_instability_is_reported_not_hidden():
    ok = {"2021": BELOW, "2022": BELOW, "2023": BELOW}
    text = findings_summary(_verdict(ok, ok, stable=False), _baseline(0.9), COHORT)
    assert "NOT stable" in text


def test_inconclusive_year_is_an_exception_too():
    v = _verdict({"2021": "inconclusive", "2022": BELOW, "2023": BELOW}, {"2021": BELOW, "2022": BELOW, "2023": BELOW})
    text = findings_summary(v, _baseline(), COHORT)
    assert "not distinguishable from it" in text and "below the chance baseline in 2022, 2023" in text
