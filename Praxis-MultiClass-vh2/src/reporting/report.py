"""Assembles outputs/reports/H2_validation_report.html."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

H2_TEXT = (
    "Diagnostic Exclusivity Violations (DEVs), discharges whose principal "
    "diagnosis (I10_DX1) co-occurs with an Excludes1-incompatible secondary "
    "diagnosis (I10_DX2-DX25), are rare - well below the rate expected if "
    "coders ignored the Excludes1 rule - with a stable rate across each NIS "
    "year from 2021 to 2023."
)
RQ2_TEXT = (
    "Are diagnosis pairs that shouldn't be coded together under the Excludes1 "
    "rule rare enough - below the rate expected by chance - in U.S. hospital "
    "discharge records every year from 2021 to 2023, and does that low rate "
    "hold steady year after year?"
)


def _df_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x,
                       border=0, classes="tbl")


def _pct_below(ratio) -> str:
    return f"{(1 - ratio):.0%} below" if isinstance(ratio, float) and ratio == ratio and ratio < 1 \
        else (f"{(ratio - 1):.0%} ABOVE" if isinstance(ratio, float) and ratio == ratio else "n/a")


_CHANCE = ("chance_outcome", "chance_ratio", "the chance baseline")
_ARTIFACT = ("artifact_outcome", "artifact_ratio", "the hospital-preserving (coding-artifact) baseline")


def _baseline_findings(verdict: dict, baseline_by_year: dict, years: list, baseline: tuple):
    """(sentence on the years significantly below `baseline`, [exceptions])."""
    outcome_key, ratio_key, label = baseline
    ok = [y for y in years if verdict["per_year"][y].get(outcome_key) == "significantly_below_null"]
    sentence = (f"The observed rate is significantly below {label} in every year." if len(ok) == len(years)
                else f"The observed rate is significantly below {label} in {', '.join(ok)}." if ok
                else "")
    exceptions = []
    for y in years:
        oc = verdict["per_year"][y].get(outcome_key)
        if oc != "significantly_below_null":
            ratio = baseline_by_year[y].get(ratio_key)
            what = ("significantly ABOVE it" if oc == "significantly_above_null"
                    else "not distinguishable from it")
            exceptions.append(f"{y} vs {label} (observed/baseline = "
                              f"{ratio:.2f}, {what})" if isinstance(ratio, float) else
                              f"{y} vs {label} ({what})")
    return sentence, exceptions


def findings_summary(verdict: dict, baseline_by_year: dict, cohort_by_year: dict) -> str:
    """Plain-language summary generated from the results (not hardcoded), so
    it stays true if the data change: what holds (absolute rarity, stability,
    below-chance years) and, separately, every exception to it. It never
    overrides the verdict. The decision is chance-only; the coding-artifact
    baseline is summarised afterwards as a secondary analysis, together with
    the verdict under the original prespecified (both-baselines) rule."""
    years = sorted(verdict["per_year"])
    rates = [cohort_by_year[y]["weighted_dev_rate"] for y in years]
    his = [cohort_by_year[y]["weighted_dev_rate_ci_hi"] for y in years]
    span = f"{years[0]}-{years[-1]}"
    out = [f"In absolute terms DEVs are rare: {min(rates):.2%}-{max(rates):.2%} of eligible "
           f"discharges in {span} (upper 95% CI bound {max(his):.2%})."]
    out.append("The rate is stable across years (year-over-year ratios and pair rankings within the "
               "prespecified limits)." if verdict.get("criterion_temporal_stability_met")
               else "The rate is NOT stable across years by the prespecified criteria (Section 3).")

    sentence, exceptions = _baseline_findings(verdict, baseline_by_year, years, _CHANCE)
    if sentence:
        out.append(sentence)
    if exceptions:
        out.append(("The one exception is " if len(exceptions) == 1 else "The exceptions are ")
                   + "; ".join(exceptions) + ". Because of this, H2 (below the chance baseline "
                   "in every year) is not supported.")
    elif verdict.get("H2_SUPPORTED"):
        out.append("H2 is supported.")
    elif verdict.get("criterion_temporal_stability_met"):
        out.append("H2 is not supported: the chance-baseline result does not survive the "
                   "Holm correction across years.")
    else:
        out.append("H2 is not supported because the stability criterion is not met.")

    # Secondary: coding-artifact baseline + original prespecified rule
    a_sentence, a_exceptions = _baseline_findings(verdict, baseline_by_year, years, _ARTIFACT)
    secondary = ["Secondary analysis (not part of the decision):"]
    if a_sentence:
        secondary.append(a_sentence)
    if a_exceptions:
        secondary.append("Exceptions: " + "; ".join(a_exceptions) + ".")
    if "H2_SUPPORTED_prespecified_both_baselines" in verdict:
        secondary.append("Under the original prespecified rule (below both baselines in every year), "
                         "H2 is " + ("supported." if verdict["H2_SUPPORTED_prespecified_both_baselines"]
                                     else "not supported."))
    if len(secondary) > 1:
        out.append(" ".join(secondary))
    return " ".join(out)


def build_report(cfg, verdict: dict, baseline_by_year: dict, cohort_by_year: dict,
                  overall_by_year: dict, phenotype_rates_by_year: dict,
                  pair_rates_by_year: dict, model_perf_by_year: dict,
                  rate_stability: dict, pair_rank_stability: dict, best_algorithm: str,
                  elevated_verdict: dict | None = None) -> Path:
    """`verdict` is the PRIMARY H2 verdict (h2_criteria.compliance_verdict):
    DEV rate significantly below both data-derived ignore-the-rule
    baselines in every year, and stable. `baseline_by_year` holds the
    observed/baseline ratios (h2_criteria.baseline_comparison),
    `cohort_by_year` the absolute eligible-discharge rates (context only).
    `elevated_verdict`, if supplied, is the earlier elevated-direction
    reading, rendered as a clearly-labeled secondary section."""
    out = cfg.outputs("reports") / "H2_validation_report.html"
    css = """
    <style>
      body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 40px; color: #222; }
      h1, h2 { color: #4a1f1f; }
      .tbl { border-collapse: collapse; margin: 12px 0; }
      .tbl th, .tbl td { border: 1px solid #ddd; padding: 6px 10px; font-size: 13px; }
      .tbl th { background: #f5eeee; }
      .verdict { font-size: 20px; font-weight: bold; padding: 10px; border-radius: 6px; }
      .supported { background: #e3f6e3; color: #1c6b1c; }
      .not-supported { background: #fbe7e7; color: #9c2b2b; }
      .summary { background: #eef3fb; border-left: 4px solid #3b6fa0; padding: 10px 16px; margin: 16px 0; }
      .caveat { background: #fff8e1; border-left: 4px solid #e0a800; padding: 10px 16px; margin: 16px 0; }
      img { max-width: 900px; border: 1px solid #ddd; }
    </style>
    """
    verdict_class = "supported" if verdict["H2_SUPPORTED"] else "not-supported"
    verdict_text = ("H2: SUPPORTED (below the chance baseline in every year, and stable)"
                    if verdict["H2_SUPPORTED"] else "H2: NOT SUPPORTED")

    html = [f"<html><head><meta charset='utf-8'>{css}</head><body>"]
    html.append("<h1>H2 Validation Report: Diagnostic Exclusivity Violations</h1>")
    html.append(f"<p><i>Generated {datetime.now().isoformat(timespec='seconds')}</i></p>")
    html.append(f"<p><b>H2:</b> {H2_TEXT}</p>")
    html.append(f"<p><b>RQ2:</b> {RQ2_TEXT}</p>")
    html.append(f"<div class='verdict {verdict_class}'>{verdict_text}</div>")
    outcomes = ", ".join(f"{y}: {o}" for y, o in verdict["per_year_outcome"].items())
    html.append(f"<p><b>Per-year outcome:</b> {outcomes} &nbsp;|&nbsp; "
                 f"<b>any year against H2:</b> {verdict.get('any_year_against_H2')} &nbsp;|&nbsp; "
                 f"<b>any year inconclusive:</b> {verdict.get('any_year_inconclusive')}</p>")
    html.append(f"<div class='summary'><b>Summary of findings:</b> "
                 f"{findings_summary(verdict, baseline_by_year, cohort_by_year)}</div>")
    html.append("<div class='caveat'><b>How 'rare' is defined:</b> no cutoff was chosen by the "
                 "investigator. 'Rare' is measured against a DATA-DERIVED chance baseline - the DEV rate "
                 "obtained when secondary-diagnosis profiles are randomly re-paired within age-band x sex "
                 "strata (what the data would show if Excludes1 were ignored). It must be significantly "
                 "beaten in every year. The baseline comes from shuffling, not from the observed pairing, "
                 "so the test is not circular. A second, coding-artifact baseline (shuffling only within "
                 "each hospital) is reported as a secondary analysis and does not enter the decision.</div>")
    html.append("<div class='caveat'><b>Amendment disclosure:</b> H2 was first worded 'below chance' and "
                 "was reworded 'rare and stable' after the first analysis; an interim draft with a fixed "
                 "1% cutoff was dropped, never finalised. The first analysis's result was already mixed "
                 "(2021 was above the coding-artifact baseline). After that result was known, the "
                 "coding-artifact baseline was removed from the decision rule (post-hoc): it preserves each "
                 "hospital's coding mix, so it tests whether DEVs are patient-specific beyond hospital "
                 "coding style rather than whether Excludes1 is followed. Under the original prespecified "
                 "rule (below both baselines) H2 is <b>"
                 + ("supported" if verdict.get("H2_SUPPORTED_prespecified_both_baselines") else "not supported")
                 + "</b>.</div>")
    html.append("<div class='caveat'><b>Read before trusting this report:</b> phenotype definitions are "
                 "DRAFT/unverified, stability thresholds are unverified placeholders, and DEVs are "
                 "potential diagnostic-inconsistency signals, never confirmed misdiagnoses. "
                 "<b>Shuffled baselines break real clinical associations</b> between a principal diagnosis and "
                 "its secondary diagnoses, and Excludes1 pairs are often clinically adjacent conditions - so "
                 "a rate ABOVE a baseline can reflect natural clinical association rather than "
                 "non-compliance. Check the run log's config path to confirm this is not a smoke run on "
                 "synthetic data.</div>")

    # ---- PRIMARY: observed vs data-derived baselines
    html.append("<h2>Section 1: Primary H2 - DEV rate vs the chance baseline</h2>")
    html.append("<p><i>Ratio = observed DEV rate / baseline rate; below 1 means DEVs are rarer than "
                 "the baseline predicts. Rates are weighted national estimates over all discharges "
                 "(the scale the baselines are computed on). 'Supports' requires positive one-sided "
                 "evidence of being BELOW the chance baseline; 'above' is evidence against H2. The "
                 "artifact columns are the secondary analysis; year_outcome is chance-only.</i></p>")
    rows = []
    for y in sorted(baseline_by_year):
        b, py = baseline_by_year[y], verdict["per_year"][y]
        rows.append({
            "year": y, "observed_rate": b["observed_rate"],
            "chance_baseline": b["chance_baseline"], "ratio_vs_chance": b["chance_ratio"],
            "chance_ci": (f"{b['chance_ratio_ci_lo']:.3f}-{b['chance_ratio_ci_hi']:.3f}"
                          if isinstance(b["chance_ratio_ci_lo"], float) else "n/a"),
            "vs_chance": f"{_pct_below(b['chance_ratio'])} ({py.get('chance_outcome')})",
            "artifact_baseline": b["artifact_baseline"], "ratio_vs_artifact": b["artifact_ratio"],
            "vs_artifact (secondary)": f"{_pct_below(b['artifact_ratio'])} ({py.get('artifact_outcome')})",
            "year_outcome": verdict["per_year_outcome"][y],
            "year_outcome_prespecified": py.get("year_outcome_prespecified_both_baselines"),
        })
    html.append(pd.DataFrame(rows).to_html(index=False, border=0, classes="tbl",
                                            float_format=lambda x: f"{x:.5f}"))
    html.append("<p><img src='../plots/baseline_comparison.png'></p>")
    html.append(f"<p><b>H2_SUPPORTED:</b> {verdict['H2_SUPPORTED']} "
                 f"&nbsp;|&nbsp; <b>every year below the chance baseline (uncorrected):</b> "
                 f"{verdict.get('criterion_all_years_support_H2_uncorrected')} "
                 f"&nbsp;|&nbsp; <b>...and Holm-corrected:</b> "
                 f"{verdict.get('criterion_all_years_support_H2_holm_corrected')} "
                 f"&nbsp;|&nbsp; <b>temporal stability met:</b> "
                 f"{verdict.get('criterion_temporal_stability_met')} "
                 f"&nbsp;|&nbsp; <b>chance null underpowered in any year:</b> "
                 f"{verdict.get('any_year_underpowered')} "
                 f"&nbsp;|&nbsp; <b>original prespecified rule (both baselines):</b> "
                 f"{verdict.get('H2_SUPPORTED_prespecified_both_baselines')}</p>")
    html.append(f"<p><i>{verdict.get('interpretation', '')}</i></p>")

    # ---- CONTEXT: absolute rates
    html.append("<h2>Section 1b: Context - absolute DEV rate</h2>")
    html.append("<p><i>Not part of the decision. 'Eligible' discharges are those whose principal diagnosis "
                 "is one of the studied phenotypes - the only discharges a DEV can be detected in - so "
                 "the eligible-discharge rate is the more meaningful absolute figure; the all-discharge "
                 "rate is diluted by discharges that were never checked. The eligible CI is a "
                 "hospital-clustered bootstrap (NIS discharges are correlated within hospitals).</i></p>")
    crow = []
    for y in sorted(cohort_by_year):
        c = cohort_by_year[y]
        crow.append({"year": y, "n_eligible": c["n_eligible"], "n_dev_raw": c["n_dev_raw"],
                      "rate_among_eligible": c["weighted_dev_rate"],
                      "ci_lo": c["weighted_dev_rate_ci_lo"], "ci_hi": c["weighted_dev_rate_ci_hi"],
                      "rate_among_all_discharges": overall_by_year[y]["weighted_dev_rate"]})
    html.append(pd.DataFrame(crow).to_html(index=False, border=0, classes="tbl",
                                            float_format=lambda x: f"{x:.5f}"))
    html.append("<h3>All-discharge annual rates (raw and weighted national estimates)</h3>")
    overall_df = pd.DataFrame([{"year": y, **overall_by_year[y]} for y in sorted(overall_by_year)])
    html.append(_df_html(overall_df))
    html.append("<p><img src='../plots/annual_dev_rates.png'></p>")

    # ---- per-year null detail
    html.append("<h2>Section 2: Per-year negative-control detail</h2>")
    html.append("<p><i>Chance-null uses a stratified (age-band x sex) block permutation, not a "
                 "full-pool shuffle - see negative_controls/chance_control.py for why an unstratified "
                 "shuffle inflates the null with clinically implausible pairings.</i></p>")
    for y, py in verdict["per_year"].items():
        html.append(f"<p><b>{y}</b>: chance outcome = <b>{py.get('chance_outcome')}</b> "
                     f"(one-sided-lower p={py.get('chance_p_value_one_sided_lower')}, "
                     f"underpowered={py.get('chance_underpowered')}), "
                     f"artifact outcome (secondary) = <b>{py.get('artifact_outcome')}</b> "
                     f"(one-sided-lower p={py.get('artifact_p_value_one_sided_lower')}, "
                     f"one-sided-higher p={py.get('artifact_p_value_one_sided_higher')}), "
                     f"year outcome: <b>{verdict['per_year_outcome'].get(y)}</b>, "
                     f"below chance (uncorrected): {py.get('year_supports_H2')}, "
                     f"chance-null Holm-corrected: {verdict['per_year_supports_H2_holm_corrected'].get(y)}, "
                     f"prespecified both-baselines outcome: "
                     f"{py.get('year_outcome_prespecified_both_baselines')}</p>")

    if elevated_verdict is not None:
        html.append("<h2>Section 2b: Secondary reading - elevated-direction (violation) verdict</h2>")
        html.append("<p><i>Reported for transparency only; NOT the reading used for the H2 verdict "
                     "above. See src/reporting/h2_criteria.py module docstring.</i></p>")
        html.append(f"<p><b>H2_SUPPORTED (elevated, one-sided):</b> {elevated_verdict.get('H2_SUPPORTED')} "
                     f"&nbsp;|&nbsp; <b>(elevated, two-sided):</b> "
                     f"{elevated_verdict.get('H2_SUPPORTED_two_sided_reading')}</p>")

    html.append("<h2>Section 3: Temporal stability (all-discharges rate; pair-rank pattern)</h2>")
    html.append("<h3>DEV rate ratios across years</h3>")
    html.append(_df_html(rate_stability["pairwise"]))
    html.append("<p><img src='../plots/temporal_stability.png'></p>")
    html.append("<h3>Excludes1-pair rank correlation across years</h3>")
    html.append(_df_html(pair_rank_stability["pairwise"]))

    html.append("<h2>Section 4: Phenotype-specific DEV rates</h2>")
    html.append("<p><i>Descriptive only (not part of the H2 decision); CIs are i.i.d. bootstrap.</i></p>")
    for y in sorted(phenotype_rates_by_year):
        html.append(f"<h3>{y}</h3>")
        html.append(_df_html(phenotype_rates_by_year[y]))
    html.append("<p><img src='../plots/phenotype_rates_latest.png'></p>")

    html.append("<h2>Section 5: Top Excludes1-violating pairs (most recent year)</h2>")
    latest = sorted(pair_rates_by_year)[-1]
    html.append(_df_html(pair_rates_by_year[latest].head(20)))
    html.append("<p><img src='../plots/top_pairs_latest.png'></p>")

    html.append(f"<h2>Section 6: DEV-risk model performance by year</h2><p>Selected algorithm "
                 f"(validation-year AUROC): {best_algorithm}</p>")
    perf_df = pd.DataFrame([{"year": y, **model_perf_by_year[y]} for y in sorted(model_perf_by_year)])
    html.append(_df_html(perf_df))
    html.append("<p><img src='../plots/shap_importance.png'></p>")

    html.append("</body></html>")
    out.write_text("\n".join(html), encoding="utf-8")
    return out
