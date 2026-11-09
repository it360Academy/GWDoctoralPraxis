"""Assembles outputs/reports/H3_validation_report.html.

The report follows the H3 findings structure:
  1. Incidence definition: DX1 or DX2 (DX1-only for criterion 8)
  2. Findings of incidence and harm comparison
  3. Published vs. re-derived mean annual incidence by phenotype
  4. Published vs. re-derived serious harm and rank by phenotype
  5. Concentration and category-level findings (top-5 / top-15 shares)
  6. Total serious-harm and sensitivity findings
  7. Monte Carlo uncertainty findings
  8. Published and re-derived hierarchical clustering
  9. Threshold sensitivity findings
followed by the eight-criterion assessment and, as a supplementary
appendix that does not enter the H3 verdict, the ML clustering-agreement
and isolation-forest analyses.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

H3_TEXT = (
    "Rebuilding disease incidence from NIS 2021-2023 with an independent "
    "extraction pipeline, using the incidence definition established by "
    "Newman-Toker et al. (2024), tests whether the published harm-"
    "concentration pattern among high-risk diseases holds steady."
)


def _fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:,.1f}" if abs(x) >= 100 else f"{x:.4f}"
    return str(x)


def _df_html(df: pd.DataFrame, **kw) -> str:
    return df.to_html(index=False, formatters={c: _fmt for c in df.columns},
                      border=0, classes="tbl", **kw)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_report(cfg, ctx: dict) -> Path:
    out = cfg.outputs("reports") / "H3_validation_report.html"
    css = """
    <style>
      body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 40px; color: #222; max-width: 1150px; }
      h1 { color: #4a1f1f; }
      h2 { color: #4a1f1f; border-bottom: 1px solid #ddd; padding-bottom: 4px; margin-top: 40px; }
      h3 { color: #333; margin-top: 24px; }
      .track-banner { font-size: 15px; font-weight: bold; color: #fff; padding: 10px 16px;
                       border-radius: 6px; margin-top: 44px; }
      .track1 { background: #3b6fa0; }
      .track2 { background: #6a4a8a; }
      .tbl { border-collapse: collapse; margin: 12px 0; font-size: 13px; }
      .tbl th, .tbl td { border: 1px solid #ddd; padding: 6px 10px; }
      .tbl th { background: #f5eeee; }
      .verdict { font-size: 20px; font-weight: bold; padding: 10px; border-radius: 6px; margin: 8px 0; }
      .supported { background: #e3f6e3; color: #1c6b1c; }
      .not-supported { background: #fbe7e7; color: #9c2b2b; }
      .ml-verdict-strong { background: #e3f6e3; color: #1c6b1c; }
      .ml-verdict-moderate { background: #fff8e1; color: #8a6d1f; }
      .ml-verdict-weak, .ml-verdict-chance { background: #fbe7e7; color: #9c2b2b; }
      .caveat { background: #fff8e1; border-left: 4px solid #e0a800; padding: 10px 16px; margin: 16px 0; }
      .definition { background: #eef3f8; border-left: 4px solid #3b6fa0; padding: 10px 16px; margin: 16px 0; }
      .exec-summary { background: #f5f3fa; border: 1px solid #ddd; padding: 16px 20px; border-radius: 6px; }
      img { max-width: 900px; border: 1px solid #ddd; margin: 8px 0; }
    </style>
    """
    v = ctx["verdict"]
    verdict_class = "supported" if v["H3_SUPPORTED"] else "not-supported"
    verdict_text = "H3: SUPPORTED" if v["H3_SUPPORTED"] else "H3: NOT SUPPORTED"
    c1, harm, c2 = ctx["criterion_1"], ctx["harm_comparison"], ctx["criterion_2"]
    c34, c5, c6, c7, c8 = (ctx["criterion_3_4"], ctx["criterion_5"], ctx["criterion_6"],
                           ctx["criterion_7"], ctx["criterion_8"])
    ssum = ctx["sensitivity_summary"]
    t = cfg["criteria_thresholds"]

    ml = ctx["ml_validation"]
    pk = ml["primary_kmeans"]
    ph = ml["primary_hierarchical"]

    def _ml_class(verdict_label: str) -> str:
        return {"strong agreement": "ml-verdict-strong",
                "moderate agreement": "ml-verdict-moderate",
                "weak agreement": "ml-verdict-weak",
                "not distinguishable from chance": "ml-verdict-chance"}.get(verdict_label, "ml-verdict-weak")

    html = [f"<html><head><meta charset='utf-8'><title>H3 Validation Report</title>{css}</head><body>"]
    html.append("<h1>H3 Validation Report: Independent Replication of the Big Three "
                 "Harm-Concentration Pattern</h1>")
    html.append(f"<p><i>Generated {datetime.now().isoformat(timespec='seconds')}</i></p>")
    html.append(f"<p><b>H3:</b> {H3_TEXT}</p>")

    # ---------------------------------------------------------- executive summary
    html.append("<div class='exec-summary'>")
    html.append("<h2 style='margin-top:0; border-bottom:none;'>Executive Summary</h2>")
    html.append(f"<div class='verdict {verdict_class}'>{verdict_text} "
                 f"({v['n_scored_criteria_met']}/{v['n_scored_criteria_total']} scored criteria met; "
                 f"{v['min_criteria_required']} required)</div>")
    html.append(_df_html(ctx["criteria_summary"]))
    html.append("</div>")

    html.append("<div class='caveat'><b>Read before trusting this report:</b> disease "
                 "definitions are DRAFT/unverified ICD-10-CM code lists; all criteria_thresholds "
                 "are unverified placeholders; only the ten vascular/infectious phenotypes have "
                 "independently NIS-derived incidence (the five cancers use published registry "
                 "incidence as a cited input); error rates, harm rates and extrapolation factors "
                 "are the paper's published values.</div>")

    # ---------------------------------------------------------- 1. definition
    html.append("<h2>1. Incidence Definition: DX1 or DX2</h2>")
    html.append("<div class='definition'>A discharge counts toward a phenotype when that "
                 "phenotype is coded in the <b>principal diagnosis (I10_DX1) or the first-listed "
                 "secondary diagnosis (I10_DX2)</b>, matching the incidence definition of "
                 "Newman-Toker et al. (2024): the two positions are often of equal, competing "
                 "clinical weight. All criteria 1-7 use this definition. A DX1-only variant is "
                 "computed and simulated separately and enters only through criterion 8, which "
                 "compares the re-derived DX1-only reduction in total harm with the published "
                 f"reduction ({c8['published_primary']:,.0f}k &rarr; "
                 f"{c8['published_dx1_only']:,.0f}k).</div>")
    html.append(_df_html(ctx["cohort_summary"]))
    html.append("<img src='../plots/eda_cohort_summary.png'>")
    html.append("<img src='../plots/eda_phenotype_age_sex.png'>")
    html.append("<img src='../plots/eda_coding_intensity.png'>")

    # ---------------------------------------------------------- 2. findings
    within = c1["relative_difference"].abs() <= t["max_incidence_relative_difference"]
    lower = int((c1["relative_difference"] < 0).sum())
    ind = harm[harm["independent"].astype(bool)]
    moved = ind[ind["rank_change_10"] != 0]
    html.append("<h2>2. Findings of Incidence and Harm Comparison</h2>")
    html.append("<ul>"
                 f"<li><b>Incidence:</b> {int(within.sum())} of {len(c1)} NIS-derived phenotypes "
                 f"fall within &plusmn;{_pct(t['max_incidence_relative_difference'])} of the published "
                 f"incidence (median relative difference "
                 f"{c1['relative_difference'].median():+.1%}); {lower} of {len(c1)} are below the "
                 f"published value.</li>"
                 f"<li><b>Harm ranking:</b> Spearman r = {c2['spearman_r']:.3f} "
                 f"(Kendall &tau; = {c2['kendall_tau']:.3f}) across the {c2['n_diseases']} "
                 f"NIS-derived phenotypes (cancers excluded: identical by design); "
                 f"{len(moved)} of them change rank (largest shift {c2['max_abs_rank_change_10']} "
                 f"places); {c2['top5_overlap_all15']} of the published top-5 harm phenotypes "
                 f"remain in the re-derived top 5 of all fifteen.</li>"
                 f"<li><b>Concentration:</b> top-5 share {_pct(c34['top5_rederived'])} vs. published "
                 f"{_pct(c34['top5_published'])}; top-15 share {_pct(c34['top15_rederived'])} vs. "
                 f"published {_pct(c34['top15_published'])}.</li>"
                 f"<li><b>Total serious harm:</b> {c6['rederived_median']:,.0f}k re-derived vs. "
                 f"{c6['published_median']:,.0f}k published ({c6['relative_difference']:+.1%}). "
                 f"Restricting to DX1 only lowers the re-derived total by "
                 f"{_pct(c8['rederived_reduction_pct'])} vs. {_pct(c8['published_reduction_pct'])} "
                 f"published.</li>"
                 "</ul>")

    # ---------------------------------------------------------- 3. incidence
    html.append("<h2>3. Published vs. Re-Derived Mean Annual Incidence by Phenotype "
                 "(Criterion 1)</h2>")
    html.append("<p>Three-year (2021-2023) mean of DISCWT-weighted annual incidence, DX1 or "
                 "DX2, in thousands; 95% CI from a discharge-level bootstrap.</p>")
    html.append(_df_html(c1))
    html.append("<img src='../plots/incidence_comparison.png'>")

    # ---------------------------------------------------------- 4. harm & rank
    html.append("<h2>4. Published vs. Re-Derived Serious Harm and Rank by Phenotype "
                 "(Criterion 2)</h2>")
    html.append("<p>Harm = incidence &times; published serious-harm rate. Published harm is the "
                 "point-estimate product on published incidence; re-derived harm uses NIS "
                 "incidence (cancers use registry incidence in both), with Monte Carlo median and "
                 "95% simulation interval. Ranks are by point harm, 1 = highest: "
                 "<code>published_rank</code>/<code>rederived_rank</code> among all fifteen, "
                 "<code>*_rank_10</code> among the ten NIS-derived phenotypes.</p>")
    html.append("<p><b>Criterion 2</b> is the Spearman correlation over the ten NIS-derived "
                 "phenotypes only. The five cancers use published registry incidence in both "
                 "structures, so their harms are identical by design; including them would inflate "
                 "the correlation without testing the independent extraction.</p>")
    html.append(_df_html(harm))
    html.append(_df_html(pd.DataFrame([c2])))
    html.append("<img src='../plots/harm_comparison.png'>")
    html.append("<img src='../plots/harm_rank_comparison.png'>")

    # ---------------------------------------------------------- 5. concentration
    html.append("<h2>5. Concentration and Category-Level Findings (Criteria 3, 4 &amp; 5)</h2>")
    html.append("<h3>Top-5 and top-15 share of total serious harm</h3>")
    html.append("<p>Top-5 share is the share of the grand total contributed by the five "
                 "highest-harm phenotypes <i>within each Monte Carlo draw</i>; "
                 "<code>top5_fixed_published_set_rederived</code> holds the published top five "
                 "(stroke, sepsis, pneumonia, VTE, lung cancer) fixed instead. Top-15 share is the "
                 "share from all fifteen named phenotypes.</p>")
    conc = pd.DataFrame([
        {"measure": "Top-5 share", "published": c34["top5_published"],
         "rederived_median": c34["top5_rederived"], "rederived_p2_5": c34["top5_rederived_p2_5"],
         "rederived_p97_5": c34["top5_rederived_p97_5"], "abs_diff_pp": c34["top5_abs_diff_pp"],
         "published_in_interval": c34["top5_published_in_interval"],
         "threshold_pp": t["max_concentration_diff_pp"], "met": v["criterion_3"]["met"]},
        {"measure": "Top-15 share", "published": c34["top15_published"],
         "rederived_median": c34["top15_rederived"], "rederived_p2_5": c34["top15_rederived_p2_5"],
         "rederived_p97_5": c34["top15_rederived_p97_5"], "abs_diff_pp": c34["top15_abs_diff_pp"],
         "published_in_interval": c34["top15_published_in_interval"],
         "threshold_pp": t["max_concentration_diff_pp"], "met": v["criterion_4"]["met"]},
    ])
    html.append(_df_html(conc))
    if "top5_fixed_published_set_rederived" in c34:
        html.append(f"<p>Published top-5 set held fixed: "
                     f"{_pct(c34['top5_fixed_published_set_rederived'])} of re-derived total harm.</p>")
    html.append("<img src='../plots/concentration_shares.png'>")
    html.append("<h3>Category-level burden shares (criterion 5, descriptive)</h3>")
    html.append("<p>Share of grand total harm by Big Three category (named phenotypes plus that "
                 "category's \"other\" bucket). The published comparison share is reconstructed "
                 "from Table 1 point inputs, since the paper does not report category shares "
                 "directly.</p>")
    html.append(_df_html(c5))
    html.append("<img src='../plots/category_shares.png'>")

    # ---------------------------------------------------------- 6. total & sensitivity
    html.append("<h2>6. Total Serious-Harm and Sensitivity Findings (Criteria 6 &amp; 8)</h2>")
    html.append("<h3>Total serious harm vs. published primary estimate (criterion 6)</h3>")
    html.append(_df_html(pd.DataFrame([c6])))
    html.append("<h3>DX1-only sensitivity comparison (criterion 8)</h3>")
    html.append("<p>Percentage reduction in grand total serious harm when incidence is restricted "
                 "from DX1-or-DX2 to DX1 only, re-derived vs. published; scored on the absolute "
                 "difference between the two reductions. The benchmark is the paper's own "
                 "principal-diagnosis-only sensitivity analysis (p.113: &ldquo;Using only principal "
                 "diagnosis NIS codes &hellip; gave lower estimates by about 30% (grand totals 1.78 M "
                 "missed diagnoses and 549000 serious harms)&rdquo;). That analysis re-measured all "
                 "vascular and infection incidence, including the &ldquo;other&rdquo; buckets, so the "
                 "re-derived DX1-only run scales the published &ldquo;other&rdquo; vascular/infection "
                 "incidence by each category's re-derived DX1-only / DX1-or-DX2 ratio "
                 "(<code>dx1_only_scale_*</code>). <code>dx1_only_relative_difference</code> is "
                 "reported for context, not scored.</p>")
    html.append(_df_html(pd.DataFrame([c8])))
    html.append("<img src='../plots/total_harm_sensitivity.png'>")

    # ---------------------------------------------------------- 7. monte carlo
    html.append("<h2>7. Monte Carlo Uncertainty Findings (Criterion 7)</h2>")
    mc = pd.DataFrame([{"metric": k, "median": m, "p2_5": lo, "p97_5": hi}
                       for k, (m, lo, hi) in ctx["mc_summary"].items()])
    html.append(_df_html(mc))
    c7_rows = [{"metric": k, **v_} for k, v_ in c7.items() if isinstance(v_, dict)]
    html.append("<h3>Overlap of re-derived and published 95% intervals</h3>")
    html.append(_df_html(pd.DataFrame(c7_rows)))
    html.append("<img src='../plots/mc_headline.png'>")
    html.append("<img src='../plots/mc_distribution_grand_total_harms.png'>")
    html.append("<h3>Rank stability across Monte Carlo draws</h3>")
    html.append(_df_html(ctx["rank_stability"]))
    html.append("<img src='../plots/rank_stability.png'>")
    html.append("<img src='../plots/top5_membership.png'>")

    # ---------------------------------------------------------- 8. clustering
    cons = ctx["consensus"]
    ca = cons["agreement"]
    html.append("<h2>8. Published and Re-Derived Hierarchical Clustering</h2>")
    html.append("<h3>Consensus hierarchical clustering over Monte Carlo draws</h3>")
    html.append(f"<p>Each of {ca['n_draws']:,} Monte Carlo draws per structure is clustered "
                 f"({ca['method']} linkage) on per-phenotype log serious harm; disease category is "
                 "deliberately not a feature, since it is identical in both structures by "
                 "construction. The published structure's draws sample the published harm- and "
                 "error-rate uncertainty at published incidence. Draws are pooled into a "
                 "co-clustering matrix (share of draws in which two phenotypes share a cluster), and "
                 "a consensus clustering is cut from it. Tiers are numbered by harm (1 = highest). "
                 f"k = {ca['k']} was selected by silhouette on the published structure only. "
                 "Structural characterization only; not an H3 criterion.</p>")
    html.append(_df_html(cons["k_selection"]))
    html.append(_df_html(pd.DataFrame([{
        "consensus_ARI_all_15": ca["observed_ari"], "permutation_p": ca["p_value_ari"],
        "null_ARI_mean": ca["null_ari_mean"], "consensus_ARI_10_NIS_derived": ca["ari_10_independent"],
        "per_draw_ARI_median": ca["per_draw_ari_median"],
        "per_draw_ARI_p2_5": ca["per_draw_ari_p2_5"], "per_draw_ARI_p97_5": ca["per_draw_ari_p97_5"],
        "same_tier_all_15": ca["n_same_tier"],
        "same_tier_10_NIS_derived": ca["n_same_tier_10_independent"]}])))
    html.append("<p><code>*_within_tier_coclustering</code>: how often a phenotype lands with the "
                 "rest of its consensus tier (higher = more stable); "
                 "<code>*_max_outside_coclustering</code>: its strongest pull toward another tier "
                 "(lower = more distinct).</p>")
    html.append(_df_html(cons["membership"]))
    html.append("<img src='../plots/consensus_coclustering.png'>")
    html.append("<img src='../plots/consensus_per_draw_ari.png'>")

    html.append("<h3>Supporting view: single Ward clustering of point estimates</h3>")
    html.append("<p>The earlier single-shot comparison: Ward clustering on standardized "
                 "incidence, harm, harm rank and category, cut at k = 4, fitted once to each "
                 "structure's point estimates. Category inflates its agreement; it is retained for "
                 "continuity.</p>")
    html.append(_df_html(ctx["structural_persistence"]))
    html.append(_df_html(ctx["clustering_membership"]))
    html.append("<img src='../plots/dendrogram_published.png'>")
    html.append("<img src='../plots/dendrogram_rederived.png'>")

    # ---------------------------------------------------------- 9. threshold sensitivity
    html.append("<h2>9. Threshold Sensitivity Findings</h2>")
    html.append("<div class='caveat'>Every value in criteria_thresholds is an unverified "
                 "placeholder. This section re-scores the same computed statistics against a grid "
                 "of alternative thresholds to show how much the verdict depends on them.</div>")
    html.append(f"<p><b>{ssum['support_rate']*100:.1f}%</b> of {ssum['n_grid_points']} alternative "
                 f"threshold configurations (full-factorial grid centred on config.yaml) yield "
                 f"H3_SUPPORTED. Criteria met per grid point ranged from {ssum['min_n_met']} to "
                 f"{ssum['max_n_met']} of 7 scored.</p>")
    html.append(_df_html(pd.DataFrame([{"criterion": k.replace("_met", ""), "share_of_grid_met": r}
                                        for k, r in ssum["per_criterion_met_rate"].items()])))
    html.append("<img src='../plots/strictness_sweep.png'>")
    html.append("<img src='../plots/criterion_fragility.png'>")
    html.append("<h3>Sensitivity to the aggregation rule (minimum criteria required)</h3>")
    html.append(_df_html(ctx["sensitivity_aggregation_rule"]))

    # ---------------------------------------------------------- criteria assessment
    html.append("<h2>Criteria Assessment</h2>")
    html.append("<p>H3 is judged jointly on the eight prespecified criteria; criterion 5 is "
                 "descriptive, the other seven are scored.</p>")
    html.append(_df_html(ctx["criteria_summary"]))
    html.append(f"<div class='verdict {verdict_class}'>{verdict_text}</div>")
    html.append(f"<pre>{pd.Series(v).to_string()}</pre>")

    # ============================================================ TRACK 2
    html.append("<div class='track-banner track2'>APPENDIX &mdash; SUPPLEMENTARY ML STRUCTURAL VALIDATION "
                 "(not an H3 criterion)</div>")
    html.append(f"<div class='verdict {_ml_class(pk['verdict'])}'>K-means: {pk['verdict'].upper()} "
                 f"&mdash; ARI = {pk['observed_ari']:.3f}, NMI = {pk['observed_nmi']:.3f}, "
                 f"permutation p = {pk['p_value_ari']:.4f}</div>")
    html.append(f"<div class='verdict {_ml_class(ph['verdict'])}'>Ward hierarchical: "
                 f"{ph['verdict'].upper()} &mdash; ARI = {ph['observed_ari']:.3f}, "
                 f"NMI = {ph['observed_nmi']:.3f}, permutation p = {ph['p_value_ari']:.4f}</div>")
    html.append("<p>The eight criteria ask whether individual summary numbers replicate. This "
                 "appendix asks a structurally different "
                 "question: does the <i>multidimensional grouping</i> of the fifteen phenotypes "
                 "by burden profile - incidence, harm, harm-uncertainty width, and disease "
                 "category jointly, not any one of them in isolation - replicate between the "
                 "published and re-derived structures? Two clusterings are fit completely "
                 "independently (never projecting one dataset's labels onto the other) and "
                 "compared by Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI), "
                 "against a permutation null appropriate to only fifteen units of observation.</p>")

    html.append("<h2>Appendix B.1: Cluster-Count (k) Selection</h2>")
    html.append("<p>k is selected by silhouette score on the <i>published</i> structure only, "
                 "then applied identically to both structures, so k is not chosen in a way that "
                 f"could favor agreement. Selected: <b>k = {ml['k_star']}</b>.</p>")
    html.append(_df_html(ml["k_selection"]))

    html.append("<h2>Appendix B.2: Primary Clustering Agreement</h2>")
    html.append("<h3>K-means</h3>")
    html.append(_df_html(pd.DataFrame([pk])))
    html.append("<h3>Ward Hierarchical</h3>")
    html.append(_df_html(pd.DataFrame([ph])))
    html.append("<img src='../plots/ml_permutation_null.png'>")
    html.append("<img src='../plots/ml_pca_clusters.png'>")
    html.append("<img src='../plots/ml_cluster_crosstab.png'>")
    html.append("<h3>Silhouette structure of each independent clustering</h3>")
    html.append("<img src='../plots/ml_silhouette.png'>")

    html.append("<h2>Appendix B.3: Robustness &mdash; Sensitivity to k and Algorithm</h2>")
    html.append(_df_html(ml["k_sensitivity"]))
    html.append("<img src='../plots/ml_k_sensitivity.png'>")

    html.append("<h2>Appendix B.4: Robustness &mdash; Sensitivity to Feature Set</h2>")
    html.append("<p>Repeats the primary clustering with features added or removed, to test which "
                 "feature groups drive the observed agreement.</p>")
    html.append(_df_html(ml["feature_set_sensitivity"]))

    html.append("<h2>Appendix B.5: Robustness &mdash; Leave-One-Phenotype-Out Stability</h2>")
    html.append("<p>Recomputes primary agreement with each of the fifteen phenotypes held out in "
                 "turn, checking whether the result depends heavily on any single disease.</p>")
    html.append("<h3>K-means</h3>")
    html.append(_df_html(ml["leave_one_out_kmeans"]))
    html.append("<img src='../plots/ml_leave_one_out.png'>")
    html.append("<h3>Ward Hierarchical</h3>")
    html.append(_df_html(ml["leave_one_out_hierarchical"]))

    html.append("<h2>Appendix B.6: Feature Matrices</h2>")
    html.append("<h3>Published structure</h3>")
    html.append(_df_html(ml["feat_published"]))
    html.append("<h3>Re-derived structure</h3>")
    html.append(_df_html(ml["feat_rederived"]))

    html.append("<h2>Appendix B.7: Supporting Evidence &mdash; Isolation-Forest Divergence "
                 "Detection</h2>")
    html.append("<p>Identifies which individual phenotypes diverge most between published and "
                 "re-derived incidence, using the full relative/absolute-difference feature space. "
                 "Exploratory only, not a decision criterion for either track.</p>")
    html.append(_df_html(ctx["divergence"].drop(columns=["category"])))
    html.append("<img src='../plots/anomaly_scores.png'>")

    html.append("<h3>Supplementary-analysis limitations</h3>")
    html.append("<ul>"
                 "<li>n = 15 phenotypes is small for any clustering method; the permutation test "
                 "addresses whether agreement exceeds chance for this sample size only.</li>"
                 "<li>Category membership is a clustering feature and is identical in both "
                 "structures by construction; see Appendix B.4 for agreement with it removed.</li>"
                 "<li>These analyses do not enter the H3_SUPPORTED determination.</li>"
                 "</ul>")

    html.append("</body></html>")
    out.write_text("\n".join(html), encoding="utf-8")
    return out
