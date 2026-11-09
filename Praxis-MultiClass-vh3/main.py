#!/usr/bin/env python
"""H3-BigThree pipeline orchestrator.

H3: Rebuilding disease incidence from NIS 2021-2023 with an independent
extraction pipeline, using the incidence definition established by
Newman-Toker et al. (2024), tests whether the published harm-
concentration pattern among high-risk diseases holds steady.

Phenotype incidence counts DX1 OR DX2 (the published definition: the two
positions are often of equal, competing clinical weight). DX1-only is
computed and simulated separately, and used only for criterion 8.

Stages: inspect -> ingest -> eda -> incidence -> simulate -> structure
        (clustering + anomaly + rank stability) -> criteria -> report

Every stage can be run standalone, in a separate invocation, not just as
part of `--stage all`: each stage that needs a prior stage's output first
checks whether that output was already produced in this same process
(the common case for `--stage all`) and, if not, loads it from the
persisted table/summary file the earlier stage wrote to disk. This
mirrors the checkpointed, resumable design of the H1 and H2 companion
projects in this series.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.analysis import (anomaly, bootstrap_rank, clustering, consensus_clustering, criteria,
                          eda, incidence, ml_validation, sensitivity)
from src.analysis.montecarlo import (dx1_only_other_scale, load_simulation,
                                     published_incidence_frame, run_simulation, save_simulation)
from src.config import load_config
from src.data.ingestion import build_year_checkpoint
from src.data.inspect import inspect_datasets
from src.features.literature_inputs import PUBLISHED_RESULTS, TABLE1_CANCERS, TABLE1_TOP5
from src.logging_utils import dump_runtimes, get_logger, setup_logging, stage_timer
from src.reporting import figures as fig
from src.reporting.report import build_report

STAGES = ["inspect", "ingest", "eda", "incidence", "simulate", "structure", "criteria",
          "threshold_sensitivity", "report"]


def stage_inspect(cfg, log):
    with stage_timer("inspect", log):
        inspect_datasets(cfg)


def stage_ingest(cfg, log, force=False):
    with stage_timer("ingest", log):
        for year in cfg.years.values():
            build_year_checkpoint(cfg, year, force=force)


def stage_eda(cfg, log):
    with stage_timer("eda", log):
        tables = cfg.outputs("tables")
        cohort = eda.cohort_summary(cfg)
        cohort.to_csv(tables / "eda_cohort_summary.csv", index=False)
        demo = eda.phenotype_demographics(cfg)
        demo.to_csv(tables / "eda_phenotype_demographics.csv", index=False)
        coding = eda.coding_intensity_distribution(cfg)
        coding.to_csv(tables / "eda_coding_intensity.csv", index=False)
        payer = eda.payer_distribution(cfg)
        payer.to_csv(tables / "eda_payer_distribution.csv", index=False)

        plots = cfg.outputs("plots")
        fig.plot_cohort_summary(cohort, plots / "eda_cohort_summary.png")
        fig.plot_phenotype_age_sex(demo, plots / "eda_phenotype_age_sex.png")
        fig.plot_coding_intensity(coding, plots / "eda_coding_intensity.png")
        return {"cohort_summary": cohort, "phenotype_demographics": demo}


def _load_eda_cohort_summary(cfg):
    return pd.read_csv(cfg.outputs("tables") / "eda_cohort_summary.csv")


def stage_incidence(cfg, log):
    with stage_timer("incidence", log):
        tables = cfg.outputs("tables")
        annual_primary = incidence.compute_annual_incidence(cfg, "PRIN_OR_SEC1")
        annual_primary.to_csv(tables / "annual_incidence_prin_or_sec1.csv", index=False)
        avg_primary = incidence.three_year_average(annual_primary)
        avg_primary.to_csv(tables / "three_year_avg_incidence_prin_or_sec1.csv", index=False)

        annual_prin = incidence.compute_annual_incidence(cfg, "PRIN_ONLY")
        annual_prin.to_csv(tables / "annual_incidence_prin_only.csv", index=False)
        avg_prin = incidence.three_year_average(annual_prin)
        avg_prin.to_csv(tables / "three_year_avg_incidence_prin_only.csv", index=False)
        return {"avg_primary": avg_primary, "avg_prin": avg_prin}


def _load_incidence(cfg):
    tables = cfg.outputs("tables")
    avg_primary = pd.read_csv(tables / "three_year_avg_incidence_prin_or_sec1.csv")
    avg_prin = pd.read_csv(tables / "three_year_avg_incidence_prin_only.csv")
    return avg_primary, avg_prin


def stage_simulate(cfg, log, avg_primary, avg_prin):
    with stage_timer("simulate", log):
        tables = cfg.outputs("tables")
        n_draws = int(cfg.get("montecarlo", "n_draws", default=10000))
        sim_primary = run_simulation(avg_primary, n_draws, cfg.seed)
        # DX1-only run: the paper's principal-only analysis also lowered the
        # "other" vascular/infection buckets, so scale them to match.
        other_scale = dx1_only_other_scale(avg_primary, avg_prin)
        with open(tables / "dx1_only_other_bucket_scale.json", "w") as fh:
            json.dump(other_scale, fh, indent=2)
        log.info("DX1-only 'other'-bucket incidence scale: %s",
                 {k: round(v, 3) for k, v in other_scale.items()})
        sim_prin = run_simulation(avg_prin, n_draws, cfg.seed + 1, other_incidence_scale=other_scale)
        save_simulation(sim_primary, tables, "primary")
        save_simulation(sim_prin, tables, "principal_only")
        return sim_primary, sim_prin


def _load_simulations(cfg):
    tables = cfg.outputs("tables")
    return load_simulation(tables, "primary"), load_simulation(tables, "principal_only")


def stage_structure(cfg, log, avg_primary, sim_primary):
    with stage_timer("structure", log):
        tables, plots = cfg.outputs("tables"), cfg.outputs("plots")
        published_inc = {n: s.published_incidence_thousands
                         for n, s in {**TABLE1_TOP5, **TABLE1_CANCERS}.items()}
        rederived_inc = criteria.rederived_incidence_map(
            criteria.criterion_1_incidence_relative_difference(avg_primary))

        # ---- Section 8 (primary): consensus hierarchical clustering over MC draws ----
        cc_cfg = cfg.get("consensus_clustering", default={}) or {}
        cc_draws = int(cc_cfg.get("n_draws", 1000))
        sim_published = run_simulation(published_incidence_frame(), cc_draws, cfg.seed + 2)
        cons = consensus_clustering.run_consensus(
            sim_published.draws, sim_primary.draws, n_draws=cc_draws,
            k_range=tuple(cc_cfg.get("k_range", [2, 3, 4, 5])),
            method=cc_cfg.get("linkage", "ward"),
            n_perm=int(cc_cfg.get("n_permutations", 10000)), seed=cfg.seed)
        cons["k_selection"].to_csv(tables / "consensus_k_selection.csv", index=False)
        cons["membership"].to_csv(tables / "consensus_membership.csv", index=False)
        cons["coclustering_published"].to_csv(tables / "consensus_coclustering_published.csv")
        cons["coclustering_rederived"].to_csv(tables / "consensus_coclustering_rederived.csv")
        cons["per_draw_ari"].to_csv(tables / "consensus_per_draw_ari.csv", index=False)
        with open(tables / "consensus_agreement.json", "w") as fh:
            json.dump(cons["agreement"], fh, indent=2)
        fig.plot_coclustering(cons["coclustering_published"], cons["coclustering_rederived"],
                              cons["membership"], plots / "consensus_coclustering.png")
        fig.plot_consensus_ari(cons["per_draw_ari"], cons["agreement"],
                               plots / "consensus_per_draw_ari.png")

        # ---- Section 8 (supporting): single Ward clustering of point estimates ----
        structures = {"published": published_inc, "rederived": rederived_inc}
        persistence = clustering.structural_persistence(structures, seed=cfg.seed)
        persistence.to_csv(tables / "clustering_structural_persistence.csv", index=False)
        membership = clustering.cluster_membership(published_inc, rederived_inc, seed=cfg.seed)
        membership.to_csv(tables / "clustering_membership.csv", index=False)

        Z_pub, labels_pub = clustering.linkage_matrix(published_inc)
        Z_red, labels_red = clustering.linkage_matrix(rederived_inc)
        fig.plot_dendrogram(Z_pub, labels_pub, "Hierarchical clustering: published structure",
                             plots / "dendrogram_published.png")
        fig.plot_dendrogram(Z_red, labels_red, "Hierarchical clustering: re-derived structure",
                             plots / "dendrogram_rederived.png")

        div = anomaly.divergence_scores(published_inc, rederived_inc, seed=cfg.seed)
        div.to_csv(tables / "anomaly_divergence_scores.csv", index=False)
        fig.plot_anomaly_scores(div, plots / "anomaly_scores.png")

        rank_df = bootstrap_rank.rank_stability(sim_primary.draws)
        rank_df.to_csv(tables / "bootstrap_rank_stability.csv", index=False)
        fig.plot_rank_stability(rank_df, plots / "rank_stability.png")
        fig.plot_top5_membership(rank_df, plots / "top5_membership.png")

        # ---- Track 2: ML structural validation (clustering agreement) ----
        n_perm = int(cfg.get("ml_validation", "n_permutations", default=10000))
        k_range = tuple(cfg.get("ml_validation", "k_range", default=[2, 3, 4]))
        ml = ml_validation.run_ml_validation(published_inc, rederived_inc, k_range=k_range,
                                              n_perm=n_perm, seed=cfg.seed)

        ml["k_selection"].to_csv(tables / "ml_k_selection.csv", index=False)
        ml["k_sensitivity"].to_csv(tables / "ml_k_sensitivity.csv", index=False)
        ml["feature_set_sensitivity"].to_csv(tables / "ml_feature_set_sensitivity.csv", index=False)
        ml["leave_one_out_kmeans"].to_csv(tables / "ml_leave_one_out_kmeans.csv", index=False)
        ml["leave_one_out_hierarchical"].to_csv(tables / "ml_leave_one_out_hierarchical.csv", index=False)
        ml["crosstab"].to_csv(tables / "ml_cluster_crosstab.csv")
        ml["feat_published"].to_csv(tables / "ml_feature_matrix_published.csv", index=False)
        ml["feat_rederived"].to_csv(tables / "ml_feature_matrix_rederived.csv", index=False)
        with open(tables / "ml_primary_kmeans.json", "w") as fh:
            json.dump(ml["primary_kmeans"], fh, indent=2)
        with open(tables / "ml_primary_hierarchical.json", "w") as fh:
            json.dump(ml["primary_hierarchical"], fh, indent=2)
        with open(tables / "ml_k_star.json", "w") as fh:
            json.dump({"k_star": ml["k_star"]}, fh, indent=2)

        fig.plot_pca_clusters(ml["pca_published"], ml["pca_rederived"], plots / "ml_pca_clusters.png")
        fig.plot_cluster_crosstab(ml["crosstab"], plots / "ml_cluster_crosstab.png")
        fig.plot_permutation_null(ml["primary_kmeans"], plots / "ml_permutation_null.png", label="K-means")
        fig.plot_k_sensitivity(ml["k_sensitivity"], plots / "ml_k_sensitivity.png")
        fig.plot_leave_one_out(ml["leave_one_out_kmeans"], ml["primary_kmeans"]["observed_ari"],
                                plots / "ml_leave_one_out.png")
        fig.plot_silhouette(ml["silhouette_published"], ml["silhouette_rederived"],
                             plots / "ml_silhouette.png")

        return {"structural_persistence": persistence, "clustering_membership": membership,
                "consensus": {k: cons[k] for k in ("k_selection", "membership", "agreement")},
                "divergence": div, "rank_stability": rank_df, "ml_validation": ml}


def _load_structure(cfg):
    tables = cfg.outputs("tables")
    with open(tables / "ml_primary_kmeans.json") as fh:
        primary_kmeans = json.load(fh)
    with open(tables / "ml_primary_hierarchical.json") as fh:
        primary_hierarchical = json.load(fh)
    with open(tables / "ml_k_star.json") as fh:
        k_star = json.load(fh)["k_star"]
    ml = {
        "k_selection": pd.read_csv(tables / "ml_k_selection.csv"),
        "k_sensitivity": pd.read_csv(tables / "ml_k_sensitivity.csv"),
        "feature_set_sensitivity": pd.read_csv(tables / "ml_feature_set_sensitivity.csv"),
        "leave_one_out_kmeans": pd.read_csv(tables / "ml_leave_one_out_kmeans.csv"),
        "leave_one_out_hierarchical": pd.read_csv(tables / "ml_leave_one_out_hierarchical.csv"),
        "crosstab": pd.read_csv(tables / "ml_cluster_crosstab.csv", index_col=0),
        "feat_published": pd.read_csv(tables / "ml_feature_matrix_published.csv"),
        "feat_rederived": pd.read_csv(tables / "ml_feature_matrix_rederived.csv"),
        "primary_kmeans": primary_kmeans, "primary_hierarchical": primary_hierarchical,
        "k_star": k_star,
    }
    with open(tables / "consensus_agreement.json") as fh:
        consensus_agreement = json.load(fh)
    consensus = {"k_selection": pd.read_csv(tables / "consensus_k_selection.csv"),
                 "membership": pd.read_csv(tables / "consensus_membership.csv"),
                 "agreement": consensus_agreement}
    return {
        "consensus": consensus,
        "structural_persistence": pd.read_csv(tables / "clustering_structural_persistence.csv"),
        "clustering_membership": pd.read_csv(tables / "clustering_membership.csv"),
        "divergence": pd.read_csv(tables / "anomaly_divergence_scores.csv"),
        "rank_stability": pd.read_csv(tables / "bootstrap_rank_stability.csv"),
        "ml_validation": ml,
    }


def stage_criteria(cfg, log, avg_primary, sim_primary, sim_prin):
    with stage_timer("criteria", log):
        tables = cfg.outputs("tables")
        c1 = criteria.criterion_1_incidence_relative_difference(avg_primary)
        c1.to_csv(tables / "criterion_1_incidence_diff.csv", index=False)
        harm = criteria.harm_comparison(c1, sim_primary.draws)
        harm.to_csv(tables / "harm_rank_comparison.csv", index=False)

        c2 = criteria.criterion_2_rank_correlation(harm)
        c34 = criteria.criterion_3_4_concentration_difference(sim_primary.summary)
        c5 = criteria.criterion_5_category_shares(sim_primary.draws)
        c5.to_csv(tables / "criterion_5_category_shares.csv", index=False)
        c6 = criteria.criterion_6_total_harm_difference(sim_primary.summary)
        c7 = criteria.criterion_7_monte_carlo_overlap(sim_primary.summary)
        c8 = criteria.criterion_8_dx1_only_sensitivity(sim_primary.summary, sim_prin.summary)
        scale_path = tables / "dx1_only_other_bucket_scale.json"
        if scale_path.exists():
            with open(scale_path) as fh:
                c8.update({f"dx1_only_scale_{k}": v for k, v in json.load(fh).items()})

        results = {"criterion_1": c1, "harm_comparison": harm, "criterion_2": c2,
                   "criterion_3_4": c34, "criterion_5": c5, "criterion_6": c6,
                   "criterion_7": c7, "criterion_8": c8}

        raw_serializable = {"criterion_2": c2, "criterion_3_4": c34, "criterion_6": c6,
                             "criterion_7": c7, "criterion_8": c8}
        with open(tables / "criteria_raw.json", "w") as fh:
            json.dump(raw_serializable, fh, indent=2, default=str)

        verdict = criteria.apply_thresholds(cfg, results)
        with open(tables / "h3_verdict.json", "w") as fh:
            json.dump(verdict, fh, indent=2, default=str)
        summary = criteria.criteria_summary(verdict, results)
        summary.to_csv(tables / "criteria_summary.csv", index=False)
        results["criteria_summary"] = summary
        log.info("H3 verdict: %s (%d/%d scored criteria met)",
                  "SUPPORTED" if verdict["H3_SUPPORTED"] else "NOT SUPPORTED",
                  verdict["n_scored_criteria_met"], verdict["n_scored_criteria_total"])
        return results, verdict


def _load_criteria(cfg):
    tables = cfg.outputs("tables")
    c1 = pd.read_csv(tables / "criterion_1_incidence_diff.csv")
    c5 = pd.read_csv(tables / "criterion_5_category_shares.csv")
    harm = pd.read_csv(tables / "harm_rank_comparison.csv")
    summary = pd.read_csv(tables / "criteria_summary.csv")
    with open(tables / "criteria_raw.json") as fh:
        raw = json.load(fh)
    with open(tables / "h3_verdict.json") as fh:
        verdict = json.load(fh)
    results = {"criterion_1": c1, "criterion_5": c5, "harm_comparison": harm,
               "criteria_summary": summary, **raw}
    return results, verdict


def stage_threshold_sensitivity(cfg, log, results):
    with stage_timer("threshold_sensitivity", log):
        tables, plots = cfg.outputs("tables"), cfg.outputs("plots")
        grid_df = sensitivity.run_grid_sensitivity(
            cfg, results["criterion_1"], results["criterion_2"], results["criterion_3_4"],
            results["criterion_6"], results["criterion_7"], results["criterion_8"])
        grid_df.to_csv(tables / "threshold_sensitivity_grid.csv", index=False)

        sweep_df = sensitivity.run_strictness_sweep(
            cfg, results["criterion_1"], results["criterion_2"], results["criterion_3_4"],
            results["criterion_6"], results["criterion_7"], results["criterion_8"])
        sweep_df.to_csv(tables / "threshold_sensitivity_sweep.csv", index=False)

        summary = sensitivity.summarize(grid_df)
        min_criteria_met = cfg["criteria_thresholds"].get("min_criteria_met", 5)
        agg_sensitivity = sensitivity.min_criteria_met_sensitivity(grid_df)
        agg_sensitivity.to_csv(tables / "threshold_sensitivity_aggregation_rule.csv", index=False)

        with open(tables / "threshold_sensitivity_summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)

        fig.plot_strictness_sweep(sweep_df, min_criteria_met, plots / "strictness_sweep.png")
        fig.plot_criterion_fragility(summary, plots / "criterion_fragility.png")

        log.info("Threshold sensitivity: %d/%d grid configurations (%.1f%%) yield H3_SUPPORTED "
                  "(baseline thresholds are the grid center, not the only point evaluated)",
                  int(summary["support_rate"] * summary["n_grid_points"]), summary["n_grid_points"],
                  summary["support_rate"] * 100)
        return {"sensitivity_grid": grid_df, "sensitivity_sweep": sweep_df,
                "sensitivity_summary": summary, "sensitivity_aggregation_rule": agg_sensitivity}


def _load_threshold_sensitivity(cfg):
    tables = cfg.outputs("tables")
    grid_df = pd.read_csv(tables / "threshold_sensitivity_grid.csv")
    sweep_df = pd.read_csv(tables / "threshold_sensitivity_sweep.csv")
    agg_df = pd.read_csv(tables / "threshold_sensitivity_aggregation_rule.csv")
    with open(tables / "threshold_sensitivity_summary.json") as fh:
        summary = json.load(fh)
    return {"sensitivity_grid": grid_df, "sensitivity_sweep": sweep_df,
            "sensitivity_summary": summary, "sensitivity_aggregation_rule": agg_df}


def stage_report(cfg, log, ctx):
    with stage_timer("report", log):
        plots = cfg.outputs("plots")
        sim = ctx["sim_primary"]
        fig.plot_incidence_comparison(ctx["criterion_1"], plots / "incidence_comparison.png")
        fig.plot_harm_comparison(ctx["harm_comparison"], plots / "harm_comparison.png")
        fig.plot_rank_comparison(ctx["harm_comparison"], plots / "harm_rank_comparison.png")
        fig.plot_concentration(ctx["criterion_3_4"], plots / "concentration_shares.png")
        fig.plot_category_shares(ctx["criterion_5"], plots / "category_shares.png")
        fig.plot_total_harm_sensitivity(ctx["criterion_6"], ctx["criterion_8"],
                                        plots / "total_harm_sensitivity.png")
        fig.plot_mc_headline(sim.summary, PUBLISHED_RESULTS, plots / "mc_headline.png")
        fig.plot_mc_distribution(sim.draws, "grand_total_harms_thousands",
                                  PUBLISHED_RESULTS["grand_total_harms_thousands"],
                                  plots / "mc_distribution_grand_total_harms.png")

        report_ctx = {k: ctx[k] for k in (
            "verdict", "cohort_summary", "criterion_1", "harm_comparison", "criterion_2",
            "criterion_3_4", "criterion_5", "criterion_6", "criterion_7", "criterion_8",
            "criteria_summary", "consensus", "structural_persistence", "clustering_membership",
            "divergence", "rank_stability", "ml_validation", "sensitivity_summary",
            "sensitivity_aggregation_rule")}
        report_ctx["mc_summary"] = sim.summary
        out = build_report(cfg, report_ctx)
        log.info("Report -> %s", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--stage", choices=["all"] + STAGES, default="all")
    ap.add_argument("--force-ingest", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = setup_logging(cfg.logs_dir, cfg.raw.get("run_name", "h3_bigthree"))

    np.random.seed(cfg.seed)
    order = STAGES if args.stage == "all" else [args.stage]
    ctx: dict = {}
    for stage in order:
        if stage == "inspect":
            stage_inspect(cfg, log)
        elif stage == "ingest":
            stage_ingest(cfg, log, force=args.force_ingest)
        elif stage == "eda":
            ctx.update(stage_eda(cfg, log))
        elif stage == "incidence":
            ctx.update(stage_incidence(cfg, log))
        elif stage == "simulate":
            if "avg_primary" not in ctx:
                ctx["avg_primary"], ctx["avg_prin"] = _load_incidence(cfg)
            ctx["sim_primary"], ctx["sim_prin"] = stage_simulate(cfg, log, ctx["avg_primary"],
                                                                 ctx["avg_prin"])
        elif stage == "structure":
            if "avg_primary" not in ctx:
                ctx["avg_primary"], ctx["avg_prin"] = _load_incidence(cfg)
            if "sim_primary" not in ctx:
                ctx["sim_primary"], ctx["sim_prin"] = _load_simulations(cfg)
            ctx.update(stage_structure(cfg, log, ctx["avg_primary"], ctx["sim_primary"]))
        elif stage == "criteria":
            if "avg_primary" not in ctx:
                ctx["avg_primary"], ctx["avg_prin"] = _load_incidence(cfg)
            if "sim_primary" not in ctx:
                ctx["sim_primary"], ctx["sim_prin"] = _load_simulations(cfg)
            results, verdict = stage_criteria(cfg, log, ctx["avg_primary"], ctx["sim_primary"],
                                              ctx["sim_prin"])
            ctx.update(results)
            ctx["verdict"] = verdict
        elif stage == "threshold_sensitivity":
            if "criterion_1" not in ctx:
                results, verdict = _load_criteria(cfg)
                ctx.update(results)
                ctx["verdict"] = verdict
            ctx.update(stage_threshold_sensitivity(cfg, log, ctx))
        elif stage == "report":
            if "cohort_summary" not in ctx:
                ctx["cohort_summary"] = _load_eda_cohort_summary(cfg)
            if "sim_primary" not in ctx:
                ctx["sim_primary"], ctx["sim_prin"] = _load_simulations(cfg)
            if "verdict" not in ctx:
                results, verdict = _load_criteria(cfg)
                ctx.update(results)
                ctx["verdict"] = verdict
            if "consensus" not in ctx:
                ctx.update(_load_structure(cfg))
            if "sensitivity_summary" not in ctx:
                ctx.update(_load_threshold_sensitivity(cfg))
            stage_report(cfg, log, ctx)

    dump_runtimes(cfg.outputs("tables"))
    log.info("Pipeline finished (stage=%s).", args.stage)


if __name__ == "__main__":
    sys.exit(main())
