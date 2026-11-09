#!/usr/bin/env python
"""H2-DEV pipeline orchestrator.

H2: Diagnostic Exclusivity Violations (DEVs), discharges whose principal
diagnosis co-occurs with an Excludes1-incompatible secondary diagnosis, are
rare - well below the rate expected if coders ignored the rule - with a
stable rate across each NIS year from 2021 to 2023.

RQ2: Are diagnosis pairs that shouldn't be coded together under the
Excludes1 rule rare enough - below the rate expected by chance - in U.S.
hospital discharge records every year from 2021 to 2023, and does that low
rate hold steady year after year?

A DEV occurs when a hospitalization's principal diagnosis (I10_DX1)
co-occurs with an ICD-10-CM Excludes1-incompatible secondary diagnosis
(DX2-DX25). DEV detection itself is deterministic (src/dev/detector.py).
"Rare" has no investigator-chosen cutoff: the DEV rate is compared with
a data-derived ignore-the-rule baseline (the stratified chance null from
the negative_controls stage) - see src/reporting/h2_criteria.py. The
coding-artifact null is computed and reported as a secondary analysis (it
was removed from the decision rule post-hoc; the original both-baselines
verdict is reported alongside). The absolute rate among eligible
discharges is reported as context.

Stages (run with --stage, or "all" for the full pipeline):
    inspect            Core file column detection
    ingest             per-year raw parquet checkpoints (DX1 + DX2-25 + covariates)
    validate           data-quality gates per year
    ontology_check     load/verify the Excludes1 ontology for every year
    detect             deterministic DEV labeling per year + rate tables
    negative_controls  chance-co-occurrence + coding-artifact nulls, per year
    model              leakage-safe DEV-risk model, dev/validation/replication split
    stability          cross-year temporal reproducibility tests + H2 verdict
    report             HTML report

Each year (2021/2022/2023) is processed INDEPENDENTLY through detect and
negative_controls - never pooled - per H2's design.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import evaluation as ev
from src.analysis import explain
from src.analysis import models as mdl
from src.config import Config, load_config
from src.data.hospital import HOSP_FEATURE_COLS
from src.data.ingestion import build_year_checkpoint, load_year
from src.data.inspect import inspect_datasets
from src.data.validation import validate_year
from src.dev.detector import detect, pairs_by_phenotype
from src.dev.rates import (cohort_rate, overall_rate, pair_rates, phenotype_rates,
                            position_rates)
from src.features.definitions import validate_definitions
from src.logging_utils import dump_runtimes, get_logger, setup_logging, stage_timer
from src.negative_controls import artifact_control, chance_control
from src.ontology.excludes1_loader import Excludes1Pair, load_pairs_csv
from src.reporting import figures as fig
from src.reporting import h2_criteria as h2c
from src.reporting import report as rpt
from src.stability import temporal_stability as ts

STAGES = ["inspect", "ingest", "validate", "ontology_check", "detect",
          "negative_controls", "model", "stability", "report"]

SEC_COLS = [f"SECDX_{i + 1}" for i in range(25)]


# --------------------------------------------------------------------- helpers
def _ontology_path_for_year(cfg: Config, year: str) -> Path:
    fy = cfg["ontology"]["nis_year_to_fy_mapping"][year]
    template = cfg["ontology"]["pairs_csv_template"]
    rel = template.format(fy=fy)
    return cfg.project_root / rel


# ----------------------------------------------------------------------- stages
def stage_inspect(cfg: Config, log):
    with stage_timer("inspect", log):
        validate_definitions()
        inspect_datasets(cfg)


def stage_ingest(cfg: Config, log, force: bool = False):
    with stage_timer("ingest", log):
        for year in cfg.years:
            build_year_checkpoint(cfg, year, force=force)


def stage_validate(cfg: Config, log):
    with stage_timer("validate", log):
        for year in cfg.years:
            validate_year(cfg, year)


def stage_ontology_check(cfg: Config, log) -> dict[str, list[Excludes1Pair]]:
    with stage_timer("ontology_check", log):
        allow_seed = cfg["ontology"].get("allow_seed_ontology", False)
        pairs_by_year = {}
        for year in cfg.years:
            path = _ontology_path_for_year(cfg, year)
            if not path.exists():
                raise FileNotFoundError(
                    f"Excludes1 ontology missing for {year} (expected {path}). Run "
                    f"src/ontology/excludes1_parser.py against the official CMS/CDC "
                    f"ICD-10-CM Tabular XML for the mapped fiscal year, then "
                    f"src/ontology/excludes1_loader.build_ontology - see ontology/SOURCE.md. "
                    f"For a smoke test, run scripts/generate_synthetic_data.py first.")
            if "SEED" in path.name and not allow_seed:
                raise RuntimeError(
                    f"{path} is a SEED/partial ontology (fabricated, smoke-test-only "
                    f"pairs) - refusing to run against it. Set "
                    f"ontology.allow_seed_ontology: true only for smoke testing.")
            pairs = load_pairs_csv(path)
            log.info("[%s] loaded %d Excludes1 pairs from %s", year, len(pairs), path.name)
            pairs_by_year[year] = pairs
        return pairs_by_year


def stage_detect(cfg: Config, log, pairs_by_year=None):
    with stage_timer("detect", log):
        pairs_by_year = pairs_by_year or stage_ontology_check(cfg, log)
        results = {}
        for year in cfg.years:
            df = load_year(cfg, year)
            pmap = pairs_by_phenotype(pairs_by_year[year])
            res = detect(df, pmap, SEC_COLS)

            overall = overall_rate(df, res.dev)
            overall["n_excludes1_pairs_in_ontology"] = len(pairs_by_year[year])
            overall["n_excludes1_pairs_with_any_dx1_cohort_overlap"] = len(pmap)
            cohort = cohort_rate(df, res.dev, seed=cfg.seed)
            ph_rates = phenotype_rates(df, res.dev)
            prs = pair_rates(res.hit_log, df)
            pos_rates = position_rates(res.hit_log)

            pd.DataFrame([overall]).to_csv(cfg.outputs("tables") / f"dev_overall_rate_{year}.csv", index=False)
            pd.DataFrame([cohort]).to_csv(cfg.outputs("tables") / f"dev_cohort_rate_{year}.csv", index=False)
            ph_rates.to_csv(cfg.outputs("tables") / f"dev_phenotype_rates_{year}.csv", index=False)
            prs.to_csv(cfg.outputs("tables") / f"dev_pair_rates_{year}.csv", index=False)
            pos_rates.to_csv(cfg.outputs("tables") / f"dev_position_rates_{year}.csv", index=False)

            log.info("[%s] raw_dev_rate=%.6f weighted_dev_rate=%.6f (n=%d, n_hosp_weighted=%.0f)",
                      year, overall["raw_dev_rate"], overall["weighted_dev_rate"],
                      len(df), overall["n_hospitalizations"])
            log.info("[%s] eligible-cohort weighted_dev_rate=%.6f 95%% CI (%.6f-%.6f) "
                      "(n_eligible=%d, %s)", year, cohort["weighted_dev_rate"],
                      cohort["weighted_dev_rate_ci_lo"], cohort["weighted_dev_rate_ci_hi"],
                      cohort["n_eligible"], cohort["ci_method"])

            results[year] = {"dev_result": res, "df": df, "overall": overall, "cohort": cohort,
                              "phenotype_rates": ph_rates, "pair_rates": prs,
                              "position_rates": pos_rates}
        return results


def stage_negative_controls(cfg: Config, log, pairs_by_year=None, detect_results=None):
    with stage_timer("negative_controls", log):
        pairs_by_year = pairs_by_year or stage_ontology_check(cfg, log)
        overwrite = cfg.get("checkpointing", "overwrite", default=False)
        out = {}
        pending_years = []
        for year in cfg.years:
            chance_path = cfg.outputs("tables") / f"negative_control_chance_{year}.json"
            artifact_path = cfg.outputs("tables") / f"negative_control_artifact_shuffle_{year}.json"
            dup_path = cfg.outputs("tables") / f"negative_control_artifact_duplicate_{year}.json"
            if not overwrite and chance_path.exists() and artifact_path.exists() and dup_path.exists():
                with open(chance_path) as fh:
                    chance = json.load(fh)
                with open(artifact_path) as fh:
                    artifact = json.load(fh)
                with open(dup_path) as fh:
                    dup = json.load(fh)
                log.info("[%s] negative controls checkpoint exists, skipping (%s)",
                          year, chance_path.name)
                out[year] = {"chance": chance, "artifact": artifact, "duplicate": dup}
            else:
                pending_years.append(year)

        if not pending_years:
            return out

        detect_results = detect_results or stage_detect(cfg, log, pairs_by_year)
        nc_cfg = cfg["negative_controls"]
        n_jobs = int(cfg.get("parallel", "n_jobs", default=1))
        for year in pending_years:
            df = detect_results[year]["df"]
            pmap = pairs_by_phenotype(pairs_by_year[year])

            chance = chance_control.run(
                df, pmap, SEC_COLS,
                n_permutations=nc_cfg["chance"]["n_permutations"],
                n_bootstrap=nc_cfg["chance"].get("n_bootstrap", 200),
                strata_cols=tuple(nc_cfg["chance"].get("strata_cols", ["AGE", "FEMALE"])),
                age_band_width=int(nc_cfg["chance"].get("age_band_width", 10)),
                seed=cfg.seed, n_jobs=n_jobs)
            artifact = artifact_control.hospital_preserving_shuffle(
                df, pmap, SEC_COLS,
                n_permutations=nc_cfg["artifact"]["hospital_shuffle_n_permutations"],
                seed=cfg.seed, n_jobs=n_jobs)
            dup = artifact_control.duplicate_profile_heuristic(
                df, SEC_COLS, detect_results[year]["dev_result"].dev)

            with open(cfg.outputs("tables") / f"negative_control_chance_{year}.json", "w") as fh:
                json.dump(chance, fh, indent=2, default=str)
            with open(cfg.outputs("tables") / f"negative_control_artifact_shuffle_{year}.json", "w") as fh:
                json.dump(artifact, fh, indent=2, default=str)
            with open(cfg.outputs("tables") / f"negative_control_artifact_duplicate_{year}.json", "w") as fh:
                json.dump(dup, fh, indent=2, default=str)

            log.info("[%s] chance: rate_ratio=%s direction=%s two_sided_p=%s | "
                      "artifact: direction=%s two_sided_p=%s | duplicate-suspect share=%s",
                      year, chance.get("rate_ratio_observed_over_null"), chance.get("direction"),
                      chance.get("p_value_two_sided"), artifact.get("direction"),
                      artifact.get("p_value_two_sided"), dup.get("share_duplicate_profile_suspect"))
            out[year] = {"chance": chance, "artifact": artifact, "duplicate": dup}
        return out


def stage_model(cfg: Config, log, detect_results=None):
    with stage_timer("model", log):
        detect_results = detect_results or stage_detect(cfg, log)
        dev_year, val_year, rep_year = (cfg.ml_years["development"],
                                         cfg.ml_years["validation"],
                                         cfg.ml_years["replication"])

        def _xy(year):
            df = detect_results[year]["df"]
            y = detect_results[year]["dev_result"].dev
            return df, y

        df_dev, y_dev = _xy(dev_year)
        X_dev, names, encoder = mdl.build_feature_matrix(df_dev)

        algorithms = cfg["models"]["algorithms"]
        perf_by_algo: dict[str, dict] = {}
        fitted: dict[str, object] = {}
        for algo in algorithms:
            try:
                model = mdl.fit_model(algo, X_dev, y_dev, seed=cfg.seed)
            except ImportError as exc:
                log.warning("Skipping algorithm %s (import failed: %s)", algo, exc)
                continue
            fitted[algo] = model
            perf_by_algo[algo] = {}
            for role, year in (("development", dev_year), ("validation", val_year),
                               ("replication", rep_year)):
                df_r, y_r = _xy(year)
                X_r, _, _ = mdl.build_feature_matrix(df_r, encoder=encoder)
                proba = mdl.predict_proba(model, X_r)
                pred = (proba >= 0.5).astype(int)
                perf_by_algo[algo][role] = ev.evaluate(y_r, pred, proba)

        if not perf_by_algo:
            raise RuntimeError("No model algorithm could be fit - check config.models.algorithms "
                                "and that the corresponding libraries are installed.")

        best_algo = max(perf_by_algo, key=lambda a: perf_by_algo[a]["validation"]["auroc"])
        log.info("Best algorithm by %s-validation AUROC: %s (%.4f)",
                  val_year, best_algo, perf_by_algo[best_algo]["validation"]["auroc"])

        perf_table = pd.DataFrame([
            {"algorithm": a, "role": role, **metrics}
            for a, roles in perf_by_algo.items() for role, metrics in roles.items()
        ])
        perf_table.to_csv(cfg.outputs("tables") / "dev_risk_model_performance.csv", index=False)

        df_rep, y_rep = _xy(rep_year)
        X_rep, _, _ = mdl.build_feature_matrix(df_rep, encoder=encoder)
        native_imp = explain.native_importance(fitted[best_algo], best_algo, names)
        native_imp.to_csv(cfg.outputs("tables") / "dev_risk_native_importance.csv", index=False)
        shap_df = explain.shap_summary(fitted[best_algo], best_algo, X_rep, names) \
            if cfg.get("explainability", "use_shap", default=True) else None
        importance_for_plot = shap_df if shap_df is not None else native_imp
        if shap_df is not None:
            shap_df.to_csv(cfg.outputs("tables") / "dev_risk_shap_summary.csv", index=False)

        cluster_labels, _ = mdl.cluster_dev_patterns(X_dev)
        pd.DataFrame({"row": np.arange(len(cluster_labels)), "cluster": cluster_labels,
                       "dev": y_dev}).to_csv(cfg.outputs("tables") / "dev_unsupervised_clusters.csv",
                                              index=False)

        perf_by_year_keyed = {dev_year: perf_by_algo[best_algo]["development"],
                               val_year: perf_by_algo[best_algo]["validation"],
                               rep_year: perf_by_algo[best_algo]["replication"]}

        return {"best_algorithm": best_algo, "perf_by_algo": perf_by_algo,
                "perf_by_year": perf_by_year_keyed, "importance": importance_for_plot}


def stage_stability(cfg: Config, log, detect_results=None, nc_results=None, model_results=None):
    with stage_timer("stability", log):
        detect_results = detect_results or stage_detect(cfg, log)
        nc_results = nc_results or stage_negative_controls(cfg, log, detect_results=detect_results)
        model_results = model_results or stage_model(cfg, log, detect_results=detect_results)

        overall_by_year = {y: detect_results[y]["overall"] for y in cfg.years}
        pair_rates_by_year = {y: detect_results[y]["pair_rates"] for y in cfg.years}
        position_rates_by_year = {y: detect_results[y]["position_rates"] for y in cfg.years}
        phenotype_rates_by_year = {y: detect_results[y]["phenotype_rates"] for y in cfg.years}

        crit = cfg["stability_criteria"]
        rate_stab = ts.rate_stability(overall_by_year, crit["rate_ratio_threshold"])
        rank_stab = ts.pair_rank_stability(pair_rates_by_year, crit["pair_rank_correlation_threshold"],
                                            top_n=int(crit.get("pair_rank_top_n", 20)))
        dominant = ts.dominant_pair_consistency(pair_rates_by_year)
        pos_stab = ts.position_pattern_stability(position_rates_by_year,
                                                  crit["pair_rank_correlation_threshold"])
        aiml_stab = ts.aiml_performance_stability(model_results["perf_by_year"], "auroc",
                                                   max_abs_diff=crit.get("aiml_performance_max_abs_diff", 0.10))

        # Context only (not part of the decision): absolute rate among eligible discharges
        cohort_by_year = {y: detect_results[y]["cohort"] for y in cfg.years}
        cohort_rate_stab = ts.rate_stability(cohort_by_year, crit["rate_ratio_threshold"])
        baseline_by_year = {y: h2c.baseline_comparison(nc_results[y]["chance"], nc_results[y]["artifact"])
                            for y in cfg.years}

        # PRIMARY H2: DEV rate is significantly BELOW the chance null in
        # every year, and stable. The coding-artifact null is passed in for
        # the secondary / original-prespecified fields only - see
        # src/reporting/h2_criteria.py.
        per_year_compliance = {
            y: h2c.per_year_compliance_verdict(nc_results[y]["chance"], nc_results[y]["artifact"],
                                                alpha=cfg["negative_controls"].get("significance_alpha", 0.05))
            for y in cfg.years
        }
        alpha = cfg["negative_controls"].get("significance_alpha", 0.05)
        compliance = h2c.compliance_verdict(per_year_compliance, rate_stab, rank_stab, alpha=alpha)

        # SECONDARY reading kept for transparency only (see docstring) -
        # does NOT drive this pipeline's H2_SUPPORTED answer.
        per_year_elevated = {
            y: h2c.per_year_verdict(nc_results[y]["chance"], nc_results[y]["artifact"])
            for y in cfg.years
        }
        elevated = h2c.h2_verdict(per_year_elevated, rate_stab, rank_stab)

        with open(cfg.outputs("tables") / "h2_verdict_compliance.json", "w") as fh:
            json.dump(compliance, fh, indent=2, default=str)
        with open(cfg.outputs("tables") / "h2_verdict_elevated_secondary.json", "w") as fh:
            json.dump(elevated, fh, indent=2, default=str)
        dominant.to_csv(cfg.outputs("tables") / "dominant_pair_consistency.csv", index=False)

        log.info("H2_SUPPORTED (below chance baseline + stable, primary) = %s | "
                  "per-year outcome = %s | original prespecified rule (both baselines) = %s | "
                  "(elevated/secondary) = %s",
                  compliance["H2_SUPPORTED"], compliance["per_year_outcome"],
                  compliance["H2_SUPPORTED_prespecified_both_baselines"], elevated["H2_SUPPORTED"])

        return {"baseline_by_year": baseline_by_year, "cohort_by_year": cohort_by_year,
                "cohort_rate_stability": cohort_rate_stab,
                "compliance_verdict": compliance, "elevated_verdict": elevated,
                "rate_stability": rate_stab, "pair_rank_stability": rank_stab,
                "dominant_pair_consistency": dominant, "position_pattern_stability": pos_stab,
                "aiml_performance_stability": aiml_stab,
                "overall_by_year": overall_by_year, "phenotype_rates_by_year": phenotype_rates_by_year,
                "pair_rates_by_year": pair_rates_by_year}


def stage_report(cfg: Config, log, detect_results=None, model_results=None, stability_results=None):
    with stage_timer("report", log):
        detect_results = detect_results or stage_detect(cfg, log)
        model_results = model_results or stage_model(cfg, log, detect_results=detect_results)
        stability_results = stability_results or stage_stability(
            cfg, log, detect_results=detect_results, model_results=model_results)

        overall_by_year = stability_results["overall_by_year"]
        phenotype_rates_by_year = stability_results["phenotype_rates_by_year"]
        pair_rates_by_year = stability_results["pair_rates_by_year"]
        latest_year = sorted(pair_rates_by_year)[-1]

        fig.plot_baseline_comparison(stability_results["baseline_by_year"],
                                      cfg.outputs("plots") / "baseline_comparison.png")
        fig.plot_annual_dev_rates(overall_by_year, cfg.outputs("plots") / "annual_dev_rates.png")
        if len(pair_rates_by_year[latest_year]):
            fig.plot_top_pairs(pair_rates_by_year[latest_year],
                                cfg.outputs("plots") / "top_pairs_latest.png")
        fig.plot_phenotype_rates(phenotype_rates_by_year[latest_year],
                                  cfg.outputs("plots") / "phenotype_rates_latest.png")
        fig.plot_temporal_stability(stability_results["rate_stability"]["pairwise"],
                                     cfg.outputs("plots") / "temporal_stability.png")
        if len(model_results["importance"]):
            fig.plot_shap_or_importance(model_results["importance"], cfg.outputs("plots") / "shap_importance.png")

        out = rpt.build_report(
            cfg, stability_results["compliance_verdict"], stability_results["baseline_by_year"],
            stability_results["cohort_by_year"], overall_by_year, phenotype_rates_by_year,
            pair_rates_by_year, model_results["perf_by_year"],
            stability_results["rate_stability"], stability_results["pair_rank_stability"],
            model_results["best_algorithm"], elevated_verdict=stability_results["elevated_verdict"])
        log.info("Report -> %s", out)
        return out


# ------------------------------------------------------------------------- CLI
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--stage", choices=["all"] + STAGES, default="all")
    parser.add_argument("--force-ingest", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging(cfg.logs_dir, "h2_pipeline")
    log.info("Config: %s | years: %s", args.config, cfg.years)

    order = STAGES if args.stage == "all" else [args.stage]
    pairs_by_year = detect_results = nc_results = model_results = stability_results = None
    for stage in order:
        if stage == "inspect":
            stage_inspect(cfg, log)
        elif stage == "ingest":
            stage_ingest(cfg, log, force=args.force_ingest)
        elif stage == "validate":
            stage_validate(cfg, log)
        elif stage == "ontology_check":
            pairs_by_year = stage_ontology_check(cfg, log)
        elif stage == "detect":
            detect_results = stage_detect(cfg, log, pairs_by_year)
        elif stage == "negative_controls":
            nc_results = stage_negative_controls(cfg, log, pairs_by_year, detect_results)
        elif stage == "model":
            model_results = stage_model(cfg, log, detect_results)
        elif stage == "stability":
            stability_results = stage_stability(cfg, log, detect_results, nc_results, model_results)
        elif stage == "report":
            stage_report(cfg, log, detect_results, model_results, stability_results)

    dump_runtimes(cfg.outputs("tables"))
    log.info("Pipeline finished (stage=%s).", args.stage)


if __name__ == "__main__":
    sys.exit(main())
