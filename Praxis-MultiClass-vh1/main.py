#!/usr/bin/env python
"""H1-MultiClass pipeline orchestrator.

H1: Within a leakage-controlled, cross-year-replicated supervised
multiclass framework for diagnostic-quality auditing, a hospitalized
patient's secondary-diagnosis profile (DX2-DX25) discriminates the
principal diagnosis (DX1) among fifteen vascular, infectious, and
oncological phenotypes in the U.S. National Inpatient Sample, 2021-2023,
under severe class imbalance.

Stages (run in order; each is checkpointed / idempotent where practical):
    inspect        - read file headers only, detect columns, sanity-check
    ingest         - stream Core CSVs -> per-year raw parquet checkpoints
    validate       - data-quality gates on each year's checkpoint
    leakage_audit  - fit the DX2-25 feature vocabulary on train, mask
                     leaky codes, write the audit table
    model          - train all configured algorithms, select on
                     validation, evaluate + negative-control on test,
                     apply the H1 decision rule
    report         - assemble the HTML report

Usage:
    python main.py --config config/config.yaml               # everything
    python main.py --stage model --algorithms xgboost         # one stage
    python main.py --config config/config_smoke.yaml          # smoke test
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import load_config
from src.data.ingestion import build_year_checkpoint, checkpoint_path, load_year
from src.data.inspect import inspect_datasets
from src.data.validation import validate_year
from src.features import dx_features as dxf
from src.features.definitions import DISEASE_ORDER, validate_definitions
from src.leakage.controls import audit_masked_categories
from src.logging_utils import dump_runtimes, get_logger, setup_logging, stage_timer
from src.analysis import evaluation as ev
from src.analysis import models as mdl
from src.analysis import negative_control as nc
from src.analysis import stats as st
from src.analysis import tuning as tn
from src.analysis.explain import global_importance
from src.reporting import figures as fig
from src.reporting import h1_criteria as h1c
from src.reporting import report as rpt

STAGES = ["inspect", "ingest", "validate", "leakage_audit", "model", "report"]


def vocab_path(cfg) -> Path:
    return cfg.data_dir / "dx_vocabulary.pkl"


def stage_inspect(cfg, log):
    with stage_timer("inspect", log):
        inspect_datasets(cfg)


def stage_ingest(cfg, log, force=False):
    with stage_timer("ingest", log):
        for role, year in cfg.years.items():
            build_year_checkpoint(cfg, year, force=force)


def stage_validate(cfg, log):
    with stage_timer("validate", log):
        for role, year in cfg.years.items():
            validate_year(cfg, year)


def stage_leakage_audit(cfg, log):
    with stage_timer("leakage_audit", log):
        train_year = cfg.years["train"]
        train_df = load_year(cfg, train_year)
        vocab = dxf.fit_vocabulary(cfg, train_df)
        with open(vocab_path(cfg), "wb") as fh:
            pickle.dump(vocab, fh)

        audit = audit_masked_categories(set(vocab.masked_codes), vocab.code_counts_train)
        audit.to_csv(cfg.outputs("tables") / "leakage_code_audit.csv", index=False)

        vocab_summary = pd.DataFrame({
            "category": vocab.categories,
            "n_occurrences_train": [vocab.category_counts_train.get(c, 0) for c in vocab.categories],
        }).sort_values("n_occurrences_train", ascending=False)
        vocab_summary.to_csv(cfg.outputs("tables") / "dx_category_vocabulary.csv", index=False)

        log.info("Vocabulary: %d categories, %d masked codes, %d covariates -> %d total features",
                  len(vocab.categories), len(vocab.masked_codes), len(vocab.covariate_cols),
                  vocab.n_features)
        return vocab


def _load_vocab(cfg):
    with open(vocab_path(cfg), "rb") as fh:
        return pickle.load(fh)


def stage_model(cfg, log, algorithms=None, tune=None):
    with stage_timer("model", log):
        algorithms = algorithms or cfg["models"]["algorithms"]
        do_tune = cfg["tuning"].get("enabled", False) if tune is None else tune

        if not vocab_path(cfg).exists():
            stage_leakage_audit(cfg, log)
        vocab = _load_vocab(cfg)

        data = {role: load_year(cfg, year) for role, year in cfg.years.items()}
        xy = {role: dxf.build_xy(cfg, df, vocab) for role, df in data.items()}
        names = dxf.feature_names(vocab)

        perf_rows = []
        fitted = {}
        proba_cache = {}
        for algo in algorithms:
            log.info("Fitting %s on %s (n=%d, %d features)", algo, cfg.years["train"],
                      xy["train"][0].shape[0], xy["train"][0].shape[1])
            params = {}
            if do_tune:
                params = tn.tune_params(algo, xy["train"][0], xy["train"][1], cfg, seed=cfg.seed)
            model = mdl.fit_model(algo, xy["train"][0], xy["train"][1], cfg, params=params, seed=cfg.seed)
            fitted[algo] = model

            for role in ("train", "validation", "test"):
                X, y, w, _ = xy[role]
                proba = mdl.predict_proba(model, algo, X)
                pred = mdl.predict_label(model, algo, X)
                agg = ev.aggregate_metrics(y, pred, proba)
                agg.update({"algorithm": algo, "year_role": role, "year": cfg.years[role]})
                perf_rows.append(agg)
                proba_cache[(algo, role)] = (y, pred, proba)

        model_perf = pd.DataFrame(perf_rows)
        model_perf.to_csv(cfg.outputs("tables") / "model_performance_all.csv", index=False)

        val_scores = model_perf[model_perf["year_role"] == "validation"].set_index("algorithm")["macro_auroc_ovr"]
        best_algo = val_scores.idxmax()
        log.info("Model selection (on %s): best algorithm = %s (macro AUROC-OVR=%.4f)",
                  cfg.years["validation"], best_algo, val_scores[best_algo])

        joblib.dump(fitted[best_algo], cfg.outputs("models") / f"best_model_{best_algo}.joblib")
        with open(cfg.outputs("models") / "best_model_meta.json", "w") as fh:
            json.dump({"algorithm": best_algo, "features": names,
                       "classes": mdl.CLASSES}, fh, indent=2)

        y_test, pred_test, proba_test = proba_cache[(best_algo, "test")]

        agg_ci = st.bootstrap_ci_aggregate(y_test, pred_test, proba_test,
                                            n_resamples=cfg["evaluation"]["bootstrap_resamples"],
                                            ci=cfg["evaluation"]["bootstrap_ci"], seed=cfg.seed)
        per_class_ci = st.bootstrap_ci_per_class_auroc(y_test, proba_test,
                                                        n_resamples=cfg["evaluation"]["bootstrap_resamples"],
                                                        ci=cfg["evaluation"]["bootstrap_ci"], seed=cfg.seed)
        neg_ctrl = nc.permutation_test(y_test, proba_test,
                                        n_permutations=cfg["negative_control"]["n_permutations"],
                                        seed=cfg.seed)

        per_class = ev.per_class_table(y_test, pred_test, proba_test)
        phenotype_table = h1c.per_phenotype_verdict(
            cfg, per_class, per_class_ci, neg_ctrl["significant_at_0.05"])
        phenotype_table.to_csv(cfg.outputs("tables") / "h1_phenotype_criteria.csv", index=False)

        test_aggregate = ev.aggregate_metrics(y_test, pred_test, proba_test)
        verdict = h1c.h1_verdict(cfg, test_aggregate, agg_ci, phenotype_table,
                                  phenotypes_present=set(pd.unique(y_test)))
        with open(cfg.outputs("tables") / "h1_verdict.json", "w") as fh:
            json.dump(verdict, fh, indent=2, default=str)

        with open(cfg.outputs("tables") / "negative_control.json", "w") as fh:
            json.dump(neg_ctrl, fh, indent=2, default=str)

        cm = ev.confusion_matrix_df(y_test, pred_test)
        cm.to_csv(cfg.outputs("tables") / "confusion_matrix_test.csv")

        imp = global_importance(fitted[best_algo], best_algo, names)
        imp.to_csv(cfg.outputs("tables") / "feature_importance_top.csv", index=False)

        log.info("H1 verdict: %s", "SUPPORTED" if verdict["H1_SUPPORTED"] else "NOT SUPPORTED")
        log.info("Test macro AUROC-OVR=%.4f MCC=%.4f balanced_acc=%.4f",
                  verdict["macro_auroc_ovr"], verdict["macro_mcc"], verdict["balanced_accuracy"])

        return {
            "best_algorithm": best_algo, "model_perf": model_perf,
            "phenotype_table": phenotype_table, "verdict": verdict,
            "negative_control": neg_ctrl, "confusion_matrix": cm,
        }


def stage_report(cfg, log, model_results=None):
    with stage_timer("report", log):
        if model_results is None:
            model_perf = pd.read_csv(cfg.outputs("tables") / "model_performance_all.csv")
            phenotype_table = pd.read_csv(cfg.outputs("tables") / "h1_phenotype_criteria.csv")
            with open(cfg.outputs("tables") / "h1_verdict.json") as fh:
                verdict = json.load(fh)
            with open(cfg.outputs("tables") / "negative_control.json") as fh:
                neg_ctrl = json.load(fh)
            with open(cfg.outputs("models") / "best_model_meta.json") as fh:
                best_algo = json.load(fh)["algorithm"]
            cm = pd.read_csv(cfg.outputs("tables") / "confusion_matrix_test.csv", index_col=0)
        else:
            model_perf = model_results["model_perf"]
            phenotype_table = model_results["phenotype_table"]
            verdict = model_results["verdict"]
            neg_ctrl = model_results["negative_control"]
            best_algo = model_results["best_algorithm"]
            cm = model_results["confusion_matrix"]

        leakage_audit = pd.read_csv(cfg.outputs("tables") / "leakage_code_audit.csv")

        fig.plot_confusion_matrix(cm, cfg.outputs("plots") / "confusion_matrix_test.png")
        per_class_for_plot = pd.DataFrame({
            "phenotype": phenotype_table["phenotype"], "auroc_ovr": phenotype_table["auroc_ovr"],
        })
        fig.plot_per_class_auroc(per_class_for_plot, cfg.outputs("plots") / "per_class_auroc_test.png")
        fig.plot_algorithm_comparison(model_perf, cfg.outputs("plots") / "algorithm_comparison.png")

        out = rpt.build_report(cfg, verdict, phenotype_table, model_perf,
                                leakage_audit, neg_ctrl, best_algo)
        log.info("Report -> %s", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--stage", choices=["all"] + STAGES, default="all")
    parser.add_argument("--algorithms", nargs="+", default=None)
    parser.add_argument("--tune", dest="tune", action="store_true", default=None)
    parser.add_argument("--no-tune", dest="tune", action="store_false")
    parser.add_argument("--force-ingest", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_definitions()
    log = setup_logging(cfg.logs_dir, "h1_pipeline")

    order = STAGES if args.stage == "all" else [args.stage]
    model_results = None
    for stage in order:
        if stage == "inspect":
            stage_inspect(cfg, log)
        elif stage == "ingest":
            stage_ingest(cfg, log, force=args.force_ingest)
        elif stage == "validate":
            stage_validate(cfg, log)
        elif stage == "leakage_audit":
            stage_leakage_audit(cfg, log)
        elif stage == "model":
            model_results = stage_model(cfg, log, algorithms=args.algorithms, tune=args.tune)
        elif stage == "report":
            stage_report(cfg, log, model_results=model_results)

    dump_runtimes(cfg.outputs("tables"))
    log.info("Pipeline finished (stage=%s).", args.stage)


if __name__ == "__main__":
    sys.exit(main())
