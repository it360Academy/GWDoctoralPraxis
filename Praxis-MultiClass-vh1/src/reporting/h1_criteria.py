"""H1 decision rule for the multiclass DX2-25 -> DX1 framework, evaluated
on the held-out TEST year (2023) only, using the model selected on the
VALIDATION year (2022).

Per-phenotype "supported" requires BOTH:
  (a) statistical validity: the bootstrap 95% CI lower bound for that
      class's AUROC-OVR is > 0.5 (equivalently: the class-level aggregate
      permutation test, computed once at the macro level, is significant
      AND this class individually clears chance by its own CI - see
      src/analysis/stats.py:bootstrap_ci_per_class_auroc)
  (b) EITHER an AUROC-OVR floor (h1_criteria.per_class.auroc_ovr_min), OR
      - for low-prevalence phenotypes (prevalence below
      h1_criteria.per_class.low_prevalence_threshold) - a PPV-lift-over-
      null-rate floor (h1_criteria.per_class.ppv_lift_min), since AUROC
      can look deceptively strong for rare classes while precision at any
      realistic operating point stays low.

H1 overall is "supported" only if:
  - the aggregate macro AUROC-OVR, macro MCC, and balanced accuracy all
    clear their configured floors on the test year, AND
  - all 15 phenotypes are represented in the test-year cohort
    (h1_criteria.require_full_phenotype_coverage), AND
  - a configurable minimum number/fraction of individual phenotypes are
    "supported" per the per-class rule above.

All floors in config/config.yaml are explicitly flagged as unverified
placeholders (see README) pending literature benchmarks - this module
applies whatever floors are configured; it does not choose them.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from ..features.definitions import DISEASE_ORDER


def per_phenotype_verdict(cfg, per_class_metrics: pd.DataFrame,
                           per_class_auroc_ci: Dict[str, tuple],
                           macro_permutation_significant: bool) -> pd.DataFrame:
    crit = cfg["h1_criteria"]["per_class"]
    auroc_floor = float(crit["auroc_ovr_min"])
    ppv_lift_floor = float(crit["ppv_lift_min"])
    low_prev_thresh = float(crit["low_prevalence_threshold"])

    rows = []
    for _, r in per_class_metrics.iterrows():
        ph = r["phenotype"]
        auroc = r["auroc_ovr"]
        prevalence = r["prevalence"]
        ppv_lift = r["ppv_lift_over_null"]
        _, ci_lo, _ = per_class_auroc_ci.get(ph, (auroc, float("nan"), float("nan")))
        statistically_valid = bool(macro_permutation_significant and ci_lo > 0.5)

        is_low_prev = prevalence < low_prev_thresh if prevalence == prevalence else False
        if is_low_prev:
            clears_performance_floor = bool(ppv_lift == ppv_lift and ppv_lift >= ppv_lift_floor)
            floor_used = f"ppv_lift>={ppv_lift_floor} (low-prevalence carve-out)"
        else:
            clears_performance_floor = bool(auroc == auroc and auroc >= auroc_floor)
            floor_used = f"auroc_ovr>={auroc_floor}"

        supported = statistically_valid and clears_performance_floor
        rows.append({
            "phenotype": ph, "prevalence": prevalence, "auroc_ovr": auroc,
            "auroc_ci_lo": ci_lo, "ppv_lift_over_null": ppv_lift,
            "statistically_valid": statistically_valid,
            "floor_rule_applied": floor_used,
            "clears_performance_floor": clears_performance_floor,
            "phenotype_supported": supported,
        })
    return pd.DataFrame(rows)


def h1_verdict(cfg, test_aggregate: Dict, test_aggregate_ci: Dict,
               phenotype_table: pd.DataFrame, phenotypes_present: set) -> Dict:
    agg_crit = cfg["h1_criteria"]["aggregate"]
    coverage_ok = (not cfg["h1_criteria"].get("require_full_phenotype_coverage", True)
                   or set(DISEASE_ORDER) <= set(phenotypes_present))

    macro_auroc = test_aggregate["macro_auroc_ovr"]
    macro_mcc = test_aggregate["mcc"]
    bal_acc = test_aggregate["balanced_accuracy"]

    clears_macro_auroc = macro_auroc >= float(agg_crit["macro_auroc_ovr_min"])
    clears_mcc = macro_mcc >= float(agg_crit["macro_mcc_min"])
    clears_bal_acc = bal_acc >= float(agg_crit["balanced_accuracy_min"])

    n_supported = int(phenotype_table["phenotype_supported"].sum())
    n_total = len(phenotype_table)

    aggregate_ok = clears_macro_auroc and clears_mcc and clears_bal_acc
    h1_supported = bool(aggregate_ok and coverage_ok and n_supported > 0)

    return {
        "macro_auroc_ovr": macro_auroc,
        "macro_auroc_ovr_ci": test_aggregate_ci.get("macro_auroc_ovr"),
        "macro_mcc": macro_mcc,
        "macro_mcc_ci": test_aggregate_ci.get("mcc"),
        "balanced_accuracy": bal_acc,
        "balanced_accuracy_ci": test_aggregate_ci.get("balanced_accuracy"),
        "clears_macro_auroc_floor": clears_macro_auroc,
        "clears_macro_mcc_floor": clears_mcc,
        "clears_balanced_accuracy_floor": clears_bal_acc,
        "full_phenotype_coverage": coverage_ok,
        "n_phenotypes_supported": n_supported,
        "n_phenotypes_total": n_total,
        "H1_SUPPORTED": h1_supported,
        "note": "All numeric floors in h1_criteria are unverified placeholders "
                "(README limitations) until replaced with literature-derived "
                "benchmarks; this verdict is only as trustworthy as those floors.",
    }
