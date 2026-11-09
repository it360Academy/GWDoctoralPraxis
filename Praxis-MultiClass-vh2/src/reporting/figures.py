"""Publication-quality figures for the H2 report."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_annual_dev_rates(overall_by_year: dict, out_path: Path) -> Path:
    years = sorted(overall_by_year.keys())
    rates = [overall_by_year[y]["weighted_dev_rate"] for y in years]
    los = [overall_by_year[y]["weighted_dev_rate_ci_lo"] for y in years]
    his = [overall_by_year[y]["weighted_dev_rate_ci_hi"] for y in years]
    errs = [[r - lo for r, lo in zip(rates, los)], [hi - r for r, hi in zip(rates, his)]]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(years, rates, yerr=errs, capsize=6, color="#a03b3b")
    ax.set_ylabel("Weighted DEV rate")
    ax.set_title("Annual DEV rate (national weighted estimate, 95% CI)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_baseline_comparison(baseline_by_year: dict, out_path: Path) -> Path:
    """PRIMARY H2 figure: observed DEV rate as a fraction of each
    data-derived 'ignore-the-rule' baseline. The dashed line at 1.0 is the
    baseline itself; bars below it mean DEVs are rarer than the baseline
    predicts. Whiskers (chance baseline only) are the bootstrap 95% CI."""
    years = sorted(baseline_by_year.keys())
    x = list(range(len(years)))
    w = 0.38
    chance = [baseline_by_year[y]["chance_ratio"] for y in years]
    lo = [baseline_by_year[y]["chance_ratio_ci_lo"] for y in years]
    hi = [baseline_by_year[y]["chance_ratio_ci_hi"] for y in years]
    art = [baseline_by_year[y]["artifact_ratio"] for y in years]
    errs = [[c - l for c, l in zip(chance, lo)], [h - c for c, h in zip(chance, hi)]]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar([i - w / 2 for i in x], chance, w, yerr=errs, capsize=5, color="#a03b3b",
           label="vs chance baseline (age x sex strata)")
    ax.bar([i + w / 2 for i in x], art, w, color="#3b6fa0",
           label="vs coding-artifact baseline (within hospital)")
    ax.axhline(1.0, color="black", linestyle="--", label="baseline = rule ignored")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Observed DEV rate / baseline rate")
    ax.set_title("DEV rate relative to the ignore-the-rule baselines")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_top_pairs(pair_rates: pd.DataFrame, out_path: Path, top_n: int = 15) -> Path:
    df = pair_rates.head(top_n).copy()
    df["pair_label"] = df["home_code"] + " x " + df["target_raw"]
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    ax.barh(df["pair_label"][::-1], df["weighted_n_hit"][::-1], color="#3b6fa0")
    ax.set_xlabel("Weighted national estimate of hospitalizations")
    ax.set_title(f"Top {top_n} Excludes1-violating diagnosis pairs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_phenotype_rates(phenotype_rates: pd.DataFrame, out_path: Path) -> Path:
    df = phenotype_rates.sort_values("weighted_dev_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df["phenotype"], df["weighted_dev_rate"], color="#4a7a4a")
    ax.set_xlabel("Weighted DEV rate")
    ax.set_title("DEV rate by phenotype")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_temporal_stability(rate_stability_pairwise: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(rate_stability_pairwise):
        ax.bar(rate_stability_pairwise["year_a"] + "->" + rate_stability_pairwise["year_b"],
               rate_stability_pairwise["rate_ratio_b_over_a"], color="#7a5a9a")
        ax.axhline(1.0, color="gray", linestyle="--")
    ax.set_ylabel("Rate ratio (year B / year A)")
    ax.set_title("Temporal stability of DEV rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_shap_or_importance(importance_df: pd.DataFrame, out_path: Path, top_n: int = 20) -> Path:
    df = importance_df.head(top_n)
    value_col = "mean_abs_shap" if "mean_abs_shap" in df.columns else (
        "importance" if "importance" in df.columns else "coefficient")
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    ax.barh(df["feature"][::-1], df[value_col][::-1], color="#c07a2a")
    ax.set_xlabel(value_col)
    ax.set_title("DEV-risk model feature importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
