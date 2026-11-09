"""All figures for the H3 report: EDA, incidence comparison, Monte Carlo,
clustering, anomaly detection, bootstrap rank stability, category shares,
and sensitivity analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

PUB_COLOR = "#8a6d1f"   # published benchmark, in every figure
RED_COLOR = "#3b6fa0"   # re-derived (NIS 2021-2023), in every figure


def _recessive(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, color="#e5e5e5", linewidth=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ------------------------------------------------------------------ EDA
def plot_cohort_summary(summary: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(summary["year"], summary["n_discharges"], color="#3b6fa0")
    ax.set_ylabel("Discharges (raw count)")
    ax.set_title("Cohort size by discharge year")
    for i, v in enumerate(summary["n_discharges"]):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


def plot_phenotype_age_sex(demo: pd.DataFrame, out: Path) -> Path:
    latest_year = demo["year"].max()
    sub = demo[demo["year"] == latest_year].sort_values("mean_age")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].barh(sub["disease"], sub["mean_age"], color="#4a7a4a")
    axes[0].set_xlabel("Mean age (years)")
    axes[0].set_title(f"Mean age by phenotype, {latest_year}")
    axes[1].barh(sub["disease"], sub["female_share"] * 100, color="#8a6d1f")
    axes[1].set_xlabel("Female share (%)")
    axes[1].set_title(f"Sex distribution by phenotype, {latest_year}")
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


def plot_coding_intensity(dist: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for year, g in dist.groupby("year"):
        ax.plot(g["n_dx_bin"], g["n"], marker="o", label=str(year))
    ax.set_xlabel("Number of diagnosis positions filled")
    ax.set_ylabel("Discharges")
    ax.set_title("Coding-intensity distribution by year")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


# ------------------------------------------------------------- incidence
def plot_incidence_comparison(c1: pd.DataFrame, out: Path) -> Path:
    df = c1.dropna(subset=["rederived_thousands"]).sort_values("published_thousands")
    x = np.arange(len(df)); w = 0.38
    err = [df["rederived_thousands"] - df["rederived_ci_low_thousands"],
           df["rederived_ci_high_thousands"] - df["rederived_thousands"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, df["published_thousands"], width=w, label="Published (DX1 or DX2)", color=PUB_COLOR)
    ax.bar(x + w / 2, df["rederived_thousands"], width=w, yerr=err, capsize=3,
           label="Re-derived (DX1 or DX2, 95% CI)", color=RED_COLOR)
    ax.set_xticks(x); ax.set_xticklabels(df["disease"], rotation=45, ha="right")
    ax.set_ylabel("Mean annual incidence (thousands)")
    ax.set_title("Mean annual incidence by phenotype: published vs. re-derived")
    ax.legend(frameon=False)
    _recessive(ax)
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


def plot_harm_comparison(harm: pd.DataFrame, out: Path) -> Path:
    df = harm.sort_values("published_harm_thousands")
    y = np.arange(len(df)); h = 0.38
    err = [df["rederived_harm_mc_median"] - df["rederived_harm_mc_p2_5"],
           df["rederived_harm_mc_p97_5"] - df["rederived_harm_mc_median"]]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(y + h / 2, df["published_harm_thousands"], height=h, color=PUB_COLOR, label="Published")
    ax.barh(y - h / 2, df["rederived_harm_mc_median"], height=h, xerr=err, capsize=2,
            color=RED_COLOR, label="Re-derived (MC median, 95% SI)")
    ax.set_yticks(y); ax.set_yticklabels(df["disease"])
    ax.set_xlabel("Serious harms (thousands)")
    ax.set_title("Serious harm by phenotype: published vs. re-derived")
    ax.legend(frameon=False, loc="lower right")
    _recessive(ax, axis="x")
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


def plot_rank_comparison(harm: pd.DataFrame, out: Path) -> Path:
    """Slope chart: harm rank among the ten NIS-derived phenotypes (the
    criterion 2 basis), published (left) vs. re-derived (right)."""
    df = harm[harm["independent"].astype(bool)]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for _, r in df.iterrows():
        pr, rr = int(r["published_rank_10"]), int(r["rederived_rank_10"])
        moved = pr != rr
        color = "#8a2b2b" if moved else "#9a9a9a"
        ax.plot([0, 1], [pr, rr], color=color, linewidth=2, marker="o", markersize=6)
        ax.text(-0.04, pr, r["disease"], ha="right", va="center", fontsize=8)
        ax.text(1.04, rr, r["disease"], ha="left", va="center", fontsize=8,
                color="#222" if moved else "#777")
    n = len(df)
    ax.set_ylim(n + 0.5, 0.5); ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Published rank", "Re-derived rank"])
    ax.set_yticks(range(1, n + 1))
    ax.set_title("Harm rank, 10 NIS-derived phenotypes (1 = highest); red = changed")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


def plot_concentration(c34: dict, out: Path) -> Path:
    labels = ["Top-5 share", "Top-15 share"]
    pub = [c34["top5_published"] * 100, c34["top15_published"] * 100]
    red = [c34["top5_rederived"] * 100, c34["top15_rederived"] * 100]
    err = [[c34["top5_rederived"] * 100 - c34["top5_rederived_p2_5"] * 100,
            c34["top15_rederived"] * 100 - c34["top15_rederived_p2_5"] * 100],
           [c34["top5_rederived_p97_5"] * 100 - c34["top5_rederived"] * 100,
            c34["top15_rederived_p97_5"] * 100 - c34["top15_rederived"] * 100]]
    x = np.arange(2); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(x - w / 2, pub, width=w, color=PUB_COLOR, label="Published")
    ax.bar(x + w / 2, red, width=w, yerr=err, capsize=4, color=RED_COLOR,
           label="Re-derived (median, 95% SI)")
    for xi, p_, r_ in zip(x, pub, red):
        ax.text(xi - w / 2, p_ + 1, f"{p_:.1f}%", ha="center", fontsize=8)
        ax.text(xi + w / 2, r_ + 1, f"{r_:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Share of grand total serious harm (%)")
    ax.set_title("Harm concentration: top-5 and top-15 phenotypes")
    ax.legend(frameon=False)
    _recessive(ax)
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


# ------------------------------------------------------------ Monte Carlo
def plot_mc_headline(sim_summary: dict, published: dict, out: Path) -> Path:
    metrics = ["big3_errors_thousands", "big3_harms_thousands",
               "grand_total_errors_thousands", "grand_total_harms_thousands"]
    labels = ["Big Three\nerrors", "Big Three\nharms", "Grand total\nerrors", "Grand total\nharms"]
    my_med = [sim_summary[m][0] for m in metrics]
    my_lo = [sim_summary[m][1] for m in metrics]
    my_hi = [sim_summary[m][2] for m in metrics]
    pub_med = [published[m][0] for m in metrics]
    pub_lo = [published[m][1] for m in metrics]
    pub_hi = [published[m][2] for m in metrics]

    x = np.arange(len(metrics)); w = 0.32
    my_err = [[m - l for m, l in zip(my_med, my_lo)], [h - m for m, h in zip(my_med, my_hi)]]
    pub_err = [[m - l for m, l in zip(pub_med, pub_lo)], [h - m for m, h in zip(pub_med, pub_hi)]]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(x - w / 2, my_med, width=w, yerr=my_err, capsize=4, color=RED_COLOR, label="Re-derived (median, 95% SI)")
    ax.bar(x + w / 2, pub_med, width=w, yerr=pub_err, capsize=4, color=PUB_COLOR, label="Published")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Thousands of cases")
    ax.set_title("Monte Carlo headline results: re-derived vs. published")
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


def plot_mc_distribution(draws: pd.DataFrame, column: str, published_range: tuple, out: Path,
                          dx1_only_published: float | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(draws[column], bins=60, color=RED_COLOR, alpha=0.8, label="Re-derived draws")
    ax.axvspan(published_range[1], published_range[2], color=PUB_COLOR, alpha=0.2, label="Published 95% range")
    ax.axvline(published_range[0], color=PUB_COLOR, linestyle="--", label="Published median")
    if dx1_only_published is not None:
        ax.axvline(dx1_only_published, color="#222222", linestyle=":", linewidth=2,
                   label="Published DX1-only estimate")
    ax.set_xlabel(column.replace("_", " "))
    ax.set_ylabel("Simulation draws")
    ax.set_title(f"Monte Carlo distribution: {column.replace('_', ' ')}")
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


# -------------------------------------------------------------- clustering
def plot_dendrogram(Z, labels, title: str, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=45, leaf_font_size=8, color_threshold=0)
    ax.set_title(title)
    ax.set_ylabel("Ward distance")
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


# ------------------------------------------------------- ML validation (Track 2)
def plot_pca_clusters(pca_pub: pd.DataFrame, pca_red: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, df, title in ((axes[0], pca_pub, "Published structure"),
                          (axes[1], pca_red, "Re-derived structure")):
        cmap = plt.get_cmap("tab10")
        for cl in sorted(df["cluster"].unique()):
            sub = df[df["cluster"] == cl]
            ax.scatter(sub["pc1"], sub["pc2"], s=90, color=cmap(cl % 10), label=f"cluster {cl}")
        for _, row in df.iterrows():
            ax.annotate(row["disease"], (row["pc1"], row["pc2"]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_title(f"K-means clusters (PCA projection): {title}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


def plot_cluster_crosstab(ct: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(ct.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(ct.columns))); ax.set_xticklabels(ct.columns)
    ax.set_yticks(range(len(ct.index))); ax.set_yticklabels(ct.index)
    ax.set_xlabel("Re-derived cluster"); ax.set_ylabel("Published cluster")
    ax.set_title("Cluster assignment crosstab (K-means)")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax.text(j, i, int(ct.to_numpy()[i, j]), ha="center", va="center",
                    color="white" if ct.to_numpy()[i, j] > ct.to_numpy().max() / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="count")
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


def plot_permutation_null(primary: dict, out: Path, label: str = "K-means") -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rng = np.random.default_rng(0)
    # reconstruct an illustrative null shape from the stored mean (exact
    # draws are not persisted; this communicates the test, not raw data)
    ax.axvline(primary["observed_ari"], color="#8a2b2b", linewidth=2,
               label=f"observed ARI = {primary['observed_ari']:.3f}")
    ax.axvline(primary["null_ari_mean"], color="#3b6fa0", linestyle="--",
               label=f"null mean ARI = {primary['null_ari_mean']:.3f}")
    ax.set_title(f"{label} permutation test: observed vs. null agreement\n"
                 f"p = {primary['p_value_ari']:.4f} ({primary['n_permutations']:,} permutations)")
    ax.set_xlabel("Adjusted Rand Index"); ax.set_yticks([])
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


def plot_k_sensitivity(k_sens: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo, marker in (("kmeans", "o"), ("hierarchical", "s")):
        sub = k_sens[k_sens["algorithm"] == algo]
        ax.plot(sub["k"], sub["adjusted_rand_index"], marker=marker, label=algo)
    ax.set_xlabel("k (number of clusters)"); ax.set_ylabel("Adjusted Rand Index")
    ax.set_title("Clustering agreement across k and algorithm")
    ax.set_xticks(k_sens["k"].unique())
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


def plot_leave_one_out(loo: pd.DataFrame, baseline_ari: float, out: Path,
                        title: str = "Leave-one-phenotype-out stability (K-means)") -> Path:
    df = loo.sort_values("adjusted_rand_index")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(df["held_out_disease"], df["adjusted_rand_index"], color="#4a7a4a")
    ax.axvline(baseline_ari, color="#8a2b2b", linestyle="--", label=f"full-sample ARI = {baseline_ari:.3f}")
    ax.set_xlabel("Adjusted Rand Index with phenotype held out")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


def plot_silhouette(sil_pub: pd.DataFrame, sil_red: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, df, title in ((axes[0], sil_pub, "Published"), (axes[1], sil_red, "Re-derived")):
        df = df.sort_values(["cluster", "silhouette"])
        colors = plt.get_cmap("tab10")(df["cluster"] % 10)
        ax.barh(df["disease"], df["silhouette"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Silhouette scores: {title}")
        ax.set_xlabel("Silhouette coefficient")
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


# ---------------------------------------------------------------- anomaly
def plot_anomaly_scores(div: pd.DataFrame, out: Path) -> Path:
    df = div.sort_values("anomaly_score")
    colors = ["#8a2b2b" if f else "#3b6fa0" for f in df["flagged_anomalous"]]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(df["disease"], df["anomaly_score"], color=colors)
    ax.set_xlabel("Isolation-forest anomaly score (higher = more anomalous)")
    ax.set_title("Divergence from published structure, by phenotype")
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


# --------------------------------------------------------------- rank stability
def plot_rank_stability(rank_df: pd.DataFrame, out: Path) -> Path:
    df = rank_df.sort_values("mean_rank")
    lo = np.clip(df["mean_rank"] - df["rank_p5"], 0, None)
    hi = np.clip(df["rank_p95"] - df["mean_rank"], 0, None)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.errorbar(df["mean_rank"], df["disease"], xerr=[lo, hi],
                fmt="o", color="#3b6fa0", ecolor="#aac4dd", capsize=3)
    ax.set_xlabel("Harm rank (1 = highest, with 90% simulation interval)")
    ax.set_title("Bootstrap rank stability across Monte Carlo draws")
    ax.invert_xaxis()
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


def plot_top5_membership(rank_df: pd.DataFrame, out: Path) -> Path:
    df = rank_df.sort_values("p_top5", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(df["disease"], df["p_top5"] * 100, color="#4a7a4a")
    ax.set_xlabel("Probability of ranking in top 5 by harm (%)")
    ax.set_title("Top-5 membership probability across Monte Carlo draws")
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


# ------------------------------------------------------------- category shares
def plot_category_shares(cat_df: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(cat_df)); w = 0.38
    err = [(cat_df["share_median"] - cat_df["share_p2_5"]) * 100,
           (cat_df["share_p97_5"] - cat_df["share_median"]) * 100]
    ax.bar(x - w / 2, cat_df["published_share"] * 100, width=w, color=PUB_COLOR,
           label="Published inputs (point reconstruction)")
    ax.bar(x + w / 2, cat_df["share_median"] * 100, width=w, yerr=err, capsize=4, color=RED_COLOR,
           label="Re-derived (median, 95% SI)")
    ax.set_xticks(x); ax.set_xticklabels(cat_df["category"])
    ax.set_ylabel("Share of grand total harm (%)")
    ax.set_title("Category-level burden shares")
    ax.legend(frameon=False)
    _recessive(ax)
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


def plot_total_harm_sensitivity(c6: dict, c8: dict, out: Path) -> Path:
    """Grand total serious harm under DX1-or-DX2 (primary) and DX1-only
    (criterion 8), published vs. re-derived, with intervals where they exist."""
    labels = ["DX1 or DX2\n(primary)", "DX1 only\n(sensitivity)"]
    pub = [c6["published_median"], c8["published_dx1_only"]]
    red = [c6["rederived_median"], c8["rederived_dx1_only"]]
    pub_lh = [tuple(c6["published_range"]), None]
    red_lh = [tuple(c6["rederived_range"]), tuple(c8["rederived_dx1_only_range"])]

    def _err(vals, lh):
        return [[v - r[0] if r else np.nan for v, r in zip(vals, lh)],
                [r[1] - v if r else np.nan for v, r in zip(vals, lh)]]

    x = np.arange(2); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(x - w / 2, pub, width=w, yerr=_err(pub, pub_lh), capsize=4, color=PUB_COLOR, label="Published")
    ax.bar(x + w / 2, red, width=w, yerr=_err(red, red_lh), capsize=4, color=RED_COLOR,
           label="Re-derived (median, 95% SI)")
    for xi, v in zip(x, pub):
        ax.text(xi - w / 2, v / 2, f"{v:,.0f}k", ha="center", fontsize=8, color="white")
    for xi, v in zip(x, red):
        ax.text(xi + w / 2, v / 2, f"{v:,.0f}k", ha="center", fontsize=8, color="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Grand total serious harms (thousands)")
    ax.set_title(f"Total serious harm; DX1-only reduction: published "
                 f"{c8['published_reduction_pct']:.1%}, re-derived {c8['rederived_reduction_pct']:.1%}",
                 fontsize=10)
    ax.legend(frameon=False)
    _recessive(ax)
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


# -------------------------------------------------- consensus clustering (Section 8)
def plot_coclustering(C_pub: pd.DataFrame, C_red: pd.DataFrame, membership: pd.DataFrame,
                      out: Path) -> Path:
    """Co-clustering matrices side by side, both ordered by published tier
    then published harm, so matching block structure reads as agreement."""
    order = membership["disease"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), gridspec_kw={"wspace": 0.55})
    for ax, C, title in ((axes[0], C_pub, "Published"), (axes[1], C_red, "Re-derived")):
        M = C.loc[order, order].to_numpy()
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=7)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7)
        ax.set_title(f"{title}: share of draws co-clustered")
    fig.colorbar(im, ax=axes, shrink=0.8, label="co-clustering probability")
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)
    return out


def plot_consensus_ari(per_draw: pd.DataFrame, agreement: dict, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(per_draw["adjusted_rand_index"], bins=40, color=RED_COLOR, alpha=0.85)
    ax.axvline(agreement["observed_ari"], color="#8a2b2b", linewidth=2,
               label=f"consensus ARI = {agreement['observed_ari']:.3f}")
    ax.axvline(agreement["null_ari_mean"], color="#777777", linestyle="--",
               label=f"permutation-null mean = {agreement['null_ari_mean']:.3f}")
    ax.set_xlabel("Adjusted Rand Index, published vs. re-derived draw")
    ax.set_ylabel("Draw pairs")
    ax.set_title(f"Clustering agreement across Monte Carlo draws (k = {agreement['k']})")
    ax.legend(frameon=False, fontsize=9)
    _recessive(ax)
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


# ------------------------------------------------------- threshold sensitivity
def plot_strictness_sweep(sweep_df: pd.DataFrame, min_criteria_met: int, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(sweep_df["strictness"], sweep_df["n_met"], marker="o", color="#3b6fa0",
            markersize=4, linewidth=1.5)
    ax.axhline(min_criteria_met, color="#8a2b2b", linestyle="--",
                label=f"Support threshold (n_met >= {min_criteria_met})")
    ax.axvline(0, color="#888888", linestyle=":", label="Baseline (config.yaml) thresholds")
    ax.fill_between(sweep_df["strictness"], sweep_df["n_met"], min_criteria_met,
                     where=(sweep_df["n_met"] >= min_criteria_met), alpha=0.15, color="#2f6b3a")
    ax.fill_between(sweep_df["strictness"], sweep_df["n_met"], min_criteria_met,
                     where=(sweep_df["n_met"] < min_criteria_met), alpha=0.15, color="#8a2b2b")
    ax.set_xlabel("Strictness (-1 = most lenient grid endpoint, +1 = most strict)")
    ax.set_ylabel("Criteria met (of 7 scored)")
    ax.set_title("H3 support vs. threshold strictness (all five thresholds moved simultaneously)")
    ax.set_ylim(0, 7.5)
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out


def plot_criterion_fragility(summary: dict, out: Path) -> Path:
    rates = summary["per_criterion_met_rate"]
    labels = [k.replace("_met", "").replace("criterion_", "Criterion ") for k in rates]
    vals = [v * 100 for v in rates.values()]
    order = np.argsort(vals)
    labels = [labels[i] for i in order]
    vals = [vals[i] for i in order]
    colors = ["#8a2b2b" if v < 60 else ("#8a6d1f" if v < 90 else "#2f6b3a") for v in vals]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.barh(labels, vals, color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of grid configurations in which this criterion is met")
    ax.set_title(f"Per-criterion threshold fragility (n={summary['n_grid_points']} grid points)")
    for i, v in enumerate(vals):
        ax.text(v + 1.5, i, f"{v:.0f}%", va="center", fontsize=8.5)
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return out
