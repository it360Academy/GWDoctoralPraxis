"""Consensus hierarchical clustering of the fifteen phenotypes over Monte
Carlo draws (report Section 8).

A single clustering of fifteen point estimates says nothing about how
stable each assignment is. Here each Monte Carlo draw - one plausible joint
sample of incidence and harm-rate uncertainty - is clustered separately on
per-phenotype log serious harm, and the results are pooled into a 15 x 15
co-clustering matrix: the share of draws in which each pair of phenotypes
lands in the same cluster. A consensus clustering is then cut from that
matrix (average linkage on 1 - co-clustering).

Design choices, stated explicitly:
  - Feature = log harm only. Disease category is deliberately excluded: it
    is identical in the published and re-derived structures by construction
    and would inflate agreement without testing anything. Logs keep the
    largest phenotypes (sepsis, stroke) from dominating the distance.
  - k is selected by silhouette on the *published* structure's median log
    harm only, then applied identically to both structures, so k cannot be
    chosen in a way that favors agreement.
  - Consensus clusters are relabeled as harm tiers (1 = highest median harm)
    so labels are comparable across structures.
  - Agreement is reported for all fifteen phenotypes and for the ten
    NIS-derived phenotypes alone, since the five cancers are identical in
    both structures by design.

Structural characterization only; not an H3 decision criterion.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score

from ..features.literature_inputs import TABLE1_CANCERS, TABLE1_TOP5
from .ml_validation import permutation_test

PHENOTYPES: List[str] = list(TABLE1_TOP5) + list(TABLE1_CANCERS)
INDEPENDENT_IDX = [i for i, d in enumerate(PHENOTYPES) if d in TABLE1_TOP5]


def log_harm_matrix(draws: pd.DataFrame, n_draws: int, seed: int) -> np.ndarray:
    """(n_draws, 15) matrix of log per-phenotype harm, subsampled from the
    Monte Carlo draws without replacement."""
    harms = draws[[f"harms_{d}_thousands" for d in PHENOTYPES]].to_numpy(dtype=float)
    if len(harms) > n_draws:
        idx = np.random.default_rng(seed).choice(len(harms), size=n_draws, replace=False)
        harms = harms[idx]
    return np.log(np.clip(harms, 1e-9, None))


def _cut(values: np.ndarray, k: int, method: str) -> np.ndarray:
    Z = linkage(values.reshape(-1, 1), method=method)
    return fcluster(Z, t=k, criterion="maxclust")


def select_k(point_log_harm: np.ndarray, k_range, method: str) -> Tuple[int, pd.DataFrame]:
    rows = []
    for k in k_range:
        if not 2 <= k <= len(point_log_harm) - 1:
            continue
        labels = _cut(point_log_harm, k, method)
        sil = (silhouette_score(point_log_harm.reshape(-1, 1), labels)
               if len(np.unique(labels)) > 1 else np.nan)
        rows.append({"k": k, "silhouette_published": float(sil)})
    table = pd.DataFrame(rows)
    return int(table.loc[table["silhouette_published"].idxmax(), "k"]), table


def coclustering(log_harm: np.ndarray, k: int, method: str) -> Tuple[np.ndarray, np.ndarray]:
    """Per-draw labels (n_draws, 15) and the co-clustering matrix (15, 15)."""
    labels = np.array([_cut(row, k, method) for row in log_harm])
    same = labels[:, :, None] == labels[:, None, :]
    return same.mean(axis=0), labels


def consensus_labels(C: np.ndarray, k: int, point_log_harm: np.ndarray) -> np.ndarray:
    """Average-linkage cut of 1 - C, relabeled so tier 1 has the highest
    mean log harm."""
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    raw = fcluster(linkage(squareform(D, checks=False), method="average"), t=k, criterion="maxclust")
    order = sorted(np.unique(raw), key=lambda c: -point_log_harm[raw == c].mean())
    remap = {c: i + 1 for i, c in enumerate(order)}
    return np.array([remap[c] for c in raw])


def _stability(C: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per phenotype: mean co-clustering with the other members of its
    consensus tier (NaN for a singleton), and the highest co-clustering
    with any phenotype outside it."""
    n = len(labels)
    within, outside = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        same = [j for j in range(n) if j != i and labels[j] == labels[i]]
        other = [j for j in range(n) if labels[j] != labels[i]]
        if same:
            within[i] = C[i, same].mean()
        if other:
            outside[i] = C[i, other].max()
    return within, outside


def run_consensus(draws_published: pd.DataFrame, draws_rederived: pd.DataFrame,
                  n_draws: int = 1000, k_range=(2, 3, 4, 5), method: str = "ward",
                  n_perm: int = 10000, seed: int = 42) -> Dict:
    L_pub = log_harm_matrix(draws_published, n_draws, seed)
    L_red = log_harm_matrix(draws_rederived, n_draws, seed + 1)
    point_pub, point_red = np.median(L_pub, axis=0), np.median(L_red, axis=0)

    k, k_table = select_k(point_pub, k_range, method)
    C_pub, per_draw_pub = coclustering(L_pub, k, method)
    C_red, per_draw_red = coclustering(L_red, k, method)
    cons_pub = consensus_labels(C_pub, k, point_pub)
    cons_red = consensus_labels(C_red, k, point_red)

    agreement = permutation_test(cons_pub, cons_red, n_perm=n_perm, seed=seed)
    agreement["ari_10_independent"] = float(adjusted_rand_score(cons_pub[INDEPENDENT_IDX],
                                                                cons_red[INDEPENDENT_IDX]))
    m = min(len(per_draw_pub), len(per_draw_red))
    per_draw_ari = np.array([adjusted_rand_score(per_draw_pub[i], per_draw_red[i]) for i in range(m)])
    agreement.update({"k": k, "n_draws": int(m), "method": method,
                      "per_draw_ari_median": float(np.median(per_draw_ari)),
                      "per_draw_ari_p2_5": float(np.quantile(per_draw_ari, 0.025)),
                      "per_draw_ari_p97_5": float(np.quantile(per_draw_ari, 0.975)),
                      "n_same_tier": int((cons_pub == cons_red).sum()),
                      "n_same_tier_10_independent": int((cons_pub[INDEPENDENT_IDX]
                                                         == cons_red[INDEPENDENT_IDX]).sum())})

    within_pub, outside_pub = _stability(C_pub, cons_pub)
    within_red, outside_red = _stability(C_red, cons_red)
    membership = pd.DataFrame({
        "disease": PHENOTYPES,
        "independent": [d in TABLE1_TOP5 for d in PHENOTYPES],
        "published_median_harm_thousands": np.exp(point_pub),
        "rederived_median_harm_thousands": np.exp(point_red),
        "published_tier": cons_pub, "rederived_tier": cons_red,
        "same_tier": cons_pub == cons_red,
        "published_within_tier_coclustering": within_pub,
        "published_max_outside_coclustering": outside_pub,
        "rederived_within_tier_coclustering": within_red,
        "rederived_max_outside_coclustering": outside_red,
    }).sort_values(["published_tier", "published_median_harm_thousands"],
                   ascending=[True, False]).reset_index(drop=True)

    return {"k_selection": k_table, "membership": membership, "agreement": agreement,
            "coclustering_published": pd.DataFrame(C_pub, index=PHENOTYPES, columns=PHENOTYPES),
            "coclustering_rederived": pd.DataFrame(C_red, index=PHENOTYPES, columns=PHENOTYPES),
            "per_draw_ari": pd.DataFrame({"adjusted_rand_index": per_draw_ari})}
