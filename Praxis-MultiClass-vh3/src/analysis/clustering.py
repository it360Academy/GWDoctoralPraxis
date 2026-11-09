"""Hierarchical clustering of the fifteen phenotypes' burden structure
(Chapter 3, Section 3.6.4): tests whether the high-burden disease
structure documented in the published benchmark persists as a similar
clustering structure across the published benchmark and each of the
three independently re-derived NIS years. Structural characterization
only, not itself an H3 decision criterion, given fifteen units of
observation is too small for large-sample inferential clustering.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from ..features.literature_inputs import (BIG_THREE_ERROR_FRACTION, OTHER_BUCKETS,
                                          TABLE1_CANCERS, TABLE1_TOP5)


def _feature_matrix(incidence_thousands: Dict[str, float]) -> pd.DataFrame:
    """One row per disease: incidence, harm (incidence x published harm
    rate), category (one-hot), and harm rank, for a given incidence
    structure (published or a specific NIS year)."""
    rows = []
    all_specs = {**TABLE1_TOP5, **TABLE1_CANCERS}
    for name, spec in all_specs.items():
        inc = incidence_thousands.get(name, spec.published_incidence_thousands or 0.0)
        harm = inc * spec.harm_rate
        rows.append({"disease": name, "category": spec.category,
                      "incidence_thousands": inc, "harm_thousands": harm})
    df = pd.DataFrame(rows)
    df["harm_rank"] = df["harm_thousands"].rank(ascending=False, method="average")
    cat_dummies = pd.get_dummies(df["category"], prefix="cat")
    feat = pd.concat([df[["disease", "incidence_thousands", "harm_thousands", "harm_rank"]],
                       cat_dummies], axis=1)
    return feat


def cluster_structure(incidence_thousands: Dict[str, float], n_clusters: int = 4,
                       seed: int = 42) -> pd.DataFrame:
    feat = _feature_matrix(incidence_thousands)
    X = feat.drop(columns=["disease"]).to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)
    Z = linkage(Xs, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    feat["cluster"] = labels
    return feat[["disease", "incidence_thousands", "harm_thousands", "harm_rank", "cluster"]]


def linkage_matrix(incidence_thousands: Dict[str, float]):
    feat = _feature_matrix(incidence_thousands)
    X = feat.drop(columns=["disease"]).to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)
    Z = linkage(Xs, method="ward")
    return Z, feat["disease"].tolist()


def structural_persistence(structures: Dict[str, Dict[str, float]], n_clusters: int = 4,
                            seed: int = 42) -> pd.DataFrame:
    """Pairwise Adjusted Rand Index between every pair of named structures
    (e.g. 'published', '2021', '2022', '2023'), quantifying how similar
    the cluster assignment is across structures. ARI = 1.0 means
    identical clustering; ARI near 0 means no more agreement than chance."""
    names = list(structures.keys())
    clustered = {n: cluster_structure(structures[n], n_clusters, seed) for n in names}
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            merged = clustered[a][["disease", "cluster"]].merge(
                clustered[b][["disease", "cluster"]], on="disease", suffixes=("_a", "_b"))
            ari = adjusted_rand_score(merged["cluster_a"], merged["cluster_b"])
            rows.append({"structure_a": a, "structure_b": b, "adjusted_rand_index": ari})
    return pd.DataFrame(rows)


def cluster_membership(published_incidence: Dict[str, float], rederived_incidence: Dict[str, float],
                        n_clusters: int = 4, seed: int = 42) -> pd.DataFrame:
    """Side-by-side Ward cluster assignment for each phenotype under the
    published and the re-derived structure."""
    pub = cluster_structure(published_incidence, n_clusters, seed)
    red = cluster_structure(rederived_incidence, n_clusters, seed)
    merged = pub[["disease", "harm_thousands", "harm_rank", "cluster"]].merge(
        red[["disease", "harm_thousands", "harm_rank", "cluster"]], on="disease",
        suffixes=("_published", "_rederived"))
    return merged.sort_values(["cluster_published", "harm_rank_published"]).reset_index(drop=True)
