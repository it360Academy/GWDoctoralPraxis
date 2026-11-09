"""Track 2 - ML structural validation of H3 (clustering agreement).

This module is a genuine ML extension of the eight prespecified,
purely-statistical H3 criteria in ``criteria.py`` (Track 1, unchanged).
Where Track 1 checks whether individual summary numbers (a rank
correlation, a percentage-point gap, an interval overlap) replicate
between the published benchmark and this project's independent
re-derivation, Track 2 asks a structurally different question: does the
*multidimensional grouping* of the fifteen phenotypes by burden profile
-- not just their scalar ranking -- replicate?

Two independent, unsupervised clusterings are fit -- one on the published
structure's feature matrix, one on the re-derived structure's feature
matrix -- and agreement between the two label sets is measured with
Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI). Both
K-means and Ward hierarchical clustering are used (agreement stable
across both, or not, is itself informative), and agreement is tested
against a permutation null (is the observed ARI/NMI higher than would
occur from randomly relabeled clusters of the same sizes, given only
fifteen units of observation).

This is exploratory/supporting evidence alongside, not a replacement
for, the eight prespecified Track 1 criteria -- consistent with the
existing (now superseded-in-scope) ``clustering.py`` module's own stated
framing that fifteen units of observation is too small for confirmatory
inferential clustering. It is reported with its own explicit verdict
language, kept separate from ``H3_SUPPORTED``, so it neither dilutes nor
duplicates Track 1's decision criteria.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                              silhouette_samples, silhouette_score)
from sklearn.preprocessing import StandardScaler

from ..features.literature_inputs import TABLE1_CANCERS, TABLE1_TOP5

ALL_SPECS = {**TABLE1_TOP5, **TABLE1_CANCERS}
DISEASE_ORDER: List[str] = list(ALL_SPECS.keys())


# --------------------------------------------------------------- features
def _triangular(rng: np.random.Generator, low: float, mode: float, high: float,
                 n: int) -> np.ndarray:
    lo, hi = min(low, mode), max(high, mode)
    if lo == hi:
        return np.full(n, mode)
    return rng.triangular(lo, mode, hi, size=n)


def simulate_harm_draws(incidence_thousands: Dict[str, float], n_draws: int = 10000,
                         seed: int = 42) -> Dict[str, np.ndarray]:
    """Per-disease harm draws (thousands) for a given incidence structure,
    sampling only the published harm-rate uncertainty (triangular about
    the point estimate, bounded by the paper's reported range) and
    holding incidence at the structure's point value. This isolates
    burden *uncertainty width* per phenotype as a clustering feature,
    independent of and lighter-weight than the full Monte Carlo pipeline
    in montecarlo.py (which additionally samples incidence and error
    rates and aggregates category totals -- not needed here, since
    Track 2 clusters the fifteen phenotypes themselves, not the grand
    total)."""
    rng = np.random.default_rng(seed)
    draws: Dict[str, np.ndarray] = {}
    for name, spec in ALL_SPECS.items():
        inc = incidence_thousands.get(name, spec.published_incidence_thousands or 0.0)
        harm_rate_draws = _triangular(rng, spec.harm_rate_low, spec.harm_rate,
                                       spec.harm_rate_high, n_draws)
        draws[name] = inc * harm_rate_draws
    return draws


def build_feature_matrix(incidence_thousands: Dict[str, float],
                          harm_draws: Dict[str, np.ndarray] | None = None,
                          include_uncertainty: bool = True,
                          include_category: bool = True) -> pd.DataFrame:
    """One row per phenotype: incidence, point harm, harm rank, optional
    95% uncertainty width (from ``harm_draws``), optional one-hot
    category. ``include_uncertainty``/``include_category`` toggle two of
    the four feature groups, used for the feature-set robustness check
    (Section: does agreement depend on which features are included)."""
    rows = []
    for name, spec in ALL_SPECS.items():
        inc = incidence_thousands.get(name, spec.published_incidence_thousands or 0.0)
        harm = inc * spec.harm_rate
        row = {"disease": name, "category": spec.category,
               "incidence_thousands": inc, "harm_thousands": harm}
        if include_uncertainty and harm_draws is not None:
            d = harm_draws[name]
            row["harm_uncertainty_width"] = float(np.quantile(d, 0.975) - np.quantile(d, 0.025))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["harm_rank"] = df["harm_thousands"].rank(ascending=False, method="average")

    feature_cols = ["incidence_thousands", "harm_thousands", "harm_rank"]
    if include_uncertainty and harm_draws is not None:
        feature_cols.append("harm_uncertainty_width")

    out = df[["disease", "category"] + feature_cols].copy()
    if include_category:
        cat_dummies = pd.get_dummies(df["category"], prefix="cat")
        out = pd.concat([out, cat_dummies], axis=1)
    return out


def _feature_cols(feat: pd.DataFrame) -> List[str]:
    return [c for c in feat.columns if c not in ("disease", "category")]


def standardized_matrix(feat: pd.DataFrame) -> np.ndarray:
    X = feat[_feature_cols(feat)].to_numpy(dtype=float)
    return StandardScaler().fit_transform(X)


# --------------------------------------------------------------- k selection
def select_k_by_silhouette(X: np.ndarray, k_range: Tuple[int, ...] = (2, 3, 4),
                            seed: int = 42) -> pd.DataFrame:
    rows = []
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        score = silhouette_score(X, labels)
        rows.append({"k": k, "silhouette_score": score})
    return pd.DataFrame(rows).sort_values("silhouette_score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------- clustering
def cluster_kmeans(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


def cluster_hierarchical(X: np.ndarray, k: int) -> np.ndarray:
    Z = linkage(X, method="ward")
    return fcluster(Z, t=k, criterion="maxclust")


def agreement_metrics(labels_a: np.ndarray, labels_b: np.ndarray) -> Dict[str, float]:
    return {
        "adjusted_rand_index": float(adjusted_rand_score(labels_a, labels_b)),
        "normalized_mutual_info": float(normalized_mutual_info_score(labels_a, labels_b)),
    }


# --------------------------------------------------------------- significance
def permutation_test(labels_a: np.ndarray, labels_b: np.ndarray, n_perm: int = 10000,
                      seed: int = 42) -> Dict[str, float]:
    """Empirical p-value for observed ARI and NMI against a null of
    randomly permuted cluster labels (same label multiset as labels_b,
    shuffled across the fifteen phenotypes). One-sided: p = share of
    permutations whose agreement with labels_a is >= the observed
    agreement. Directly addresses the "could fifteen items agree this
    much by chance" concern that small-n clustering invites."""
    rng = np.random.default_rng(seed)
    obs_ari = adjusted_rand_score(labels_a, labels_b)
    obs_nmi = normalized_mutual_info_score(labels_a, labels_b)
    n = len(labels_b)
    perm_ari = np.empty(n_perm)
    perm_nmi = np.empty(n_perm)
    b = np.asarray(labels_b)
    for i in range(n_perm):
        shuffled = rng.permutation(b)
        perm_ari[i] = adjusted_rand_score(labels_a, shuffled)
        perm_nmi[i] = normalized_mutual_info_score(labels_a, shuffled)
    p_ari = float((np.sum(perm_ari >= obs_ari) + 1) / (n_perm + 1))
    p_nmi = float((np.sum(perm_nmi >= obs_nmi) + 1) / (n_perm + 1))
    return {"observed_ari": float(obs_ari), "observed_nmi": float(obs_nmi),
            "p_value_ari": p_ari, "p_value_nmi": p_nmi,
            "null_ari_mean": float(perm_ari.mean()), "null_nmi_mean": float(perm_nmi.mean()),
            "n_permutations": n_perm}


def _verdict_label(ari: float, p_value: float) -> str:
    if p_value >= 0.05:
        return "not distinguishable from chance"
    if ari >= 0.6:
        return "strong agreement"
    if ari >= 0.3:
        return "moderate agreement"
    return "weak agreement"


# --------------------------------------------------------------- robustness
def k_sensitivity(feat_pub: pd.DataFrame, feat_red: pd.DataFrame, k_range: Tuple[int, ...],
                   seed: int) -> pd.DataFrame:
    Xp, Xr = standardized_matrix(feat_pub), standardized_matrix(feat_red)
    rows = []
    for k in k_range:
        for algo, fn in (("kmeans", lambda X: cluster_kmeans(X, k, seed)),
                          ("hierarchical", lambda X: cluster_hierarchical(X, k))):
            la, lb = fn(Xp), fn(Xr)
            m = agreement_metrics(la, lb)
            rows.append({"k": k, "algorithm": algo, **m})
    return pd.DataFrame(rows)


def feature_set_sensitivity(published_inc: Dict[str, float], rederived_inc: Dict[str, float],
                             harm_draws_pub: Dict[str, np.ndarray], harm_draws_red: Dict[str, np.ndarray],
                             k: int, seed: int) -> pd.DataFrame:
    variants = [
        ("full (incidence+harm+rank+uncertainty+category)", True, True),
        ("no uncertainty", False, True),
        ("no category", True, False),
        ("minimal (incidence+harm+rank only)", False, False),
    ]
    rows = []
    for label, inc_unc, inc_cat in variants:
        fp = build_feature_matrix(published_inc, harm_draws_pub, inc_unc, inc_cat)
        fr = build_feature_matrix(rederived_inc, harm_draws_red, inc_unc, inc_cat)
        Xp, Xr = standardized_matrix(fp), standardized_matrix(fr)
        for algo, fn in (("kmeans", lambda X: cluster_kmeans(X, k, seed)),
                          ("hierarchical", lambda X: cluster_hierarchical(X, k))):
            la, lb = fn(Xp), fn(Xr)
            m = agreement_metrics(la, lb)
            rows.append({"feature_set": label, "algorithm": algo, **m})
    return pd.DataFrame(rows)


def leave_one_out_stability(feat_pub: pd.DataFrame, feat_red: pd.DataFrame, k: int,
                             algorithm: str, seed: int) -> pd.DataFrame:
    """Recompute primary agreement with each phenotype removed in turn,
    to check whether the overall agreement result depends heavily on any
    single disease."""
    cluster_fn = {"kmeans": lambda X: cluster_kmeans(X, k, seed),
                  "hierarchical": lambda X: cluster_hierarchical(X, k)}[algorithm]
    rows = []
    diseases = feat_pub["disease"].tolist()
    for held_out in diseases:
        fp = feat_pub[feat_pub["disease"] != held_out].reset_index(drop=True)
        fr = feat_red[feat_red["disease"] != held_out].reset_index(drop=True)
        Xp, Xr = standardized_matrix(fp), standardized_matrix(fr)
        la, lb = cluster_fn(Xp), cluster_fn(Xr)
        m = agreement_metrics(la, lb)
        rows.append({"held_out_disease": held_out, **m})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- viz support
def pca_projection(feat: pd.DataFrame, labels: np.ndarray, seed: int = 42) -> pd.DataFrame:
    X = standardized_matrix(feat)
    n_components = min(2, X.shape[1])
    coords = PCA(n_components=n_components, random_state=seed).fit_transform(X)
    out = feat[["disease", "category"]].copy()
    out["pc1"] = coords[:, 0]
    out["pc2"] = coords[:, 1] if n_components > 1 else 0.0
    out["cluster"] = labels
    return out


def silhouette_table(feat: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    X = standardized_matrix(feat)
    vals = silhouette_samples(X, labels)
    out = feat[["disease", "category"]].copy()
    out["cluster"] = labels
    out["silhouette"] = vals
    return out


def crosstab(labels_pub: np.ndarray, labels_red: np.ndarray, diseases: List[str]) -> pd.DataFrame:
    df = pd.DataFrame({"disease": diseases, "published_cluster": labels_pub,
                        "rederived_cluster": labels_red})
    return pd.crosstab(df["published_cluster"], df["rederived_cluster"])


# --------------------------------------------------------------- orchestration
def run_ml_validation(published_incidence: Dict[str, float], rederived_incidence: Dict[str, float],
                       k_range: Tuple[int, ...] = (2, 3, 4), n_perm: int = 10000,
                       seed: int = 42) -> Dict:
    """Full Track 2 pipeline: k-selection, primary K-means + hierarchical
    clustering agreement (with permutation significance), and the three
    robustness checks (k, algorithm, feature set, leave-one-out)."""
    harm_draws_pub = simulate_harm_draws(published_incidence, seed=seed)
    harm_draws_red = simulate_harm_draws(rederived_incidence, seed=seed)

    feat_pub = build_feature_matrix(published_incidence, harm_draws_pub)
    feat_red = build_feature_matrix(rederived_incidence, harm_draws_red)
    Xp, Xr = standardized_matrix(feat_pub), standardized_matrix(feat_red)

    k_select = select_k_by_silhouette(Xp, k_range, seed=seed)
    k_star = int(k_select.iloc[0]["k"])

    labels_pub_km = cluster_kmeans(Xp, k_star, seed)
    labels_red_km = cluster_kmeans(Xr, k_star, seed)
    labels_pub_hc = cluster_hierarchical(Xp, k_star)
    labels_red_hc = cluster_hierarchical(Xr, k_star)

    primary_kmeans = permutation_test(labels_pub_km, labels_red_km, n_perm=n_perm, seed=seed)
    primary_hier = permutation_test(labels_pub_hc, labels_red_hc, n_perm=n_perm, seed=seed + 1)

    primary_kmeans["verdict"] = _verdict_label(primary_kmeans["observed_ari"], primary_kmeans["p_value_ari"])
    primary_hier["verdict"] = _verdict_label(primary_hier["observed_ari"], primary_hier["p_value_ari"])

    k_sens = k_sensitivity(feat_pub, feat_red, k_range, seed)
    feat_sens = feature_set_sensitivity(published_incidence, rederived_incidence,
                                         harm_draws_pub, harm_draws_red, k_star, seed)
    loo_kmeans = leave_one_out_stability(feat_pub, feat_red, k_star, "kmeans", seed)
    loo_hier = leave_one_out_stability(feat_pub, feat_red, k_star, "hierarchical", seed)

    pca_pub = pca_projection(feat_pub, labels_pub_km, seed)
    pca_red = pca_projection(feat_red, labels_red_km, seed)
    sil_pub = silhouette_table(feat_pub, labels_pub_km)
    sil_red = silhouette_table(feat_red, labels_red_km)
    ct = crosstab(labels_pub_km, labels_red_km, feat_pub["disease"].tolist())

    return {
        "k_selection": k_select, "k_star": k_star,
        "primary_kmeans": primary_kmeans, "primary_hierarchical": primary_hier,
        "k_sensitivity": k_sens, "feature_set_sensitivity": feat_sens,
        "leave_one_out_kmeans": loo_kmeans, "leave_one_out_hierarchical": loo_hier,
        "pca_published": pca_pub, "pca_rederived": pca_red,
        "silhouette_published": sil_pub, "silhouette_rederived": sil_red,
        "crosstab": ct,
        "feat_published": feat_pub, "feat_rederived": feat_red,
        "labels_published_kmeans": labels_pub_km, "labels_rederived_kmeans": labels_red_km,
        "labels_published_hierarchical": labels_pub_hc, "labels_rederived_hierarchical": labels_red_hc,
    }
