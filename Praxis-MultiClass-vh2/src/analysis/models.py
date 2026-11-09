"""DEV-risk characterization model - binary classifier predicting DEV
likelihood from features that are NOT part of the DEV definition.

Leakage boundary (confirmed): predictors are coding-intensity, hospital
characteristics, demographics, admission circumstances, and DX1's BROAD
phenotype category (which of the 15 phenotypes) - never the exact DX1
code and never any DX2-25 code, since those directly define DEV by
construction (predicting DEV from the codes that define it is circular,
not a modeling result). This makes the model a genuine "who is more
DEV-prone" characterization tool, not a DEV detector (detection is
already deterministic via src/dev/detector.py).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

from ..logging_utils import get_logger

log = get_logger("h2.models")

NUMERIC_FEATURES = ["AGE", "FEMALE", "ELECTIVE", "AWEEKEND", "PAY1", "N_DX_FILLED"]
CATEGORICAL_FEATURES = ["LABEL", "HOSP_REGION", "HOSP_TEACH", "HOSP_BEDSIZE"]


def build_feature_matrix(df: pd.DataFrame, encoder: OneHotEncoder | None = None
                          ) -> Tuple[sparse.csr_matrix, List[str], OneHotEncoder]:
    num = df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    num_medians = num.median()
    num = num.fillna(num_medians)

    cat = df[CATEGORICAL_FEATURES].astype(object).fillna("MISSING").astype(str)
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        cat_enc = encoder.fit_transform(cat)
    else:
        cat_enc = encoder.transform(cat)

    num_sparse = sparse.csr_matrix(num.to_numpy(dtype=np.float64))
    X = sparse.hstack([num_sparse, cat_enc], format="csr")
    names = list(NUMERIC_FEATURES) + list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return X, names, encoder


def fit_model(algorithm: str, X: sparse.csr_matrix, y: np.ndarray,
              params: Dict[str, Any] | None = None, seed: int = 42):
    params = dict(params or {})
    sw = compute_sample_weight("balanced", y)

    if algorithm == "logistic_regression":
        model = LogisticRegression(solver="lbfgs", max_iter=2000,
                                    C=params.get("C", 1.0), random_state=seed, n_jobs=-1)
        model.fit(X, y, sample_weight=sw)
        return model

    if algorithm == "random_forest":
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 400),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            n_jobs=-1, random_state=seed,
        )
        model.fit(X, y, sample_weight=sw)
        return model

    if algorithm == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=params.get("n_estimators", 500),
            learning_rate=params.get("learning_rate", 0.05),
            num_leaves=params.get("num_leaves", 63),
            random_state=seed, n_jobs=-1, verbose=-1,
        )
        model.fit(X, y, sample_weight=sw)
        return model

    if algorithm == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier(
            objective="binary:logistic", n_estimators=params.get("n_estimators", 500),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 6), tree_method="hist",
            random_state=seed, eval_metric="logloss", n_jobs=-1,
        )
        model.fit(X, y, sample_weight=sw)
        return model

    raise ValueError(f"Unknown algorithm: {algorithm}")


def predict_proba(model, X: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(model.predict_proba(X))[:, 1]


# -------------------------------------------------------- unsupervised
def cluster_dev_patterns(X: sparse.csr_matrix, n_clusters: int = 5, seed: int = 42):
    """Unsupervised characterization: k-means over the same leakage-safe
    feature space, to see whether DEV-prone hospitalizations form
    identifiable subgroups (e.g. a high-coding-intensity cluster) rather
    than being uniformly distributed - a complementary, exploratory view
    to the supervised risk model, not a replacement for the prespecified
    Excludes1 detection rule."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(X.toarray() if X.shape[1] < 200 else X)
    return labels, km
