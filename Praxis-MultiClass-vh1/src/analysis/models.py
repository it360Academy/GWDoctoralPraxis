"""Multiclass model training (logistic regression / random forest /
LightGBM / XGBoost), 15-class softmax over the phenotype label.

Class imbalance is handled uniformly across all four algorithms via
sklearn's `compute_sample_weight("balanced", y)` on the TRAINING data
only, rather than relying on each library's own (inconsistent)
`class_weight` support - this keeps "cost-sensitive weighting" a single,
auditable definition regardless of which algorithm produced a result.
DISCWT (the NIS survey weight) is reserved for national-estimate
reporting (weighted prevalence) and is NOT used as a model-fitting
weight, matching standard practice: sampling-design weights and
class-imbalance weights answer different questions and conflating them
would bias both.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from ..features.definitions import DISEASE_ORDER
from ..logging_utils import get_logger

log = get_logger("h1.models")

CLASSES: List[str] = list(DISEASE_ORDER)


def _label_encode(y: np.ndarray) -> np.ndarray:
    idx = {c: i for i, c in enumerate(CLASSES)}
    return np.array([idx[v] for v in y], dtype=np.int64)


def fit_model(algorithm: str, X: sparse.csr_matrix, y: np.ndarray, cfg,
              params: Dict[str, Any] | None = None, seed: int = 42):
    params = dict(params or {})
    sw = compute_sample_weight("balanced", y)
    y_enc = _label_encode(y)
    n_classes = len(CLASSES)

    if algorithm == "logistic_regression":
        # sklearn >=1.5 removed the `multi_class` kwarg; lbfgs already
        # fits a true multinomial (softmax) model for >2 classes by
        # default, so nothing else needs to change.
        model = LogisticRegression(
            solver="lbfgs", max_iter=2000,
            C=params.get("C", 1.0), random_state=seed, n_jobs=-1,
        )
        model.fit(X, y_enc, sample_weight=sw)
        return model

    if algorithm == "random_forest":
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 400),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            n_jobs=-1, random_state=seed,
        )
        model.fit(X, y_enc, sample_weight=sw)
        return model

    if algorithm == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            objective="multiclass", num_class=n_classes,
            n_estimators=params.get("n_estimators", 500),
            learning_rate=params.get("learning_rate", 0.05),
            num_leaves=params.get("num_leaves", 63),
            random_state=seed, n_jobs=-1, verbose=-1,
        )
        model.fit(X, y_enc, sample_weight=sw)
        return model

    if algorithm == "xgboost":
        import xgboost as xgb
        tree_method = "gpu_hist" if cfg["models"].get("xgboost_gpu", False) else "hist"
        model = xgb.XGBClassifier(
            objective="multi:softprob", num_class=n_classes,
            n_estimators=params.get("n_estimators", 500),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 6),
            tree_method=tree_method, random_state=seed,
            eval_metric="mlogloss", n_jobs=-1,
        )
        model.fit(X, y_enc, sample_weight=sw)
        return model

    raise ValueError(f"Unknown algorithm: {algorithm}")


def predict_proba(model, algorithm: str, X: sparse.csr_matrix) -> np.ndarray:
    """Always returns an (n, 15) probability matrix in CLASSES order."""
    proba = model.predict_proba(X)
    return np.asarray(proba)


def predict_label(model, algorithm: str, X: sparse.csr_matrix) -> np.ndarray:
    proba = predict_proba(model, algorithm, X)
    idx = proba.argmax(axis=1)
    return np.array([CLASSES[i] for i in idx], dtype=object)
