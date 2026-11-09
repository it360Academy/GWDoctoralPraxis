"""Optional randomized-search tuning, off by default (config:
tuning.enabled, or --tune/--no-tune on the CLI). Tuned on the TRAINING
year via cross-validation only - the validation/test years are never
touched during tuning, only during model selection/final evaluation."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy import sparse
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight

from ..logging_utils import get_logger
from .models import CLASSES

log = get_logger("h1.tuning")

_PARAM_DISTRIBUTIONS = {
    "logistic_regression": {"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
    "random_forest": {
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 10, 20, 40],
        "min_samples_leaf": [1, 3, 5, 10],
    },
    "lightgbm": {
        "n_estimators": [200, 400, 600],
        "learning_rate": [0.02, 0.05, 0.1],
        "num_leaves": [31, 63, 127],
    },
    "xgboost": {
        "n_estimators": [200, 400, 600],
        "learning_rate": [0.02, 0.05, 0.1],
        "max_depth": [4, 6, 8],
    },
}


def _base_estimator(algorithm: str, cfg, seed: int):
    if algorithm == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(solver="lbfgs", max_iter=2000,
                                   random_state=seed, n_jobs=-1)
    if algorithm == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=seed, n_jobs=-1)
    if algorithm == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(objective="multiclass", num_class=len(CLASSES),
                                   random_state=seed, n_jobs=-1, verbose=-1)
    if algorithm == "xgboost":
        import xgboost as xgb
        tree_method = "gpu_hist" if cfg["models"].get("xgboost_gpu", False) else "hist"
        return xgb.XGBClassifier(objective="multi:softprob", num_class=len(CLASSES),
                                  tree_method=tree_method, random_state=seed,
                                  eval_metric="mlogloss", n_jobs=-1)
    raise ValueError(algorithm)


def tune_params(algorithm: str, X: sparse.csr_matrix, y: np.ndarray, cfg, seed: int = 42) -> Dict[str, Any]:
    idx = {c: i for i, c in enumerate(CLASSES)}
    y_enc = np.array([idx[v] for v in y], dtype=np.int64)
    sw = compute_sample_weight("balanced", y_enc)

    dist = _PARAM_DISTRIBUTIONS[algorithm]
    n_iter = min(int(cfg["tuning"].get("n_iter", 20)), int(np.prod([len(v) for v in dist.values()])))
    cv = int(cfg["tuning"].get("cv_folds", 3))

    search = RandomizedSearchCV(
        _base_estimator(algorithm, cfg, seed), param_distributions=dist,
        n_iter=n_iter, cv=cv, scoring="f1_macro", random_state=seed, n_jobs=-1,
    )
    fit_params = {"sample_weight": sw}
    log.info("Tuning %s: n_iter=%d cv=%d", algorithm, n_iter, cv)
    search.fit(X, y_enc, **fit_params)
    log.info("Tuning %s: best params=%s best_cv_f1_macro=%.4f", algorithm,
              search.best_params_, search.best_score_)
    return dict(search.best_params_)
