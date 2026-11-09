"""Interpretability for the DEV-risk model: SHAP where the library and
model type support it efficiently, impurity/coefficient-based importance
as a robust fallback (always computed, so a report never has an empty
explainability section because SHAP failed to import or timed out)."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger("h2.explain")


def native_importance(model, algorithm: str, names: List[str], top_n: int = 30) -> pd.DataFrame:
    if algorithm in ("random_forest", "lightgbm", "xgboost"):
        importances = np.asarray(model.feature_importances_, dtype=np.float64)
        df = pd.DataFrame({"feature": names, "importance": importances})
        return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    if algorithm == "logistic_regression":
        coef = np.asarray(model.coef_).ravel()
        df = pd.DataFrame({"feature": names, "coefficient": coef})
        return df.reindex(df["coefficient"].abs().sort_values(ascending=False).index).head(top_n).reset_index(drop=True)
    return pd.DataFrame(columns=["feature", "importance"])


def shap_summary(model, algorithm: str, X_sample, names: List[str], max_rows: int = 2000) -> pd.DataFrame | None:
    """Returns a mean-|SHAP value| per feature table, or None if SHAP
    isn't available/applicable for this algorithm (never raises - falls
    back gracefully so the pipeline keeps running)."""
    if algorithm not in ("random_forest", "lightgbm", "xgboost"):
        return None
    try:
        import shap
    except ImportError:
        log.warning("shap not installed; skipping SHAP summary (native_importance still computed).")
        return None
    try:
        X_dense = X_sample[:max_rows]
        if hasattr(X_dense, "toarray"):
            X_dense = X_dense.toarray()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_dense)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        mean_abs = np.abs(np.asarray(shap_values)).mean(axis=0)
        return pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs}).sort_values(
            "mean_abs_shap", ascending=False).reset_index(drop=True)
    except Exception as exc:  # SHAP has many model/version-specific failure modes
        log.warning("SHAP computation failed (%s); falling back to native importance only.", exc)
        return None
