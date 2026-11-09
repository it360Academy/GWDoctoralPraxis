"""Global feature importance for the fitted model (top predictive
DX2-25 categories per phenotype)."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .models import CLASSES


def global_importance(model, algorithm: str, names: List[str], top_n: int = 30) -> pd.DataFrame:
    if algorithm in ("random_forest", "lightgbm", "xgboost"):
        importances = np.asarray(model.feature_importances_, dtype=np.float64)
        df = pd.DataFrame({"feature": names, "importance": importances})
        return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    if algorithm == "logistic_regression":
        coef = np.asarray(model.coef_)  # (n_classes, n_features)
        rows = []
        for i, cls in enumerate(CLASSES[: coef.shape[0]]):
            for j in np.argsort(-np.abs(coef[i]))[:top_n]:
                rows.append({"phenotype": cls, "feature": names[j], "coefficient": float(coef[i, j])})
        return pd.DataFrame(rows)

    return pd.DataFrame(columns=["feature", "importance"])
