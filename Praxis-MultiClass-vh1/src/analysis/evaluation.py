"""Multiclass evaluation metrics for the 15-phenotype DX1 label."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                              f1_score, matthews_corrcoef, precision_score,
                              recall_score, roc_auc_score)

from .models import CLASSES


def _one_hot(y: np.ndarray) -> np.ndarray:
    idx = {c: i for i, c in enumerate(CLASSES)}
    n = len(y)
    out = np.zeros((n, len(CLASSES)), dtype=np.float64)
    for i, v in enumerate(y):
        out[i, idx[v]] = 1.0
    return out


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    onehot = _one_hot(y_true)
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def per_class_auroc_ovr(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    onehot = _one_hot(y_true)
    out = {}
    for i, c in enumerate(CLASSES):
        y_c = onehot[:, i]
        if y_c.sum() == 0 or y_c.sum() == len(y_c):
            out[c] = float("nan")
            continue
        out[c] = float(roc_auc_score(y_c, proba[:, i]))
    return out


def per_class_table(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    onehot = _one_hot(y_true)
    aurocs = per_class_auroc_ovr(y_true, proba)
    rows = []
    for i, c in enumerate(CLASSES):
        y_c = onehot[:, i]
        pred_c = (y_pred == c).astype(np.float64)
        tp = float(((y_c == 1) & (pred_c == 1)).sum())
        fp = float(((y_c == 0) & (pred_c == 1)).sum())
        fn = float(((y_c == 1) & (pred_c == 0)).sum())
        tn = float(((y_c == 0) & (pred_c == 0)).sum())
        n_pos = float(y_c.sum())
        prevalence = n_pos / len(y_c) if len(y_c) else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) else float("nan")
        npv = tn / (tn + fn) if (tn + fn) else float("nan")
        null_rate = prevalence  # PPV a naive "always predict this class" rule would get by chance
        ppv_lift = precision / null_rate if null_rate not in (0, float("nan")) and not np.isnan(null_rate) and null_rate > 0 else float("nan")
        rows.append({
            "phenotype": c, "n_positive": int(n_pos), "prevalence": prevalence,
            "auroc_ovr": aurocs[c], "precision_ppv": precision, "recall_sensitivity": recall,
            "specificity": specificity, "npv": npv, "ppv_lift_over_null": ppv_lift,
            "f1": (2 * precision * recall / (precision + recall)
                   if precision and recall and not np.isnan(precision) and not np.isnan(recall)
                   and (precision + recall) > 0 else float("nan")),
        })
    return pd.DataFrame(rows)


def confusion_matrix_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    return pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES],
                         columns=[f"pred_{c}" for c in CLASSES])


def aggregate_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    aurocs = per_class_auroc_ovr(y_true, proba)
    valid_aurocs = [v for v in aurocs.values() if not np.isnan(v)]
    return {
        "macro_auroc_ovr": float(np.mean(valid_aurocs)) if valid_aurocs else float("nan"),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier_multiclass": multiclass_brier(y_true, proba),
        "n": int(len(y_true)),
        "n_classes_present": int(len(set(y_true))),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> Dict:
    return {
        "aggregate": aggregate_metrics(y_true, y_pred, proba),
        "per_class": per_class_table(y_true, y_pred, proba),
        "confusion_matrix": confusion_matrix_df(y_true, y_pred),
    }
