"""Binary evaluation metrics for the DEV-risk characterization model."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                              brier_score_loss, confusion_matrix, f1_score,
                              matthews_corrcoef, precision_score, recall_score,
                              roc_auc_score)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> Dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_pos = int((y_true == 1).sum())
    prevalence = n_pos / len(y_true) if len(y_true) else float("nan")
    auroc = float(roc_auc_score(y_true, proba)) if 0 < n_pos < len(y_true) else float("nan")
    auprc = float(average_precision_score(y_true, proba)) if 0 < n_pos < len(y_true) else float("nan")
    return {
        "n": int(len(y_true)), "n_positive": n_pos, "prevalence": prevalence,
        "auroc": auroc, "auprc": auprc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, proba)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
