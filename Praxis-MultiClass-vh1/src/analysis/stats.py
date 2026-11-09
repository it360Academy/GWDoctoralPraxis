"""Stratified-bootstrap confidence intervals and paired model comparison."""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

from .evaluation import aggregate_metrics
from .models import CLASSES


def _stratified_resample_idx(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample with replacement WITHIN each class, preserving the
    original class proportions in every bootstrap draw - plain iid
    resampling under severe imbalance can (rarely but non-negligibly)
    drop a whole rare class from a resample, corrupting macro-averaged
    metrics for that draw."""
    idx_parts = []
    for c in CLASSES:
        c_idx = np.flatnonzero(y == c)
        if len(c_idx) == 0:
            continue
        idx_parts.append(rng.choice(c_idx, size=len(c_idx), replace=True))
    return np.concatenate(idx_parts)


def bootstrap_ci_aggregate(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray,
                            n_resamples: int = 1000, ci: float = 0.95,
                            seed: int = 42) -> Dict[str, Tuple[float, float, float]]:
    """Returns {metric_name: (point_estimate, lo, hi)} for every scalar key
    in aggregate_metrics()."""
    rng = np.random.default_rng(seed)
    point = aggregate_metrics(y_true, y_pred, proba)
    samples: Dict[str, list] = {k: [] for k in point if isinstance(point[k], float)}
    for _ in range(n_resamples):
        idx = _stratified_resample_idx(y_true, rng)
        m = aggregate_metrics(y_true[idx], y_pred[idx], proba[idx])
        for k in samples:
            v = m[k]
            if not np.isnan(v):
                samples[k].append(v)
    alpha = (1 - ci) / 2
    out = {}
    for k, vals in samples.items():
        if not vals:
            out[k] = (point[k], float("nan"), float("nan"))
            continue
        lo, hi = np.quantile(vals, [alpha, 1 - alpha])
        out[k] = (point[k], float(lo), float(hi))
    return out


def bootstrap_ci_per_class_auroc(y_true: np.ndarray, proba: np.ndarray,
                                  n_resamples: int = 1000, ci: float = 0.95,
                                  seed: int = 42) -> Dict[str, Tuple[float, float, float]]:
    """Per-class AUROC-OVR bootstrap CI. A class's lower CI bound > 0.5 is
    used by the H1 decision rule as its "statistically distinguishable
    from chance" test (cheaper than a full per-class permutation test,
    and a standard way to read discrimination significance off a CI)."""
    from .evaluation import per_class_auroc_ovr

    rng = np.random.default_rng(seed)
    point = per_class_auroc_ovr(y_true, proba)
    samples: Dict[str, list] = {k: [] for k in point}
    for _ in range(n_resamples):
        idx = _stratified_resample_idx(y_true, rng)
        m = per_class_auroc_ovr(y_true[idx], proba[idx])
        for k in samples:
            if not np.isnan(m[k]):
                samples[k].append(m[k])
    alpha = (1 - ci) / 2
    out = {}
    for k, vals in samples.items():
        if not vals:
            out[k] = (point[k], float("nan"), float("nan"))
            continue
        lo, hi = np.quantile(vals, [alpha, 1 - alpha])
        out[k] = (point[k], float(lo), float(hi))
    return out


def paired_bootstrap_auroc_diff(y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray,
                                 n_resamples: int = 1000, seed: int = 42) -> Dict[str, float]:
    """Paired-bootstrap comparison of macro AUROC-OVR between two models'
    predicted probabilities on the SAME test rows (e.g. best vs. runner-up
    algorithm, or DX2-25-only vs. +covariates sensitivity variant).
    A DeLong test is not directly defined for multiclass macro-OVR AUROC,
    so we use the paired bootstrap here for both binary and multiclass
    comparisons - a single consistent method throughout."""
    from .evaluation import per_class_auroc_ovr

    def macro(y, p):
        aurocs = per_class_auroc_ovr(y, p)
        vals = [v for v in aurocs.values() if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    rng = np.random.default_rng(seed)
    obs_diff = macro(y_true, proba_a) - macro(y_true, proba_b)
    diffs = []
    n = len(y_true)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        d = macro(y_true[idx], proba_a[idx]) - macro(y_true[idx], proba_b[idx])
        if not np.isnan(d):
            diffs.append(d)
    diffs = np.array(diffs)
    p_value = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())) if len(diffs) else float("nan")
    return {
        "observed_diff": obs_diff,
        "ci_lo": float(np.quantile(diffs, 0.025)) if len(diffs) else float("nan"),
        "ci_hi": float(np.quantile(diffs, 0.975)) if len(diffs) else float("nan"),
        "p_value_two_sided": min(p_value, 1.0) if not np.isnan(p_value) else float("nan"),
    }
