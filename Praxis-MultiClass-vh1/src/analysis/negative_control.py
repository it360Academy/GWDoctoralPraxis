"""Negative control: is the observed test-year discrimination better than
chance alignment between the fitted model's predicted probabilities and
the true labels?

Implementation: hold the fitted model's predicted-probability matrix on
the test year FIXED, and repeatedly permute the true label vector against
it, recomputing macro AUROC-OVR each time. This tests the null hypothesis
that there is no real association between the model's predictions and the
true labels, without the cost of refitting the model per permutation -
appropriate here because the quantity under test is discrimination
(ranking), which only depends on the (proba, label) pairing, not on
retraining.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .evaluation import per_class_auroc_ovr


def _macro_auroc(y: np.ndarray, proba: np.ndarray) -> float:
    aurocs = per_class_auroc_ovr(y, proba)
    vals = [v for v in aurocs.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def permutation_test(y_true: np.ndarray, proba: np.ndarray, n_permutations: int = 200,
                      seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    observed = _macro_auroc(y_true, proba)
    null_dist = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm = rng.permutation(y_true)
        null_dist[i] = _macro_auroc(perm, proba)

    n_at_or_above = int((null_dist >= observed).sum())
    p_value = (n_at_or_above + 1) / (n_permutations + 1)

    # Smallest achievable p-value with n_permutations resamples is
    # 1/(n_permutations+1); flag runs that structurally cannot reach the
    # conventional 0.05 threshold rather than silently reporting a p-value
    # that looks precise but isn't.
    min_achievable_p = 1 / (n_permutations + 1)
    underpowered = min_achievable_p > 0.05

    return {
        "observed_macro_auroc_ovr": observed,
        "null_mean": float(np.mean(null_dist)),
        "null_std": float(np.std(null_dist)),
        "n_permutations": n_permutations,
        "p_value": p_value,
        "min_achievable_p": min_achievable_p,
        "underpowered": underpowered,
        "significant_at_0.05": bool(p_value < 0.05 and not underpowered),
    }
