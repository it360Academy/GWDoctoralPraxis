"""Isolation forest divergence detection (Chapter 3, Section 3.6.4):
identifies which phenotypes' re-derived profile diverges unusually
strongly from the published structure, supporting identification of
which specific diseases drive any observed replication instability.
Given only fifteen units of observation, this is supporting exploratory
evidence alongside the eight prespecified criteria in Table 3.3, not
itself an H3 decision criterion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..features.literature_inputs import TABLE1_CANCERS, TABLE1_TOP5


def divergence_scores(published_incidence: dict, rederived_incidence: dict,
                       seed: int = 42) -> pd.DataFrame:
    """One row per phenotype: published incidence, re-derived incidence,
    relative difference, and an isolation-forest anomaly score computed
    over the full feature space (relative difference, absolute difference,
    published incidence, re-derived incidence) so a disease can be flagged
    as anomalous either for a large relative shift or a large absolute
    one."""
    all_specs = {**TABLE1_TOP5, **TABLE1_CANCERS}
    rows = []
    for name, spec in all_specs.items():
        pub = published_incidence.get(name, spec.published_incidence_thousands or np.nan)
        red = rederived_incidence.get(name, np.nan)
        rel_diff = (red - pub) / pub if pub else np.nan
        abs_diff = red - pub if pub == pub and red == red else np.nan
        rows.append({"disease": name, "category": spec.category,
                      "published_incidence_thousands": pub,
                      "rederived_incidence_thousands": red,
                      "relative_difference": rel_diff, "absolute_difference": abs_diff})
    df = pd.DataFrame(rows)

    feat_cols = ["relative_difference", "absolute_difference",
                 "published_incidence_thousands", "rederived_incidence_thousands"]
    X = df[feat_cols].fillna(0.0).to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)

    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=seed)
    iso.fit(Xs)
    # more negative decision_function = more anomalous; flip sign so
    # higher score = more anomalous, which reads more naturally in a report
    df["anomaly_score"] = -iso.decision_function(Xs)
    df["flagged_anomalous"] = iso.predict(Xs) == -1
    return df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
