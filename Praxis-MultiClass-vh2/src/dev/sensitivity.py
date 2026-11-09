"""Sensitivity analyses for H2, run alongside the primary analysis (never
replacing it): diagnosis position, coding intensity / number of secondary
diagnoses, weighted vs. raw, and direction-of-note subsets. Excludes1
"exceptions" and ICD-10 annual-code-set changes are handled structurally
elsewhere (ontology/SOURCE.md's exception-note preservation and the
per-fiscal-year ontology versioning respectively) rather than as a
post-hoc sensitivity run here.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .rates import coding_intensity_stratified_rate, overall_rate


def position_sensitivity(df: pd.DataFrame, dev: np.ndarray, first_position: np.ndarray,
                          weight_col: str = "DISCWT", cutoffs=(5, 10, 15, 25)) -> pd.DataFrame:
    """Recomputes the overall DEV rate restricting to violations found
    only within the first K secondary-diagnosis positions - tests whether
    the rate is driven by early (more clinically salient) vs. late
    (more administrative/lower-priority) positions."""
    rows = []
    for k in cutoffs:
        restricted_dev = dev & (first_position > 0) & (first_position <= k)
        r = overall_rate(df, restricted_dev.astype(np.uint8), weight_col)
        r["max_secondary_position"] = k
        rows.append(r)
    return pd.DataFrame(rows)


def weighting_sensitivity(df: pd.DataFrame, dev: np.ndarray, weight_col: str = "DISCWT") -> pd.DataFrame:
    raw = overall_rate(df, dev, weight_col)
    unweighted_as_weighted = overall_rate(
        df.assign(**{weight_col: 1.0}), dev, weight_col)
    return pd.DataFrame([
        {"scheme": "NIS discharge weight (DISCWT)", **raw},
        {"scheme": "unweighted (all weights=1)", **unweighted_as_weighted},
    ])


def direction_sensitivity(dev_by_direction: Dict[str, np.ndarray], df: pd.DataFrame,
                           weight_col: str = "DISCWT") -> pd.DataFrame:
    """DEV rate recomputed with the ontology restricted to one direction
    at a time (internal_to_phenotype / phenotype_is_home /
    phenotype_is_target) - the three types have different clinical
    interpretations. Caller runs detect() three times, once per
    direction-filtered pairs_map (see main.py), and passes the resulting
    dev arrays in here keyed by direction name."""
    rows = []
    for direction, dev in dev_by_direction.items():
        r = overall_rate(df, dev, weight_col)
        r["direction_subset"] = direction
        rows.append(r)
    return pd.DataFrame(rows)


def coding_intensity_sensitivity(df: pd.DataFrame, dev: np.ndarray) -> pd.DataFrame:
    return coding_intensity_stratified_rate(df, dev)
