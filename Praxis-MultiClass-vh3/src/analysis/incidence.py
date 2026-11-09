"""Independently computed DISCWT-weighted national incidence for the ten
vascular/infectious phenotypes, three-year average (2021-2023) - the
genuinely independent, NIS-computed component of the H3 replication.
Compared directly against the published incidence in Table 1 of
Newman-Toker et al. (2024) (literature_inputs.py TABLE1_TOP5).

Primary incidence counts a discharge when the phenotype is coded in DX1 OR
DX2, matching the published Table 1 definition; the DX1-only variant is
computed separately for criterion 8.

**Limitation, stated explicitly**: the source paper used SAS
PROC SURVEYMEANS, which models NIS's full complex survey design (strata +
sampled hospital clusters) for design-based variance. This project uses a
discharge-level bootstrap, which understates true sampling uncertainty
somewhat since it does not account for within-hospital clustering. Point
estimates should be comparable; CI widths may differ. This is the same
documented limitation as the H1/H2 projects in this series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..data.ingestion import VASC_INFECT_DISEASES, load_year
from ..features.definitions import DISEASE_ORDER
from ..logging_utils import get_logger

_LABELS = {d: d.replace("_", " ").title() for d in VASC_INFECT_DISEASES}


def _bootstrap_ci(y: np.ndarray, w: np.ndarray, n_boot: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    n = len(y)
    estimates = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        estimates[b] = (y[idx] * w[idx]).sum()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def compute_annual_incidence(cfg: Config, column_prefix: str = "PRIN_OR_SEC1") -> pd.DataFrame:
    """Per-year weighted incidence for each of the ten computed phenotypes.

    ``column_prefix``: "PRIN_OR_SEC1" (primary definition, DX1 or DX2,
    matching the published paper's main analysis) or "PRIN_ONLY" (DX1-only
    sensitivity variant, used only for criterion 8, Chapter 3 Section 3.6.3).
    """
    log = get_logger("h3.incidence")
    n_boot = int(cfg.get("incidence", "bootstrap_iterations", default=200))
    bootstrap_max_rows = int(cfg.get("incidence", "bootstrap_max_rows", default=1_000_000))
    rows = []
    for role, year in cfg.years.items():
        df = load_year(cfg, year)
        w = df["DISCWT"].to_numpy()
        for disease in VASC_INFECT_DISEASES:
            y = df[f"{column_prefix}_{disease}"].to_numpy()
            weighted_n = float((y * w).sum())
            if len(df) > bootstrap_max_rows:
                rng = np.random.default_rng(cfg.seed)
                idx = rng.choice(len(df), size=bootstrap_max_rows, replace=False)
                scale = len(df) / bootstrap_max_rows
                lo, hi = _bootstrap_ci(y[idx], w[idx], n_boot, cfg.seed)
                lo, hi = lo * scale, hi * scale
            else:
                lo, hi = _bootstrap_ci(y, w, n_boot, cfg.seed)
            rows.append({"role": role, "year": year, "disease": disease,
                         "label": _LABELS[disease], "weighted_incidence": weighted_n,
                         "weighted_incidence_ci_low": lo, "weighted_incidence_ci_high": hi})
        log.info("[%s/%s] incidence computed for %d phenotypes (definition=%s)",
                  role, year, len(VASC_INFECT_DISEASES), column_prefix)
    return pd.DataFrame(rows)


def three_year_average(annual: pd.DataFrame) -> pd.DataFrame:
    """Three-year average incidence per phenotype, matching the published
    paper's stated approach ("a three-year average was chosen to improve
    stability of incidence measures for rare conditions")."""
    agg = annual.groupby(["disease", "label"]).agg(
        mean_annual_incidence=("weighted_incidence", "mean"),
        mean_ci_low=("weighted_incidence_ci_low", "mean"),
        mean_ci_high=("weighted_incidence_ci_high", "mean"),
    ).reset_index()
    agg["mean_annual_incidence_thousands"] = agg["mean_annual_incidence"] / 1000
    agg["mean_ci_low_thousands"] = agg["mean_ci_low"] / 1000
    agg["mean_ci_high_thousands"] = agg["mean_ci_high"] / 1000
    return agg
