"""Exploratory data analysis on the ingested NIS cohorts: demographic and
admission patterns, and coding intensity, by phenotype and by year. This
is descriptive context supporting interpretation of the incidence and
replication results; it is not itself part of the eight prespecified H3
criteria.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from ..config import Config
from ..data.ingestion import PRIN_ONLY_COLS, VASC_INFECT_DISEASES, load_year
from ..logging_utils import get_logger


def cohort_summary(cfg: Config) -> pd.DataFrame:
    """Overall discharge counts, weighted totals, and covariate coverage
    per year."""
    rows = []
    for role, year in cfg.years.items():
        df = load_year(cfg, year)
        rows.append({
            "role": role, "year": year, "n_discharges": len(df),
            "weighted_n_discharges": float(df["DISCWT"].sum()),
            "mean_n_dx_filled": float(df["N_DX_FILLED"].mean()),
            "age_coverage": float(df["AGE"].notna().mean()),
            "mean_age": float(df["AGE"].mean()),
            "female_share": float(df["FEMALE"].mean()),
            "elective_share": float(df["ELECTIVE"].mean()),
            "weekend_share": float(df["AWEEKEND"].mean()),
        })
    return pd.DataFrame(rows)


def phenotype_demographics(cfg: Config) -> pd.DataFrame:
    """Age, sex, elective-admission, and coding-intensity profile for
    discharges principally coded (DX1) to each of the ten computed
    phenotypes, by year."""
    rows = []
    for role, year in cfg.years.items():
        df = load_year(cfg, year)
        for disease, col in zip(VASC_INFECT_DISEASES, PRIN_ONLY_COLS):
            sub = df[df[col] == 1]
            if len(sub) == 0:
                continue
            rows.append({
                "role": role, "year": year, "disease": disease, "n": len(sub),
                "weighted_n": float(sub["DISCWT"].sum()),
                "mean_age": float(sub["AGE"].mean()),
                "female_share": float(sub["FEMALE"].mean()),
                "elective_share": float(sub["ELECTIVE"].mean()),
                "weekend_share": float(sub["AWEEKEND"].mean()),
                "mean_n_dx_filled": float(sub["N_DX_FILLED"].mean()),
            })
    return pd.DataFrame(rows)


def coding_intensity_distribution(cfg: Config, bins=(0, 5, 10, 15, 20, 25, 30, 41)) -> pd.DataFrame:
    """Distribution of the number of filled diagnosis positions per
    discharge, by year - context for interpreting how much of each
    discharge's diagnostic detail the DX1-or-DX2 incidence definition is
    actually drawing from."""
    rows = []
    for role, year in cfg.years.items():
        df = load_year(cfg, year)
        binned = pd.cut(df["N_DX_FILLED"], bins=bins, include_lowest=True)
        counts = binned.value_counts().sort_index()
        for interval, n in counts.items():
            rows.append({"role": role, "year": year, "n_dx_bin": str(interval), "n": int(n)})
    return pd.DataFrame(rows)


def payer_distribution(cfg: Config) -> pd.DataFrame:
    rows = []
    for role, year in cfg.years.items():
        df = load_year(cfg, year)
        counts = df["PAY1"].value_counts(dropna=True).sort_index()
        for payer, n in counts.items():
            rows.append({"role": role, "year": year, "pay1_code": payer, "n": int(n)})
    return pd.DataFrame(rows)
