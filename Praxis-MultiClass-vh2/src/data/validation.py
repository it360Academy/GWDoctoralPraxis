"""Data-quality validation on the per-year raw checkpoints."""
from __future__ import annotations

import pandas as pd

from ..config import Config
from ..features.definitions import DISEASE_ORDER
from ..logging_utils import get_logger
from .ingestion import load_year


class DataValidationError(Exception):
    pass


def validate_year(cfg: Config, year: str) -> pd.DataFrame:
    log = get_logger("h2.validate")
    df = load_year(cfg, year)
    checks = []

    def check(name: str, ok: bool, detail: str = ""):
        checks.append({"year": year, "check": name, "passed": bool(ok), "detail": detail})
        (log.info if ok else log.error)("[%s] %s: %s %s", year, name,
                                         "PASS" if ok else "FAIL", detail)

    min_rows = int(cfg.get("validation_min_rows", default=100_000))
    check("nonempty", len(df) >= min_rows, f"n={len(df):,} (min={min_rows:,})")
    check("weights_positive", (df["DISCWT"] > 0).mean() > 0.99,
          f"share>0 = {(df['DISCWT'] > 0).mean():.4f}")
    check("dx1_raw_present", df["DX1_RAW"].notna().mean() > 0.99,
          f"share non-null = {df['DX1_RAW'].notna().mean():.4f}")

    n_cohort = df["LABEL"].notna().sum()
    check("cohort_nonempty", n_cohort > 0, f"15-phenotype DX1 cohort = {n_cohort:,}/{len(df):,}")
    label_counts = df["LABEL"].value_counts()
    missing_classes = [d for d in DISEASE_ORDER if d not in label_counts.index]
    check("all_15_phenotypes_observed", len(missing_classes) == 0,
          f"missing: {missing_classes}" if missing_classes else "all 15 present")

    n_ambiguous = int(df["AMBIGUOUS_DX1"].sum())
    check("ambiguous_dx1_rare", n_ambiguous / max(len(df), 1) < 0.01,
          f"{n_ambiguous:,} discharges matched >1 phenotype on I10_DX1")

    check("n_dx_filled_plausible", df["N_DX_FILLED"].between(0, 41).mean() > 0.999,
          f"in[0,41] share = {df['N_DX_FILLED'].between(0, 41).mean():.4f}")

    res = pd.DataFrame(checks)
    out = cfg.outputs("tables") / f"data_validation_{year}.csv"
    res.to_csv(out, index=False)
    if not res["passed"].all():
        raise DataValidationError(f"Validation failed for {year}; see {out}")
    return res
