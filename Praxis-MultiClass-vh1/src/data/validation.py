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
    log = get_logger("h1.validate")
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
    check("age_plausible", df["AGE"].dropna().between(0, 120).mean() > 0.999,
          f"in[0,120] = {df['AGE'].dropna().between(0, 120).mean():.4f}")
    check("female_binary", df["FEMALE"].dropna().isin([0, 1]).all())

    n_cohort = df["LABEL"].notna().sum()
    check("cohort_nonempty", n_cohort > 0, f"15-phenotype cohort = {n_cohort:,}/{len(df):,}")
    label_counts = df["LABEL"].value_counts()
    missing_classes = [d for d in DISEASE_ORDER if d not in label_counts.index]
    check("all_15_phenotypes_observed", len(missing_classes) == 0,
          f"missing: {missing_classes}" if missing_classes else "all 15 present")
    if n_cohort > 0:
        min_class_n = int(label_counts.reindex(DISEASE_ORDER).fillna(0).min())
        log.info("[%s] smallest phenotype class has %d discharges (severe "
                  "class imbalance is expected and is part of H1's framing, "
                  "not itself a validation failure).", year, min_class_n)

    n_ambiguous = int(df["AMBIGUOUS_DX1"].sum())
    check("ambiguous_dx1_rare", n_ambiguous / max(len(df), 1) < 0.01,
          f"{n_ambiguous:,} discharges ({n_ambiguous / max(len(df), 1):.4%}) "
          f"matched >1 phenotype on I10_DX1 - excluded from cohort")

    res = pd.DataFrame(checks)
    out = cfg.outputs("tables") / f"data_validation_{year}.csv"
    res.to_csv(out, index=False)
    if not res["passed"].all():
        raise DataValidationError(f"Validation failed for {year}; see {out}")
    return res
