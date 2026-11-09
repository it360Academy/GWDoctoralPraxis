#!/usr/bin/env python
"""Generate synthetic ICD-10-shaped NIS Core CSVs (2021/2022/2023) so the
pipeline's CODE can be verified end-to-end without real HCUP data.

Smoke output is SYNTHETIC and is NEVER H1 evidence - it only proves the
pipeline runs without error on data of the right shape.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.definitions import DISEASE_ORDER, DISEASES  # noqa: E402

# A pool of comorbidity-like 3-4 char ICD-10-CM codes that do NOT overlap
# any of the 15 phenotype definitions - the realistic secondary-diagnosis
# "signal" pool.
COMORBID_POOL = [
    "E119", "E1122", "I10", "I5023", "N189", "F329", "J449", "E669",
    "Z9861", "E785", "K219", "M1990", "G4733", "D649", "R05", "R0602",
    "N390", "E871", "R7303", "I2510", "M549", "Z7901", "F419", "E039",
]
# A pool of codes that DO overlap phenotype definitions (used to inject
# some same-/cross-phenotype secondary codes, so the leakage-masking
# stage has something real to catch).
LEAKY_POOL_BY_PHENOTYPE = {d: [p + "0" if len(p) < 4 else p for p in spec["icd10_prefixes"]]
                           for d, spec in DISEASES.items()}

rng = np.random.default_rng(7)


def _sample_principal(n: int, no_phenotype_share: float = 0.55) -> np.ndarray:
    """Simulate severe class imbalance: most discharges are NOT one of the
    15 phenotypes; among the 15, prevalence varies ~40x between the most
    and least common."""
    weights = rng.uniform(1.0, 40.0, size=len(DISEASE_ORDER))
    weights = weights / weights.sum() * (1 - no_phenotype_share)
    choices = list(DISEASE_ORDER) + ["OTHER"]
    probs = list(weights) + [no_phenotype_share]
    probs = np.array(probs) / np.sum(probs)
    picks = rng.choice(choices, size=n, p=probs)
    codes = []
    for p in picks:
        if p == "OTHER":
            codes.append(rng.choice(["Z0000", "M5450", "K5900", "R101", "H9190"]))
        else:
            prefix = rng.choice(DISEASES[p]["icd10_prefixes"])
            suffix = rng.choice(["", "0", "1", "9", "X1"])
            codes.append((prefix + suffix)[:6])
    return np.array(codes, dtype=object), picks


def _sample_secondary(principal_labels: np.ndarray, n_secondary: int = 25,
                       leak_rate: float = 0.05) -> np.ndarray:
    n = len(principal_labels)
    out = np.full((n, n_secondary), "", dtype=object)
    for i, label in enumerate(principal_labels):
        k = rng.integers(0, 12)  # not every discharge fills all 25 secondary slots
        codes = list(rng.choice(COMORBID_POOL, size=k, replace=True))
        # inject a mild, label-correlated signal
        if label != "OTHER" and rng.random() < 0.5:
            codes.append(rng.choice(COMORBID_POOL[hash(label) % 8: hash(label) % 8 + 4]
                                     or COMORBID_POOL[:4]))
        # occasionally inject a genuinely leaky (phenotype-defining) code
        if rng.random() < leak_rate and label != "OTHER":
            leak_from = rng.choice(list(DISEASE_ORDER))
            codes.append(rng.choice(LEAKY_POOL_BY_PHENOTYPE[leak_from]))
        rng.shuffle(codes)
        for j, c in enumerate(codes[:n_secondary]):
            out[i, j] = c
    return out


def generate_year(n_rows: int) -> pd.DataFrame:
    principal, labels = _sample_principal(n_rows)
    secondary = _sample_secondary(labels, n_secondary=40)

    df = pd.DataFrame(index=range(n_rows))
    df["KEY_NIS"] = np.arange(1, n_rows + 1)
    df["HOSP_NIS"] = rng.integers(1, 500, size=n_rows)
    df["I10_DX1"] = principal
    for j in range(40):
        df[f"I10_DX{j+2}"] = secondary[:, j]
    df["AGE"] = rng.integers(0, 95, size=n_rows)
    df["FEMALE"] = rng.integers(0, 2, size=n_rows)
    df["ELECTIVE"] = rng.integers(0, 2, size=n_rows)
    df["AWEEKEND"] = rng.integers(0, 2, size=n_rows)
    df["PAY1"] = rng.integers(1, 6, size=n_rows)
    df["DISCWT"] = rng.uniform(0.5, 3.0, size=n_rows).round(4)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--rows", type=int, default=20000)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in ("2021", "2022", "2023"):
        df = generate_year(args.rows)
        path = out_dir / f"NIS_{year}_Core.csv"
        df.to_csv(path, index=False)
        print(f"wrote {path} ({len(df):,} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
