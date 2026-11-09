#!/usr/bin/env python
"""Generates small synthetic NIS-shaped Core/Hospital CSVs, plus a
real-but-partial SEED Excludes1 ontology (sepsis pairs only), so the H2
pipeline can be smoke-tested end-to-end without real HCUP NIS data.

NEVER treat output from this synthetic data as research evidence -
diagnosis codes, weights, and Excludes1 pairs here are FABRICATED for
code-path coverage only. Real DEV rates, p-values, and the H2 verdict
must come from real NIS 2021-2023 data run through config/config.yaml.

Usage:
    python scripts/generate_synthetic_data.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.definitions import DISEASE_ORDER, DISEASES  # noqa: E402

N_HOSPITALS = 40
N_PER_YEAR = 3000
MAX_SEC_DX = 25
YEARS = ["2021", "2022", "2023"]

# A handful of secondary "noise" codes with no Excludes1 relationship to
# anything, used to fill unused secondary-DX slots realistically.
NOISE_CODES = ["E119", "I10", "Z794", "F329", "N390", "J449", "E785", "M549"]

# Fabricated sepsis Excludes1 pairs for the SEED ontology (smoke-testing
# only - see ontology/SOURCE.md). Real pairs must be parsed from the
# official ICD-10-CM Tabular XML via src/ontology/excludes1_parser.py.
SEED_PAIRS = [
    {
        "fiscal_year": "SEED", "phenotype": "sepsis", "direction": "phenotype_is_home",
        "home_code": "A41", "home_desc": "Other sepsis", "home_level": "category",
        "target_raw": "P368", "target_kind": "prefix", "target_lo": "P368", "target_hi": "P368",
        "note_text": "Excludes1 (fabricated, smoke-test only): neonatal sepsis (P36.-)",
        "source_chapter": "1", "source_section": "A00-A49",
    },
    {
        "fiscal_year": "SEED", "phenotype": "sepsis", "direction": "phenotype_is_home",
        "home_code": "A40", "home_desc": "Streptococcal sepsis", "home_level": "category",
        "target_raw": "R6521", "target_kind": "prefix", "target_lo": "R6521", "target_hi": "R6521",
        "note_text": "Excludes1 (fabricated, smoke-test only): severe sepsis (R65.2-)",
        "source_chapter": "1", "source_section": "A00-A49",
    },
]


def write_seed_ontology(ontology_dir: Path) -> Path:
    ontology_dir.mkdir(parents=True, exist_ok=True)
    out_path = ontology_dir / "excludes1_pairs_vSEED.csv"
    cols = ["fiscal_year", "phenotype", "direction", "home_code", "home_desc",
            "home_level", "target_raw", "target_kind", "target_lo", "target_hi",
            "note_text", "source_chapter", "source_section"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in SEED_PAIRS:
            writer.writerow(row)
    print(f"Wrote SEED ontology -> {out_path} ({len(SEED_PAIRS)} pairs)")
    return out_path


def _sample_code_for_phenotype(rng: np.random.Generator, phenotype: str) -> str:
    prefixes = DISEASES[phenotype]["icd10_prefixes"]
    prefix = rng.choice(prefixes)
    # pad with plausible trailing digits so codes look like real ICD-10-CM
    suffix = "".join(str(d) for d in rng.integers(0, 10, size=max(0, 4 - len(prefix))))
    return f"{prefix}{suffix}"


def generate_year(rng: np.random.Generator, year: str, dev_fraction: float) -> pd.DataFrame:
    n = N_PER_YEAR
    rows = []
    hosp_ids = rng.integers(1, N_HOSPITALS + 1, size=n)
    # DX1 drawn evenly across the 15 phenotypes so validate_year's
    # "all_15_phenotypes_observed" check passes.
    phenotype_choices = rng.choice(DISEASE_ORDER, size=n)

    for i in range(n):
        phenotype = phenotype_choices[i]
        dx1 = _sample_code_for_phenotype(rng, phenotype)
        sec = list(rng.choice(NOISE_CODES, size=MAX_SEC_DX))

        # Deliberately inject a real Excludes1 violation into a slice of
        # the sepsis cohort, so DEV detection, negative controls, and the
        # DEV-risk model all have non-degenerate (non-all-zero) signal to
        # exercise during the smoke test.
        if phenotype == "sepsis" and rng.random() < dev_fraction:
            violating_code = "P3681" if dx1.startswith("A41") else "R6521"
            pos = rng.integers(0, min(10, MAX_SEC_DX))  # bias toward early positions
            sec[pos] = violating_code

        row = {
            "KEY_NIS": (int(year) * 1_000_000) + i,
            "HOSP_NIS": int(hosp_ids[i]),
            "AGE": int(rng.integers(0, 95)),
            "FEMALE": int(rng.integers(0, 2)),
            "ELECTIVE": int(rng.integers(0, 2)),
            "AWEEKEND": int(rng.integers(0, 2)),
            "PAY1": int(rng.integers(1, 6)),
            "DISCWT": float(rng.uniform(1.5, 8.0)),
        }
        row["I10_DX1"] = dx1
        for j, code in enumerate(sec, start=2):
            row[f"I10_DX{j}"] = code
        rows.append(row)

    return pd.DataFrame(rows)


def generate_hospital(year: str) -> pd.DataFrame:
    rows = []
    for h in range(1, N_HOSPITALS + 1):
        rows.append({
            "HOSP_NIS": h,
            "HOSP_REGION": ((h - 1) % 4) + 1,
            "HOSP_TEACH": (h % 2),
            "HOSP_BEDSIZE": ((h - 1) % 3) + 1,
        })
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(42)
    out_dir = ROOT / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Vary the injected DEV fraction slightly by year so the temporal-
    # stability stage has something non-trivial (but still within the
    # 1.5x stability threshold) to measure.
    dev_fractions = {"2021": 0.09, "2022": 0.10, "2023": 0.095}

    for year in YEARS:
        core = generate_year(rng, year, dev_fractions[year])
        core_path = out_dir / f"NIS_{year}_Core.csv"
        core.to_csv(core_path, index=False)
        print(f"Wrote {core_path} ({len(core):,} rows)")

        hosp = generate_hospital(year)
        hosp_path = out_dir / f"NIS_{year}_Hospital.csv"
        hosp.to_csv(hosp_path, index=False)
        print(f"Wrote {hosp_path} ({len(hosp):,} rows)")

    write_seed_ontology(ROOT / "ontology")


if __name__ == "__main__":
    main()
