"""Target-leakage controls for the multiclass DX2-DX25 -> DX1 framework.

Two complementary controls, both real (code-level), not semantic
judgment calls:

1. **Positional separation.** The label comes from I10_DX1 ONLY. Predictors
   come exclusively from I10_DX2..I10_DX(1+max_secondary_dx). DX1's own
   code never appears among the predictors by construction (see
   src/data/ingestion.py).

2. **Strict phenotype-code masking.** Positional separation alone is not
   enough: a secondary code that is ITSELF one of the 15 phenotypes'
   defining codes (e.g. DX1=MI, but an I21-family code also sits in DX2 -
   a real, clinically legitimate recurrent/bilateral-event pattern) would
   let a classifier learn "presence of a phenotype-defining code in a
   secondary position" as a near-direct proxy for the true class, for
   that class AND for cross-phenotype confounding (e.g. an aortic-
   aneurysm code in DX2 is evidence the record is vascular-service
   patient generally, indirectly informative about several vascular
   labels at once). We therefore drop ANY DX2-25 code that matches ANY of
   the 15 phenotype definitions (same prefix+exclude logic as
   definitions.py) from the feature vocabulary entirely - for every row,
   regardless of that row's true label. This is stricter than excluding
   only same-class codes, and is fully auditable at the code/category
   level (see `audit_masked_categories`).

This is a genuine trade-off: legitimate recurrent-event signal is
discarded along with the leakage risk. That trade-off is the point - H1
is framed as "leakage-controlled" specifically because the alternative
(keeping same-phenotype secondary codes as features) would inflate
discrimination in a way that is difficult to distinguish from a real
comorbidity-driven signal after the fact.
"""
from __future__ import annotations

from typing import Dict, List, Set

import numpy as np
import pandas as pd

from ..features.definitions import DISEASE_ORDER, DISEASES
from ..features.icd_classifier import CodeClassifier


def phenotype_code_classifier() -> CodeClassifier:
    return CodeClassifier(
        {d: s["icd10_prefixes"] for d, s in DISEASES.items()},
        exclude={d: s.get("icd10_exclude", []) for d, s in DISEASES.items()},
    )


def leaky_codes(unique_codes: List[str]) -> Set[str]:
    """Codes (secondary-position, normalized, no dot) that match at least
    one of the 15 phenotype definitions and must therefore be masked out
    of the feature vocabulary."""
    clf = phenotype_code_classifier()
    mapping = clf.classify_unique(np.asarray(unique_codes, dtype=object))
    return {code for code, hits in mapping.items() if len(hits) > 0}


def which_phenotypes(code: str) -> List[str]:
    """Which phenotype(s) a single code matches - used only for audit
    reporting, not on the hot path."""
    clf = phenotype_code_classifier()
    hits = clf._classify_one(code)  # noqa: SLF001 (internal, audit use only)
    return [DISEASE_ORDER[i] for i in hits]


def audit_masked_categories(masked_codes: Set[str], code_counts: Dict[str, int]) -> pd.DataFrame:
    """One row per masked code: which phenotype(s) it matches, how often
    it occurred in the secondary-position corpus it was computed from.
    Written to outputs/tables/leakage_code_audit.csv."""
    rows = []
    for code in sorted(masked_codes):
        rows.append({
            "code": code,
            "matches_phenotypes": ";".join(which_phenotypes(code)),
            "n_occurrences_secondary_position": code_counts.get(code, 0),
        })
    df = pd.DataFrame(rows).sort_values(
        "n_occurrences_secondary_position", ascending=False
    ).reset_index(drop=True)
    return df
