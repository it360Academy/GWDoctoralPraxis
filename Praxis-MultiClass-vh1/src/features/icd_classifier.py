"""Generic ICD-10-CM prefix classifier, used to build target-disease flags
(TGT_*/SEC_*) from raw I10_DX columns. Comorbidities no longer use this
module (see src/features/comorbidity.py) - they come from DX_PR_GRPS's
official CMR_* flags instead. This is kept as a standalone, reusable
component since target-disease classification still requires scanning raw
diagnosis codes against a prefix list.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


class CodeClassifier:
    """Classify normalized ICD-10-CM codes into named prefix groups.

    Each *unique* code observed in a chunk is classified once and cached,
    so multi-million-row columns reduce to fast ``Series.isin`` calls.
    """

    def __init__(self, groups: Dict[str, Sequence[str]],
                 exclude: Dict[str, Sequence[str]] | None = None):
        self.group_names: List[str] = list(groups.keys())
        self._prefixes: List[Tuple[str, Tuple[str, ...]]] = [
            (g, tuple(groups[g])) for g in self.group_names
        ]
        self._exclude: Dict[str, Tuple[str, ...]] = {
            g: tuple((exclude or {}).get(g, [])) for g in self.group_names
        }
        self._cache: Dict[str, Tuple[int, ...]] = {}

    def _classify_one(self, code: str) -> Tuple[int, ...]:
        hits = []
        for gi, (g, prefixes) in enumerate(self._prefixes):
            matched = any(code.startswith(p) for p in prefixes)
            if not matched:
                continue
            excluded = any(code.startswith(p) for p in self._exclude[g])
            if not excluded:
                hits.append(gi)
        return tuple(hits)

    def classify_unique(self, codes: np.ndarray) -> Dict[str, Tuple[int, ...]]:
        out = {}
        for code in codes:
            if code in self._cache:
                out[code] = self._cache[code]
                continue
            res = self._classify_one(code)
            self._cache[code] = res
            out[code] = res
        return out

    def flags_for_dx_frame(self, dx: pd.DataFrame) -> np.ndarray:
        """Return an (n_rows, n_groups) uint8 matrix: any DX column matches group."""
        n, g = len(dx), len(self.group_names)
        flags = np.zeros((n, g), dtype=np.uint8)
        for col in dx.columns:
            s = dx[col]
            uniq = s.dropna().unique()
            if len(uniq) == 0:
                continue
            mapping = self.classify_unique(np.asarray(uniq, dtype=object))
            group_codes: Dict[int, set] = {}
            for code, hits in mapping.items():
                for gi in hits:
                    group_codes.setdefault(gi, set()).add(code)
            for gi, codeset in group_codes.items():
                flags[:, gi] |= s.isin(codeset).to_numpy(dtype=np.uint8)
        return flags


def normalize_dx(series: pd.Series) -> pd.Series:
    """Normalize a raw DX column: strip, uppercase, blank/invalid -> NaN."""
    s = series.astype("string").str.strip().str.upper()
    s = s.replace({"": pd.NA, "INVL": pd.NA, "INCN": pd.NA, "-9": pd.NA,
                   "NAN": pd.NA, "NONE": pd.NA})
    return s
