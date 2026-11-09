"""Diagnostic Exclusivity Violation (DEV) detection.

DEV = 1 for a hospitalization when I10_DX1 co-occurs, anywhere in
I10_DX2-DX25, with a code that the Excludes1 ontology says is
incompatible with DX1. Detection is DETERMINISTIC given the ontology -
this module contains no statistical/ML component and no discretion; the
ontology (src/ontology/, built only from the official CMS/CDC source) is
the sole determinant of what counts as a violation.

For each hospitalization, DX1 is tested against every ontology pair for
DX1's phenotype: a pair matches if DX1 falls in the pair's `home` side
and any DX2-25 code falls in the `target` side, OR DX1 falls in the
`target` side and any DX2-25 code falls in the `home` side (Excludes1
exclusivity is applied symmetrically regardless of which code the
official note happened to be attached to - see ontology/SOURCE.md).

Vectorized over pandas string ops (not a per-row Python loop) so this
scales to real NIS volume (tens of millions of discharges across 3
years x up to 25 secondary-DX columns).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from ..ontology.excludes1_loader import Excludes1Pair


def _norm(s: pd.Series) -> pd.Series:
    out = s.astype("string")
    out = out.str.replace(".", "", regex=False).str.replace(" ", "", regex=False).str.upper()
    return out


def _side_mask(series: pd.Series, kind: str, lo: str, hi: str) -> pd.Series:
    series = series.fillna("")
    if kind == "prefix":
        return series.str.startswith(lo)
    return ((series >= lo) & (series <= hi)) | series.str.startswith(lo) | series.str.startswith(hi)


@dataclass
class DevResult:
    dev: np.ndarray
    n_violations: np.ndarray
    first_pair_desc: List[str]
    first_secondary_position: np.ndarray
    hit_log: pd.DataFrame


def pairs_by_phenotype(pairs: List[Excludes1Pair]) -> Dict[str, List[Excludes1Pair]]:
    out: Dict[str, List[Excludes1Pair]] = {}
    for p in pairs:
        out.setdefault(p.phenotype, []).append(p)
    return out


def detect(df: pd.DataFrame, pairs_map: Dict[str, List[Excludes1Pair]],
           sec_cols: List[str]) -> DevResult:
    n = len(df)
    dev = np.zeros(n, dtype=np.uint8)
    n_violations = np.zeros(n, dtype=np.int32)
    first_desc = np.full(n, "", dtype=object)
    first_pos = np.zeros(n, dtype=np.int16)
    hit_frames: List[pd.DataFrame] = []

    dx1 = _norm(df["DX1_RAW"])
    labels = df["LABEL"].astype(object)
    sec_norm = {c: _norm(df[c]) for c in sec_cols}

    for phenotype, pairs in pairs_map.items():
        cohort_mask = (labels == phenotype).to_numpy()
        if not cohort_mask.any():
            continue
        cohort_idx = np.flatnonzero(cohort_mask)
        dx1_cohort = dx1.iloc[cohort_idx]

        for pair in pairs:
            if pair.home_code.startswith(("CHAPTER_", "SECTION_")):
                continue
            home_lo = pair.home_code.replace(".", "").upper()

            dx1_is_home = dx1_cohort.fillna("").str.startswith(home_lo)
            dx1_is_target = _side_mask(dx1_cohort, pair.target_kind, pair.target_lo, pair.target_hi)
            if not (dx1_is_home.any() or dx1_is_target.any()):
                continue

            any_sec_target = pd.Series(False, index=dx1_cohort.index)
            any_sec_home = pd.Series(False, index=dx1_cohort.index)
            per_col_target_hit = {}
            per_col_home_hit = {}
            for c in sec_cols:
                s = sec_norm[c].iloc[cohort_idx]
                t_hit = _side_mask(s, pair.target_kind, pair.target_lo, pair.target_hi)
                h_hit = s.fillna("").str.startswith(home_lo)
                per_col_target_hit[c] = t_hit
                per_col_home_hit[c] = h_hit
                any_sec_target |= t_hit
                any_sec_home |= h_hit

            matched = (dx1_is_home & any_sec_target) | (dx1_is_target & any_sec_home)
            if not matched.any():
                continue

            matched_local_idx = np.flatnonzero(matched.to_numpy())
            matched_global_idx = cohort_idx[matched_local_idx]
            dev[matched_global_idx] = 1
            n_violations[matched_global_idx] += 1

            desc = f"{pair.home_code} Excludes1 {pair.target_raw}"
            need_desc_mask = first_desc[matched_global_idx] == ""
            for gi in matched_global_idx[need_desc_mask]:
                first_desc[gi] = desc

            for pos, c in enumerate(sec_cols, start=1):
                col_matched = (dx1_is_home & per_col_target_hit[c]) | (dx1_is_target & per_col_home_hit[c])
                if not col_matched.any():
                    continue
                local_idx = np.flatnonzero(col_matched.to_numpy())
                global_idx = cohort_idx[local_idx]
                need_pos = first_pos[global_idx] == 0
                first_pos[global_idx[need_pos]] = pos
                hit_frames.append(pd.DataFrame({
                    "row": global_idx, "phenotype": phenotype,
                    "dx1": dx1.iloc[global_idx].to_numpy(),
                    "secondary_position": pos,
                    "secondary_code": sec_norm[c].iloc[global_idx].to_numpy(),
                    "home_code": pair.home_code, "target_raw": pair.target_raw,
                    "note_text": pair.note_text, "fiscal_year": pair.fiscal_year,
                }))

    hit_log = (pd.concat(hit_frames, ignore_index=True) if hit_frames else
               pd.DataFrame(columns=["row", "phenotype", "dx1", "secondary_position",
                                      "secondary_code", "home_code", "target_raw",
                                      "note_text", "fiscal_year"]))
    return DevResult(dev=dev, n_violations=n_violations,
                      first_pair_desc=list(first_desc),
                      first_secondary_position=first_pos, hit_log=hit_log)
