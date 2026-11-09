"""Loads a raw Excludes1-notes CSV (from excludes1_parser.py) and
produces the final, version-controlled ontology consumed by DEV
detection: pairs where at least one side falls within one of H1's 15
phenotype code definitions (the DX1 scope decision).

Two matching primitives, both string-based on DOT-STRIPPED codes (matching
how HCUP NIS stores I10_DX fields, e.g. "A41.81" -> "A4181"):
  - prefix match: "A40.-" or a bare code like "A41.81" -> startswith("A4081"/"A4181")
  - range match: "A39.2-A39.4" -> lexicographic bounds check (valid because
    ICD-10-CM range notation only appears within a single category, where
    codes share length and character-position semantics)

A pair is RETAINED if:
  - the home_code (the code the official note is attached to) falls
    within a phenotype's icd10_prefixes, OR
  - the parsed target falls within a phenotype's icd10_prefixes
  (this catches both directions: phenotype-code-bears-the-note, and
  phenotype-code-is-the-excluded-partner-under-another-code's-note -
  directionality is preserved in the `direction` column either way).

Notes with parse_status != "ok" (no cleanly-extractable target code) are
EXCLUDED from the operational ontology and written to a separate
`_unparsed_needs_review.csv` file instead - they are never silently
dropped, and never auto-resolved by guessing.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from ..features.definitions import DISEASE_ORDER, DISEASES


def _strip(code: str) -> str:
    return code.replace(".", "").replace(" ", "").upper()


def _parse_target(target_raw: str) -> Tuple[str, str, str]:
    """Returns (kind, lo, hi). kind in {'prefix', 'range'}. For a prefix,
    lo==hi==the stripped prefix (trailing '-' wildcard removed)."""
    t = target_raw.strip()
    if "-" in t and not t.endswith("-"):
        # a genuine range, e.g. "A39.2-A39.4" (not a trailing wildcard "P36.-")
        lo_raw, hi_raw = t.split("-", 1)
        return "range", _strip(lo_raw), _strip(hi_raw)
    return "prefix", _strip(t.rstrip("-")), _strip(t.rstrip("-"))


def _phenotype_prefixes() -> List[Tuple[str, str]]:
    """(phenotype, stripped_prefix) for every prefix across all 15
    phenotypes - used to test whether a home_code or target falls inside
    a phenotype's own definition."""
    out = []
    for name, spec in DISEASES.items():
        for p in spec["icd10_prefixes"]:
            out.append((name, _strip(p)))
    return out


def _matches_any_phenotype(code_stripped: str, phenotype_prefixes: List[Tuple[str, str]]) -> List[str]:
    return [ph for ph, prefix in phenotype_prefixes if code_stripped.startswith(prefix)]


def _target_overlaps_phenotype(kind: str, lo: str, hi: str,
                                phenotype_prefixes: List[Tuple[str, str]]) -> List[str]:
    if kind == "prefix":
        return _matches_any_phenotype(lo, phenotype_prefixes)
    # range: a phenotype prefix "overlaps" the range if the prefix's own
    # code sits lexicographically within [lo, hi], OR the range sits
    # within the phenotype's prefix span. Conservative (string-bound)
    # check - flagged in ontology output for manual spot-check since
    # ICD-10-CM lexicographic order does not always equal clinical
    # category order across chapter boundaries.
    hits = []
    for ph, prefix in phenotype_prefixes:
        if lo <= prefix <= hi or prefix <= lo <= (prefix + "\uffff"):
            hits.append(ph)
    return hits


@dataclass
class Excludes1Pair:
    fiscal_year: str
    home_code: str
    home_desc: str
    home_level: str
    target_raw: str
    target_kind: str
    target_lo: str
    target_hi: str
    note_text: str
    direction: str          # "phenotype_is_home" | "phenotype_is_target"
    phenotype: str
    source_chapter: str
    source_section: str


def build_ontology(raw_notes_csv: Path) -> Tuple[List[Excludes1Pair], List[dict]]:
    phenotype_prefixes = _phenotype_prefixes()
    pairs: List[Excludes1Pair] = []
    unparsed: List[dict] = []

    with open(raw_notes_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["parse_status"] != "ok" or not row["target_raw"]:
                unparsed.append(row)
                continue

            home_stripped = _strip(row["home_code"]) if not row["home_code"].startswith(("CHAPTER_", "SECTION_")) else ""
            home_hits = _matches_any_phenotype(home_stripped, phenotype_prefixes) if home_stripped else []

            kind, lo, hi = _parse_target(row["target_raw"])
            target_hits = _target_overlaps_phenotype(kind, lo, hi, phenotype_prefixes)

            # A single note can put the SAME phenotype on both sides (e.g.
            # home=A40 and target=A41.81 are both within "sepsis"). That
            # is one underlying Excludes1 relationship, not two - emit it
            # once as "internal_to_phenotype", not once per direction, or
            # detect() (which already applies pairs symmetrically) would
            # double-count the same violation.
            internal = set(home_hits) & set(target_hits)
            home_only = set(home_hits) - internal
            target_only = set(target_hits) - internal

            for ph, direction in (
                [(p, "internal_to_phenotype") for p in internal]
                + [(p, "phenotype_is_home") for p in home_only]
                + [(p, "phenotype_is_target") for p in target_only]
            ):
                pairs.append(Excludes1Pair(
                    fiscal_year=row["fiscal_year"], home_code=row["home_code"],
                    home_desc=row["home_desc"], home_level=row["home_level"],
                    target_raw=row["target_raw"], target_kind=kind, target_lo=lo, target_hi=hi,
                    note_text=row["note_text"], direction=direction,
                    phenotype=ph, source_chapter=row["chapter"], source_section=row["section_id"],
                ))

    return pairs, unparsed


def write_pairs_csv(pairs: List[Excludes1Pair], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fiscal_year", "phenotype", "direction", "home_code", "home_desc",
                          "home_level", "target_raw", "target_kind", "target_lo", "target_hi",
                          "note_text", "source_chapter", "source_section"])
        for p in pairs:
            writer.writerow([p.fiscal_year, p.phenotype, p.direction, p.home_code, p.home_desc,
                              p.home_level, p.target_raw, p.target_kind, p.target_lo, p.target_hi,
                              p.note_text, p.source_chapter, p.source_section])


def load_pairs_csv(path: Path) -> List[Excludes1Pair]:
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(Excludes1Pair(
                fiscal_year=row["fiscal_year"], phenotype=row["phenotype"],
                direction=row["direction"], home_code=row["home_code"],
                home_desc=row["home_desc"], home_level=row["home_level"],
                target_raw=row["target_raw"], target_kind=row["target_kind"],
                target_lo=row["target_lo"], target_hi=row["target_hi"],
                note_text=row["note_text"], source_chapter=row["source_chapter"],
                source_section=row["source_section"],
            ))
    return out
