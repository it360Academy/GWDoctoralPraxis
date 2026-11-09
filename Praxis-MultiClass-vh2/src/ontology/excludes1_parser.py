"""Parses the OFFICIAL CDC/CMS ICD-10-CM Tabular List XML into a
version-controlled Excludes1 pairs table.

This is the ONLY legitimate source for H2's enemy-pair ontology - per the
research design, pairs are never data-mined (never inferred from NIS
co-occurrence statistics). Every row in the output CSV traces back to a
specific <excludes1><note> element in a specific year's official XML file.

Source (one file per fiscal year, structure verified against a real
excerpt in tests/fixtures/):
    https://ftp.cdc.gov/pub/health_statistics/nchs/publications/ICD10CM/<FY>/icd10cm_tabular_<FY>.xml

*** THIS SCRIPT REQUIRES INTERNET ACCESS TO ftp.cdc.gov ***
It is meant to be run from an environment with normal internet access
(a researcher's own machine, an unrestricted CI job, etc.) - NOT from
inside a network-sandboxed pipeline execution environment. Run it once
per fiscal year needed; commit the resulting ontology/excludes1_notes_raw_v<FY>.csv
to version control. main.py's `ontology_build` stage will refuse to
proceed if these files are missing, rather than silently falling back to
an unsourced default.

Fiscal-year vs. calendar-year note: ICD-10-CM is versioned by fiscal year
(Oct 1-Sep 30); NIS is calendar-year. A NIS calendar year spans two FYs
(e.g. NIS 2022 = FY2022 Jan-Sep + FY2023 Oct-Dec). This module lets you
fetch either FY; src/ontology/excludes1_loader.py documents which FY is
used to represent each NIS year and flags the boundary-month limitation -
see config.yaml `ontology.nis_year_to_fy_mapping`.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional
from urllib.request import urlopen

SOURCE_URL_TEMPLATE = (
    "https://ftp.cdc.gov/pub/health_statistics/nchs/publications/"
    "ICD10CM/{fy}/icd10cm_tabular_{fy}.xml"
)

_CODE_TOKEN = r"[A-Z]\d[0-9A-Z](?:\.?[0-9A-Z]{0,4})?\.?-?"
_SINGLE_OR_RANGE = r"(?:" + _CODE_TOKEN + r")(?:-(?:" + _CODE_TOKEN + r"))?"
# A trailing parenthetical holding one or more codes/ranges, comma-
# separated (real CMS notes commonly list several excluded codes in one
# note, e.g. "(C51-C52, C60.-, C63.2)") - this SUBSUMES the single-code
# and single-range cases (both are lists of length 1).
_PAREN_LIST_RE = re.compile(
    r"\(\s*(" + _SINGLE_OR_RANGE + r"(?:\s*,\s*" + _SINGLE_OR_RANGE + r")*)\s*\)\s*$")
# A handful of official notes end in a bare trailing code with NO
# parentheses at all (e.g. "carcinoma in situ of nose NOS D09.8") - only
# tried when the parenthetical form doesn't match, and anchored so it
# can't accidentally grab a code-shaped fragment out of running prose.
_BARE_TRAILING_CODE_RE = re.compile(r"(?<![(\w])(" + _CODE_TOKEN + r")\s*$")


@dataclass
class ExcludesNote:
    fiscal_year: str
    home_code: str
    home_desc: str
    home_level: str
    chapter: str
    section_id: str
    note_text: str
    target_raw: Optional[str]
    parse_status: str


def _extract_targets(note_text: str) -> tuple[List[str], str]:
    """Returns (target_codes, status). A single note can legitimately
    exclude MULTIPLE codes at once (comma-separated list in one
    parenthetical) - all are returned, to be emitted as separate
    Excludes1Pair relationships sharing the same note_text, rather than
    losing the note entirely (an earlier version of this parser only
    matched a single code or a single range and routed every
    comma-separated-list note to manual review, silently dropping ~4%
    of all real Excludes1 notes - including several affecting this
    project's oncological phenotypes, e.g. "(C51-C52, C60.-, C63.2)")."""
    text = note_text.strip()
    m = _PAREN_LIST_RE.search(text)
    if m:
        parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
        if parts:
            return parts, "ok"
    m2 = _BARE_TRAILING_CODE_RE.search(text)
    if m2:
        return [m2.group(1)], "ok"
    return [], "no_code_found"


def _walk_diag(elem: ET.Element, fy: str, chapter: str, section_id: str,
                depth: int = 0) -> Iterator[ExcludesNote]:
    name_el = elem.find("name")
    desc_el = elem.find("desc")
    code = name_el.text.strip() if name_el is not None and name_el.text else ""
    desc = desc_el.text.strip() if desc_el is not None and desc_el.text else ""
    level = {0: "category"}.get(depth, "subcategory" if depth == 1 else "code")

    for excl in elem.findall("excludes1"):
        for note in excl.findall("note"):
            text = (note.text or "").strip()
            if not text:
                continue
            targets, status = _extract_targets(text)
            if status != "ok":
                yield ExcludesNote(
                    fiscal_year=fy, home_code=code, home_desc=desc, home_level=level,
                    chapter=chapter, section_id=section_id,
                    note_text=text, target_raw=None, parse_status=status,
                )
                continue
            for target in targets:
                yield ExcludesNote(
                    fiscal_year=fy, home_code=code, home_desc=desc, home_level=level,
                    chapter=chapter, section_id=section_id,
                    note_text=text, target_raw=target, parse_status=status,
                )

    for child in elem.findall("diag"):
        yield from _walk_diag(child, fy, chapter, section_id, depth + 1)


def parse_tabular_xml(xml_path: Path, fiscal_year: str) -> List[ExcludesNote]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    notes: List[ExcludesNote] = []

    for chapter_el in root.findall("chapter"):
        chapter_name_el = chapter_el.find("name")
        chapter = chapter_name_el.text.strip() if chapter_name_el is not None and chapter_name_el.text else "?"

        for excl in chapter_el.findall("excludes1"):
            for note in excl.findall("note"):
                text = (note.text or "").strip()
                if not text:
                    continue
                targets, status = _extract_targets(text)
                targets = targets if status == "ok" else [None]
                for target in targets:
                    notes.append(ExcludesNote(
                        fiscal_year=fiscal_year, home_code=f"CHAPTER_{chapter}",
                        home_desc=chapter_el.findtext("desc", default="").strip(),
                        home_level="chapter", chapter=chapter, section_id="",
                        note_text=text, target_raw=target, parse_status=status,
                    ))

        for section_el in chapter_el.findall("section"):
            section_id = section_el.get("id", "")
            for excl in section_el.findall("excludes1"):
                for note in excl.findall("note"):
                    text = (note.text or "").strip()
                    if not text:
                        continue
                    targets, status = _extract_targets(text)
                    targets = targets if status == "ok" else [None]
                    for target in targets:
                        notes.append(ExcludesNote(
                            fiscal_year=fiscal_year, home_code=f"SECTION_{section_id}",
                            home_desc=section_el.findtext("desc", default="").strip(),
                            home_level="section", chapter=chapter, section_id=section_id,
                            note_text=text, target_raw=target, parse_status=status,
                        ))
            for diag_el in section_el.findall("diag"):
                notes.extend(_walk_diag(diag_el, fiscal_year, chapter, section_id))

    return notes


def write_ontology_csv(notes: List[ExcludesNote], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fiscal_year", "home_code", "home_desc", "home_level",
                          "chapter", "section_id", "note_text", "target_raw",
                          "parse_status"])
        for n in notes:
            writer.writerow([n.fiscal_year, n.home_code, n.home_desc, n.home_level,
                              n.chapter, n.section_id, n.note_text,
                              n.target_raw or "", n.parse_status])


def download_tabular_xml(fiscal_year: str, dest: Path) -> Path:
    url = SOURCE_URL_TEMPLATE.format(fy=fiscal_year)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest} (requires internet access to ftp.cdc.gov)")
    with urlopen(url, timeout=120) as resp, open(dest, "wb") as out:
        out.write(resp.read())
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fiscal-year", required=True, help="e.g. 2021, 2022, 2023")
    ap.add_argument("--xml-path", default=None,
                     help="Local path to an already-downloaded Tabular XML "
                          "(skips download - use this if this environment "
                          "cannot reach ftp.cdc.gov directly).")
    ap.add_argument("--out", default=None,
                     help="Output CSV path (default: ontology/excludes1_notes_raw_v<FY>.csv)")
    args = ap.parse_args()

    fy = args.fiscal_year
    xml_path = Path(args.xml_path) if args.xml_path else Path(f"data/icd10cm_tabular_{fy}.xml")
    if not xml_path.exists():
        download_tabular_xml(fy, xml_path)

    notes = parse_tabular_xml(xml_path, fy)
    out = Path(args.out) if args.out else Path(f"ontology/excludes1_notes_raw_v{fy}.csv")
    write_ontology_csv(notes, out)

    n_ok = sum(1 for n in notes if n.parse_status == "ok")
    n_unparsed = len(notes) - n_ok
    print(f"FY{fy}: {len(notes)} Excludes1 notes parsed ({n_ok} with a clean "
          f"target code, {n_unparsed} needing manual review) -> {out}")
    if n_unparsed:
        print(f"  {n_unparsed} notes had no cleanly-extractable trailing code "
              f"reference and are marked parse_status='no_code_found' - "
              f"review these manually before using them.")


if __name__ == "__main__":
    sys.exit(main())
