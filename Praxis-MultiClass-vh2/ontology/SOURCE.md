# Excludes1 ontology: sourcing, versioning, and known limitations

## What lives here

`excludes1_pairs_v<fy>.csv` — one file per ICD-10-CM fiscal year, each a
flat list of `Excludes1Pair` rows (see `src/ontology/excludes1_loader.py`)
consumed directly by `src/dev/detector.py`. These are the **only**
files `main.py`'s `ontology_check` / `detect` stages read; nothing else
in `src/ontology/` is loaded at pipeline run time.

## How a real (non-SEED) file is built

1. Download the official CMS/CDC ICD-10-CM Tabular XML for the target
   fiscal year (`src/ontology/excludes1_parser.py:download_tabular_xml`,
   or fetch manually if the machine has no internet access).
2. Run `src/ontology/excludes1_parser.py` against that XML to extract
   every `<Excludes1>` note in the tabular list, producing a raw
   `*_notes.csv` (one row per note, before any phenotype filtering).
3. Run `src/ontology/excludes1_loader.build_ontology()` on that raw CSV.
   This is where phenotype filtering happens: a pair is retained only if
   the home code or the parsed target falls inside one of the 15
   phenotype definitions in `src/features/definitions.py` (the H2 "DX1
   scope decision" — DX1 is restricted to those 15 phenotypes, same
   restriction H1 uses). Notes that can't be cleanly parsed into a target
   code/range are written to a separate `*_unparsed_needs_review.csv`
   file and are **never** silently dropped or guessed at.
4. Run `write_pairs_csv()` to produce the final
   `excludes1_pairs_v<fy>.csv` this pipeline reads.

`config/config.yaml`'s `ontology.nis_year_to_fy_mapping` controls which
fiscal-year file each NIS calendar year uses. **Known misalignment**:
ICD-10-CM fiscal years run October 1–September 30, so a calendar year's
Q4 (Oct–Dec) discharges technically fall under the *next* fiscal year's
code set. This pipeline maps each NIS calendar year to its
same-numbered fiscal year rather than resolving this discharge-by-
discharge; this is a simplification, not a resolved edge case.

## The SEED file

`excludes1_pairs_vSEED.csv` (written by
`scripts/generate_synthetic_data.py`) contains a small number of
**fabricated** sepsis-only pairs, for smoke-testing the code path only.
`main.py`'s `stage_ontology_check` refuses to run against any file whose
name contains `SEED` unless `ontology.allow_seed_ontology: true` is set
(as `config/config_smoke.yaml` does) — this is a deliberate guard against
accidentally treating a smoke run as research evidence.

## Before treating any real run as final

- Confirm the downloaded Tabular XML is the CMS/CDC-published version for
  the intended fiscal year, not a mirrored or user-edited copy.
- Spot-check a sample of `*_unparsed_needs_review.csv` rows — a
  systematically-unparseable note format (rather than a genuinely rare
  edge case) would mean real Excludes1 relationships are silently
  missing from the operational ontology.
- Confirm the phenotype code lists in `src/features/definitions.py` are
  clinically reviewed (every entry there is flagged `verified: False` as
  shipped — see that file's header).

## Build record (this copy)

The operational ontology in this directory was built from the real,
official CMS/CDC ICD-10-CM Tabular List XML for FY2021, FY2022, and
FY2023 (`icd10cm_tabular_<fy>.xml`, verified against the CDC FTP
directory listing — sizes and filenames match exactly).

| FY | Raw Excludes1 notes (rows) | Parsed OK | Unparsed (needs review) | Operational pairs (15 phenotypes) |
|----|----:|----:|----:|----:|
| 2021 | 5,852 | 5,655 (96.6%) | 197 (3.4%) | 135 |
| 2022 | 5,822 | 5,631 (96.7%) | 191 (3.3%) | 140 |
| 2023 | 5,848 | 5,654 (96.7%) | 194 (3.3%) | 137 |

**Parser fixes made during this build** (see `src/ontology/excludes1_parser.py`
and `tests/test_excludes1_parser.py`): the original single-code/single-range
regex silently routed every comma-separated multi-code note (a common real
CMS format, e.g. `(C51-C52, C60.-, C63.2)`) to manual review — about 4% of
all real notes, concentrated in the oncology chapter and directly relevant
to this project's 5 cancer phenotypes. Fixed to extract every code in the
list as a separate target. Also fixed: the code-token regex excluded the
letter "U" entirely, which silently broke extraction of COVID-19 codes
(U07.1) wherever they appeared as an Excludes1 target. The remaining ~3.3%
unparsed per year are genuinely malformed/ambiguous source text (list
headers ending in ":", "see body system-related chapters" cross-references,
codes missing their enclosing parenthesis in the published XML itself) —
these are written to `excludes1_notes_raw_v<fy>_unparsed_needs_review.csv`
for manual review, never guessed at or silently dropped.

**Important finding, not a bug:** `arterial_te` (arterial thromboembolism,
ICD-10-CM I74) has **zero** Excludes1 relationships anywhere in the FY2021
Tabular List — verified exhaustively (every I74.x child code checked for
its own notes, and the full raw-notes table checked for any note whose
target references I74). This means DEV cannot occur for this phenotype by
construction in any year — not because of coding behavior, but because no
Excludes1 rule exists for it to violate. This should be addressed
explicitly in the dissertation (e.g., report it separately, or note the
effective phenotype count for DEV purposes is 14, not 15) rather than
silently averaged into an overall rate where it can only ever contribute
zeros.

Per-phenotype operational pair counts (FY2021): stroke 10, vte 12,
arterial_te **0**, aortic_aneurysm 5, mi 5, sepsis 10, pneumonia 46,
meningitis_encephalitis 20, spinal_abscess 1, endocarditis 2,
lung_cancer 2, breast_cancer 8, colorectal_cancer 5, melanoma 8,
prostate_cancer 1. FY2022/FY2023 counts are close but not identical
(e.g. aortic_aneurysm drops to 1 pair by FY2023) — a real annual
ICD-10-CM code-set change, which is exactly why the ontology is versioned
per fiscal year rather than reused across years.
