"""ICD-10-CM definitions for the 15 target diagnoses (NIS 2021-2023).

REWRITTEN from the original ICD-9-CM version (NIS 2012-2014). NIS switched
to ICD-10-CM/PCS exclusively starting October 1, 2015; no ICD-9 code can
match anything in 2021-2023 data, so this file could not simply be
"updated" - every code list is a new derivation.

*** RESEARCHER VERIFICATION REQUIRED ***
These ICD-10-CM code lists are DRAFT starting points, derived from general
ICD-10-CM chapter/category knowledge, NOT copied from a validated published
crosswalk. Every entry is flagged `verified: False`. Before treating any
result as final, a clinician/coder should confirm:
  1. Code ranges are complete for the FY2021-FY2023 ICD-10-CM releases
     (annual updates each October 1 may add/retire codes within this
     window).
  2. Exclusion logic matches the intended clinical definition (e.g. whether
     "personal history of" or "sequela" codes should be excluded).
  3. Whether principal-diagnosis-only (as used here) is the right target
     definition, or whether some phenotypes should also allow secondary-dx
     confirmation.

A discharge matches a disease if any listed prefix is a prefix of the
diagnosis code (dot removed, as stored in HCUP NIS I10_DX fields, e.g.
"I21.4" -> "I214").
"""
from __future__ import annotations

from typing import Dict, List

PHENOTYPES: Dict[str, List[str]] = {
    "vascular":   ["stroke", "vte", "arterial_te", "aortic_aneurysm", "mi"],
    "infectious": ["sepsis", "pneumonia", "meningitis_encephalitis",
                   "spinal_abscess", "endocarditis"],
    "oncological": ["lung_cancer", "breast_cancer", "colorectal_cancer",
                    "melanoma", "prostate_cancer"],
}

DISEASES: Dict[str, Dict] = {
    # ----------------------------------------------------------- vascular
    "stroke": {
        "label": "Stroke (ischemic + hemorrhagic, acute)",
        "phenotype": "vascular",
        "icd10_prefixes": [
            "I60", "I61",            # nontraumatic subarachnoid/intracerebral hemorrhage
            "I63",                    # cerebral infarction
            "I64",                    # stroke, not specified as hemorrhage or infarction
        ],
        "icd10_exclude": ["I69"],     # sequelae of cerebrovascular disease (historical, not acute)
        "verified": False,
    },
    "vte": {
        "label": "Venous thromboembolism (PE + DVT)",
        "phenotype": "vascular",
        "icd10_prefixes": [
            "I26",                    # pulmonary embolism
            "I80", "I82",             # phlebitis/thrombophlebitis; other venous embolism/thrombosis
        ],
        "icd10_exclude": [],
        "verified": False,
    },
    "arterial_te": {
        "label": "Arterial thromboembolism",
        "phenotype": "vascular",
        "icd10_prefixes": ["I74"],    # arterial embolism and thrombosis
        "icd10_exclude": [],
        "verified": False,
    },
    "aortic_aneurysm": {
        "label": "Aortic aneurysm / dissection",
        "phenotype": "vascular",
        "icd10_prefixes": ["I71"],
        "icd10_exclude": [],
        "verified": False,
    },
    "mi": {
        "label": "Acute myocardial infarction",
        "phenotype": "vascular",
        "icd10_prefixes": ["I21", "I22"],   # STEMI/NSTEMI; subsequent MI
        "icd10_exclude": [],
        "verified": False,
    },
    # --------------------------------------------------------- infectious
    "sepsis": {
        "label": "Sepsis / septicemia",
        "phenotype": "infectious",
        "icd10_prefixes": ["A40", "A41"],   # streptococcal sepsis; other sepsis
        "icd10_exclude": [],
        "verified": False,
        "notes": "Multiple competing validated ICD-10 sepsis algorithms exist "
                 "(CDC Adult Sepsis Event, CMS SEP-1, Angus implicit criteria); "
                 "this is the narrowest explicit-code definition. Confirm this "
                 "matches your intended definition before finalizing.",
    },
    "pneumonia": {
        "label": "Pneumonia",
        "phenotype": "infectious",
        "icd10_prefixes": ["J12", "J13", "J14", "J15", "J16", "J18"],
        "icd10_exclude": [],
        "verified": False,
    },
    "meningitis_encephalitis": {
        "label": "Meningitis / encephalitis",
        "phenotype": "infectious",
        "icd10_prefixes": ["G00", "G03", "G04", "A87"],
        "icd10_exclude": [],
        "verified": False,
    },
    "spinal_abscess": {
        "label": "Spinal (intraspinal) abscess",
        "phenotype": "infectious",
        "icd10_prefixes": ["G061", "G062"],  # intraspinal abscess; extradural/subdural abscess
        "icd10_exclude": [],
        "verified": False,
    },
    "endocarditis": {
        "label": "Infective endocarditis",
        "phenotype": "infectious",
        "icd10_prefixes": ["I33"],
        "icd10_exclude": [],
        "verified": False,
    },
    # -------------------------------------------------------- oncological
    "lung_cancer": {
        "label": "Lung cancer (bronchus/lung)",
        "phenotype": "oncological",
        "icd10_prefixes": ["C34"],
        "icd10_exclude": ["C780"],    # secondary/metastatic lung neoplasm - primary site only
        "verified": False,
    },
    "breast_cancer": {
        "label": "Breast cancer",
        "phenotype": "oncological",
        "icd10_prefixes": ["C50"],
        "icd10_exclude": [],
        "verified": False,
    },
    "colorectal_cancer": {
        "label": "Colorectal cancer",
        "phenotype": "oncological",
        "icd10_prefixes": ["C18", "C19", "C20"],
        "icd10_exclude": [],
        "verified": False,
    },
    "melanoma": {
        "label": "Melanoma of skin",
        "phenotype": "oncological",
        "icd10_prefixes": ["C43"],
        "icd10_exclude": ["C44"],     # C44 = other (non-melanoma) skin cancers - must not overlap
        "verified": False,
    },
    "prostate_cancer": {
        "label": "Prostate cancer",
        "phenotype": "oncological",
        "icd10_prefixes": ["C61"],
        "icd10_exclude": [],
        "verified": False,
    },
}

DISEASE_ORDER: List[str] = [d for ph in ("vascular", "infectious", "oncological")
                            for d in PHENOTYPES[ph]]


def disease_label(name: str) -> str:
    """Look up a disease's display label. Also handles keys built as
    "{disease}_{feature_set}" (e.g. "stroke_hospital", used for per-
    feature-set figure filenames/titles) by stripping the known suffixes
    before lookup, falling back to the raw string if still unrecognized."""
    if name in DISEASES:
        return DISEASES[name]["label"]
    for suffix in ("_official", "_hospital"):
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            if base in DISEASES:
                variant = suffix.strip("_").capitalize()
                return f"{DISEASES[base]['label']} ({variant} feature set)"
    return name


def unverified_diseases() -> List[str]:
    return [d for d, spec in DISEASES.items() if not spec.get("verified", False)]


def validate_definitions() -> None:
    """Sanity checks used by unit tests and at pipeline start-up."""
    assert len(DISEASES) == 15, "Expected exactly 15 target diseases"
    assert sum(len(v) for v in PHENOTYPES.values()) == 15
    for name, spec in DISEASES.items():
        assert spec["icd10_prefixes"], f"{name} has no ICD-10 prefixes"
        assert spec["phenotype"] in PHENOTYPES
        for p in spec["icd10_prefixes"]:
            assert p[0].isalpha() and p[0].isupper(), (
                f"bad ICD-10-CM prefix '{p}' in {name}: must start with an "
                f"uppercase letter (ICD-10-CM codes never start with a digit "
                f"- if this looks like an ICD-9 code, the rewrite is incomplete)"
            )
