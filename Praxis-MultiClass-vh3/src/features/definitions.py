"""ICD-10-CM definitions for the 15 target diagnoses (NIS 2021-2023).

*** RESEARCHER VERIFICATION REQUIRED ***
These ICD-10-CM code lists are DRAFT starting points, derived from general
ICD-10-CM chapter/category knowledge, NOT copied from a validated published
code list. Every entry is flagged `verified: False`. Before treating any
result as final, a clinician/coder should confirm:
  1. Code ranges are complete for the FY2021-FY2023 ICD-10-CM releases
     (annual updates each October 1 may add/retire codes within this
     window).
  2. Exclusion logic matches the intended clinical definition (e.g. whether
     "personal history of" or "sequela" codes should be excluded).
  3. Whether the DX1-or-DX2 incidence definition (principal or first-listed
     secondary diagnosis, as used here) is the right target definition for
     every phenotype.

Source definitions: Newman-Toker et al. (2024), with exact lists in the
paper's online supplemental file 1-A3 (not printed in the article). The ten
NIS-derived phenotypes below are this project's ICD-10-CM definitions; the
paper's own supplemental 1-A3 is still needed to confirm them, especially
for arterial thromboembolism.

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
    # Vascular and infectious phenotypes: ICD-10-CM prefixes. History/sequela
    # codes are excluded because H3 counts incident cases.
    # ----------------------------------------------------------- vascular
    "stroke": {
        "label": "Stroke (acute cerebrovascular disease)",
        "phenotype": "vascular",
        "icd10_prefixes": [
            "I60", "I61", "I62",     # SAH, ICH, other nontraumatic intracranial hemorrhage
            "I63",                   # cerebral infarction
            "G436",                  # migraine with infarction
        ],
        "icd10_exclude": ["I69"],    # sequelae of cerebrovascular disease
        "verified": False,
    },
    "vte": {
        "label": "Venous thromboembolism (PE + DVT)",
        "phenotype": "vascular",
        "icd10_prefixes": [
            "I26",                   # pulmonary embolism
            "I80",                   # phlebitis/thrombophlebitis
            "I81",                   # portal vein thrombosis
            "I82",                   # other venous embolism/thrombosis
        ],
        "icd10_exclude": [],
        "verified": False,
        "notes": "History-of-VTE codes (Z86.71x) are deliberately not included.",
    },
    "arterial_te": {
        "label": "Arterial thromboembolism",
        "phenotype": "vascular",
        "icd10_prefixes": [
            "I74",                   # arterial embolism and thrombosis
            "I75",                   # atheroembolism
        ],
        "icd10_exclude": [],
        "verified": False,
        "notes": "I74/I75 alone do not plausibly reach the published 173k/yr; the "
                 "paper's grouping may have been broader. Unresolved until the "
                 "paper's own supplemental file 1-A3 is available.",
    },
    "aortic_aneurysm": {
        "label": "Aortic aneurysm / dissection",
        "phenotype": "vascular",
        "icd10_prefixes": [
            "I71",                   # aortic aneurysm and dissection
            "I72",                   # other aneurysm
            "I777",                  # dissection of other arteries
            "I7781",                 # aortic ectasia
        ],
        "icd10_exclude": [],
        "verified": False,
    },
    "mi": {
        "label": "Acute myocardial infarction",
        "phenotype": "vascular",
        "icd10_prefixes": ["I21", "I22"],
        "icd10_exclude": [],
        "verified": False,
    },
    # --------------------------------------------------------- infectious
    "sepsis": {
        "label": "Sepsis / septicemia",
        "phenotype": "infectious",
        "icd10_prefixes": [
            "A40", "A41",            # streptococcal / other sepsis
            "A021",                  # Salmonella sepsis
            "A207",                  # septicemic plague
            "A227",                  # anthrax sepsis
            "A392", "A393", "A394",  # meningococcemia
            "B007",                  # disseminated herpesviral disease
            "I76",                   # septic arterial embolism
            "P36",                   # bacterial sepsis of newborn
            "R7881",                 # bacteremia
            "R652",                  # severe sepsis, with/without shock
        ],
        "icd10_exclude": [],
        "verified": False,
        "notes": "Labor/puerperal sepsis is not included. R65.2x (severe sepsis, "
                 "including R65.21 with septic shock) cannot be sequenced as "
                 "principal, so it affects DX2 only.",
    },
    "pneumonia": {
        "label": "Pneumonia (except TB or STD)",
        "phenotype": "infectious",
        "icd10_prefixes": [
            "J12", "J13", "J14", "J15", "J16", "J17", "J18",
            "A0222",                 # Salmonella pneumonia
            "A202",                  # pneumonic plague
            "A212",                  # pulmonary tularemia
            "A221",                  # pulmonary anthrax
            "A310",                  # pulmonary mycobacterial infection
            "A420",                  # pulmonary actinomycosis
            "A70",                   # ornithosis
            "B012",                  # varicella pneumonia
            "B052",                  # measles pneumonia
            "B371",                  # pulmonary candidiasis
            "B380", "B381", "B382",  # pulmonary coccidioidomycosis
            "B390", "B391", "B392",  # pulmonary histoplasmosis
            "B583",                  # pulmonary toxoplasmosis
            "B59",                   # pneumocystosis
            "J851", "J852",          # lung abscess with / without pneumonia
        ],
        "icd10_exclude": [],
        "verified": False,
        "notes": "J12.82 (pneumonia due to COVID-19, FY2021+) is included via the "
                 "J12 prefix. Influenza (J09-J11) and aspiration pneumonitis (J69) "
                 "are not.",
    },
    "meningitis_encephalitis": {
        "label": "Meningitis / encephalitis",
        "phenotype": "infectious",
        "icd10_prefixes": [
            "G00", "G01", "G02", "G03",          # meningitis
            "G04", "G05",                        # encephalitis/myelitis
            "G373",                              # acute transverse myelitis
            "A0221",                             # Salmonella meningitis
            "A390", "A3981",                     # meningococcal meningitis/encephalitis
            "A83", "A84", "A85", "A86", "A87",   # viral (arthropod-borne) encephalitis/meningitis
            "B003", "B004",                      # herpesviral meningitis/encephalitis
            "B011",                              # varicella encephalitis/myelitis
            "B021",                              # zoster meningitis
            "B050",                              # measles encephalitis
            "B261", "B262",                      # mumps meningitis/encephalitis
            "B375",                              # candidal meningitis
            "B384",                              # coccidioidal meningitis
            "B451",                              # cryptococcal meningitis
            "B582",                              # toxoplasma meningoencephalitis
            "A9231",                             # West Nile encephalitis
        ],
        "icd10_exclude": [],
        "verified": False,
        "notes": "Sequela codes are not incident cases and are not included.",
    },
    "spinal_abscess": {
        "label": "Spinal (intraspinal) abscess",
        "phenotype": "infectious",
        "icd10_prefixes": ["G061"],  # intraspinal abscess and granuloma
        "icd10_exclude": [],
        "verified": False,
        "notes": "G06.2 (extradural/subdural abscess, unspecified site) is not "
                 "included.",
    },
    "endocarditis": {
        "label": "Infective endocarditis",
        "phenotype": "infectious",
        "icd10_prefixes": [
            "I33",                   # acute and subacute endocarditis
            "A3951",                 # meningococcal endocarditis
            "B376",                  # candidal endocarditis
            "I011",                  # acute rheumatic endocarditis
        ],
        "icd10_exclude": [],
        "verified": False,
        "notes": "Pericarditis, myocarditis and cardiomyopathy are not "
                 "endocarditis and are excluded; I38/I39 are treated as heart "
                 "valve disorders and excluded.",
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
                f"uppercase letter (ICD-10-CM codes never start with a digit)"
            )
