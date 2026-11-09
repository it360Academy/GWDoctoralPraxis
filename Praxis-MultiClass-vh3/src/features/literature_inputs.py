"""Literature-derived inputs from Newman-Toker DE, Nassery N, Schaffer AC,
et al. "Burden of Serious Harms from Diagnostic Error in the United
States: A National, Cross-Sectional Analysis Using 'The Big Three'."
BMJ Qual Saf. 2024;33(2):109-120. doi:10.1136/bmjqs-2021-014130 (PMC10792094).

**None of the numbers in this file are computed by this project.** They
are transcribed directly from the paper's Table 1 and Results section, and
exist here so the Monte Carlo model in montecarlo.py has literature-based
error rates, harm rates, and extrapolation factors to combine with this
project's own NIS-derived incidence (see incidence.py). No dataset of
diagnosis codes alone -- NIS or otherwise -- can tell you how often a
diagnosis was missed; that requires the clinical-studies and
malpractice-claims literature the source paper synthesizes.

Error/harm rate uncertainty is given as the paper's own reported ranges:
95% CI where statistically derived, PR (plausible range, expert-informed)
for the 5 cancers where delay-length definitions varied across source
studies. These are sampled directly in the Monte Carlo simulation
(approximated as normal/lognormal about the point estimate scaled to the
reported range, since the paper's own per-draw distributional family is
not published -- see montecarlo.py docstring for this approximation's
limitations).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

# ---------------------------------------------------------------------
# Table 1, vascular events (top 5) and infections (top 5): published NIS
# 2012-2014 incidence (thousands) is included ONLY for side-by-side
# comparison against this project's own computed incidence -- it is NOT
# used as a Monte Carlo input; see incidence.py for the numbers actually
# used.
# ---------------------------------------------------------------------

@dataclass
class DiseaseRates:
    category: str                       # "vascular" | "infection" | "cancer"
    published_incidence_thousands: Optional[float]   # None for cancers (see below)
    error_rate: float                   # point estimate (proportion)
    error_rate_low: float
    error_rate_high: float
    harm_rate: float
    harm_rate_low: float
    harm_rate_high: float


# Disease keys match src/features/definitions.py DISEASES for the 10
# vascular/infectious diseases this project computes incidence for.
TABLE1_TOP5: Dict[str, DiseaseRates] = {
    "stroke": DiseaseRates("vascular", 952, 0.175, 0.095, 0.273, 0.098, 0.053, 0.155),
    "vte": DiseaseRates("vascular", 320, 0.204, 0.170, 0.239, 0.109, 0.089, 0.131),
    "arterial_te": DiseaseRates("vascular", 173, 0.239, 0.189, 0.295, 0.127, 0.099, 0.160),
    "aortic_aneurysm": DiseaseRates("vascular", 96, 0.356, 0.210, 0.517, 0.221, 0.130, 0.325),
    "mi": DiseaseRates("vascular", 1242, 0.015, 0.010, 0.022, 0.008, 0.005, 0.012),
    "sepsis": DiseaseRates("infection", 1345, 0.099, 0.028, 0.206, 0.059, 0.017, 0.123),
    "pneumonia": DiseaseRates("infection", 1469, 0.095, 0.023, 0.143, 0.046, 0.011, 0.070),
    "meningitis_encephalitis": DiseaseRates("infection", 47, 0.256, 0.208, 0.308, 0.147, 0.118, 0.181),
    "spinal_abscess": DiseaseRates("infection", 14, 0.621, 0.546, 0.692, 0.367, 0.315, 0.422),
    "endocarditis": DiseaseRates("infection", 34, 0.255, 0.217, 0.296, 0.138, 0.115, 0.164),
}

# Cancers: incidence is 2014 NAACCR/ACS cancer-registry counts (thousands),
# NOT derivable from NIS discharge data (most cancer care is outpatient;
# inpatient discharges would badly undercount true incidence). Taken here
# as a fixed, cited input.
TABLE1_CANCERS: Dict[str, DiseaseRates] = {
    "lung_cancer": DiseaseRates("cancer", 224, 0.225, 0.113, 0.378, 0.142, 0.071, 0.241),
    "breast_cancer": DiseaseRates("cancer", 235, 0.089, 0.085, 0.263, 0.045, 0.042, 0.134),
    "colorectal_cancer": DiseaseRates("cancer", 137, 0.096, 0.084, 0.477, 0.056, 0.049, 0.281),
    "melanoma": DiseaseRates("cancer", 76, 0.136, 0.068, 0.250, 0.057, 0.028, 0.106),
    "prostate_cancer": DiseaseRates("cancer", 233, 0.024, 0.017, 0.138, 0.013, 0.009, 0.074),
}

# "Other" (non-top-5) diseases within each Big Three category: published
# incidence + harm rate (thousands, proportion); error rate is the
# incidence-weighted mean of that category's top-5 error rates (paper's
# own stated method for diseases with no literature-derivable rate).
# These come directly from Table 1 and are NOT computed by this project --
# the full HCUP Clinical Classification Software vascular/infection
# category groupings needed to compute them are not reproduced here.
OTHER_BUCKETS = {
    "other_vascular": {"category": "vascular", "published_incidence_thousands": 3173,
                       "incidence_low": 3131, "incidence_high": 3215,
                       "harm_rate": 0.014, "harm_rate_low": 0.011, "harm_rate_high": 0.019},
    "other_infection": {"category": "infection", "published_incidence_thousands": 3286,
                        "incidence_low": 3249, "incidence_high": 3323,
                        "harm_rate": 0.033, "harm_rate_low": 0.014, "harm_rate_high": 0.047},
    "other_cancer": {"category": "cancer", "published_incidence_thousands": 640,
                     "incidence_low": None, "incidence_high": None,   # registry census, no CI
                     "harm_rate": 0.074, "harm_rate_low": 0.067, "harm_rate_high": 0.142},
}

# Extrapolation factors from Newman-Toker DE, Schaffer AC, Yu-Moe CW, et al.
# "Serious misdiagnosis-related harms in malpractice claims: The 'Big
# Three'." Diagnosis (Berl). 2019;6(3):227-240 -- a DIFFERENT data source
# (malpractice claims + clinical practice-based studies), not NIS and not
# computed by this project. The paper divides the Big-Three total by these
# fractions to reach the "grand total across all dangerous diseases."
BIG_THREE_ERROR_FRACTION = 0.585      # errors attributable to Big Three
BIG_THREE_HARM_FRACTION = 0.758       # serious harms attributable to Big Three

# Published headline results, for direct comparison against this
# project's replication (see report.py).
PUBLISHED_RESULTS = {
    "big3_errors_thousands": (1514, 1122, 1889),
    "big3_harms_thousands": (603, 454, 776),
    "grand_total_errors_thousands": (2588, 1918, 3230),
    "grand_total_harms_thousands": (795, 598, 1023),
    # Principal-diagnosis-only sensitivity analysis (paper p.113: "Using only
    # principal diagnosis NIS codes ... gave lower estimates by about 30%
    # (grand totals 1.78 M missed diagnoses and 549000 serious harms")
    "sensitivity_conservative_harms_thousands": 549,
    "top15_harms_share": 0.507,
    "top5_harms_share": 0.387,
}

# The five named phenotypes with the highest published serious-harm burden.
PUBLISHED_TOP5_HARM_DISEASES = ["stroke", "sepsis", "pneumonia", "vte", "lung_cancer"]
