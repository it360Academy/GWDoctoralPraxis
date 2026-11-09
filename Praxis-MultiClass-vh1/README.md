# H1-MultiClass — NIS 2021–2023 (ICD-10-CM), DX2–DX25 → DX1 Multiclass Discrimination

Production-grade research pipeline testing:

> **H1:** Within a leakage-controlled, cross-year-replicated supervised
> multiclass framework for diagnostic-quality auditing, a hospitalized
> patient's secondary-diagnosis profile (DX2–DX25) discriminates the
> principal diagnosis (DX1) among fifteen vascular, infectious, and
> oncological phenotypes in the U.S. National Inpatient Sample, 2021–2023,
> under severe class imbalance.

> **Research Question 1:** Can a supervised multiclass model, using
> DX2–DX25, accurately predict DX1 across fifteen phenotypes in NIS
> 2021–2023?

Train on **NIS 2021**, model-select on **NIS 2022**, final independent
test on **NIS 2023**. "Discriminates" means out-of-sample classification
performance on held-out data — not temporal forecasting.

---

## This is a rewrite of an earlier binary/per-disease H1 pipeline

This project descends from a prior codebase that tested a *different*
H1: fifteen separate **binary** one-vs-rest models (does this patient
have stroke, yes/no — repeated per disease) using AHRQ's pre-aggregated
`CMR_*` comorbidity flags as predictors, plus a comorbidity co-occurrence
network (Hidalgo/Goh methodology) layered on top.

| | Prior H1 (binary + network) | This H1 (multiclass, this repo) |
|---|---|---|
| Task | 15 independent binary classifiers | **One** 15-class softmax classifier |
| Predictors | 20 AHRQ `CMR_*` aggregated flags | Raw **DX2–DX25** codes directly |
| Framework | + comorbidity-network replication (RR/phi, cross-year edges) | Not part of this H1 — dropped |
| Cohort | Every discharge, per-disease binary label | Only discharges whose DX1 is one of the 15 phenotypes |
| Cross-year design | Train 2021 / select 2022 / test 2023 | **Unchanged** |

If you're looking for the network-methodology version, it lives in the
prior `H1-Comorbidity` codebase — it is a genuinely different H1 and is
not reproduced here.

---

## Feature representation: DX2–DX25 → 3-character ICD-10-CM category

DX2–DX25 are raw ICD-10-CM codes: tens of thousands of distinct values,
very sparse per patient. This pipeline encodes them as **3-character ICD-
10-CM category multi-hot features** (e.g. "I21.4" → "I21"), chosen over
two alternatives:

- *Full-precision code multi-hot* — more granular, but dimensionality
  explodes and overfitting risk rises sharply under the already-severe
  class imbalance H1 explicitly frames as a study condition.
- *AHRQ CCSR clinical categories* — the most clinically standard grouping,
  but requires downloading AHRQ's public CCSR crosswalk file, which this
  environment doesn't ship. 3-character categories are self-contained
  (no external file required) and match the granularity `definitions.py`
  already uses for the 15 phenotype code lists, making the leakage audit
  directly comparable.

The category **vocabulary is fit on the training year (2021) only** —
`min_category_frequency` in config controls the frequency floor for a
category to become a feature column — and applied unchanged to
2022/2023; a category never seen in training contributes nothing to a
later year's feature matrix (this is a real train/test boundary on
feature *selection*, not just model fitting).

## Leakage control: strict code-level phenotype masking

Positional separation (label from DX1 only, predictors from DX2–DX25
only) is the primary defense, but it isn't sufficient by itself: a
secondary code that is *itself* one of the 15 phenotypes' defining codes
(e.g. DX1=MI, but an I21-family code also sits in DX2 — a real recurrent/
bilateral-event pattern) would let a classifier learn "presence of a
phenotype-defining code in a secondary position" as a near-direct proxy
for the true class.

`src/leakage/controls.py` therefore **masks any DX2–25 code that matches
ANY of the 15 phenotype definitions** (same prefix+exclude logic as
`definitions.py`) out of the feature vocabulary entirely — for every row,
not just rows of the matching class. This is a real trade-off (legitimate
recurrent-event signal is discarded along with the leakage risk), and is
the specific thing "leakage-controlled" in H1's wording refers to. Every
masked code is logged with which phenotype(s) it matched and how often it
occurred (`outputs/tables/leakage_code_audit.csv`).

---

## ⚠️ Before treating any result as final

1. **All 15 ICD-10-CM disease definitions are DRAFT** (`src/features/
   definitions.py`, every entry `verified: False`) — inherited unchanged
   from the prior project. A clinician/coder should review before
   results are reported; `sepsis` in particular has multiple competing
   validated definitions in the literature.
2. **`h1_criteria` floors in `config/config.yaml` are unverified
   placeholders**, not derived from a published benchmark for this exact
   task formulation. `src/reporting/h1_criteria.py` applies whatever
   floors are configured; it does not choose them.
3. **The 3-character category granularity is a modeling choice, not a
   clinical standard.** If you obtain AHRQ's public CCSR crosswalk, a
   drop-in `representation: "ccsr"` mode is a natural extension point in
   `src/features/dx_features.py` (not implemented here — flagged rather
   than silently assumed equivalent).
4. **This environment has never touched real NIS files.** Everything here
   was built and verified against synthetic ICD-10-shaped data
   (`scripts/generate_synthetic_smoke_data.py`) — the full pipeline (all
   4 algorithms, all 6 stages) runs clean on synthetic data, which
   demonstrates the *code* works, not that the *science* is correct. Real
   execution requires your NIS 2021/2022/2023 Core files at the
   configured path.
5. **Ambiguous-DX1 discharges are dropped, not adjudicated.** A discharge
   whose principal diagnosis matches more than one of the 15 phenotype
   definitions (should be rare/zero given the code lists are meant to be
   mutually exclusive — checked in `tests/test_definitions.py` and
   reported per year in data validation) is excluded from the cohort
   rather than assigned a label by tie-breaking rule.

---

## 1. Setup

```bash
cd H1-MultiClass
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Real data is expected at the path in `config/config.yaml`
(`paths.nis_root`) as `NIS_<year>/NIS_<year>_Core.csv`. Only the Core
file is needed — this study does not use DX_PR_GRPS or Hospital files.

## 2. Verify the code first (no real data needed)

```bash
pytest tests/ -q
python scripts/generate_synthetic_smoke_data.py --out data/synthetic --rows 20000
python main.py --config config/config_smoke.yaml
```

**Smoke outputs are synthetic and are never H1 evidence.**

## 3. Full run on real NIS data

```bash
python main.py --stage inspect
python main.py --stage ingest
python main.py --stage validate
python main.py --stage leakage_audit
python main.py --stage model              # trains all 4 algorithms, selects on 2022, evaluates on 2023
python main.py --stage report
# or all stages at once:
python main.py
```

```bash
# Restrict algorithms while debugging:
python main.py --stage model --algorithms logistic_regression

# Tuning is expensive at real scale - off by default. Override explicitly:
python main.py --tune
python main.py --no-tune
```

Checkpoints (`data/raw_<year>.parquet`, `data/dx_vocabulary.pkl`) let
interrupted runs resume; set `checkpointing.overwrite: true` to force
recompute.

## 4. Key outputs

| Path | Content |
|---|---|
| `outputs/reports/H1_validation_report.html` | Full report: verdict, per-phenotype criteria, algorithm comparison |
| `outputs/tables/h1_verdict.json` | Aggregate H1 decision (macro AUROC/MCC/balanced accuracy vs. floors, coverage) |
| `outputs/tables/h1_phenotype_criteria.csv` | Per-phenotype statistical-validity + performance-floor verdicts |
| `outputs/tables/model_performance_all.csv` | Every algorithm × year: full aggregate metric suite |
| `outputs/tables/negative_control.json` | Permutation-test null-distribution comparison |
| `outputs/tables/leakage_code_audit.csv` | Every DX2-25 code masked for phenotype overlap, with counts |
| `outputs/tables/dx_category_vocabulary.csv` | The fitted (post-mask) feature vocabulary |
| `outputs/tables/confusion_matrix_test.csv` | 15×15 confusion matrix, test year |
| `outputs/tables/feature_importance_top.csv` | Top predictive DX categories per the selected model |
| `outputs/models/best_model_<algorithm>.joblib` | Persisted best model (selected on 2022) |

## 5. Methodology summary

- **Label**: principal diagnosis (`I10_DX1`), collapsed to one of 15
  phenotype names via `src/features/definitions.py`; discharges matching
  zero or >1 phenotypes are excluded from the cohort.
- **Predictors**: DX2–DX25 3-character ICD-10-CM category multi-hot
  matrix (post leakage-masking), fit on 2021 only; optionally + AGE,
  FEMALE, ELECTIVE, AWEEKEND, PAY1 as covariates
  (`features.include_demographic_covariates`).
- **Leakage controls**: positional separation + strict code-level
  phenotype masking (see above).
- **Imbalance**: `sklearn.utils.class_weight.compute_sample_weight
  ("balanced", y)` on training data only, applied uniformly across all
  four algorithms (not each library's own inconsistent `class_weight`
  support). NIS's `DISCWT` survey weight is reserved for national-
  estimate reporting, not model fitting.
- **Models**: multinomial logistic regression, random forest, LightGBM,
  XGBoost (optionally GPU); optional randomized-search tuning (off by
  default); model selection on 2022 macro AUROC-OVR.
- **Evaluation**: macro/weighted F1, MCC, balanced accuracy, multiclass
  Brier score, per-class AUROC-OVR/precision/recall/specificity/NPV/PPV-
  lift-over-null, 15×15 confusion matrix; stratified-bootstrap 95% CIs
  throughout.
- **Statistics**: permutation-based negative control (labels permuted
  against a fixed, already-fitted model's predicted probabilities, with
  an explicit `underpowered` flag when the permutation count can't
  mathematically reach p<0.05); paired bootstrap for algorithm-vs-
  algorithm AUROC comparison.
- **H1 decision rule** (`src/reporting/h1_criteria.py`): a phenotype is
  "supported" if its AUROC-OVR bootstrap CI excludes chance AND it
  clears its AUROC floor (or, for low-prevalence phenotypes, a PPV-
  lift-over-null-rate floor). H1 overall requires the aggregate macro
  AUROC/MCC/balanced-accuracy floors, full 15-phenotype test-year
  coverage, and at least one individually-supported phenotype.
- **Explainability**: feature importance (impurity-based for tree
  models, coefficient magnitude per class for logistic regression).

## 6. Reproducibility

Deterministic seeds throughout, config-driven behavior, checkpointed
stages, structured logging, runtime tracking, data-validation gates, unit
tests (`pytest tests/`). No result in the report is hard-coded.
