# H2-DEV: Diagnostic Exclusivity Violations in NIS 2021–2023

Tests whether hospital discharge coding is consistent with the ICD-10-CM
**Excludes1** convention, using the National Inpatient Sample (NIS)
2021–2023.

## Hypothesis and research question

**H2:** Diagnostic Exclusivity Violations (DEVs), discharges whose
principal diagnosis co-occurs with an Excludes1-incompatible secondary
diagnosis, are rare — well below the rate expected if coders ignored the
Excludes1 rule — with a stable rate across each NIS year from 2021 to
2023.

**RQ2:** Are diagnosis pairs that shouldn't be coded together under the
Excludes1 rule rare enough — below the rate expected by chance — in U.S.
hospital discharge records every year from 2021 to 2023, and does that low
rate hold steady year after year?

**Definition.** A Diagnostic Exclusivity Violation (DEV) occurs when a
hospitalization's principal diagnosis (`I10_DX1`) co-occurs — anywhere in
`I10_DX2`–`I10_DX25` — with a secondary diagnosis that the ICD-10-CM
Excludes1 note attached to the principal diagnosis's code says must never
be coded together with it. DEV *detection* is fully deterministic given
the Excludes1 ontology (`src/dev/detector.py` has no statistical or ML
component).

### How H2 is tested

`src/reporting/h2_criteria.py:compliance_verdict()` — `H2_SUPPORTED` in
`outputs/tables/h2_verdict_compliance.json` and the top of the HTML report.

**No cutoff is chosen by the investigator.** "Rare" is measured against
a **data-derived baseline** — the DEV rate you get when diagnosis
profiles are randomly re-paired (what the data would show if Excludes1
were ignored), from `src/negative_controls/`:

- the **chance baseline** (decides H2): shuffles secondary-diagnosis
  profiles within age-band × sex strata;
- the **coding-artifact baseline** (secondary analysis only, not part of
  the decision): shuffles only within each hospital, so hospital-level
  coding style is preserved.

The baselines come from shuffling, not from the observed pairing, so the
test is not circular (a cutoff computed from the observed rate itself
would pass by construction). H2 is supported only when, in **all three
years**:

1. the observed rate is **significantly below the chance baseline**
   (one-sided test; Holm-corrected across years) — positive evidence, not
   merely "not significantly above" (see the correction log in
   `h2_criteria.py`);
2. the rate is **stable** (consecutive-year ratios within 1.5×,
   `stability_criteria.rate_ratio_threshold`) and the leading
   Excludes1 pairs keep a consistent ranking (Spearman ≥ 0.70).

The report shows each year's observed/baseline ratio (e.g. "0.84 = 16%
below the chance baseline") and, as **context only**, the absolute rate:
among *eligible* discharges (principal diagnosis is one of the 15 studied
phenotypes — the only ones a DEV can be detected in;
`src/dev/rates.py:cohort_rate`, hospital-clustered bootstrap CI) and among
all discharges (diluted by the ~86% never checked).

**Interpretation limits.** "Rare" here is a prevalence/relative claim, not
proof that hospitals apply Excludes1 correctly (low rates can also reflect
low comorbidity, under-coding, or an ontology limited to the studied
phenotypes). Both shuffled baselines break real clinical associations between
a principal diagnosis and its secondaries; Excludes1 pairs are often
clinically adjacent conditions, so a rate *above* a baseline can reflect
natural association rather than non-compliance. Published coding-accuracy
research (e.g. Burns et al., *J Public Health* 2012, DOI
10.1093/pubmed/fdr054) measures agreement with the medical record — a
different quantity from internal code conflicts — so it is context, not a
benchmark for this test.

**Reporting convention.** The HTML report opens with a *Summary of
findings* generated from the results (`report.py:findings_summary`), which
states separately (a) the absolute rate, (b) whether the rate is stable,
(c) which years fall significantly below the chance baseline, and (d)
every exception. If any year is not below the chance baseline, the
summary says H2 is not supported — it never softens the verdict. It then
reports, as a secondary analysis, the coding-artifact baseline's results
and the verdict under the original prespecified rule (below both
baselines). In the NIS 2021–2023 results from the 200-permutation run, the
absolute rate is about 0.7% of eligible discharges and stable, and below
the chance baseline in all three years; 2021 is above the
hospital-preserving baseline, so the original prespecified rule is not
met.

### Amendment history

H2 was first worded "below chance", then reworded "rare and stable" after
the first full analysis. An interim draft used a fixed 1% cutoff; it was
dropped (never finalised) in favour of the data-derived baselines above.
The statistical test has been the same throughout, and the first analysis
already showed a mixed result (2021 slightly *above* the coding-artifact
baseline). The report states this.

**Second amendment (post-hoc).** After those results were known, the
coding-artifact baseline was removed from the decision rule, which is now
chance-only. Rationale: the hospital-preserving shuffle keeps each
hospital's secondary-code mix, so it asks whether DEVs are
patient-specific beyond hospital coding style — a different question from
whether Excludes1 is followed — and a rate above it fits the clinical
adjacency of Excludes1 pairs. Because the change was made after seeing
that this was the only failing criterion, the original prespecified
verdict is still computed and reported
(`H2_SUPPORTED_prespecified_both_baselines` in
`h2_verdict_compliance.json`, and in the report's summary and amendment
disclosure).

`h2_criteria.py:h2_verdict()` — an even
earlier elevated-direction reading — is kept for transparency only
(`outputs/tables/h2_verdict_elevated_secondary.json`, Section 2b).

## Architecture

```
main.py                    Pipeline orchestrator (9 stages, --stage to run one)
config/
  config.yaml               Real-run config (edit paths.nis_root to your NIS data)
  config_smoke.yaml         Synthetic-data smoke-test config
scripts/
  generate_synthetic_data.py   Builds data/synthetic/ + a fabricated SEED ontology
src/
  config.py                  YAML config loading + validation + path management
  logging_utils.py           Structured logging, per-stage timing
  features/
    definitions.py            15 target phenotypes' ICD-10-CM code lists (DRAFT - see below)
    icd_classifier.py          Generic prefix-based ICD-10-CM code classifier
  data/
    csv_io.py                  Encoding/delimiter-robust NIS CSV reader
    inspect.py                 Column auto-detection on the raw Core file
    ingestion.py                Streams Core CSV -> per-year parquet checkpoint
    hospital.py                 Hospital-file loader (region/teaching/bedsize)
    validation.py               Data-quality gates on each year's checkpoint
  ontology/
    excludes1_parser.py         Parses official ICD-10-CM Tabular XML -> raw notes CSV
    excludes1_loader.py         Raw notes -> phenotype-filtered Excludes1Pair list
  dev/
    detector.py                  Deterministic DEV labeling (the core of H2)
    rates.py                     Weighted/raw rate aggregation + bootstrap CIs
    sensitivity.py               Position/weighting/direction/coding-intensity sensitivity
  negative_controls/
    chance_control.py            Stratified chance-co-occurrence null (parallelized)
    artifact_control.py          Hospital-preserving shuffle + duplicate-profile heuristic
    investigate_2021_artifact.py 2021 artifact-null diagnostic: hospital structure + all-year rerun
  analysis/
    models.py                    Leakage-safe DEV-risk characterization model
    evaluation.py                 Binary classification metrics
    explain.py                    SHAP + native feature importance
  stability/
    temporal_stability.py        Cross-year rate/rank/position/model-performance stability
  reporting/
    h2_criteria.py                H2 decision rule (below-chance verdict + prespecified + elevated verdicts)
    figures.py                    Report plots
    report.py                     Assembles outputs/reports/H2_validation_report.html
tests/                       Unit tests (pytest)
ontology/SOURCE.md           How the Excludes1 ontology is sourced/versioned
```

## Running it

### 1. Smoke test (synthetic data — always do this first)

```bash
pip install -r requirements.txt
python scripts/generate_synthetic_data.py
python main.py --config config/config_smoke.yaml --stage all
```

This writes to `outputs_smoke/` and takes well under a minute. It proves
the *code* works end-to-end; it proves nothing about real DEV rates —
the diagnosis codes, weights, hospitals, and Excludes1 pairs are all
fabricated. The synthetic generator deliberately injects real violations
into a slice of the sepsis cohort so the negative-control and stability
stages have non-degenerate signal to exercise, which is why the smoke
run's `H2_SUPPORTED` will typically come back `False` (the synthetic
"hospitals" are, by construction, *not* Excludes1-compliant) — that is
the smoke test working correctly, not a bug.

### 2. Real run

1. Edit `config/config.yaml` (and `config/config_real_dryrun.yaml`, which
   should mirror it): set `paths.nis_root` to your local NIS 2021–2023
   CSV directory and confirm the `files:` block matches your actual
   filenames.
2. Build the real Excludes1 ontology for each fiscal year — see
   `ontology/SOURCE.md` for the exact steps
   (`excludes1_parser.py` → `excludes1_loader.build_ontology()`). A real
   run refuses to proceed against the SEED file.
3. **Dry-run first, on real data, before the full 200-permutation run.**
   `negative_controls` re-runs `detect()` `n_permutations` times per
   null per year — it dominates total runtime, and running the full
   count blind, on data you haven't confirmed loads cleanly, wastes
   hours if something upstream is misconfigured.
   ```bash
   python main.py --config config/config_real_dryrun.yaml --stage all
   ```
   Check the logged per-permutation timing in `logs/` — that tells you
   what the full run will cost (see "Performance" below) — and confirm
   `outputs_smoke`-equivalent output (here, `outputs/`) looks sane.
   **Do not report `config_real_dryrun.yaml`'s p-values as a result** —
   at 30 permutations, `min_achievable_p` (≈0.032) is too coarse to
   reach a Holm-corrected significance threshold; it exists purely to
   validate the pipeline runs, not to test H2.
4. Full run:
   ```bash
   python main.py --config config/config.yaml --stage all
   ```
   Or run one stage at a time (`--stage inspect`, `--stage ingest`, ...)
   — each stage's outputs are checkpointed so a later stage will build
   its own dependencies on demand if you skip straight to it.
5. Read `outputs/reports/H2_validation_report.html`. For an automated
   consumer of the verdict, read
   `outputs/tables/h2_verdict_compliance.json` directly.

### Performance

`negative_controls` is the dominant cost: it calls `detect()` once per
permutation (`n_permutations`, default 200) per null (chance + artifact)
per year — up to 1,200 full `detect()` calls for a real run. Each
permutation is fully independent, so `parallel.n_jobs: -1` (all CPU
cores, the shipped default) gives a near-linear speedup over the
single-core smoke-test default. If a single `detect()` call takes T
seconds on your real data (the dry-run's log tells you this), a rough
estimate for `negative_controls` alone is
`(200 permutations × 2 nulls × 3 years × T) / n_jobs_used`. `ingest`
and `model` are comparatively cheap (minutes, not hours) at NIS scale.

### 2021 coding-artifact investigation (post-hoc diagnostic)

`src/negative_controls/investigate_2021_artifact.py` looks into the one
exception, 2021 against the hospital-preserving baseline. It does **not**
change the H2 verdict or overwrite the pipeline's prespecified
`negative_control_*.json`. Everything goes to
`outputs/investigations/2021_artifact/`.

- **Part 1, structure (descriptive):** per-year missing `HOSP_NIS`,
  hospital count, discharges per hospital, singleton hospitals, and the
  share of discharges the hospital shuffle leaves unpermuted (missing
  + singleton). 2021 is flagged on any measure more than 50% away from
  the 2022/2023 mean. This is a screening heuristic, not a test.
- **Part 2, rerun:** both nulls, **all three years**, at a higher
  `--n-permutations` (default 2000). Every other setting (seed, strata,
  bootstrap, ontology, `n_jobs`) comes from the config, so the only
  thing that changes is the permutation count. It saves per-year
  checkpoints named with the n, and writes a
  `comparison_original_vs_n<N>.csv` that lists every year, whatever the
  result.

```bash
# Part 1 only (fast)
python -m src.negative_controls.investigate_2021_artifact --config config/config.yaml --structure-only
# Full, memory-capped and detached: roughly 10x the negative_controls stage runtime at n=2000
nohup scripts/run_pipeline.sh -m src.negative_controls.investigate_2021_artifact \
    --config config/config.yaml --n-permutations 2000 > investigation_stdout.log 2>&1 &
```

Report the rerun for all three years as it comes out. Don't raise n
again for one year alone.

### 3. Tests

```bash
pytest tests/ -q
```

## Limitations (read before treating any result as final)

- **Phenotype definitions are DRAFT.** Every ICD-10-CM code list in
  `src/features/definitions.py` is flagged `verified: False` — derived
  from general chapter/category knowledge, not a validated published
  crosswalk. A clinician/coder should review before final use (see that
  file's header for the specific things to check).
- **All stability and negative-control thresholds are unverified
  placeholders** (`config.yaml`'s `stability_criteria` and
  `negative_controls.significance_alpha`) — prespecified (fixed before
  results are examined) but not yet grounded in a published benchmark or
  committee sign-off. `H2_SUPPORTED` is only as trustworthy as these

- **DEVs are potential diagnostic-inconsistency *coding* signals, never
  confirmed misdiagnoses**, regardless of which way the verdict comes
  out. A low DEV rate is evidence of Excludes1-consistent *coding*, not
  proof of diagnostic accuracy — those are different claims.
- **The Excludes1 ontology's fiscal-year mapping is a simplification**
  around the October–September ICD-10-CM release cycle vs. the
  January–December NIS calendar year — see `ontology/SOURCE.md`.
- **DX1 scope decision:** this analysis only considers hospitalizations
  whose principal diagnosis falls into one of 15 vascular/infectious/
  oncological phenotypes, not all possible NIS principal diagnoses.
