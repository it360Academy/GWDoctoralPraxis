"""Diagnostics for the 2021 coding-artifact negative control.

This script does two things, corresponding to the two legitimate lines
of investigation for the 2021 result (NOT an attempt to change the
result itself):

  PART 1 - STRUCTURAL DIAGNOSTIC
  Checks whether 2021's HOSP_NIS field and hospital-year groupings look
  structurally different from 2022 and 2023 in ways that could affect
  the hospital-preserving shuffle specifically (e.g. more singleton
  hospitals, more missing HOSP_NIS, a different hospital count). This
  is descriptive only - it does not change or rerun the test. If 2021
  looks structurally different, that is a reportable data-quality
  finding for your limitations section. If it looks the same as the
  other two years, that itself is useful: it means the 2021 result is
  not attributable to a hospital-grouping artifact.

  PART 2 - HIGHER-PRECISION RERUN
  Reruns BOTH negative controls (chance and artifact) for ALL THREE
  years at a higher n_permutations (default 2000 vs. the original 200),
  so the comparison stays apples-to-apples across years rather than
  bumping precision for 2021 alone. Report the results for all three
  years, not just 2021, regardless of what they show.

Integration with the pipeline: everything except n_permutations is taken
from the same config main.py uses - the year checkpoints (load_year), the
per-year Excludes1 ontology (main.stage_ontology_check, including its
SEED-file guard), SEC_COLS, random_seed, the chance null's strata /
age_band_width / n_bootstrap, and parallel.n_jobs. So the ONLY thing that
differs from the prespecified negative_controls stage is the permutation
count. Results go to <outputs_dir>/investigations/2021_artifact/ and never
overwrite the pipeline's negative_control_*.json (the prespecified
200-permutation results remain the ones the H2 verdict is built from).

Usage (from the project root):
    python -m src.negative_controls.investigate_2021_artifact \\
        --config config/config.yaml --n-permutations 2000

    # memory-capped, survives VS Code closing (recommended for real data):
    nohup scripts/run_pipeline.sh -m src.negative_controls.investigate_2021_artifact \\
        --config config/config.yaml --n-permutations 2000 > investigation_stdout.log 2>&1 &

    # Part 1 only (fast, descriptive):
    python -m src.negative_controls.investigate_2021_artifact \\
        --config config/config.yaml --structure-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..dev.detector import detect
from ..reporting import h2_criteria as h2c
from .artifact_control import duplicate_profile_heuristic, hospital_preserving_shuffle
from .chance_control import run as run_chance_control

FOCUS_YEAR = "2021"
OUTLIER_REL_DIFF = 0.50


# ---------------------------------------------------------------------
# PART 1: Structural diagnostic - does 2021 look different?
# ---------------------------------------------------------------------

def hospital_structure_summary(df: pd.DataFrame, hosp_col: str = "HOSP_NIS",
                                weight_col: str = "DISCWT", year_label: str = "") -> Dict:
    """Describes the hospital-grouping structure the hospital-preserving
    shuffle actually depends on. None of this is a p-value or a test -
    it's the descriptive groundwork for deciding whether 2021 is
    structurally unusual, which is the legitimate question here.

    Rows with missing HOSP_NIS and rows in singleton hospitals are both
    left in place by hospital_preserving_shuffle (groupby drops NaN keys;
    size-1 groups are skipped), so share_of_discharges_not_permuted is
    the fraction of the year that contributes nothing to the artifact
    null."""
    n = len(df)
    if hosp_col not in df.columns:
        return {"year": year_label, "n_discharges": n,
                "error": f"'{hosp_col}' not in checkpoint - artifact null cannot run."}

    missing_hosp = int(df[hosp_col].isna().sum())
    missing_share = missing_hosp / n if n else float("nan")

    non_missing = df[~df[hosp_col].isna()]
    group_sizes = non_missing.groupby(hosp_col).size()

    n_hospitals = int(group_sizes.shape[0])
    n_singleton_hospitals = int((group_sizes == 1).sum())
    singleton_share_of_hospitals = n_singleton_hospitals / n_hospitals if n_hospitals else float("nan")

    discharges_in_singleton_hospitals = int(group_sizes[group_sizes == 1].sum())
    singleton_share_of_discharges = (discharges_in_singleton_hospitals / len(non_missing)
                                      if len(non_missing) else float("nan"))
    not_permuted = missing_hosp + discharges_in_singleton_hospitals

    weights = non_missing[weight_col].to_numpy(dtype=np.float64)
    total_weight = float(weights.sum())

    return {
        "year": year_label,
        "n_discharges": n,
        "n_missing_hosp_nis": missing_hosp,
        "missing_hosp_nis_share": missing_share,
        "n_distinct_hospitals": n_hospitals,
        "median_discharges_per_hospital": float(group_sizes.median()) if n_hospitals else float("nan"),
        "p10_discharges_per_hospital": float(group_sizes.quantile(0.10)) if n_hospitals else float("nan"),
        "p90_discharges_per_hospital": float(group_sizes.quantile(0.90)) if n_hospitals else float("nan"),
        "n_singleton_hospitals": n_singleton_hospitals,
        "singleton_share_of_hospitals": singleton_share_of_hospitals,
        "discharges_in_singleton_hospitals": discharges_in_singleton_hospitals,
        "singleton_share_of_discharges": singleton_share_of_discharges,
        "n_discharges_not_permuted": not_permuted,
        "share_of_discharges_not_permuted": not_permuted / n if n else float("nan"),
        "total_discharge_weight": total_weight,
    }


def compare_years_structurally(summaries: List[Dict], focus_year: str = FOCUS_YEAR,
                                threshold: float = OUTLIER_REL_DIFF) -> Dict:
    """Flags whether the focus year (2021) is an outlier among the three
    years on each structural measure, using the simple, transparent rule
    'more than 50% relatively different from the average of the other two
    years.' This threshold is arbitrary and should be treated as a
    starting point for judgment, not a formal statistical test - there is
    no p-value here, only a screening heuristic to decide what deserves a
    closer look.

    When the other years average exactly 0 (e.g. no missing HOSP_NIS at
    all) a relative difference is undefined; any nonzero focus-year value
    is then flagged (relative_difference=inf) rather than silently
    skipped, since "2021 has missing hospital IDs and the others have
    none" is exactly the kind of difference this screen exists to catch."""
    by_year = {s["year"]: s for s in summaries}
    if focus_year not in by_year:
        return {"error": f"No {focus_year} summary provided."}

    other_years = [y for y in by_year if y != focus_year]
    flags = {}
    numeric_fields = [
        "missing_hosp_nis_share", "n_distinct_hospitals",
        "median_discharges_per_hospital", "singleton_share_of_hospitals",
        "singleton_share_of_discharges", "share_of_discharges_not_permuted",
    ]
    for field in numeric_fields:
        v_focus = by_year[focus_year].get(field)
        others = [by_year[y].get(field) for y in other_years
                  if by_year[y].get(field) is not None and by_year[y].get(field) == by_year[y].get(field)]
        if v_focus is None or v_focus != v_focus or not others:
            continue
        other_mean = float(np.mean(others))
        if other_mean:
            rel_diff = (v_focus - other_mean) / other_mean
        else:
            rel_diff = 0.0 if v_focus == 0 else float("inf")
        flags[field] = {
            f"{focus_year}_value": v_focus,
            "other_years_mean": other_mean,
            "relative_difference": rel_diff,
            "flagged_as_outlier": bool(abs(rel_diff) > threshold),
        }
    return flags


# ---------------------------------------------------------------------
# PART 2: Higher-precision rerun, all three years, both controls
# ---------------------------------------------------------------------

def _checkpointed(path: Path | None, compute):
    """Load a control's result from `path` if it exists, else compute and
    save it - so a crash partway through a year (each control is many
    hours at n=2000) only loses the control that was running."""
    if path is not None and path.exists():
        return json.loads(path.read_text())
    res = compute()
    if path is not None:
        _dump(res, path)
    return res


def rerun_controls_one_year(df: pd.DataFrame, pairs_map: Dict, sec_cols: List[str],
                             nc_cfg: Dict, n_permutations: int, seed: int,
                             n_jobs: int, ckpt_prefix: Path | None = None) -> Dict:
    """Both controls for one year, with every setting except n_permutations
    taken from the pipeline config (same call shape as main.py's
    stage_negative_controls), plus the duplicate-profile heuristic (which
    has no permutation count to increase, but is re-included so the
    year's full artifact-control picture is in one place). With
    `ckpt_prefix`, each control is checkpointed separately as
    <prefix>_chance.json / <prefix>_artifact.json."""
    def _path(control):
        return None if ckpt_prefix is None else ckpt_prefix.with_name(f"{ckpt_prefix.name}_{control}.json")

    chance_cfg = nc_cfg["chance"]
    chance = _checkpointed(_path("chance"), lambda: run_chance_control(
        df, pairs_map, sec_cols,
        n_permutations=n_permutations,
        n_bootstrap=chance_cfg.get("n_bootstrap", 200),
        strata_cols=tuple(chance_cfg.get("strata_cols", ["AGE", "FEMALE"])),
        age_band_width=int(chance_cfg.get("age_band_width", 10)),
        seed=seed, n_jobs=n_jobs))
    artifact = _checkpointed(_path("artifact"), lambda: hospital_preserving_shuffle(
        df, pairs_map, sec_cols,
        n_permutations=n_permutations, seed=seed, n_jobs=n_jobs))
    observed = detect(df, pairs_map, sec_cols)
    dup_profile = duplicate_profile_heuristic(df, sec_cols, observed.dev)
    return {
        "n_permutations": n_permutations,
        "seed": seed,
        "chance_control": chance,
        "artifact_control": artifact,
        "duplicate_profile_heuristic": dup_profile,
    }


def comparison_table(rerun_by_year: Dict[str, Dict], original_by_year: Dict[str, Dict],
                     alpha: float = 0.05) -> pd.DataFrame:
    """One row per year x control: original (prespecified) vs. rerun
    p-values and direction, plus the per-year three-way outcome from
    h2_criteria.per_year_compliance_verdict for each run. Every year is
    included regardless of what it shows."""
    rows = []
    for year in sorted(rerun_by_year):
        new = rerun_by_year[year]
        old = original_by_year.get(year) or {}
        new_verdict = h2c.per_year_compliance_verdict(new["chance_control"], new["artifact_control"], alpha)
        old_verdict = (h2c.per_year_compliance_verdict(old["chance"], old["artifact"], alpha)
                       if old else {})
        for control, new_res, old_res, p_lo, p_hi in (
            ("chance", new["chance_control"], old.get("chance", {}),
             "p_value_one_sided_lower_than_chance", "p_value_one_sided_higher_than_chance"),
            ("artifact", new["artifact_control"], old.get("artifact", {}),
             "p_value_one_sided_lower_than_artifact_null", "p_value"),
        ):
            rows.append({
                "year": year,
                "control": control,
                "original_n_permutations": old_res.get("n_permutations"),
                "rerun_n_permutations": new_res.get("n_permutations"),
                "observed_weighted_dev_rate": new_res.get("observed_weighted_dev_rate"),
                "original_null_mean": old_res.get("null_mean_weighted_dev_rate"),
                "rerun_null_mean": new_res.get("null_mean_weighted_dev_rate"),
                "original_direction": old_res.get("direction"),
                "rerun_direction": new_res.get("direction"),
                "original_p_lower": old_res.get(p_lo),
                "rerun_p_lower": new_res.get(p_lo),
                "original_p_higher": old_res.get(p_hi),
                "rerun_p_higher": new_res.get(p_hi),
                "original_outcome": old_verdict.get(f"{control}_outcome"),
                "rerun_outcome": new_verdict[f"{control}_outcome"],
                "rerun_min_achievable_p": new_res.get("min_achievable_p"),
            })
    return pd.DataFrame(rows)


def _load_original_controls(tables: Path, years: List[str]) -> Dict[str, Dict]:
    """The prespecified negative_controls stage outputs in `tables`, if they exist."""
    out = {}
    for year in years:
        paths = {"chance": tables / f"negative_control_chance_{year}.json",
                 "artifact": tables / f"negative_control_artifact_shuffle_{year}.json"}
        if all(p.exists() for p in paths.values()):
            out[year] = {k: json.loads(p.read_text()) for k, p in paths.items()}
    return out


def _dump(obj, path: Path) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--n-permutations", type=int, default=2000,
                    help="Applied identically to BOTH controls in ALL years (default 2000).")
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="Override config parallel.n_jobs (still RAM-capped by parallel_utils).")
    ap.add_argument("--out-dir", default=None,
                    help="Default: <outputs_dir>/investigations/2021_artifact")
    ap.add_argument("--originals-dir", default=None,
                    help="Directory holding the prespecified 200-permutation "
                         "negative_control_*.json to compare against "
                         "(default: <outputs_dir>/tables).")
    ap.add_argument("--structure-only", action="store_true",
                    help="Run Part 1 (descriptive) only; skip the permutation rerun.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute years whose rerun checkpoint already exists.")
    args = ap.parse_args(argv)

    # main.py lives at the project root; imported here (not at module
    # level) so the library functions above stay importable without it.
    from main import SEC_COLS, stage_ontology_check
    from src.config import load_config
    from src.data.ingestion import load_year
    from src.dev.detector import pairs_by_phenotype
    from src.logging_utils import dump_runtimes, setup_logging, stage_timer

    cfg = load_config(args.config)
    log = setup_logging(cfg.logs_dir, "h2_investigate_2021_artifact")
    years = cfg.years
    if FOCUS_YEAR not in years:
        raise SystemExit(f"{FOCUS_YEAR} not in config years {years}")

    out_dir = Path(args.out_dir) if args.out_dir else cfg.outputs("investigations") / "2021_artifact"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Config: %s | years: %s | out: %s", args.config, years, out_dir)

    # --- Part 1: structural diagnostic, all three years ---
    with stage_timer("investigate_structure", log):
        cols = ["HOSP_NIS", "DISCWT"]
        summaries = [hospital_structure_summary(load_year(cfg, y, columns=cols), year_label=y)
                     for y in years]
        for s in summaries:
            log.info("[%s] structure: %s", s["year"], json.dumps(s, default=str))
        outlier_flags = compare_years_structurally(summaries)
        flagged = [f for f, v in outlier_flags.items()
                   if isinstance(v, dict) and v.get("flagged_as_outlier")]
        log.info("%s structural outlier flags (>%.0f%% from other-year mean): %s",
                 FOCUS_YEAR, OUTLIER_REL_DIFF * 100, flagged or "none")
        _dump({"summaries": summaries, "outlier_flags": outlier_flags,
               "outlier_threshold_relative": OUTLIER_REL_DIFF},
              out_dir / "hospital_structure_summary.json")
        pd.DataFrame(summaries).to_csv(out_dir / "hospital_structure_summary.csv", index=False)

    if args.structure_only:
        dump_runtimes(out_dir)
        log.info("Structure-only run finished -> %s", out_dir)
        return 0

    # --- Part 2: higher-precision rerun, all three years ---
    n_perm = args.n_permutations
    nc_cfg = cfg["negative_controls"]
    alpha = nc_cfg.get("significance_alpha", 0.05)
    n_jobs = args.n_jobs if args.n_jobs is not None else int(cfg.get("parallel", "n_jobs", default=1))
    originals_dir = Path(args.originals_dir) if args.originals_dir else cfg.outputs("tables")
    originals = _load_original_controls(originals_dir, years)
    missing = [y for y in years if y not in originals]
    if missing:
        log.warning("No original negative_control_*.json for %s in %s - the comparison table "
                    "will leave those years' original_* columns empty (pass --originals-dir).",
                    missing, originals_dir)
    else:
        log.info("Comparing against original controls in %s", originals_dir)

    with stage_timer(f"investigate_rerun_n{n_perm}", log):
        pairs_by_year = stage_ontology_check(cfg, log)
        rerun_by_year: Dict[str, Dict] = {}
        for year in years:
            # per-year checkpoint: a multi-hour run that dies in 2023 keeps
            # 2021/2022; the n is in the filename so different-n results
            # can never be mixed into one comparison
            ckpt = out_dir / f"rerun_n{n_perm}_{year}.json"
            if ckpt.exists() and not args.overwrite:
                log.info("[%s] rerun checkpoint exists, loading (%s)", year, ckpt.name)
                rerun_by_year[year] = json.loads(ckpt.read_text())
                continue
            # per-control checkpoints within the year (see _checkpointed)
            ckpt_prefix = out_dir / f"rerun_n{n_perm}_{year}"
            for control in ("chance", "artifact"):
                part = out_dir / f"{ckpt_prefix.name}_{control}.json"
                if part.exists():
                    if args.overwrite:
                        part.unlink()
                    else:
                        log.info("[%s] %s control checkpoint exists, loading (%s)",
                                 year, control, part.name)
            log.info("[%s] rerunning chance + artifact controls at n_permutations=%d", year, n_perm)
            df = load_year(cfg, year)
            res = rerun_controls_one_year(df, pairs_by_phenotype(pairs_by_year[year]), SEC_COLS,
                                          nc_cfg, n_perm, cfg.seed, n_jobs, ckpt_prefix=ckpt_prefix)
            del df
            _dump(res, ckpt)
            rerun_by_year[year] = res
            c, a = res["chance_control"], res["artifact_control"]
            log.info("[%s] chance: rate_ratio=%s p_lower=%s | artifact: direction=%s "
                     "p_lower=%s p_higher=%s", year, c.get("rate_ratio_observed_over_null"),
                     c.get("p_value_one_sided_lower_than_chance"), a.get("direction"),
                     a.get("p_value_one_sided_lower_than_artifact_null"), a.get("p_value"))

        _dump(rerun_by_year, out_dir / f"rerun_n{n_perm}.json")
        table = comparison_table(rerun_by_year, originals, alpha)
        table.to_csv(out_dir / f"comparison_original_vs_n{n_perm}.csv", index=False)
        log.info("Original vs. rerun, all years:\n%s", table.to_string(index=False))

    dump_runtimes(out_dir)
    log.info("Done. Results written to %s/", out_dir)
    log.info("Read hospital_structure_summary.json first: if %s is NOT flagged as an "
             "outlier there, the higher-precision rerun result should be reported as-is, "
             "whatever it shows, for all three years.", FOCUS_YEAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
