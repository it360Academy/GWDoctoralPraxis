import math

import numpy as np
import pandas as pd

from src.negative_controls.investigate_2021_artifact import (compare_years_structurally,
                                                               comparison_table,
                                                               hospital_structure_summary)


def _df(hosp):
    return pd.DataFrame({"HOSP_NIS": hosp, "DISCWT": np.ones(len(hosp))})


def test_structure_summary_counts_singletons_and_missing():
    # hospital 1: 3 rows, hospital 2: 1 row (singleton), 1 missing
    s = hospital_structure_summary(_df([1, 1, 1, 2, np.nan]), year_label="2021")
    assert s["n_discharges"] == 5
    assert s["n_missing_hosp_nis"] == 1
    assert s["n_distinct_hospitals"] == 2
    assert s["n_singleton_hospitals"] == 1
    assert s["singleton_share_of_hospitals"] == 0.5
    assert s["discharges_in_singleton_hospitals"] == 1
    assert s["singleton_share_of_discharges"] == 0.25
    # missing + singleton rows are the ones the hospital shuffle leaves in place
    assert s["n_discharges_not_permuted"] == 2
    assert s["share_of_discharges_not_permuted"] == 0.4


def test_structure_summary_without_hosp_column():
    s = hospital_structure_summary(pd.DataFrame({"DISCWT": [1.0]}), year_label="2021")
    assert "error" in s


def test_compare_flags_only_the_outlier_field():
    base = {"missing_hosp_nis_share": 0.0, "n_distinct_hospitals": 100,
            "median_discharges_per_hospital": 50.0, "singleton_share_of_hospitals": 0.02,
            "singleton_share_of_discharges": 0.001, "share_of_discharges_not_permuted": 0.001}
    flags = compare_years_structurally([
        {**base, "year": "2021", "singleton_share_of_hospitals": 0.10},
        {**base, "year": "2022"},
        {**base, "year": "2023"},
    ])
    assert flags["singleton_share_of_hospitals"]["flagged_as_outlier"]
    assert not flags["n_distinct_hospitals"]["flagged_as_outlier"]
    assert flags["missing_hosp_nis_share"]["relative_difference"] == 0.0


def test_compare_flags_nonzero_when_other_years_are_zero():
    base = {"missing_hosp_nis_share": 0.0}
    flags = compare_years_structurally([
        {**base, "year": "2021", "missing_hosp_nis_share": 0.01},
        {**base, "year": "2022"},
        {**base, "year": "2023"},
    ])
    assert math.isinf(flags["missing_hosp_nis_share"]["relative_difference"])
    assert flags["missing_hosp_nis_share"]["flagged_as_outlier"]


def test_compare_requires_focus_year():
    assert "error" in compare_years_structurally([{"year": "2022"}, {"year": "2023"}])


def test_comparison_table_includes_every_year_even_without_originals():
    chance = {"n_permutations": 2000, "p_value_one_sided_lower_than_chance": 0.001,
              "p_value_one_sided_higher_than_chance": 0.999, "min_achievable_p": 1 / 2001,
              "direction": "suppressed_vs_chance"}
    artifact = {"n_permutations": 2000, "p_value_one_sided_lower_than_artifact_null": 0.9,
                "p_value": 0.1, "min_achievable_p": 1 / 2001,
                "direction": "elevated_vs_artifact_null"}
    rerun = {y: {"chance_control": chance, "artifact_control": artifact} for y in ("2021", "2022", "2023")}
    table = comparison_table(rerun, original_by_year={})
    assert sorted(table["year"].unique()) == ["2021", "2022", "2023"]
    assert len(table) == 6
    assert set(table.loc[table.control == "chance", "rerun_outcome"]) == {"significantly_below_null"}
    assert set(table.loc[table.control == "artifact", "rerun_outcome"]) == {"inconclusive"}
    assert table["original_n_permutations"].isna().all()


def test_checkpointed_saves_then_resumes_without_recomputing(tmp_path):
    from src.negative_controls.investigate_2021_artifact import _checkpointed
    path = tmp_path / "rerun_n2000_2021_chance.json"
    calls = []

    def compute():
        calls.append(1)
        return {"p_value": 0.5}

    assert _checkpointed(path, compute) == {"p_value": 0.5}
    assert path.exists()
    assert _checkpointed(path, compute) == {"p_value": 0.5}   # loaded, not recomputed
    assert len(calls) == 1
