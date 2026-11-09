import numpy as np
import pandas as pd

from src.dev import rates


def test_cohort_rate_uses_eligible_discharges_only_as_denominator():
    # 10 discharges, 4 eligible (LABEL set), 1 DEV among them; weights all 1
    df = pd.DataFrame({"LABEL": ["mi"] * 4 + [np.nan] * 6, "DISCWT": 1.0,
                       "HOSP_NIS": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]})
    dev = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    out = rates.cohort_rate(df, dev)
    assert out["n_eligible"] == 4 and out["n_all_discharges"] == 10
    assert out["weighted_dev_rate"] == 0.25          # 1/4, not 1/10
    assert out["n_clusters"] == 2                    # only hospitals 1 and 2 are eligible
    assert "clustered" in out["ci_method"]


def test_cohort_rate_weights_respected():
    df = pd.DataFrame({"LABEL": ["mi", "mi"], "DISCWT": [1.0, 3.0], "HOSP_NIS": [1, 2]})
    out = rates.cohort_rate(df, np.array([1, 0], dtype=np.uint8))
    assert out["weighted_dev_rate"] == 0.25          # 1 / (1+3)


def test_cluster_ci_is_wider_than_iid_when_events_cluster_by_hospital():
    """The reason for the cluster bootstrap: if DEVs concentrate in a few
    hospitals, an i.i.d. bootstrap overstates precision."""
    rng = np.random.default_rng(0)
    n_hosp, per = 200, 100
    hosp = np.repeat(np.arange(n_hosp), per)
    hosp_rate = np.where(rng.random(n_hosp) < 0.1, 0.05, 0.0)        # 10% of hospitals have all the DEVs
    dev = (rng.random(n_hosp * per) < hosp_rate[hosp]).astype(np.uint8)
    w = np.ones(len(dev))
    _, lo_c, hi_c, _ = rates._clustered_weighted_rate_ci(dev, w, pd.Series(hosp), n_resamples=400)
    _, lo_i, hi_i = rates._weighted_rate_ci(dev, w, n_resamples=400)
    assert (hi_c - lo_c) > 2 * (hi_i - lo_i)


def test_cohort_rate_falls_back_when_no_hospital_column():
    df = pd.DataFrame({"LABEL": ["mi"] * 5, "DISCWT": 1.0})
    out = rates.cohort_rate(df, np.array([1, 0, 0, 0, 0], dtype=np.uint8))
    assert "i.i.d." in out["ci_method"] and out["n_clusters"] is None
