import numpy as np
import pandas as pd

from src.dev.rates import overall_rate, phenotype_rates, position_rates


def test_overall_rate_basic():
    df = pd.DataFrame({"DISCWT": [1.0, 1.0, 2.0, 2.0]})
    dev = np.array([0, 1, 0, 1], dtype=np.uint8)
    out = overall_rate(df, dev)
    assert out["n_hospitalizations"] == 4
    assert out["n_dev_raw"] == 2
    assert out["raw_dev_rate"] == 0.5
    # weighted: (1*0 + 1*1 + 2*0 + 2*1) / (1+1+2+2) = 3/6 = 0.5
    assert abs(out["weighted_dev_rate"] - 0.5) < 1e-9


def test_overall_rate_empty():
    df = pd.DataFrame({"DISCWT": []})
    dev = np.array([], dtype=np.uint8)
    out = overall_rate(df, dev)
    assert out["n_hospitalizations"] == 0
    assert out["raw_dev_rate"] != out["raw_dev_rate"]  # NaN


def test_phenotype_rates_groups_by_label():
    df = pd.DataFrame({"LABEL": pd.Categorical(["sepsis", "sepsis", "mi"]),
                        "DISCWT": [1.0, 1.0, 1.0]})
    dev = np.array([1, 0, 0], dtype=np.uint8)
    out = phenotype_rates(df, dev)
    sepsis_row = out[out["phenotype"] == "sepsis"].iloc[0]
    assert sepsis_row["n"] == 2
    assert sepsis_row["n_dev_raw"] == 1


def test_position_rates_dedupes_same_row_same_position():
    hit_log = pd.DataFrame({
        "row": [0, 0, 1], "secondary_position": [1, 1, 2],
    })
    out = position_rates(hit_log)
    # the duplicate (row=0, position=1) should be counted once
    assert out[out["secondary_position"] == 1]["n_hits"].iloc[0] == 1
    assert out[out["secondary_position"] == 2]["n_hits"].iloc[0] == 1


def test_position_rates_empty_hit_log():
    hit_log = pd.DataFrame(columns=["row", "secondary_position"])
    out = position_rates(hit_log)
    assert out.empty
