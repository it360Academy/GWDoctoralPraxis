import pandas as pd

from src.config import Config
from src.features import dx_features as dxf


def _tiny_cfg():
    raw = {
        "random_seed": 42,
        "paths": {"nis_root": ".", "data_dir": "/tmp/h1test_data",
                   "logs_dir": "/tmp/h1test_logs", "outputs_dir": "/tmp/h1test_out",
                   "files": {}},
        "years": {"train": "2021", "validation": "2022", "test": "2023"},
        "ingestion": {"chunksize": 1000, "dx_regex": "^I10_DX([0-9]+)$",
                      "max_secondary_dx": 3, "key_candidates": ["KEY_NIS"],
                      "hosp_candidates": ["HOSP_NIS"], "weight_candidates": ["DISCWT"]},
        "features": {"representation": "icd10_category3", "min_category_frequency": 1,
                     "include_demographic_covariates": False, "demographic_cols": [],
                     "admission_cols": []},
        "models": {"algorithms": [], "xgboost_gpu": False, "class_weight": "balanced"},
        "evaluation": {"bootstrap_resamples": 10, "bootstrap_ci": 0.95},
        "h1_criteria": {"per_class": {"auroc_ovr_min": 0.7, "ppv_lift_min": 2.0,
                                       "low_prevalence_threshold": 0.01},
                         "aggregate": {"macro_auroc_ovr_min": 0.7, "macro_mcc_min": 0.3,
                                       "balanced_accuracy_min": 0.3},
                         "require_full_phenotype_coverage": False},
    }
    from pathlib import Path
    return Config(raw=raw, project_root=Path("/tmp"))


def test_vocabulary_masks_leaky_codes_and_respects_min_freq():
    cfg = _tiny_cfg()
    df = pd.DataFrame({
        "LABEL": ["mi", "mi", "sepsis", None],
        "DISCWT": [1.0, 1.0, 1.0, 1.0],
        "SECDX_1": ["E119", "E119", "I214", "Z0000"],  # I214 is MI-defining -> must be masked
        "SECDX_2": ["I10", "I10", "A410", None],
        "SECDX_3": [None, None, None, None],
    })
    vocab = dxf.fit_vocabulary(cfg, df)
    assert "I21" not in vocab.categories  # masked (matches MI definition)
    assert "E11" in vocab.categories
    assert "I10" in vocab.categories
    assert "I214" in vocab.masked_codes


def test_encode_shape_and_cohort_restriction():
    cfg = _tiny_cfg()
    df = pd.DataFrame({
        "LABEL": ["mi", "mi", "sepsis", None],
        "DISCWT": [1.0, 1.0, 1.0, 1.0],
        "SECDX_1": ["E119", "E119", "A410", "Z0000"],
        "SECDX_2": ["I10", "I10", None, None],
        "SECDX_3": [None, None, None, None],
    })
    vocab = dxf.fit_vocabulary(cfg, df)
    X, y, w, mask = dxf.build_xy(cfg, df, vocab)
    assert X.shape[0] == 3  # the 4th row (LABEL=None) is excluded from the cohort
    assert len(y) == 3
    assert set(y) == {"mi", "sepsis"}
