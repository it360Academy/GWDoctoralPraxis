import pandas as pd

from src.data.ingestion import _derive_label, _phenotype_classifier


def test_derive_label_single_match():
    clf = _phenotype_classifier()
    principal = pd.DataFrame({"DX1": ["I214", "A410", "Z0000"]})
    out = _derive_label(principal, clf)
    labels = out["LABEL"].tolist()
    assert labels[0] == "mi"
    assert labels[1] == "sepsis"
    assert pd.isna(labels[2])
    assert list(out["NO_PHENOTYPE_DX1"]) == [0, 0, 1]
    assert list(out["AMBIGUOUS_DX1"]) == [0, 0, 0]


def test_derive_label_handles_missing_code():
    clf = _phenotype_classifier()
    principal = pd.DataFrame({"DX1": [pd.NA]})
    out = _derive_label(principal, clf)
    assert pd.isna(out["LABEL"].iloc[0])
    assert out["NO_PHENOTYPE_DX1"].iloc[0] == 1
