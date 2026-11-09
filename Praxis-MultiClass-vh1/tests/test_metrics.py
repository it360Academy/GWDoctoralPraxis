import numpy as np

from src.analysis import evaluation as ev
from src.analysis.models import CLASSES


def test_perfect_predictions_score_well():
    n = 60
    y = np.array([CLASSES[i % len(CLASSES)] for i in range(n)], dtype=object)
    proba = np.zeros((n, len(CLASSES)))
    for i, label in enumerate(y):
        proba[i, CLASSES.index(label)] = 1.0
    agg = ev.aggregate_metrics(y, y.copy(), proba)
    assert agg["macro_f1"] == 1.0
    assert agg["mcc"] > 0.99
    assert agg["balanced_accuracy"] == 1.0


def test_confusion_matrix_shape():
    n = 30
    y = np.array([CLASSES[i % len(CLASSES)] for i in range(n)], dtype=object)
    pred = np.roll(y, 1)
    proba = np.full((n, len(CLASSES)), 1.0 / len(CLASSES))
    cm = ev.confusion_matrix_df(y, pred)
    assert cm.shape == (len(CLASSES), len(CLASSES))
    assert cm.to_numpy().sum() == n


def test_per_class_table_has_all_phenotypes():
    n = 45
    y = np.array([CLASSES[i % len(CLASSES)] for i in range(n)], dtype=object)
    proba = np.full((n, len(CLASSES)), 1.0 / len(CLASSES))
    table = ev.per_class_table(y, y.copy(), proba)
    assert set(table["phenotype"]) == set(CLASSES)
    assert len(table) == 15
