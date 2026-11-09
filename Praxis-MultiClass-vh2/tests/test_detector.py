import pandas as pd

from src.dev.detector import detect, pairs_by_phenotype
from src.ontology.excludes1_loader import Excludes1Pair

SEC_COLS = [f"SECDX_{i+1}" for i in range(3)]


def _pair(home="A41", target="P368", phenotype="sepsis"):
    return Excludes1Pair(
        fiscal_year="TEST", home_code=home, home_desc="test", home_level="category",
        target_raw=target, target_kind="prefix", target_lo=target, target_hi=target,
        note_text="test note", direction="phenotype_is_home", phenotype=phenotype,
        source_chapter="1", source_section="A00-A49",
    )


def test_detect_flags_a_true_violation():
    df = pd.DataFrame({
        "DX1_RAW": ["A419", "A419", "I219"],
        "LABEL": ["sepsis", "sepsis", "mi"],
        "SECDX_1": ["P3681", "E119", "I10"],
        "SECDX_2": [None, None, None],
        "SECDX_3": [None, None, None],
    })
    pmap = pairs_by_phenotype([_pair()])
    res = detect(df, pmap, SEC_COLS)
    assert list(res.dev) == [1, 0, 0]
    assert res.n_violations[0] == 1
    assert "A41 Excludes1 P368" in res.first_pair_desc[0]


def test_detect_finds_nothing_when_no_pairs_match():
    df = pd.DataFrame({
        "DX1_RAW": ["A419"], "LABEL": ["sepsis"],
        "SECDX_1": ["E119"], "SECDX_2": [None], "SECDX_3": [None],
    })
    pmap = pairs_by_phenotype([_pair()])
    res = detect(df, pmap, SEC_COLS)
    assert res.dev.sum() == 0


def test_detect_ignores_pairs_outside_the_dx1_cohort():
    """A pair for 'sepsis' must not fire when DX1's LABEL is a different
    phenotype, even if the raw code superficially overlaps."""
    df = pd.DataFrame({
        "DX1_RAW": ["A419"], "LABEL": ["mi"],  # mismatched cohort label
        "SECDX_1": ["P3681"], "SECDX_2": [None], "SECDX_3": [None],
    })
    pmap = pairs_by_phenotype([_pair()])
    res = detect(df, pmap, SEC_COLS)
    assert res.dev.sum() == 0


def test_pairs_by_phenotype_groups_correctly():
    pairs = [_pair(phenotype="sepsis"), _pair(home="I21", target="I269", phenotype="mi")]
    grouped = pairs_by_phenotype(pairs)
    assert set(grouped.keys()) == {"sepsis", "mi"}
    assert len(grouped["sepsis"]) == 1
