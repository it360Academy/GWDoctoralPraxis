from src.features.definitions import DISEASE_ORDER, DISEASES, validate_definitions


def test_fifteen_diseases():
    validate_definitions()
    assert len(DISEASES) == 15
    assert len(DISEASE_ORDER) == 15


def test_no_duplicate_prefixes_across_diseases():
    """A raw prefix used as a defining code for one disease should not be
    silently reused as a defining prefix for a different disease - that
    would make the cohort label ambiguous by construction."""
    seen = {}
    for name, spec in DISEASES.items():
        for p in spec["icd10_prefixes"]:
            assert p not in seen, f"prefix {p!r} used by both {seen.get(p)} and {name}"
            seen[p] = name
