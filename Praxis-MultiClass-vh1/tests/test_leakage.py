from src.leakage.controls import leaky_codes, which_phenotypes


def test_phenotype_defining_code_is_masked():
    # I214 is an MI-defining code (I21 prefix)
    masked = leaky_codes(["I214", "E119", "I10"])
    assert "I214" in masked
    assert "E119" not in masked
    assert "I10" not in masked


def test_which_phenotypes_reports_the_match():
    hits = which_phenotypes("I214")
    assert "mi" in hits


def test_definitionally_adjacent_code_not_conflated():
    # C440 (non-melanoma skin cancer) must not be classified as melanoma
    # (C43) - a real historical bug risk if prefix matching were sloppy.
    hits = which_phenotypes("C440")
    assert "melanoma" not in hits


def test_non_phenotype_code_not_masked():
    assert leaky_codes(["Z0000", "M5450"]) == set()
