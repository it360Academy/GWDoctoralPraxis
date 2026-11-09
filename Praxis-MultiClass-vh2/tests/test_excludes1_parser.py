from src.ontology.excludes1_parser import _extract_targets


def test_single_code_in_parens():
    targets, status = _extract_targets("lobar pneumonia, unspecified organism (J18.1)")
    assert status == "ok"
    assert targets == ["J18.1"]


def test_single_range_in_parens():
    targets, status = _extract_targets("chronic obstructive bronchitis (J44.-)")
    assert status == "ok"
    assert targets == ["J44.-"]


def test_comma_separated_list_in_parens_yields_multiple_targets():
    """Regression test: an earlier version of this parser only matched a
    single code or a single range and routed every comma-separated-list
    note (a real, common CMS format, e.g. multi-site cancer exclusions)
    to manual review, silently losing ~4% of all real Excludes1 notes."""
    targets, status = _extract_targets(
        "malignant neoplasm of skin of genital organs (C51-C52, C60.-, C63.2)")
    assert status == "ok"
    assert targets == ["C51-C52", "C60.-", "C63.2"]


def test_two_code_comma_list():
    targets, status = _extract_targets("current child sexual abuse (T74.22, T76.22)")
    assert status == "ok"
    assert targets == ["T74.22", "T76.22"]


def test_bare_trailing_code_no_parens():
    targets, status = _extract_targets("carcinoma in situ of nose NOS D09.8")
    assert status == "ok"
    assert targets == ["D09.8"]


def test_u_prefixed_codes_parse():
    """Regression test: the code-token regex used to exclude the letter
    'U' entirely, which silently broke extraction of COVID-19 codes
    (U07.1 etc.) - a real, permanent part of ICD-10-CM."""
    targets, status = _extract_targets("COVID-19 (U07.1)")
    assert status == "ok"
    assert targets == ["U07.1"]


def test_genuinely_unparseable_note_is_flagged_not_guessed():
    targets, status = _extract_targets("certain localized infections - see body system-related chapters")
    assert status == "no_code_found"
    assert targets == []


def test_list_header_with_no_trailing_code_is_flagged():
    targets, status = _extract_targets("disorders of:")
    assert status == "no_code_found"
    assert targets == []
