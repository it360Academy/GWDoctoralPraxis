from pathlib import Path

import pytest
import yaml

from src.config import Config, ConfigError, load_config
from src.features.definitions import DISEASE_ORDER, DISEASES, validate_definitions


def test_validate_definitions_passes_on_shipped_definitions():
    validate_definitions()  # should not raise
    assert len(DISEASE_ORDER) == 15
    assert len(set(DISEASE_ORDER)) == 15  # no duplicate names


def test_all_prefixes_are_icd10_shaped_not_icd9():
    for name, spec in DISEASES.items():
        for p in spec["icd10_prefixes"]:
            assert p[0].isalpha() and p[0].isupper(), f"{name} prefix {p} looks non-ICD-10-CM"


def test_load_config_rejects_missing_section(tmp_path: Path):
    bad = {"paths": {}, "years": ["2021", "2022", "2023"],
           "ml_years": {"development": "2021", "validation": "2022", "replication": "2023"}}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump(bad))
    with pytest.raises(ConfigError):
        load_config(p)


def test_load_config_rejects_wrong_year_count(tmp_path: Path):
    bad = {
        "paths": {"data_dir": "d", "logs_dir": "l", "outputs_dir": "o", "nis_root": ".", "files": {}},
        "years": ["2021", "2022"],  # only 2, must be exactly 3
        "ml_years": {"development": "2021", "validation": "2022", "replication": "2023"},
        "ingestion": {}, "ontology": {}, "models": {}, "evaluation": {},
        "negative_controls": {}, "stability_criteria": {},
    }
    p = tmp_path / "bad_years.yaml"
    p.write_text(yaml.dump(bad))
    with pytest.raises(ConfigError):
        load_config(p)


def test_load_config_accepts_the_real_configs():
    root = Path(__file__).resolve().parents[1]
    for name in ("config/config.yaml", "config/config_smoke.yaml"):
        cfg = load_config(root / name)
        assert isinstance(cfg, Config)
        assert len(cfg.years) == 3
        assert set(cfg.ml_years) == {"development", "validation", "replication"}
