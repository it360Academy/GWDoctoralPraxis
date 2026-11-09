"""Configuration loading, validation and path management."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(Exception):
    """Raised when the configuration file is invalid."""


@dataclass
class Config:
    raw: Dict[str, Any]
    project_root: Path
    config_path: Path | None = None
    _resolved: Dict[str, Path] = field(default_factory=dict)

    # ------------------------------------------------------------------ access
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    @property
    def seed(self) -> int:
        return int(self.raw.get("random_seed", 42))

    # ------------------------------------------------------------------- paths
    def _dir(self, key: str) -> Path:
        p = self.project_root / self.raw["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_dir(self) -> Path:
        return self._dir("data_dir")

    @property
    def logs_dir(self) -> Path:
        return self._dir("logs_dir")

    @property
    def outputs_dir(self) -> Path:
        return self._dir("outputs_dir")

    def outputs(self, sub: str) -> Path:
        p = self.outputs_dir / sub
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def nis_root(self) -> Path:
        return Path(os.environ.get("NIS_ROOT", self.raw["paths"]["nis_root"]))

    def nis_file(self, year: str, kind: str) -> Path:
        try:
            fname = self.raw["paths"]["files"][str(year)][kind]
        except KeyError as exc:
            raise ConfigError(f"No {kind} file configured for year {year}") from exc
        return self.nis_root / fname

    @property
    def years(self) -> list[str]:
        """H2 analyzes years independently (2021/2022/2023), not as a
        train/validation/test dict like H1 - see `ml_years` for the
        DEV-risk model's development/validation/replication split."""
        return [str(y) for y in self.raw["years"]]

    @property
    def ml_years(self) -> Dict[str, str]:
        return {k: str(v) for k, v in self.raw["ml_years"].items()}


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    required = ["paths", "years", "ml_years", "ingestion", "ontology",
                "models", "evaluation", "negative_controls", "stability_criteria"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ConfigError(f"Config missing required sections: {missing}")

    if not isinstance(raw["years"], list) or len(raw["years"]) != 3:
        raise ConfigError("years must be a list of exactly 3 NIS calendar years")
    ml_roles = set(raw["ml_years"])
    if ml_roles != {"development", "validation", "replication"}:
        raise ConfigError("ml_years must define exactly development/validation/replication")

    project_root = path.resolve().parent.parent
    return Config(raw=raw, project_root=project_root,
                  config_path=path.resolve())
