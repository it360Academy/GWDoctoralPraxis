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
    def years(self) -> Dict[str, str]:
        return {k: str(v) for k, v in self.raw["years"].items()}


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    required = ["paths", "years", "ingestion", "incidence", "montecarlo",
                "criteria_thresholds"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ConfigError(f"Config missing required sections: {missing}")

    roles = set(raw["years"])
    if roles != {"train", "validation", "test"}:
        raise ConfigError("years must define exactly train/validation/test "
                           "(used here as development/validation/replication)")

    project_root = path.resolve().parent.parent
    return Config(raw=raw, project_root=project_root,
                  config_path=path.resolve())
