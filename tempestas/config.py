"""Leitura da configuração do Tempestas."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "tempestas.yaml"


@dataclass(frozen=True)
class Domain:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@dataclass
class Config:
    raw: dict[str, Any]

    @property
    def paths(self) -> dict[str, Path]:
        return {k: Path(v) for k, v in self.raw["paths"].items()}

    @property
    def domain(self) -> Domain:
        return Domain(**self.raw["domain"])

    @property
    def models(self) -> dict[str, dict[str, Any]]:
        return self.raw["models"]

    @property
    def observed(self) -> dict[str, dict[str, Any]]:
        return self.raw["observed"]

    @property
    def verification(self) -> dict[str, Any]:
        return self.raw["verification"]


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if overrides:
        for section, values in overrides.items():
            raw.setdefault(section, {}).update(values)
    return Config(raw)
