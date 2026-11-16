"""Análise num ponto (município): leque das previsões × chuva observada por satélite e estações."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from tempestas.config import Config
from tempestas.download.inmet import daily_precip
from tempestas.io.observed import OBSERVED_READERS
from tempestas.io.standardize import regrid, target_grid
from tempestas.io.stations import _READERS
from tempestas.verify.probabilistic import daily_ensemble

log = logging.getLogger(__name__)

PLACES = {  # lat, lon (centro aproximado)
    "Porto Alegre": (-30.03, -51.21),
}


def _km(lat0, lon0, lat, lon):
    dy = (np.asarray(lat) - lat0) * 111.2
    dx = (np.asarray(lon) - lon0) * 111.2 * np.cos(np.radians(lat0))
    return np.hypot(dx, dy)


@dataclass
class PointResult:
    name: str
    lat: float
    lon: float
    radius_km: float
    start: date
    end: date
    members: pd.DataFrame                 # dia × membro (mm/dia), previsões na célula do ponto
    satellite: pd.DataFrame               # dia × produto (mm/dia)
    stations: pd.DataFrame                # dia, rede, codigo, estacao, dist_km, precip_mm (dia UTC, só dias completos)
    skipped: dict[str, str] = field(default_factory=dict)

    def network_daily(self) -> pd.DataFrame:
        """Média diária por rede (estações com o dia completo)."""
        if self.stations.empty:
            return pd.DataFrame()
        return self.stations.groupby(["dia", "rede"])["precip_mm"].mean().unstack("rede")

    def summary(self) -> pd.DataFrame:
        """Por dia: leque das previsões, observados e posição de cada observado."""
        m = self.members
        out = pd.DataFrame({
            "n_prev": m.notna().sum(axis=1), "min": m.min(axis=1), "p10": m.quantile(0.1, axis=1),
            "mediana": m.median(axis=1), "p90": m.quantile(0.9, axis=1), "max": m.max(axis=1),
        })
        obs = pd.concat([self.satellite, self.network_daily()], axis=1)
        for col in obs.columns:
            o = obs[col].reindex(out.index)
            out[col] = o
            pos = np.where(o.isna(), "", np.where(o < out["min"], "abaixo", np.where(o > out["max"], "acima",
                           np.where((o >= out["p10"]) & (o <= out["p90"]), "P10–P90", "leque"))))
            out[f"pos_{col}"] = pos
        return out


def point_analysis(
    cfg: Config, name: str, start: date, end: date, lat: float | None = None, lon: float | None = None,
    radius_km: float = 15.0, models: list[str] | None = None,
) -> PointResult:
    if lat is None or lon is None:
        lat, lon = PLACES[name]
    lats, lons = target_grid(cfg.domain, cfg.verification["grid"])
    skipped = {}

    ens = daily_ensemble(cfg, start, end, lats, lons, models=models)
    cell = ens.sel(lat=lat, lon=lon, method="nearest")
    members = cell.to_pandas().T  # time × member
    members.index = pd.DatetimeIndex(members.index).date
    members = members.dropna(axis=1, how="all")

    sat = {}
    for prod, reader in OBSERVED_READERS.items():
        try:
            da = regrid(reader(cfg, start, end), lats, lons).sel(lat=lat, lon=lon, method="nearest")
            s = da.to_series()
            s.index = pd.DatetimeIndex(s.index).date
            sat[prod] = s
        except FileNotFoundError as exc:
            skipped[prod] = str(exc)
    satellite = pd.DataFrame(sat)

    frames = []
    for network, reader in _READERS.items():
        try:
            hourly = pd.concat([reader(cfg, y) for y in range(start.year, end.year + 1)], ignore_index=True)
        except (FileNotFoundError, ValueError) as exc:
            skipped[network] = str(exc)
            continue
        if hourly.empty:
            skipped[network] = "sem dados"
            continue
        hourly = hourly.dropna(subset=["lat", "lon"])
        hourly["dist_km"] = _km(lat, lon, hourly["lat"].astype(float), hourly["lon"].astype(float))
        near = hourly[hourly["dist_km"] <= radius_km]
        if near.empty:
            skipped[network] = f"nenhuma estação a até {radius_km:g} km"
            continue
        d = daily_precip(near.drop(columns="dist_km"), day_start_hour_utc=0)
        d = d[(d["dia"] >= pd.Timestamp(start)) & (d["dia"] <= pd.Timestamp(end))].dropna(subset=["precip_mm"])
        dist = near.groupby("codigo")["dist_km"].first()
        d["dist_km"] = d["codigo"].map(dist)
        d["dia"] = d["dia"].dt.date
        d.insert(0, "rede", network)
        frames.append(d[["dia", "rede", "codigo", "estacao", "dist_km", "precip_mm"]])
    stations = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["dia", "rede", "codigo", "estacao", "dist_km", "precip_mm"])
    return PointResult(name, lat, lon, radius_km, start, end, members, satellite, stations, skipped)
