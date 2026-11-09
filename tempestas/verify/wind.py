"""Caracterização do vento observado num período (sem previsão).

Fontes, cada uma mantida separada:
- ERA5: rajada a 10 m (fg10) e vento médio, 0,25°, horário;
- ERA5-Land: vento médio a 10 m, 0,1°, horário;
- INMET: velocidade horária e rajada máxima por estação;
- METAR (aeroportos): vento médio e rajada.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from tempestas.config import Config
from tempestas.download.era5 import target_file

log = logging.getLogger(__name__)

_TIME_NAMES = ("valid_time", "time")


@dataclass
class WindResult:
    start: date
    end: date
    grids: dict[str, xr.DataArray] = field(default_factory=dict)   # máximo no período (m/s)
    stations: pd.DataFrame | None = None                            # máximos por estação/aeroporto
    skipped: dict[str, str] = field(default_factory=dict)


def _open_hourly(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path)
    for t in _TIME_NAMES:
        if t in ds.dims and t != "time":
            ds = ds.rename({t: "time"})
    drop = [v for v in ("number", "expver") if v in ds.coords or v in ds.variables]
    return ds.drop_vars(drop, errors="ignore")


def _standardize_hourly(da: xr.DataArray, cfg: Config) -> xr.DataArray:
    """Coordenadas padronizadas (lat/lon crescentes, −180..180), tempo horário preservado."""
    times = da["time"].values
    lat = "latitude" if "latitude" in da.coords else "lat"
    lon = "longitude" if "longitude" in da.coords else "lon"
    da = da.rename({lat: "lat", lon: "lon"})
    if float(da["lon"].max()) > 180:
        da = da.assign_coords(lon=(((da["lon"] + 180) % 360) - 180))
    da = da.sortby("lon").sortby("lat").assign_coords(time=times)
    d = cfg.domain
    return da.sel(lat=slice(d.lat_min - 0.5, d.lat_max + 0.5), lon=slice(d.lon_min - 0.5, d.lon_max + 0.5))


def era5_period_max(cfg: Config, product: str, start: date, end: date) -> dict[str, xr.DataArray]:
    """Máximos no período: rajada (se houver) e vento médio horário."""
    root = cfg.paths["work_dir"] / "dados"
    parts = []
    d = start
    while d <= end:
        p = target_file(root, product, d)
        if not p.exists():
            raise FileNotFoundError(f"{product}: falta {p}")
        parts.append(_open_hourly(p))
        d += timedelta(days=1)
    ds = xr.concat(parts, dim="time")
    out = {}
    if "u10" in ds and "v10" in ds:
        speed = np.hypot(ds["u10"], ds["v10"])
        out[f"{product} vento médio máx."] = _standardize_hourly(speed, cfg).max("time").load()
    if "fg10" in ds:
        out[f"{product} rajada máx."] = _standardize_hourly(ds["fg10"], cfg).max("time").load()
    for k in out:
        out[k].attrs["units"] = "m/s"
    return out


def stations_period_max(cfg: Config, start: date, end: date) -> pd.DataFrame:
    """Vento médio e rajada máximos por estação INMET e aeroporto (METAR)."""
    t0 = pd.Timestamp(start, tz="UTC")
    t1 = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    frames = []
    data = cfg.paths["work_dir"] / "dados"

    for year in range(start.year, end.year + 1):
        f = data / "INMET" / f"INMET_RS_{year}.csv"
        if f.exists():
            df = pd.read_csv(f)
            if not {"wind_ms", "gust_ms"} <= set(df.columns):
                continue
            df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
            df = df[(df.time_utc > t0) & (df.time_utc <= t1)]
            g = df.groupby(["codigo", "estacao", "lat", "lon"])
            agg = g.agg(vento_max_ms=("wind_ms", "max"), rajada_max_ms=("gust_ms", "max")).reset_index()
            idx = df.loc[df["gust_ms"].notna()].groupby("codigo")["gust_ms"].idxmax()
            when = df.loc[idx, ["codigo", "time_utc"]].rename(columns={"time_utc": "quando_rajada_utc"})
            agg = agg.merge(when, on="codigo", how="left")
            agg.insert(0, "rede", "INMET")
            frames.append(agg)

    months = pd.period_range(start, end, freq="M")
    metar = [data / "METAR" / f"METAR_RS_{p.year}{p.month:02d}.csv" for p in months]
    metar = [f for f in metar if f.exists()]
    if metar:
        df = pd.concat([pd.read_csv(f) for f in metar], ignore_index=True)
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
        df = df[(df.time_utc >= t0) & (df.time_utc < t1)]
        g = df.groupby(["icao", "nome"])
        agg = g.agg(vento_max_ms=("wind_ms", "max"), rajada_max_ms=("gust_ms", "max")).reset_index()
        agg = agg.rename(columns={"icao": "codigo", "nome": "estacao"})
        agg.insert(0, "rede", "METAR")
        frames.append(agg)

    if not frames:
        raise FileNotFoundError("sem dados de vento em estações (INMET/METAR) para o período")
    return pd.concat(frames, ignore_index=True).sort_values("rajada_max_ms", ascending=False)


def characterize_wind(cfg: Config, start: date, end: date) -> WindResult:
    res = WindResult(start, end)
    for product in ("ERA5", "ERA5-Land"):
        try:
            res.grids.update(era5_period_max(cfg, product, start, end))
        except (FileNotFoundError, OSError, KeyError) as exc:
            res.skipped[product] = str(exc)
            log.warning("vento %s ignorado: %s", product, exc)
    try:
        res.stations = stations_period_max(cfg, start, end)
    except FileNotFoundError as exc:
        res.skipped["estações (vento)"] = str(exc)
    return res
