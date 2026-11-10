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
    metar_weather: pd.DataFrame | None = None                      # horas com chuva/trovoada por aeroporto


def _open_hourly(path: Path) -> xr.Dataset:
    from tempestas.download.era5 import open_cds

    ds = open_cds(path)
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
    gust = next((v for v in ("fg10", "i10fg", "gust") if v in ds), None)
    if gust:
        out[f"{product} rajada máx."] = _standardize_hourly(ds[gust], cfg).max("time").load()
    for k in out:
        out[k].attrs["units"] = "m/s"
    return out


WIND_QC_TEXT = (
    "Controle de qualidade das estações: descartados vento médio acima de 40 m/s, rajada acima de 60 m/s, "
    "rajada menor que o vento médio da mesma hora e picos isolados de vento médio (acima de 20 m/s com as "
    "horas vizinhas abaixo da metade). Rajadas convectivas isoladas são mantidas."
)
ERA5_NOTE = (
    "A rajada do ERA5 é uma estimativa da reanálise numa grade de cerca de 30 km e suaviza rajadas convectivas "
    "locais; por isso fica tipicamente bem abaixo das rajadas medidas nas estações. Os mapas mostram o padrão "
    "espacial, e os valores pontuais devem ser lidos nas estações."
)


def qc_wind(df: pd.DataFrame, wind: str = "wind_ms", gust: str = "gust_ms", key: str = "codigo") -> tuple[pd.DataFrame, int]:
    """Controle de qualidade simples do vento horário. Devolve (dados, n de valores descartados)."""
    d = df.sort_values([key, "time_utc"]).copy()
    w, g = d[wind].astype(float), d[gust].astype(float)
    bad_w = (w < 0) | (w > 40)
    bad_g = (g < 0) | (g > 60) | (g.notna() & w.notna() & (g < w))
    prev = d.groupby(key)[wind].shift(1)
    nxt = d.groupby(key)[wind].shift(-1)
    neigh = pd.concat([prev, nxt], axis=1).max(axis=1)
    bad_w |= (w > 20) & neigh.notna() & (neigh < w / 2)
    n = int(bad_w.sum() + bad_g.sum())
    d.loc[bad_w, wind] = float("nan")
    d.loc[bad_g, gust] = float("nan")
    return d, n


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
            df, n_bad = qc_wind(df)
            if n_bad:
                log.info("vento INMET: %d valores descartados no controle de qualidade", n_bad)
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
        df, _ = qc_wind(df, key="icao")
        g = df.groupby(["icao", "nome"])
        agg = g.agg(vento_max_ms=("wind_ms", "max"), rajada_max_ms=("gust_ms", "max")).reset_index()
        agg = agg.rename(columns={"icao": "codigo", "nome": "estacao"})
        agg.insert(0, "rede", "METAR")
        frames.append(agg)

    if not frames:
        raise FileNotFoundError("sem dados de vento em estações (INMET/METAR) para o período")
    return pd.concat(frames, ignore_index=True).sort_values("rajada_max_ms", ascending=False)


def metar_weather(cfg: Config, start: date, end: date) -> pd.DataFrame:
    from tempestas.download.metar import weather_hours

    data = cfg.paths["work_dir"] / "dados" / "METAR"
    files = [data / f"METAR_RS_{p.year}{p.month:02d}.csv" for p in pd.period_range(start, end, freq="M")]
    files = [f for f in files if f.exists()]
    if not files:
        raise FileNotFoundError("sem METAR no período")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    t0 = pd.Timestamp(start, tz="UTC")
    t1 = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return weather_hours(df[(df.time_utc >= t0) & (df.time_utc < t1)])


def characterize_wind(cfg: Config, start: date, end: date) -> WindResult:
    res = WindResult(start, end)
    try:
        res.metar_weather = metar_weather(cfg, start, end)
    except (FileNotFoundError, KeyError) as exc:
        log.warning("METAR (tempo presente) ignorado: %s", exc)
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
