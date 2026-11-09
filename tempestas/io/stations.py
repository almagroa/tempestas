"""Chuva observada em estações (referência pontual).

Por enquanto: INMET automáticas. Lê os CSV baixados por ``tempestas baixar``
(``<work_dir>/dados/INMET/INMET_RS_AAAA.csv``) e, para anos anteriores, os zips
anuais do portal guardados em ``paths.inmet_root`` (somente leitura).
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from tempestas.config import Config
from tempestas.download.inmet import daily_precip, read_zip

log = logging.getLogger(__name__)


def _inmet_year(cfg: Config, year: int) -> pd.DataFrame:
    csv = cfg.paths["work_dir"] / "dados" / "INMET" / f"INMET_RS_{year}.csv"
    if csv.exists():
        df = pd.read_csv(csv, parse_dates=["time_utc"])
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
        return df
    z = cfg.paths["inmet_root"] / f"{year}.zip"
    if z.exists():
        return read_zip(z, "RS")
    return pd.DataFrame()


def inmet_period_totals(cfg: Config, start: date, end: date) -> pd.DataFrame:
    """Total de chuva por estação no período (dia UTC, como IMERG e modelos).

    Só entram estações com todos os dias completos no período.
    """
    frames = [_inmet_year(cfg, y) for y in range(start.year, end.year + 1)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise FileNotFoundError(f"sem dados INMET para {start}–{end}")
    hourly = pd.concat(frames, ignore_index=True)
    daily = daily_precip(hourly, day_start_hour_utc=0)
    daily = daily[(daily["dia"] >= pd.Timestamp(start)) & (daily["dia"] <= pd.Timestamp(end))]
    n_days = (end - start).days + 1
    g = daily.groupby(["codigo", "estacao", "lat", "lon"])
    tot = g["precip_mm"].agg(obs_mm="sum", dias_validos="count").reset_index()
    tot = tot[tot["dias_validos"] == n_days].drop(columns="dias_validos")
    if tot.empty:
        raise ValueError(f"nenhuma estação INMET com dados completos em {start}–{end}")
    return tot
