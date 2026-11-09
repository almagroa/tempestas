"""Leitura das rodadas de previsão arquivadas (somente leitura).

Estrutura esperada: <forecast_root>/<pasta do modelo>/<YYYY-MM-DD>/<arquivo>.
A maioria dos arquivos traz a chuva ACUMULADA desde o início da rodada
(``accumulated: true`` na configuração); ela é convertida em totais diários.
O primeiro passo é o próprio dia da rodada (antecedência 1 dia).
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import xarray as xr

from tempestas.config import Config
from tempestas.io.standardize import standardize


def deaccumulate(da: xr.DataArray) -> xr.DataArray:
    """Converte acumulado desde o início da rodada em totais diários."""
    daily = da.diff("time", label="upper")
    daily = xr.concat([da.isel(time=[0]), daily], dim="time").clip(min=0)
    daily.attrs = da.attrs
    return daily


def run_path(cfg: Config, model: str, init: date) -> Path:
    m = cfg.models[model]
    fname = m["file"].format(date=init.isoformat())
    return cfg.paths["forecast_root"] / m["folder"] / init.isoformat() / fname


def available_runs(cfg: Config, model: str, start: date, end: date) -> list[date]:
    """Datas de rodada existentes entre start e end (inclusive)."""
    out = []
    d = start
    while d <= end:
        if run_path(cfg, model, d).exists():
            out.append(d)
        d += timedelta(days=1)
    return out


def open_run(cfg: Config, model: str, init: date, pad: float = 0.5) -> xr.DataArray:
    """Abre uma rodada, padronizada e recortada no domínio.

    Devolve DataArray (time, lat, lon) com coordenadas extras ``init`` e
    ``lead`` (dias; 1 = dia da rodada).
    """
    m = cfg.models[model]
    path = run_path(cfg, model, init)
    with xr.open_dataset(path) as ds:
        da = standardize(ds[m["var"]], units=m.get("units"), domain=cfg.domain, pad=pad).load()
    if m.get("accumulated", False):
        da = deaccumulate(da)
    leads = (pd.DatetimeIndex(da["time"].values) - pd.Timestamp(init)).days + 1
    da = da.assign_coords(lead=("time", leads.astype(int)), init=pd.Timestamp(init))
    da.name = model
    return da


def forecast_for_period(
    cfg: Config,
    model: str,
    start: date,
    end: date,
    max_lead: int | None = None,
) -> xr.DataArray:
    """Total previsto para o período [start, end] por cada rodada disponível.

    Só entram rodadas que cobrem o período inteiro. Resultado: (init, lat, lon),
    com coordenada ``lead_start`` = antecedência (dias) do primeiro dia do período.
    """
    n_days = (end - start).days + 1
    horizon = max_lead or 16
    first_init = start - timedelta(days=horizon - 1)
    totals = []
    for init in available_runs(cfg, model, first_init, start):
        da = open_run(cfg, model, init)
        sel = da.sel(time=slice(pd.Timestamp(start), pd.Timestamp(end)))
        if sel.sizes["time"] < n_days:
            continue  # rodada não alcança o fim do período
        tot = sel.sum("time")
        tot = tot.expand_dims(init=[pd.Timestamp(init)])
        tot = tot.assign_coords(lead_start=("init", [(start - init).days + 1]))
        totals.append(tot.drop_vars("lead", errors="ignore"))
    if not totals:
        raise FileNotFoundError(f"nenhuma rodada de {model} cobre {start}–{end}")
    out = xr.concat(totals, dim="init")
    out.name = model
    out.attrs = {"units": "mm", "period_start": str(start), "period_end": str(end)}
    return out
