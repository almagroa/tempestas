"""Leitura dos produtos observados. Cada produto é mantido separado (sem fusão)."""
from __future__ import annotations

from datetime import date

import pandas as pd
import xarray as xr

from tempestas.config import Config
from tempestas.io.standardize import standardize


def _months(start: date, end: date) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def open_imerg(cfg: Config, start: date, end: date, pad: float = 0.5) -> xr.DataArray:
    """IMERG diário (mm/dia) entre start e end, recortado no domínio."""
    spec = cfg.observed["IMERG"]
    root = cfg.paths["imerg_root"]
    parts = []
    for ym in _months(start, end):
        path = root / spec["file"].format(yyyymm=ym)
        if not path.exists():
            raise FileNotFoundError(path)
        with xr.open_dataset(path) as ds:
            da = standardize(ds[spec["var"]], units="mm/day", domain=cfg.domain, pad=pad).load()
        parts.append(da)
    da = xr.concat(parts, dim="time").sel(time=slice(pd.Timestamp(start), pd.Timestamp(end)))
    expected = pd.date_range(start, end, freq="D")
    missing = expected.difference(pd.DatetimeIndex(da["time"].values))
    da.attrs["missing_days"] = [d.strftime("%Y-%m-%d") for d in missing]
    da.name = "IMERG"
    return da


OBSERVED_READERS = {"IMERG": open_imerg}


def observed_total(cfg: Config, product: str, start: date, end: date) -> xr.DataArray:
    da = OBSERVED_READERS[product](cfg, start, end)
    if da.attrs.get("missing_days"):
        raise ValueError(f"{product}: faltam dias {da.attrs['missing_days']}")
    tot = da.sum("time")
    tot.name = product
    tot.attrs = {"units": "mm", "period_start": str(start), "period_end": str(end)}
    return tot
