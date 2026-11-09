"""Padronização de campos de precipitação vindos de fontes diferentes.

Cada fonte chega com convenções próprias (nomes de coordenadas, longitude 0–360
ou −180–180, latitude crescente ou decrescente, unidades em m ou mm, ordem das
dimensões). Tudo sai daqui como DataArray (time, lat, lon), lat crescente,
lon em −180–180, valores em mm/dia e time como data (00:00) do dia acumulado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from tempestas.config import Domain

_LAT_NAMES = ("lat", "latitude", "y")
_LON_NAMES = ("lon", "longitude", "x")

_UNIT_FACTORS = {
    "mm": 1.0,
    "mm/day": 1.0,
    "kg m**-2": 1.0,
    "kg/m^2": 1.0,
    "kg m-2": 1.0,
    "m": 1000.0,
}


def _find(names, candidates):
    for n in candidates:
        if n in names:
            return n
    raise KeyError(f"coordenada não encontrada entre {candidates}; disponíveis: {list(names)}")


def standardize(
    da: xr.DataArray,
    units: str | None = None,
    domain: Domain | None = None,
    pad: float = 0.0,
) -> xr.DataArray:
    """Devolve o campo padronizado (mm/dia, time/lat/lon, recortado no domínio)."""
    lat = _find(da.coords, _LAT_NAMES)
    lon = _find(da.coords, _LON_NAMES)
    da = da.rename({lat: "lat", lon: "lon"})

    # longitude em −180..180
    if float(da["lon"].max()) > 180:
        da = da.assign_coords(lon=(((da["lon"] + 180) % 360) - 180))
    da = da.sortby("lon").sortby("lat")

    if "time" in da.dims:
        da = da.transpose("time", "lat", "lon", ...)
        # O acumulado diário vem carimbado no fim do dia (23:59:59) ou no início (00:00).
        # Em ambos os casos o dia de referência é a data do carimbo.
        da = da.assign_coords(time=pd.DatetimeIndex(da["time"].values).normalize())
    else:
        da = da.transpose("lat", "lon", ...)

    units = units or da.attrs.get("units", "mm")
    if units not in _UNIT_FACTORS:
        raise ValueError(f"unidade desconhecida: {units!r}")
    factor = _UNIT_FACTORS[units]
    attrs = dict(da.attrs)
    if factor != 1.0:
        da = da * factor
    da = da.clip(min=0).astype("float32")
    attrs["units"] = "mm/day"
    da.attrs = attrs

    if domain is not None:
        da = da.sel(
            lat=slice(domain.lat_min - pad, domain.lat_max + pad),
            lon=slice(domain.lon_min - pad, domain.lon_max + pad),
        )
    return da


def target_grid(domain: Domain, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Centros de célula da grade comum de verificação."""
    lats = np.round(np.arange(domain.lat_min + res / 2, domain.lat_max, res), 4)
    lons = np.round(np.arange(domain.lon_min + res / 2, domain.lon_max, res), 4)
    return lats, lons


def regrid(da: xr.DataArray, lats: np.ndarray, lons: np.ndarray) -> xr.DataArray:
    """Reamostra para a grade comum.

    Fontes mais finas que a grade-alvo (CPT-WRF, CHIRPS-GEFS) são agregadas por
    média dentro de cada célula, o que preserva o volume de chuva. Fontes mais
    grossas (0,25°) são interpoladas bilinearmente.
    """
    res_src = float(np.abs(np.diff(da["lat"].values)).mean())
    res_tgt = float(np.abs(np.diff(lats)).mean())
    if res_src < res_tgt * 0.75:
        half = res_tgt / 2
        lat_bins = np.append(lats - half, lats[-1] + half)
        lon_bins = np.append(lons - half, lons[-1] + half)
        out = (
            da.groupby_bins("lat", lat_bins, labels=lats).mean()
            .groupby_bins("lon", lon_bins, labels=lons).mean()
            .rename({"lat_bins": "lat", "lon_bins": "lon"})
        )
        out = out.assign_coords(lat=lats.astype(float), lon=lons.astype(float))
        dims = [d for d in ("time", "lat", "lon") if d in out.dims]
        out = out.transpose(*dims, ...)
    else:
        out = da.interp(lat=lats, lon=lons, method="linear")
    out.attrs = da.attrs
    return out
