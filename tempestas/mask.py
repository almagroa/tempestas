"""Máscaras da área de análise (RS inteiro e regiões hidrográficas) sobre a grade comum.

O contorno padrão é ``tempestas/data/rs_regioes.geojson``: as três regiões
hidrográficas do RS (Guaíba, Litoral, Uruguai), cuja união é o estado.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr
from shapely import contains_xy
from shapely.geometry import shape
from shapely.ops import unary_union

DEFAULT_REGIONS = Path(__file__).resolve().parent / "data" / "rs_regioes.geojson"
ALL = "RS"


def load_regions(path: str | Path | None = None, name_field: str = "regiao") -> dict:
    """Lê o GeoJSON e devolve {nome: geometria}, incluindo a união como 'RS'."""
    path = Path(path) if path else DEFAULT_REGIONS
    with open(path, encoding="utf-8") as fh:
        gj = json.load(fh)
    regions = {f["properties"][name_field]: shape(f["geometry"]) for f in gj["features"]}
    regions = {ALL: unary_union(list(regions.values())), **regions}
    return regions


def region_masks(lats: np.ndarray, lons: np.ndarray, path: str | Path | None = None) -> dict[str, xr.DataArray]:
    """{região: máscara booleana (lat, lon)}; célula entra se o centro cai no polígono."""
    lon2d, lat2d = np.meshgrid(lons, lats)
    coords = {"lat": lats.astype(float), "lon": lons.astype(float)}
    return {
        name: xr.DataArray(contains_xy(geom, lon2d, lat2d), coords=coords, dims=("lat", "lon"))
        for name, geom in load_regions(path).items()
    }
