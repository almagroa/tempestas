"""Máscara da área de análise (RS ou sub-regiões) sobre a grade comum."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def polygon_mask(lats: np.ndarray, lons: np.ndarray, vector_path: str | Path | None) -> xr.DataArray:
    """True nas células cujo centro cai dentro do polígono.

    Sem arquivo vetorial, usa o retângulo inteiro do domínio (todas as células).
    Aceita shapefile, GeoPackage ou GeoJSON (lidos com geopandas).
    """
    da = xr.DataArray(
        np.ones((lats.size, lons.size), dtype=bool),
        coords={"lat": lats.astype(float), "lon": lons.astype(float)},
        dims=("lat", "lon"),
    )
    if not vector_path:
        return da
    import geopandas as gpd
    from shapely import contains_xy

    gdf = gpd.read_file(vector_path).to_crs(4326)
    geom = gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union
    lon2d, lat2d = np.meshgrid(lons, lats)
    da.values = contains_xy(geom, lon2d, lat2d)
    return da
