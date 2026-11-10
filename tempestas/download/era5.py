"""Download de vento do ERA5 (rajada) e do ERA5-Land (vento médio a 10 m) via CDS.

Requer o pacote ``cdsapi`` e a chave do Copernicus em ``~/.cdsapirc``:

    url: https://cds.climate.copernicus.eu/api
    key: <sua-chave>

O ERA5-Land não tem rajada; por isso a rajada vem do ERA5 (0,25°) e o vento
médio em alta resolução vem do ERA5-Land (0,1°). Os dois chegam com ~5 dias de
atraso (versão preliminar ERA5T).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Norte, Oeste, Sul, Leste (RS com folga)
RS_AREA = [-26.5, -58.5, -34.5, -49.0]

DATASETS = {
    "ERA5": {
        "name": "reanalysis-era5-single-levels",
        "extra": {"product_type": ["reanalysis"]},
        "variables": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "10m_wind_gust_since_previous_post_processing",
        ],
    },
    "ERA5-Land": {
        "name": "reanalysis-era5-land",
        "extra": {},
        "variables": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
    },
}


def target_file(out_root: Path, product: str, day: date) -> Path:
    return out_root / product / f"{day:%Y-%m}" / f"{product}_vento_RS_{day:%Y%m%d}.nc"


def open_cds(path: str | Path):
    """Abre a resposta do CDS: NetCDF simples ou ZIP com vários .nc (instant + max), unidos num Dataset."""
    import tempfile
    import zipfile

    import xarray as xr

    path = Path(path)
    if not zipfile.is_zipfile(path):
        return xr.open_dataset(path).load()
    parts = []
    with zipfile.ZipFile(path) as zf, tempfile.TemporaryDirectory() as tmp:
        for name in zf.namelist():
            if name.endswith(".nc"):
                zf.extract(name, tmp)
                with xr.open_dataset(Path(tmp) / name) as ds:
                    parts.append(ds.drop_vars(["expver", "number"], errors="ignore").load())
    return xr.merge(parts, compat="override", join="outer")


def download_days(product: str, days: list[date], out_root: str | Path, overwrite: bool = False) -> list[Path]:
    """Baixa vários dias do MESMO mês num único pedido ao CDS e grava um arquivo por dia.

    Um pedido por mês reduz muito o tempo de fila do CDS em comparação com um
    pedido por dia.
    """
    import cdsapi

    out_root = Path(out_root)
    todo = [d for d in days if overwrite or not target_file(out_root, product, d).exists()]
    if not todo:
        return []
    months = {(d.year, d.month) for d in todo}
    if len(months) != 1:
        raise ValueError("download_days espera dias de um único mês")
    year, month = months.pop()
    spec = DATASETS[product]
    request = {
        **spec["extra"],
        "variable": spec["variables"],
        "year": [f"{year}"],
        "month": [f"{month:02d}"],
        "day": sorted({f"{d.day:02d}" for d in todo}),
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": RS_AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    chunk = out_root / product / f"{year}-{month:02d}" / f"_pedido_{todo[0]:%Y%m%d}_{todo[-1]:%Y%m%d}.part"
    chunk.parent.mkdir(parents=True, exist_ok=True)
    if not chunk.exists():  # reaproveita um pedido já baixado
        cdsapi.Client(quiet=True).retrieve(spec["name"], request, str(chunk))

    written = []
    with open_cds(chunk) as ds:
        tname = "valid_time" if "valid_time" in ds.dims else "time"
        stamps = ds[tname].dt.floor("D").values
        for d in todo:
            sel = ds.isel({tname: (stamps == np.datetime64(d.isoformat()))})
            if sel.sizes[tname] == 0:
                log.warning("%s %s: dia ausente na resposta do CDS", product, d)
                continue
            out = target_file(out_root, product, d)
            tmp = out.with_suffix(".tmp")
            sel.load().to_netcdf(tmp)
            tmp.replace(out)
            written.append(out)
            log.info("baixado %s", out)
    chunk.unlink(missing_ok=True)
    return written


def update(out_root: str | Path, products=("ERA5", "ERA5-Land"), days_back: int = 15, lag_days: int = 5) -> list[Path]:
    """Garante os últimos ``days_back`` dias (respeitando o atraso do ERA5T).

    Faz um pedido por produto e por mês. Falhas (fila, indisponibilidade) são
    registradas e o dia é tentado de novo na próxima execução.
    """
    end = date.today() - timedelta(days=lag_days)
    days = [end - timedelta(days=k) for k in range(days_back)][::-1]
    by_month: dict[tuple[int, int], list[date]] = {}
    for d in days:
        by_month.setdefault((d.year, d.month), []).append(d)
    done = []
    for product in products:
        for chunk in by_month.values():
            try:
                done += download_days(product, chunk, out_root)
            except Exception as exc:  # o CDS falha de forma intermitente
                log.warning("%s %s–%s: %s", product, chunk[0], chunk[-1], exc)
    return done
