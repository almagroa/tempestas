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


def download_day(product: str, day: date, out_root: str | Path, overwrite: bool = False) -> Path | None:
    import cdsapi

    out = target_file(Path(out_root), product, day)
    if out.exists() and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    spec = DATASETS[product]
    request = {
        **spec["extra"],
        "variable": spec["variables"],
        "year": [f"{day.year}"],
        "month": [f"{day.month:02d}"],
        "day": [f"{day.day:02d}"],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": RS_AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    tmp = out.with_suffix(".part")
    cdsapi.Client(quiet=True).retrieve(spec["name"], request, str(tmp))
    tmp.replace(out)
    log.info("baixado %s", out)
    return out


def update(out_root: str | Path, products=("ERA5", "ERA5-Land"), days_back: int = 15, lag_days: int = 5) -> list[Path]:
    """Garante os últimos ``days_back`` dias (respeitando o atraso do ERA5T)."""
    end = date.today() - timedelta(days=lag_days)
    done = []
    for product in products:
        for k in range(days_back):
            day = end - timedelta(days=k)
            try:
                p = download_day(product, day, out_root)
                if p:
                    done.append(p)
            except Exception as exc:  # o CDS falha de forma intermitente; tenta de novo na próxima execução
                log.warning("%s %s: %s", product, day, exc)
    return done
