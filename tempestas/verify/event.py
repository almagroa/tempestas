"""Verificação de um evento (período) de chuva: todos os modelos × todos os observados."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import xarray as xr

from tempestas.config import Config
from tempestas.io.forecasts import forecast_for_period
from tempestas.io.observed import observed_total
from tempestas.io.standardize import regrid, target_grid
from tempestas.mask import region_masks
from tempestas.io.stations import NETWORKS, period_totals, station_inventory
from tempestas.verify.metrics import point_table, score_table

log = logging.getLogger(__name__)


@dataclass
class EventResult:
    start: date
    end: date
    observed: dict[str, xr.DataArray]
    forecasts: dict[str, xr.DataArray]
    masks: dict[str, xr.DataArray]
    scores: pd.DataFrame
    skipped: dict[str, str] = field(default_factory=dict)
    stations: pd.DataFrame | None = None
    station_scores: pd.DataFrame | None = None
    inventory: pd.DataFrame | None = None


def verify_event(
    cfg: Config,
    start: date,
    end: date,
    models: list[str] | None = None,
    observed_products: list[str] | None = None,
) -> EventResult:
    res = cfg.verification["grid"]
    lats, lons = target_grid(cfg.domain, res)
    masks = region_masks(lats, lons, cfg.raw.get("regions_vector"))

    observed, skipped = {}, {}
    for prod in observed_products or list(cfg.observed):
        try:
            observed[prod] = regrid(observed_total(cfg, prod, start, end), lats, lons)
        except (FileNotFoundError, ValueError) as exc:
            skipped[prod] = str(exc)
            log.warning("observado %s ignorado: %s", prod, exc)
    if not observed:
        raise RuntimeError(f"nenhum produto observado disponível: {skipped}")

    forecasts = {}
    for model in models or list(cfg.models):
        try:
            forecasts[model] = regrid(forecast_for_period(cfg, model, start, end), lats, lons)
        except FileNotFoundError as exc:
            skipped[model] = str(exc)
            log.warning("modelo %s ignorado: %s", model, exc)

    thresholds = cfg.verification["thresholds_mm"]
    tables = []
    for region, mask in masks.items():
        t = score_table(forecasts, observed, thresholds, mask)
        t.insert(0, "region", region)
        tables.append(t)
    scores = pd.concat(tables, ignore_index=True)

    station_frames, score_frames = [], []
    fields = {**forecasts, **{f"{k} (grade)": v for k, v in observed.items()}}
    for network in NETWORKS:
        try:
            st = period_totals(cfg, network, start, end)
        except (FileNotFoundError, ValueError) as exc:
            skipped[network] = str(exc)
            log.warning("estações %s ignoradas: %s", network, exc)
            continue
        sc = point_table(st, fields, thresholds)
        sc.insert(0, "rede", network)
        station_frames.append(st)
        score_frames.append(sc)
    stations = pd.concat(station_frames, ignore_index=True) if station_frames else None
    station_scores = pd.concat(score_frames, ignore_index=True) if score_frames else None
    try:
        inventory = station_inventory(cfg, start, end, stations)
    except Exception as exc:  # inventário é só para o mapa; não pode derrubar a verificação
        log.warning("inventário de estações: %s", exc)
        inventory = None
    return EventResult(start, end, observed, forecasts, masks, scores, skipped, stations, station_scores, inventory)
