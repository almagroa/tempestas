"""Relatório mensal: chuva observada no mês, eventos detectados, desempenho diário dos modelos e vento."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
import xarray as xr

from tempestas.config import Config
from tempestas.io.observed import OBSERVED_READERS
from tempestas.io.standardize import regrid, target_grid
from tempestas.io.stations import NETWORKS, period_totals, station_inventory
from tempestas.mask import region_masks
from tempestas.verify.daily import daily_verification
from tempestas.verify.event import EventResult, verify_event
from tempestas.verify.wind import WindResult, characterize_wind

log = logging.getLogger(__name__)


@dataclass
class MonthResult:
    year: int
    month: int
    start: date
    end: date
    obs_daily: dict[str, xr.DataArray]           # (time, lat, lon) por produto, na grade comum
    obs_total: dict[str, xr.DataArray]           # total do mês
    masks: dict[str, xr.DataArray]
    regional_daily: pd.DataFrame                 # chuva média diária por região e produto
    events: list[tuple[date, date]]
    event_results: list[EventResult]
    daily: pd.DataFrame                          # verificação dia a dia
    wind: WindResult | None
    stations: pd.DataFrame | None                # totais mensais por estação (completas)
    inventory: pd.DataFrame | None
    availability: pd.DataFrame                   # rodadas disponíveis por modelo
    skipped: dict[str, str] = field(default_factory=dict)
    prob: dict | None = None                     # métricas probabilísticas do mês (ensemble multimodelo defasado)
    event_ens: list = field(default_factory=list)  # ensemble do total de cada evento (member, lat, lon)


def detect_events(
    daily_obs: xr.DataArray,
    mask: xr.DataArray,
    heavy_mm: float = 50.0,
    min_area_frac: float = 0.05,
    wet_mm: float = 20.0,
    wet_area_frac: float = 0.25,
) -> tuple[list[tuple[date, date]], pd.DataFrame]:
    """Dias de chuva forte no RS e agrupamento em eventos (dias consecutivos).

    Um dia é "de evento" se ≥ ``min_area_frac`` da área teve ≥ ``heavy_mm`` no dia,
    ou ≥ ``wet_area_frac`` da área teve ≥ ``wet_mm``.
    """
    inside = daily_obs.where(mask)
    n = float(mask.sum())
    frac_heavy = (inside >= heavy_mm).sum(("lat", "lon")) / n
    frac_wet = (inside >= wet_mm).sum(("lat", "lon")) / n
    table = pd.DataFrame({
        "dia": pd.DatetimeIndex(daily_obs["time"].values).date,
        "media_mm": inside.mean(("lat", "lon")).values,
        "max_mm": inside.max(("lat", "lon")).values,
        "area_ge_heavy": frac_heavy.values,
        "area_ge_wet": frac_wet.values,
    })
    table["evento"] = (table.area_ge_heavy >= min_area_frac) | (table.area_ge_wet >= wet_area_frac)
    events, cur = [], None
    for d, flag in zip(table["dia"], table["evento"]):
        if flag:
            cur = (cur[0], d) if cur and (d - cur[1]).days == 1 else (d, d)
            if not events or events[-1][0] != cur[0]:
                events.append(cur)
            else:
                events[-1] = cur
        else:
            cur = None
    return events, table


def _ranges(days: list[date]) -> str:
    """Datas em intervalos compactos: 01–05/10, 08/10."""
    if not days:
        return "–"
    out, a, b = [], days[0], days[0]
    for d in days[1:] + [None]:
        if d is not None and (d - b).days == 1:
            b = d
            continue
        out.append(f"{a:%d/%m}" if a == b else f"{a:%d}–{b:%d/%m}")
        if d is not None:
            a = b = d
    return ", ".join(out)


def run_availability(cfg: Config, start: date, end: date) -> pd.DataFrame:
    from tempestas.io.forecasts import available_runs

    rows = []
    n = (end - start).days + 1
    for model in cfg.models:
        runs = set(available_runs(cfg, model, start, end))
        missing = [start + timedelta(days=k) for k in range(n) if start + timedelta(days=k) not in runs]
        rows.append({"modelo": model, "rodadas": len(runs), "dias_no_mes": n,
                     "faltando": _ranges(missing)})
    return pd.DataFrame(rows)


def verify_month(cfg: Config, year: int, month: int, models: list[str] | None = None) -> MonthResult:
    start = date(year, month, 1)
    end = (pd.Timestamp(start) + pd.offsets.MonthEnd(0)).date()
    lats, lons = target_grid(cfg.domain, cfg.verification["grid"])
    masks = region_masks(lats, lons, cfg.raw.get("regions_vector"))
    skipped = {}

    obs_daily, obs_total, reg_rows = {}, {}, []
    for prod, reader in OBSERVED_READERS.items():
        try:
            da = regrid(reader(cfg, start, end), lats, lons)
        except FileNotFoundError as exc:
            skipped[prod] = str(exc)
            continue
        obs_daily[prod] = da
        obs_total[prod] = da.sum("time")
        for region, m in masks.items():
            s = da.where(m).mean(("lat", "lon")).to_series()
            reg_rows += [{"produto": prod, "regiao": region, "dia": t.date(), "media_mm": float(v)} for t, v in s.items()]
    if not obs_daily:
        raise RuntimeError(f"sem produto observado para {start:%m/%Y}: {skipped}")
    ref = next(iter(obs_daily))

    events, _ = detect_events(obs_daily[ref], masks["RS"])
    log.info("eventos detectados: %s", events)
    event_results = []
    for s, e in events:
        try:
            event_results.append(verify_event(cfg, s, e, models=models))
        except Exception as exc:
            skipped[f"evento {s}–{e}"] = str(exc)
            log.warning("evento %s–%s: %s", s, e, exc)

    daily = daily_verification(cfg, start, end, models=models)

    from tempestas.verify.probabilistic import daily_ensemble, event_ensemble, monthly_probabilistic

    prob, event_ens = None, []
    try:
        ens = daily_ensemble(cfg, start, end, lats, lons, models=models)
        prob = monthly_probabilistic(ens, obs_daily[ref], masks["RS"])
    except Exception as exc:
        skipped["probabilístico"] = str(exc)
        log.warning("probabilístico: %s", exc)
    for ev in event_results:
        try:
            event_ens.append(event_ensemble(ev.forecasts))
        except Exception as exc:
            event_ens.append(None)
            log.warning("ensemble do evento %s: %s", ev.start, exc)

    max_missing = int(cfg.verification.get("monthly_max_missing_days", 0))
    st_frames = []
    for network in NETWORKS:
        try:
            st_frames.append(period_totals(cfg, network, start, end, max_missing_days=max_missing))
        except (FileNotFoundError, ValueError) as exc:
            skipped[network] = str(exc)
    stations = pd.concat(st_frames, ignore_index=True) if st_frames else None
    try:
        inventory = station_inventory(cfg, start, end, stations)
    except Exception as exc:
        log.warning("inventário: %s", exc)
        inventory = None

    try:
        wind = characterize_wind(cfg, start, end)
    except Exception as exc:
        log.warning("vento: %s", exc)
        wind = None

    return MonthResult(
        year, month, start, end, obs_daily, obs_total, masks, pd.DataFrame(reg_rows), events, event_results,
        daily, wind, stations, inventory, run_availability(cfg, start, end), skipped, prob, event_ens,
    )
