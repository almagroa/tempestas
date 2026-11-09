"""Verificação dia a dia dentro de um período.

Complementa a verificação do acumulado do período: cada dia é comparado com a
previsão feita 1, 2, 3... dias antes. Assim modelos de horizonte curto (ICON e
CPT-WRF, 7 dias) também entram em eventos longos.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import xarray as xr

from tempestas.config import Config
from tempestas.io.forecasts import available_runs, open_run
from tempestas.io.observed import OBSERVED_READERS
from tempestas.io.standardize import regrid, target_grid
from tempestas.mask import region_masks
from tempestas.verify.metrics import categorical, continuous

log = logging.getLogger(__name__)

DAILY_THRESHOLDS = [10, 25, 50]


def daily_verification(
    cfg: Config,
    start: date,
    end: date,
    leads: list[int] | None = None,
    models: list[str] | None = None,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Uma linha por região × dia × modelo × antecedência × observado × limiar."""
    leads = leads or [1, 2, 3, 5, 7]
    thresholds = thresholds or DAILY_THRESHOLDS
    lats, lons = target_grid(cfg.domain, cfg.verification["grid"])
    masks = region_masks(lats, lons, cfg.raw.get("regions_vector"))

    observed = {}
    for prod, reader in OBSERVED_READERS.items():
        try:
            observed[prod] = regrid(reader(cfg, start, end), lats, lons)
        except FileNotFoundError as exc:
            log.warning("observado %s ignorado: %s", prod, exc)

    rows = []
    first_init = start - timedelta(days=max(leads) - 1)
    for model in models or list(cfg.models):
        runs = {}
        for init in available_runs(cfg, model, first_init, end):
            try:
                runs[init] = regrid(open_run(cfg, model, init), lats, lons)
            except Exception as exc:
                log.warning("%s %s: %s", model, init, exc)
        day = start
        while day <= end:
            for lead in leads:
                init = day - timedelta(days=lead - 1)
                if init not in runs:
                    continue
                fc_run = runs[init]
                t = pd.Timestamp(day)
                if t not in fc_run["time"].values:
                    continue
                fc = fc_run.sel(time=t)
                for obs_name, ob_all in observed.items():
                    if t not in ob_all["time"].values:
                        continue
                    ob = ob_all.sel(time=t)
                    for region, mask in masks.items():
                        cont = continuous(fc, ob, mask)
                        for thr in thresholds:
                            rows.append({
                                "region": region, "day": day, "model": model, "lead": lead,
                                "init": init, "observed": obs_name, **cont, **categorical(fc, ob, thr, mask),
                            })
            day += timedelta(days=1)
    return pd.DataFrame(rows)


def summarize_by_lead(daily: pd.DataFrame, region: str = "RS", threshold: float = 25) -> pd.DataFrame:
    """Agrega os dias: soma a tabela de contingência e faz média dos erros, por modelo × antecedência."""
    d = daily[(daily.region == region) & (daily.threshold_mm == threshold)]
    if d.empty:
        return d
    g = d.groupby(["observed", "model", "lead"])
    out = g.agg(
        dias=("day", "nunique"), obs_mean=("obs_mean", "mean"), fc_mean=("fc_mean", "mean"),
        mae=("mae", "mean"), hits=("hits", "sum"), misses=("misses", "sum"), false_alarms=("false_alarms", "sum"),
    ).reset_index()
    out["bias_pct"] = (100 * (out.fc_mean - out.obs_mean) / out.obs_mean).where(out.obs_mean >= 1.0)
    out["pod"] = out.hits / (out.hits + out.misses)
    out["far"] = out.false_alarms / (out.hits + out.false_alarms)
    out["csi"] = out.hits / (out.hits + out.misses + out.false_alarms)
    return out
