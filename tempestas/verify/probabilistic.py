"""Verificação probabilística com um ensemble multimodelo defasado.

Os modelos arquivados são determinísticos (o GEFS guardado é a média do ensemble).
A incerteza vem de juntar, para o mesmo dia ou evento, todos os modelos e as
rodadas de dias anteriores (antecedências 1..N). Cada par modelo × antecedência
é um "membro". O espalhamento mede a discordância entre modelos e rodadas,
não a incerteza calibrada de um ensemble de perturbações.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import xarray as xr

from tempestas.config import Config
from tempestas.io.forecasts import available_runs, open_run
from tempestas.io.standardize import regrid

log = logging.getLogger(__name__)

MIN_MEMBERS = 3
POSITION_LABELS = {0: "abaixo de todas as previsões", 1: "dentro do leque", 2: "acima de todas as previsões"}


def daily_ensemble(
    cfg: Config, start: date, end: date, lats, lons, leads: list[int] | None = None, models: list[str] | None = None,
) -> xr.DataArray:
    """Chuva diária prevista (member, time, lat, lon); member = 'MODELO dN'. Falta = NaN."""
    leads = leads or [1, 2, 3, 4, 5]
    days = pd.date_range(start, end, freq="D")
    members = []
    for model in models or list(cfg.models):
        runs = {}
        for init in available_runs(cfg, model, start - timedelta(days=max(leads) - 1), end):
            try:
                runs[init] = regrid(open_run(cfg, model, init), lats, lons)
            except Exception as exc:
                log.warning("%s %s: %s", model, init, exc)
        for lead in leads:
            fields = []
            for t in days:
                init = (t - pd.Timedelta(days=lead - 1)).date()
                run = runs.get(init)
                if run is not None and t in run["time"].values:
                    fields.append(run.sel(time=t).drop_vars(["lead", "init"], errors="ignore").values)
                else:
                    fields.append(np.full((len(lats), len(lons)), np.nan, dtype="float32"))
            if all(np.isnan(f).all() for f in fields):
                continue
            members.append(xr.DataArray(np.stack(fields), dims=("time", "lat", "lon"),
                                        coords={"time": days, "lat": lats, "lon": lons},
                                        name=f"{model} d{lead}").expand_dims(member=[f"{model} d{lead}"]))
    if not members:
        raise FileNotFoundError(f"nenhuma rodada para o ensemble de {start}–{end}")
    return xr.concat(members, dim="member")


def event_ensemble(forecasts: dict[str, xr.DataArray], max_lead: int = 5) -> xr.DataArray:
    """Total do evento por membro, a partir das rodadas por modelo de ``verify_event`` (dim init)."""
    members = []
    for model, da in forecasts.items():
        if "init" not in da.dims:
            continue
        sel = da.where(da["lead_start"] <= max_lead, drop=True) if "lead_start" in da.coords else da
        for k in range(sel.sizes.get("init", 0)):
            m = sel.isel(init=k)
            lead = int(m["lead_start"]) if "lead_start" in m.coords else k + 1
            if lead < 1:
                continue
            m = m.drop_vars([c for c in m.coords if c not in ("lat", "lon")])
            members.append(m.expand_dims(member=[f"{model} d{lead}"]))
    if not members:
        raise ValueError("sem rodadas para o ensemble do evento")
    return xr.concat(members, dim="member")


def probability(ens: xr.DataArray, threshold: float) -> xr.DataArray:
    """Fração dos membros com chuva ≥ limiar (NaN onde há menos de MIN_MEMBERS)."""
    n = ens.notnull().sum("member")
    p = (ens >= threshold).where(ens.notnull()).sum("member") / n
    return p.where(n >= MIN_MEMBERS)


def obs_position(ens: xr.DataArray, obs: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """(categoria, percentil) do observado no leque dos membros.

    categoria: 0 abaixo do menor membro, 1 dentro do leque, 2 acima do maior.
    percentil: fração dos membros abaixo do observado (empates contam metade).
    """
    n = ens.notnull().sum("member")
    lo, hi = ens.min("member"), ens.max("member")
    cat = xr.where(obs < lo, 0, xr.where(obs > hi, 2, 1)).where(obs.notnull() & (n >= MIN_MEMBERS))
    below = (ens < obs).where(ens.notnull()).sum("member")
    ties = (ens == obs).where(ens.notnull()).sum("member")
    pct = ((below + 0.5 * ties) / n).where(obs.notnull() & (n >= MIN_MEMBERS))
    return cat, pct


def brier(p: np.ndarray, o: np.ndarray) -> dict:
    """Brier score, climatologia da amostra e skill score."""
    ok = np.isfinite(p) & np.isfinite(o)
    p, o = p[ok], o[ok]
    if not len(p):
        return {"n": 0, "brier": np.nan, "brier_clim": np.nan, "bss": np.nan, "freq_obs": np.nan}
    bs = float(np.mean((p - o) ** 2))
    base = float(o.mean())
    bs_clim = base * (1 - base)
    return {"n": int(len(p)), "brier": bs, "brier_clim": bs_clim,
            "bss": 1 - bs / bs_clim if bs_clim > 0 else np.nan, "freq_obs": base}


def reliability(p: np.ndarray, o: np.ndarray, bins: int = 5) -> pd.DataFrame:
    """Frequência observada por faixa de probabilidade prevista."""
    ok = np.isfinite(p) & np.isfinite(o)
    p, o = p[ok], o[ok]
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for k in range(bins):
        sel = idx == k
        rows.append({"faixa": f"{edges[k]:.0%}–{edges[k + 1]:.0%}", "p_media": float(p[sel].mean()) if sel.any() else np.nan,
                     "freq_obs": float(o[sel].mean()) if sel.any() else np.nan, "n": int(sel.sum())})
    return pd.DataFrame(rows)


def monthly_probabilistic(
    ens: xr.DataArray, obs: xr.DataArray, mask: xr.DataArray, thresholds: list[float] = (25, 50),
) -> dict:
    """Métricas do mês (dias × pontos dentro da máscara)."""
    ens = ens.sel(time=ens["time"].isin(obs["time"].values))
    obs = obs.sel(time=ens["time"])
    m = mask.broadcast_like(obs).values.astype(bool)
    out = {"brier": [], "reliability": {}}
    for thr in thresholds:
        p = probability(ens, thr).values[m]
        o = (obs >= thr).where(obs.notnull()).values[m].astype(float)
        out["brier"].append({"limiar_mm": thr, **brier(p, o)})
        out["reliability"][thr] = reliability(p, o)
    out["brier"] = pd.DataFrame(out["brier"])
    cat, pct = obs_position(ens, obs)
    # só casos com chuva observada ou prevista (≥ 1 mm): "zero dentro de zero" inflaria o "dentro do leque"
    active = ((obs >= 1) | (ens.max("member") >= 1)).values & m
    c = cat.values[active]
    c = c[np.isfinite(c)]
    out["position"] = {POSITION_LABELS[k]: float((c == k).mean()) if len(c) else np.nan for k in (0, 1, 2)}
    # histograma de posição só com dias de chuva (obs ≥ 1 mm), para não encher a faixa de "zero" empatado
    wet = (obs >= 1).values[m]
    pv = pct.values[m][wet]
    pv = pv[np.isfinite(pv)]
    out["rank_hist"] = np.histogram(pv, bins=10, range=(0, 1))[0]
    daily_in = ((cat == 1).where(cat.notnull())).where(mask).mean(("lat", "lon")).to_series()
    out["daily_inside"] = daily_in
    out["n_members"] = int(ens.sizes["member"])
    out["members"] = list(map(str, ens["member"].values))
    return out


def event_summary(ens: xr.DataArray, obs: xr.DataArray, mask: xr.DataArray, thresholds=(50, 100)) -> dict:
    """Resumo de um evento: leque da média regional e probabilidade onde o limiar foi observado."""
    reg = ens.where(mask).mean(("lat", "lon"))
    reg = reg[np.isfinite(reg.values)]
    o_reg = float(obs.where(mask).mean(("lat", "lon")))
    out = {"n": int(reg.size), "obs_media": o_reg}
    if reg.size:
        v = np.sort(reg.values)
        out.update(p10=float(np.percentile(v, 10)), p50=float(np.percentile(v, 50)), p90=float(np.percentile(v, 90)),
                   minimo=float(v[0]), maximo=float(v[-1]),
                   pct_obs=float(((v < o_reg).sum() + 0.5 * (v == o_reg).sum()) / v.size),
                   por_membro=reg.to_series().sort_values())
    cat, pct = obs_position(ens, obs)
    c = cat.where(mask).values
    c = c[np.isfinite(c)]
    out["area_posicao"] = {POSITION_LABELS[k]: float((c == k).mean()) if len(c) else np.nan for k in (0, 1, 2)}
    out["limiares"] = []
    m = mask.values.astype(bool)
    for thr in thresholds:
        p = probability(ens, thr)
        hit = (obs >= thr).values & m
        out["limiares"].append({
            "limiar_mm": thr,
            "area_obs": float(hit.sum() / m.sum()),
            "prob_media_onde_ocorreu": float(np.nanmean(p.values[hit])) if hit.any() else np.nan,
            "prob_max": float(np.nanmax(p.where(mask).values)) if np.isfinite(p.where(mask).values).any() else np.nan,
            "area_prob_ge_50pct": float(np.nansum(p.values[m] >= 0.5) / m.sum()),
        })
    out["cat"], out["pct"] = cat, pct
    return out
