"""Métricas de verificação previsto × observado sobre uma grade comum."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def continuous(fc: xr.DataArray, ob: xr.DataArray, mask: xr.DataArray | None = None) -> dict:
    """Métricas contínuas sobre as células válidas (e dentro da máscara)."""
    valid = np.isfinite(fc) & np.isfinite(ob)
    if mask is not None:
        valid = valid & mask
    f = fc.where(valid).values.ravel()
    o = ob.where(valid).values.ravel()
    keep = np.isfinite(f) & np.isfinite(o)
    f, o = f[keep], o[keep]
    if f.size == 0:
        return {k: np.nan for k in ("n", "obs_mean", "fc_mean", "bias_mm", "bias_pct", "mae", "rmse", "corr")}
    err = f - o
    return {
        "n": int(f.size),
        "obs_mean": float(o.mean()),
        "fc_mean": float(f.mean()),
        "bias_mm": float(err.mean()),
        "bias_pct": float(100 * err.mean() / o.mean()) if o.mean() > 0 else np.nan,
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "corr": float(np.corrcoef(f, o)[0, 1]) if f.std() > 0 and o.std() > 0 else np.nan,
    }


def categorical(fc: xr.DataArray, ob: xr.DataArray, threshold: float, mask: xr.DataArray | None = None) -> dict:
    """Tabela de contingência para ``>= threshold`` e índices derivados.

    POD = acertos / (acertos + perdas); FAR = alarmes falsos / (acertos + alarmes);
    CSI = acertos / (acertos + perdas + alarmes); FBI = (acertos + alarmes) / (acertos + perdas).
    """
    valid = np.isfinite(fc) & np.isfinite(ob)
    if mask is not None:
        valid = valid & mask
    f = (fc >= threshold) & valid
    o = (ob >= threshold) & valid
    hits = int((f & o).sum())
    misses = int((~f & o & valid).sum())
    false_alarms = int((f & ~o).sum())
    cn = int((~f & ~o & valid).sum())

    def div(a, b):
        return a / b if b else np.nan

    return {
        "threshold_mm": threshold,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": cn,
        "pod": div(hits, hits + misses),
        "far": div(false_alarms, hits + false_alarms),
        "csi": div(hits, hits + misses + false_alarms),
        "fbi": div(hits + false_alarms, hits + misses),
    }


def score_table(
    forecasts: dict[str, xr.DataArray],
    observed: dict[str, xr.DataArray],
    thresholds: list[float],
    mask: xr.DataArray | None = None,
) -> pd.DataFrame:
    """Uma linha por modelo × rodada × produto observado × limiar.

    ``forecasts[model]`` tem dimensão ``init`` (saída de ``forecast_for_period``
    já reamostrada); ``observed[produto]`` é o total observado (lat, lon).
    """
    rows = []
    for model, fc_all in forecasts.items():
        for i in range(fc_all.sizes["init"]):
            fc = fc_all.isel(init=i)
            base = {
                "model": model,
                "init": pd.Timestamp(fc["init"].values).date(),
                "lead_start": int(fc["lead_start"].values),
            }
            for obs_name, ob in observed.items():
                cont = continuous(fc, ob, mask)
                for thr in thresholds:
                    rows.append({**base, "observed": obs_name, **cont, **categorical(fc, ob, thr, mask)})
    return pd.DataFrame(rows)
