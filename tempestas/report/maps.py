"""Mapas estáticos para os relatórios (PNG).

Cores: chuva acumulada usa uma escala sequencial de um só matiz (claro → escuro);
a diferença previsto − observado usa uma escala divergente de dois matizes com
centro neutro em zero (marrom = previu menos, verde-azulado = previu mais).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402

PRECIP_LEVELS = [0, 5, 10, 25, 50, 75, 100, 150, 200, 300]
DIFF_LEVELS = [-150, -100, -50, -25, -10, 10, 25, 50, 100, 150]
INK = "#1f2328"
MUTED = "#6e7781"
# Cor fixa por modelo (a cor segue o modelo, nunca a posição na lista)
MODEL_COLORS = {
    "GEFS": "#2a6fdb",
    "IFS": "#e0782a",
    "AIFS": "#2a9d8f",
    "ICON": "#9b5de5",
    "CPT-WRF": "#c9184a",
    "CHIRPS-GEFS": "#6c757d",
}


def _outline_lines(vector_path: str | Path | None) -> list[np.ndarray]:
    """Lê contornos de um GeoJSON (lon/lat) como listas de coordenadas."""
    if not vector_path or not Path(vector_path).exists():
        return []
    with open(vector_path, encoding="utf-8") as fh:
        gj = json.load(fh)
    lines = []
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    for f in feats:
        g = f.get("geometry", f)
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            for ring in poly:
                lines.append(np.asarray(ring))
    return lines


def _panel(ax, da, levels, cmap, outline, title, mask=None):
    norm = BoundaryNorm(levels, plt.get_cmap(cmap).N, extend="both")
    data = da.where(mask) if mask is not None else da
    im = ax.pcolormesh(da["lon"], da["lat"], data, cmap=cmap, norm=norm, shading="nearest")
    for ring in outline:
        ax.plot(ring[:, 0], ring[:, 1], color=INK, lw=0.6)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_aspect(1 / math.cos(math.radians(float(da["lat"].mean()))))
    ax.tick_params(labelsize=7, colors=MUTED)
    for s in ax.spines.values():
        s.set_color("#d0d7de")
    return im


def event_maps(
    observed: dict[str, xr.DataArray],
    forecasts: dict[str, xr.DataArray],
    out_path: str | Path,
    lead_start: int = 1,
    reference: str | None = None,
    outline_path: str | Path | None = None,
    mask: xr.DataArray | None = None,
    suptitle: str = "",
) -> Path:
    """Painéis: observado(s), cada modelo e a diferença de cada modelo para ``reference``.

    Para cada modelo usa a rodada com antecedência ``lead_start`` (ou a mais
    próxima disponível acima dela).
    """
    outline = _outline_lines(outline_path)
    reference = reference or next(iter(observed))
    ob_ref = observed[reference]

    fc_sel = {}
    for name, fc in forecasts.items():
        leads = fc["lead_start"].values
        ok = np.where(leads >= lead_start)[0]
        if ok.size:
            i = ok[np.argmin(leads[ok])]
            fc_sel[name] = (fc.isel(init=i), int(leads[i]))

    ncols = max(len(observed), len(fc_sel), 1)
    fig, axes = plt.subplots(3, ncols, figsize=(2.6 * ncols, 8.4), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)

    for j, (name, ob) in enumerate(observed.items()):
        ax = axes[0, j]
        ax.set_visible(True)
        im_p = _panel(ax, ob, PRECIP_LEVELS, "Blues", outline, f"Observado: {name}", mask)
    for j, (name, (fc, lead)) in enumerate(fc_sel.items()):
        ax = axes[1, j]
        ax.set_visible(True)
        im_p = _panel(ax, fc, PRECIP_LEVELS, "Blues", outline, f"{name} (antec. {lead} d)", mask)
        ax = axes[2, j]
        ax.set_visible(True)
        im_d = _panel(ax, fc - ob_ref, DIFF_LEVELS, "BrBG", outline, f"{name} − {reference}", mask)

    fig.colorbar(im_p, ax=axes[:2, :].ravel().tolist(), shrink=0.6, label="Chuva acumulada (mm)")
    if fc_sel:
        fig.colorbar(im_d, ax=axes[2, :].ravel().tolist(), shrink=0.8, label="Previsto − observado (mm)")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, color=INK, x=0.02, ha="left")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def skill_by_lead(scores, threshold: float, observed: str, metric: str, out_path: str | Path) -> Path:
    """Linha por modelo: métrica × antecedência (dias), para um limiar e um observado."""
    sub = scores[(scores.threshold_mm == threshold) & (scores.observed == observed)]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for model, g in sub.groupby("model", sort=False):
        g = g.sort_values("lead_start")
        ax.plot(g.lead_start, g[metric], lw=2, marker="o", ms=4, color=MODEL_COLORS.get(model, INK), label=model)
    ax.set_xlabel("Antecedência do início do evento (dias)", color=MUTED, fontsize=8)
    ax.set_ylabel(metric.upper(), color=MUTED, fontsize=8)
    ax.set_title(f"{metric.upper()} para ≥ {threshold:g} mm × {observed}", loc="left", fontsize=10, color=INK)
    ax.grid(color="#eaeef2", lw=0.6)
    ax.tick_params(labelsize=7, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False, ncol=3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


WIND_LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40]


def wind_maps(grids: dict[str, xr.DataArray], out_path: str | Path, outline_path=None, suptitle: str = "") -> Path:
    """Um painel por campo de vento máximo no período (m/s)."""
    outline = _outline_lines(outline_path)
    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), squeeze=False)
    im = None
    for ax, (name, da) in zip(axes[0], grids.items()):
        im = _panel(ax, da, WIND_LEVELS, "Oranges", outline, name)
    if im is not None:
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="Vento máximo no período (m/s)")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, color=INK, x=0.02, ha="left")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
