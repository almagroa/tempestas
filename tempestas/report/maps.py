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
import pandas as pd  # noqa: E402
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


def daily_series(daily, out_path, observed: str, lead: int = 1, region: str = "RS") -> Path:
    """Chuva média diária na região: observado × cada modelo numa antecedência fixa."""
    d = daily[(daily.region == region) & (daily.observed == observed)]
    d = d.drop_duplicates(["day", "model", "lead"])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ob = d.drop_duplicates("day").sort_values("day")
    ax.bar(ob["day"], ob["obs_mean"], color="#c8d6e5", width=0.8, label=f"{observed} (observado)")
    for model, g in d[d.lead == lead].groupby("model", sort=False):
        g = g.sort_values("day")
        ax.plot(g["day"], g["fc_mean"], lw=2, marker="o", ms=4, color=MODEL_COLORS.get(model, INK), label=model)
    ax.set_ylabel("Chuva média (mm/dia)", color=MUTED, fontsize=8)
    ax.set_title(f"Chuva diária média – {region} – previsão com antecedência de {lead} dia(s)", loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color="#eaeef2", lw=0.6)
    ax.tick_params(labelsize=7, colors=MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.autofmt_xdate()
    ax.legend(fontsize=7, frameon=False, ncol=4)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _sample(fld: xr.DataArray, stations) -> np.ndarray:
    lat = xr.DataArray(stations["lat"].values, dims="st")
    lon = xr.DataArray(stations["lon"].values, dims="st")
    return fld.sel(lat=lat, lon=lon, method="nearest").values


def station_maps(
    stations,
    observed: dict[str, xr.DataArray],
    forecasts: dict[str, xr.DataArray],
    out_path: str | Path,
    lead_start: int = 1,
    outline_path=None,
    mask: xr.DataArray | None = None,
    suptitle: str = "",
) -> Path:
    """Estações plotadas.

    Linha 1: chuva acumulada nas estações (círculos) sobre o campo do produto em
    grade, mesma escala de cores; um painel por rede.
    Linhas seguintes: erro de cada modelo em cada estação (previsto − estação),
    escala divergente.
    """
    outline = _outline_lines(outline_path)
    networks = list(stations["rede"].unique())
    bg_name, bg = next(iter(observed.items()))

    fc_sel = {}
    for name, fc in forecasts.items():
        leads = fc["lead_start"].values
        ok = np.where(leads >= lead_start)[0]
        if ok.size:
            i = ok[np.argmin(leads[ok])]
            fc_sel[name] = (fc.isel(init=i), int(leads[i]))

    ncols = max(len(networks), 3)
    nrows_fc = int(np.ceil(len(fc_sel) / ncols))
    fig, axes = plt.subplots(1 + nrows_fc, ncols, figsize=(3.3 * ncols, 3.4 * (1 + nrows_fc)), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)

    p_norm = BoundaryNorm(PRECIP_LEVELS, plt.get_cmap("Blues").N, extend="both")
    d_norm = BoundaryNorm(DIFF_LEVELS, plt.get_cmap("BrBG").N, extend="both")
    sc_p = sc_d = None
    for j, net in enumerate(networks):
        st = stations[stations.rede == net]
        ax = axes[0, j]
        ax.set_visible(True)
        _panel(ax, bg, PRECIP_LEVELS, "Blues", outline, f"{net} ({len(st)} est.) · fundo {bg_name}", mask)
        sc_p = ax.scatter(st["lon"], st["lat"], c=st["obs_mm"], cmap="Blues", norm=p_norm, s=14,
                          edgecolors=INK, linewidths=0.4, zorder=3)

    for k, (name, (fc, lead)) in enumerate(fc_sel.items()):
        ax = axes[1 + k // ncols, k % ncols]
        ax.set_visible(True)
        err = _sample(fc, stations) - stations["obs_mm"].values
        _panel(ax, fc * np.nan, PRECIP_LEVELS, "Blues", outline, f"{name} − est. ({lead} d)")
        order = np.argsort(np.abs(err))  # maiores erros por cima
        sc_d = ax.scatter(stations["lon"].values[order], stations["lat"].values[order], c=err[order],
                          cmap="BrBG", norm=d_norm, s=12, edgecolors="#ffffff", linewidths=0.3, zorder=3)

    if sc_p is not None:
        fig.colorbar(sc_p, ax=axes[0, :].tolist(), shrink=0.8, label="Chuva acumulada (mm)")
    if sc_d is not None:
        fig.colorbar(sc_d, ax=axes[1:, :].ravel().tolist(), shrink=0.8, label="Previsto − estação (mm)")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, color=INK, x=0.02, ha="left")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


NETWORK_MARKERS = {"Cemaden": "o", "INMET": "s", "ANA": "^", "METAR": "*"}


def network_map(inventory, background: xr.DataArray, out_path, bg_name: str = "IMERG", outline_path=None,
                mask=None, suptitle: str = "") -> Path:
    """Todas as estações de todas as redes num só mapa.

    Símbolo = rede. Preenchido e colorido pela chuva = período completo;
    vazado em cinza = estação sem dados completos no período; estrela = aeroporto (METAR, sem medida de chuva).
    """
    from matplotlib.lines import Line2D

    outline = _outline_lines(outline_path)
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    _panel(ax, background, PRECIP_LEVELS, "Blues", outline, f"Fundo: {bg_name}", mask)
    norm = BoundaryNorm(PRECIP_LEVELS, plt.get_cmap("Blues").N, extend="both")
    handles = []
    sc = None
    for net, g in inventory.groupby("rede", sort=False):
        mk = NETWORK_MARKERS.get(net, "D")
        size = 170 if mk == "*" else 26
        z = 6 if net == "METAR" else 0  # aeroportos por cima das redes densas
        miss = g[g["obs_mm"].isna()]
        ok = g[g["obs_mm"].notna()]
        if len(miss):
            face = "#f2c94c" if net == "METAR" else "none"
            ax.scatter(miss["lon"], miss["lat"], marker=mk, s=size, facecolors=face,
                       edgecolors="#6e7781" if net != "METAR" else INK, linewidths=0.8, zorder=3 + z)
        if len(ok):
            part = ok.get("situacao", pd.Series("completa", index=ok.index)).eq("parcial")
            sc = ax.scatter(ok["lon"], ok["lat"], marker=mk, s=size, c=ok["obs_mm"], cmap="Blues", norm=norm,
                            edgecolors=np.where(part, "#d9822b", INK), linewidths=np.where(part, 1.1, 0.5),
                            zorder=4 + z)
        n_ok, n_all = len(ok), len(g)
        label = f"{net}: {n_all} aeroportos (vento e tempo presente)" if net == "METAR" else f"{net}: {n_ok} de {n_all} com dados completos"
        handles.append(Line2D([], [], marker=mk, ls="", ms=9 if mk == "*" else 6, mfc="#f2c94c" if net == "METAR" else "#9ecae1",
                              mec=INK, label=label))
    if "situacao" in inventory and inventory["situacao"].eq("parcial").any():
        handles.append(Line2D([], [], marker="o", ls="", ms=6, mfc="#9ecae1", mec="#d9822b", mew=1.1,
                              label="borda laranja = parcial (até 3 dias faltando)"))
    handles.append(Line2D([], [], marker="o", ls="", ms=6, mfc="none", mec="#6e7781", label="vazado = sem dados completos no período"))
    ax.legend(handles=handles, loc="lower left", fontsize=7.5, frameon=True, framealpha=0.9)
    if sc is not None:
        fig.colorbar(sc, ax=ax, shrink=0.75, label="Chuva acumulada (mm)")
    if suptitle:
        ax.set_title(suptitle, fontsize=11, color=INK, loc="left")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
