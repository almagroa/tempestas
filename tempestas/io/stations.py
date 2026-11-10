"""Chuva observada em estações (referência pontual).

Por enquanto: INMET automáticas. Lê os CSV baixados por ``tempestas baixar``
(``<work_dir>/dados/INMET/INMET_RS_AAAA.csv``) e, para anos anteriores, os zips
anuais do portal guardados em ``paths.inmet_root`` (somente leitura).
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from tempestas.config import Config
from tempestas.download.inmet import daily_precip, read_zip

log = logging.getLogger(__name__)


def _inmet_year(cfg: Config, year: int) -> pd.DataFrame:
    csv = cfg.paths["work_dir"] / "dados" / "INMET" / f"INMET_RS_{year}.csv"
    if csv.exists():
        df = pd.read_csv(csv, parse_dates=["time_utc"])
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
        return df
    z = cfg.paths["inmet_root"] / f"{year}.zip"
    if z.exists():
        return read_zip(z, "RS")
    return pd.DataFrame()


def _csv_year(folder: str, prefix: str):
    def reader(cfg: Config, year: int) -> pd.DataFrame:
        csv = cfg.paths["work_dir"] / "dados" / folder / f"{prefix}_RS_{year}.csv"
        if not csv.exists():
            return pd.DataFrame()
        df = pd.read_csv(csv, dtype={"codigo": str})
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
        return df
    return reader


_READERS = {"INMET": _inmet_year, "Cemaden": _csv_year("CEMADEN", "CEMADEN"), "ANA": _csv_year("ANA", "ANA")}
NETWORKS = list(_READERS)


def period_totals(cfg: Config, network: str, start: date, end: date, max_missing_days: int = 0) -> pd.DataFrame:
    """Total de chuva por estação de uma rede no período (dia UTC). Redes não são fundidas.

    Entram estações com no máximo ``max_missing_days`` dias incompletos; os dias que
    faltam não são preenchidos (coluna ``dias_faltando``).
    """
    frames = [_READERS[network](cfg, y) for y in range(start.year, end.year + 1)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise FileNotFoundError(f"sem dados {network} para {start}–{end}")
    hourly = pd.concat(frames, ignore_index=True)
    daily = daily_precip(hourly, day_start_hour_utc=0)
    daily = daily[(daily["dia"] >= pd.Timestamp(start)) & (daily["dia"] <= pd.Timestamp(end))]
    n_days = (end - start).days + 1
    g = daily.groupby(["codigo", "estacao", "lat", "lon"])
    tot = g["precip_mm"].agg(obs_mm="sum", dias_validos="count").reset_index()
    tot["dias_faltando"] = n_days - tot.pop("dias_validos")
    tot = tot[tot["dias_faltando"] <= max_missing_days]
    tot = tot.dropna(subset=["lat", "lon"])
    if tot.empty:
        raise ValueError(f"nenhuma estação {network} com dados completos em {start}–{end}")
    tot.insert(0, "rede", network)
    return tot


def inmet_period_totals(cfg: Config, start: date, end: date) -> pd.DataFrame:
    """Total de chuva por estação no período (dia UTC, como IMERG e modelos).

    Só entram estações com todos os dias completos no período.
    """
    frames = [_inmet_year(cfg, y) for y in range(start.year, end.year + 1)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise FileNotFoundError(f"sem dados INMET para {start}–{end}")
    hourly = pd.concat(frames, ignore_index=True)
    daily = daily_precip(hourly, day_start_hour_utc=0)
    daily = daily[(daily["dia"] >= pd.Timestamp(start)) & (daily["dia"] <= pd.Timestamp(end))]
    n_days = (end - start).days + 1
    g = daily.groupby(["codigo", "estacao", "lat", "lon"])
    tot = g["precip_mm"].agg(obs_mm="sum", dias_validos="count").reset_index()
    tot = tot[tot["dias_validos"] == n_days].drop(columns="dias_validos")
    if tot.empty:
        raise ValueError(f"nenhuma estação INMET com dados completos em {start}–{end}")
    return tot


# Coordenadas aproximadas dos aeroportos do RS com METAR (graus decimais)
AIRPORTS_RS = {
    "SBPA": ("Porto Alegre", -29.994, -51.171),
    "SBCO": ("Canoas", -29.946, -51.144),
    "SBCX": ("Caxias do Sul", -29.197, -51.188),
    "SBSM": ("Santa Maria", -29.711, -53.688),
    "SBPF": ("Passo Fundo", -28.244, -52.327),
    "SBPK": ("Pelotas", -31.718, -52.328),
    "SBRG": ("Rio Grande", -32.082, -52.166),
    "SBUG": ("Uruguaiana", -29.782, -57.038),
    "SBBG": ("Bagé", -31.390, -54.112),
    "SBNM": ("Santo Ângelo", -28.282, -54.169),
}


def station_inventory(cfg: Config, start: date, end: date, complete: pd.DataFrame | None = None) -> pd.DataFrame:
    """Todas as estações conhecidas de todas as redes, com a situação no período.

    Colunas: rede, codigo, estacao, lat, lon, obs_mm (só para estações com o
    período completo), situacao ('completa', 'incompleta/sem dados', 'aeroporto').
    """
    rows = []
    for network in NETWORKS:
        frames = [_READERS[network](cfg, y) for y in range(start.year, end.year + 1)]
        frames = [f for f in frames if not f.empty]
        if not frames:
            continue
        meta = pd.concat(frames, ignore_index=True)[["codigo", "estacao", "lat", "lon"]]
        meta = meta.dropna(subset=["lat", "lon"]).drop_duplicates("codigo")
        meta.insert(0, "rede", network)
        rows.append(meta)
    inv = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["rede", "codigo", "estacao", "lat", "lon"])
    inv["codigo"] = inv["codigo"].astype(str)
    if complete is not None and len(complete):
        c = complete[["rede", "codigo", "obs_mm"] + (["dias_faltando"] if "dias_faltando" in complete else [])].copy()
        c["codigo"] = c["codigo"].astype(str)
        inv = inv.merge(c, on=["rede", "codigo"], how="left")
    else:
        inv["obs_mm"] = float("nan")
    inv["situacao"] = inv["obs_mm"].notna().map({True: "completa", False: "incompleta/sem dados"})
    if "dias_faltando" in inv:
        inv.loc[inv["obs_mm"].notna() & (inv["dias_faltando"] > 0), "situacao"] = "parcial"
    ap = pd.DataFrame([
        {"rede": "METAR", "codigo": k, "estacao": v[0], "lat": v[1], "lon": v[2], "obs_mm": float("nan"), "situacao": "aeroporto"}
        for k, v in AIRPORTS_RS.items()
    ])
    return pd.concat([inv, ap], ignore_index=True)
