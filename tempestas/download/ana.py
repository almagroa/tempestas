"""Estações telemétricas da ANA (pluviômetros), via webservice público, sem cadastro.

Webservice: http://telemetriaws1.ana.gov.br/ServiceANA.asmx
- ListaEstacoesTelemetricas: lista de estações telemétricas;
- DadosHidrometeorologicos: dados de 15 em 15 minutos (chuva, nível, vazão).

As datas/horas do webservice estão em horário de Brasília (UTC−3); aqui são
convertidas para UTC, como o restante do Tempestas.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

WS = "http://telemetriaws1.ana.gov.br/ServiceANA.asmx"


def _rows(xml_text: str, tag: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for el in root.iter():
        if el.tag.split("}")[-1] == tag:
            out.append({c.tag.split("}")[-1]: (c.text or "").strip() for c in el})
    return out


def list_stations(uf: str = "RS", timeout: int = 120) -> pd.DataFrame:
    import requests

    r = requests.get(f"{WS}/ListaEstacoesTelemetricas", params={"statusEstacoes": "0", "origem": ""}, timeout=timeout)
    r.raise_for_status()
    df = pd.DataFrame(_rows(r.text, "Table"))
    if df.empty:
        return df
    uf_col = next((c for c in df.columns if c.lower() in ("nomeestado", "uf", "sgestado")), None)
    if uf_col:
        df = df[df[uf_col].str.upper().isin([uf, "RIO GRANDE DO SUL"])]
    for c in ("Latitude", "Longitude"):
        if c in df:
            df[c] = pd.to_numeric(df[c].str.replace(",", "."), errors="coerce")
    return df


def fetch_station(code: str, start: date, end: date, timeout: int = 120) -> pd.DataFrame:
    import requests

    params = {"codEstacao": code, "dataInicio": f"{start:%d/%m/%Y}", "dataFim": f"{end:%d/%m/%Y}"}
    r = requests.get(f"{WS}/DadosHidrometeorologicos", params=params, timeout=timeout)
    r.raise_for_status()
    df = pd.DataFrame(_rows(r.text, "DadosHidrometereologicos"))
    if df.empty or "Chuva" not in df:
        return pd.DataFrame()
    local = pd.to_datetime(df["DataHora"], errors="coerce")
    out = pd.DataFrame({
        "time_utc": (local + pd.Timedelta(hours=3)).dt.tz_localize("UTC"),
        "precip_mm": pd.to_numeric(df["Chuva"].str.replace(",", "."), errors="coerce"),
        "codigo": code,
    })
    return out.dropna(subset=["time_utc"])


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega os registros de 15 min em horas (carimbo no fim da hora, como o INMET)."""
    if df.empty:
        return df
    d = df.copy()
    d["time_utc"] = d["time_utc"].dt.ceil("h")
    g = d.groupby(["codigo", "time_utc"])
    out = g["precip_mm"].agg(["sum", "count"]).reset_index()
    out = out.rename(columns={"sum": "precip_mm", "count": "n"})
    out.loc[out["n"] < 3, "precip_mm"] = float("nan")  # hora com falhas
    return out.drop(columns="n")


def update_recent(out_root, days: int = 35, uf: str = "RS") -> Path:
    """Baixa os últimos ``days`` dias de todas as telemétricas da UF e salva por ano."""
    end = date.today()
    start = end - timedelta(days=days)
    st = list_stations(uf)
    if st.empty:
        raise RuntimeError("lista de estações telemétricas da ANA vazia")
    code_col = next(c for c in st.columns if c.lower().startswith("codestacao") or c.lower() == "codigo")
    name_col = next((c for c in st.columns if c.lower().startswith("nomeestacao")), code_col)
    frames = []
    for row in st.itertuples(index=False):
        code = str(getattr(row, code_col))
        try:
            h = to_hourly(fetch_station(code, start, end))
        except Exception as exc:
            log.warning("ANA %s: %s", code, exc)
            continue
        if h.empty:
            continue
        h["estacao"] = str(getattr(row, name_col)).upper()
        h["lat"] = getattr(row, "Latitude", None)
        h["lon"] = getattr(row, "Longitude", None)
        h["uf"] = uf
        frames.append(h)
    if not frames:
        raise RuntimeError("nenhum dado de chuva retornado pela ANA")
    new = pd.concat(frames, ignore_index=True)
    out = Path(out_root) / "ANA" / f"ANA_{uf}_{end.year}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = pd.read_csv(out)
        old["time_utc"] = pd.to_datetime(old["time_utc"], utc=True)
        new = pd.concat([old, new], ignore_index=True)
    new = new[new["time_utc"].dt.year == end.year]
    new = new.drop_duplicates(["codigo", "time_utc"], keep="last").sort_values(["codigo", "time_utc"])
    new.to_csv(out, index=False)
    log.info("ANA: %s, %d estações, até %s", out, new["codigo"].nunique(), new["time_utc"].max())
    return out
