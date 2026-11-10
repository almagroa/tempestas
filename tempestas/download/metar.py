"""Observações de aeroportos (METAR) do RS pelo arquivo do Iowa Environmental Mesonet.

Gratuito, sem chave. Usado principalmente para vento médio e rajada; o METAR
brasileiro raramente informa a quantidade de chuva.
"""
from __future__ import annotations

import io
import logging
from datetime import date
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Aeroportos do RS com METAR
RS_AIRPORTS = {
    "SBPA": "Porto Alegre",
    "SBCO": "Canoas",
    "SBCX": "Caxias do Sul",
    "SBSM": "Santa Maria",
    "SBPF": "Passo Fundo",
    "SBPK": "Pelotas",
    "SBRG": "Rio Grande",
    "SBUG": "Uruguaiana",
    "SBBG": "Bagé",
    "SBNM": "Santo Ângelo",
}

KT_TO_MS = 0.514444


def fetch(start: date, end: date, stations=None, timeout: int = 120) -> pd.DataFrame:
    """Vento (m/s), direção e rajada por aeroporto, em UTC."""
    import requests

    stations = list(stations or RS_AIRPORTS)
    params = [("station", s) for s in stations] + [
        ("data", "drct"), ("data", "sknt"), ("data", "gust"), ("data", "p01m"), ("data", "metar"),
        ("year1", start.year), ("month1", start.month), ("day1", start.day),
        ("year2", end.year), ("month2", end.month), ("day2", end.day),
        ("tz", "Etc/UTC"), ("format", "onlycomma"), ("missing", "M"), ("trace", "T"),
        ("report_type", "3"), ("report_type", "4"),
    ]
    r = requests.get(IEM_URL, params=params, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), na_values=["M"])
    if df.empty:
        return df
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    for col in ("sknt", "gust"):
        df[col] = pd.to_numeric(df[col], errors="coerce") * KT_TO_MS
    df = df.rename(columns={"station": "icao", "valid": "time_utc", "drct": "dir_deg", "sknt": "wind_ms", "gust": "gust_ms"})
    df["nome"] = df["icao"].map(RS_AIRPORTS)
    return df


def update_month(out_root: str | Path, year: int, month: int) -> Path:
    start = date(year, month, 1)
    end = (pd.Timestamp(start) + pd.offsets.MonthEnd(0)).date()
    end = min(end, date.today())
    out = Path(out_root) / "METAR" / f"METAR_RS_{year}{month:02d}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = fetch(start, end + pd.Timedelta(days=1).to_pytimedelta())
    df.to_csv(out, index=False)
    log.info("METAR %s-%02d: %d registros", year, month, len(df))
    return out


def weather_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Horas com chuva e com trovoada por aeroporto, a partir do tempo presente do METAR.

    O METAR brasileiro quase nunca informa a quantidade de chuva, mas informa o
    fenômeno (RA, SHRA, TSRA, +RA etc.). Serve como indicação qualitativa.
    """
    d = df.dropna(subset=["metar"]).copy()
    d["hora"] = d["time_utc"].dt.floor("h")
    txt = d["metar"].str.upper()
    d["chuva"] = txt.str.contains(r"(?:^|\s)[+-]?(?:VC)?(?:TS|SH)?RA\b|(?:^|\s)[+-]?DZ\b", regex=True)
    d["forte"] = txt.str.contains(r"(?:^|\s)\+(?:TS|SH)?RA\b", regex=True)
    d["trovoada"] = txt.str.contains(r"(?:^|\s)[+-]?(?:VC)?TS", regex=True)
    h = d.groupby(["icao", "nome", "hora"])[["chuva", "forte", "trovoada"]].any().reset_index()
    return h.groupby(["icao", "nome"])[["chuva", "forte", "trovoada"]].sum().reset_index().rename(
        columns={"chuva": "horas_chuva", "forte": "horas_chuva_forte", "trovoada": "horas_trovoada"})
