"""Estações automáticas do INMET no RS.

Duas fontes, com o mesmo leitor:
- arquivos anuais do portal (``https://portal.inmet.gov.br/uploads/dadoshistoricos/AAAA.zip``),
  que já existem em E:/INMET até 2025 e são atualizados ao longo do ano corrente;
- a pasta local E:/INMET (só leitura).

Cada CSV do portal tem 8 linhas de cabeçalho com metadados da estação e depois
dados horários (UTC), separados por ';' e com vírgula decimal.
"""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

PORTAL_URL = "https://portal.inmet.gov.br/uploads/dadoshistoricos/{year}.zip"

_COLS = {
    "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": "precip_mm",
    "VENTO, VELOCIDADE HORARIA (m/s)": "wind_ms",
    "VENTO, RAJADA MAXIMA (m/s)": "gust_ms",
    "VENTO, DIREÇÃO HORARIA (gr) (° (gr))": "dir_deg",
}


def _parse_header(lines: list[str]) -> dict:
    meta = {}
    for ln in lines:
        if ":" in ln:
            k, v = ln.split(":", 1)
            meta[k.strip().upper()] = v.strip().strip(";").replace(",", ".")
    return {
        "codigo": meta.get("CODIGO (WMO)"),
        "estacao": meta.get("ESTACAO"),
        "uf": meta.get("UF"),
        "lat": float(meta["LATITUDE"]) if meta.get("LATITUDE") else None,
        "lon": float(meta["LONGITUDE"]) if meta.get("LONGITUDE") else None,
        "alt": float(meta["ALTITUDE"]) if meta.get("ALTITUDE") else None,
    }


def parse_csv(raw: bytes) -> tuple[dict, pd.DataFrame]:
    text = raw.decode("latin-1")
    lines = text.splitlines()
    meta = _parse_header(lines[:8])
    df = pd.read_csv(io.StringIO("\n".join(lines[8:])), sep=";", decimal=",", na_values=["-9999", ""])
    df.columns = [c.strip() for c in df.columns]
    date_col = next(c for c in df.columns if c.upper().startswith("DATA"))
    hour_col = next(c for c in df.columns if c.upper().startswith("HORA"))
    hour = df[hour_col].astype(str).str.replace(" UTC", "").str.replace(":", "").str.zfill(4)
    df["time_utc"] = pd.to_datetime(df[date_col].astype(str).str.replace("/", "-") + " " + hour, format="%Y-%m-%d %H%M", utc=True)
    keep = {c: v for c, v in _COLS.items() if c in df.columns}
    out = df[["time_utc", *keep]].rename(columns=keep)
    for k, v in meta.items():
        out[k] = v
    return meta, out


def read_zip(path_or_bytes, uf: str = "RS") -> pd.DataFrame:
    """Lê todas as estações de uma UF de um zip anual do portal."""
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    frames = []
    with zipfile.ZipFile(src) as zf:
        for name in zf.namelist():
            if f"_{uf}_" not in Path(name).name.upper() or not name.upper().endswith(".CSV"):
                continue
            try:
                _, df = parse_csv(zf.read(name))
                frames.append(df)
            except Exception as exc:
                log.warning("falha ao ler %s: %s", name, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def download_year(year: int, out_root: str | Path, uf: str = "RS", timeout: int = 600) -> Path:
    """Baixa o zip anual do portal e salva só a UF em parquet/CSV em ``out_root/INMET``."""
    import requests

    r = requests.get(PORTAL_URL.format(year=year), timeout=timeout)
    r.raise_for_status()
    df = read_zip(r.content, uf)
    out = Path(out_root) / "INMET" / f"INMET_{uf}_{year}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("INMET %s %s: %d registros, %d estações", uf, year, len(df), df["codigo"].nunique() if len(df) else 0)
    return out


def daily_precip(df: pd.DataFrame, day_start_hour_utc: int = 12) -> pd.DataFrame:
    """Chuva diária por estação.

    O dia pluviométrico padrão no Brasil vai de 12 UTC a 12 UTC (09h–09h local),
    rotulado pelo dia em que termina. Use ``day_start_hour_utc=0`` para o dia UTC,
    que é a convenção do IMERG e dos modelos.
    """
    d = df.dropna(subset=["precip_mm"]).copy()
    # o carimbo horário marca o FIM da hora acumulada
    shifted = d["time_utc"] - pd.Timedelta(hours=day_start_hour_utc) - pd.Timedelta(minutes=1)
    dia = shifted.dt.floor("D").dt.tz_localize(None)
    if day_start_hour_utc > 0:
        dia = dia + pd.Timedelta(days=1)
    d["dia"] = dia
    g = d.groupby(["codigo", "estacao", "lat", "lon", "dia"])
    out = g["precip_mm"].agg(precip_mm="sum", n_horas="count").reset_index()
    out.loc[out["n_horas"] < 22, "precip_mm"] = float("nan")  # dia com falhas
    return out
