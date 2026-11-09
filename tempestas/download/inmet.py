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


# ---------------------------------------------------------------------------
# Dados recentes pela API do INMET (o zip do portal atrasa ~1 semana a 1 mês)
# ---------------------------------------------------------------------------
API_BASE = "https://apitempo.inmet.gov.br"


def _api_get(path: str, token: str | None, timeout: int = 120):
    import requests

    url = f"{API_BASE}/token{path}/{token}" if token else f"{API_BASE}{path}"
    r = requests.get(url, timeout=timeout)
    if r.status_code == 204:
        return []
    r.raise_for_status()
    try:
        return r.json()
    except ValueError:
        raise RuntimeError(f"API do INMET não devolveu JSON: {r.text[:200]!r}") from None


def api_stations(uf: str = "RS", token: str | None = None) -> pd.DataFrame:
    """Estações automáticas operantes de uma UF."""
    import requests

    r = requests.get(f"{API_BASE}/estacoes/T", timeout=120)
    r.raise_for_status()
    try:
        payload = r.json()
    except ValueError:
        raise RuntimeError(f"API do INMET não devolveu JSON: {r.text[:200]!r}") from None
    st = pd.DataFrame(payload)
    return st[st["SG_ESTADO"] == uf]


def api_fetch(start, end, uf: str = "RS", token: str | None = None) -> pd.DataFrame:
    """Dados horários de todas as estações automáticas da UF entre start e end."""
    st = api_stations(uf, token)
    frames = []
    for code in st["CD_ESTACAO"]:
        try:
            data = _api_get(f"/estacao/{start:%Y-%m-%d}/{end:%Y-%m-%d}/{code}", token)
        except Exception as exc:
            log.warning("INMET API %s: %s", code, exc)
            continue
        if data:
            frames.append(pd.DataFrame(data))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    hour = df["HR_MEDICAO"].astype(str).str.zfill(4)
    out = pd.DataFrame({
        "time_utc": pd.to_datetime(df["DT_MEDICAO"] + " " + hour, format="%Y-%m-%d %H%M", utc=True),
        "precip_mm": pd.to_numeric(df.get("CHUVA"), errors="coerce"),
        "wind_ms": pd.to_numeric(df.get("VEN_VEL"), errors="coerce"),
        "gust_ms": pd.to_numeric(df.get("VEN_RAJ"), errors="coerce"),
        "dir_deg": pd.to_numeric(df.get("VEN_DIR"), errors="coerce"),
        "codigo": df["CD_ESTACAO"],
        "estacao": df["DC_NOME"].str.upper(),
        "uf": df["UF"],
        "lat": pd.to_numeric(df["VL_LATITUDE"], errors="coerce"),
        "lon": pd.to_numeric(df["VL_LONGITUDE"], errors="coerce"),
        "alt": pd.to_numeric(df.get("VL_ALTITUDE"), errors="coerce"),
    })
    return out


def update_recent(out_root, days: int = 35, uf: str = "RS", token: str | None = None) -> Path:
    """Completa o CSV do ano corrente com os últimos ``days`` dias da API.

    Junta com o que veio do portal e remove duplicatas (estação × hora).
    O token (opcional) pode vir da variável de ambiente INMET_TOKEN.
    """
    import os
    from datetime import date, timedelta

    token = token or os.environ.get("INMET_TOKEN")
    end = date.today()
    start = end - timedelta(days=days)
    new = api_fetch(start, end, uf, token)
    out = Path(out_root) / "INMET" / f"INMET_{uf}_{end.year}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = pd.read_csv(out)
        old["time_utc"] = pd.to_datetime(old["time_utc"], utc=True)
        new = pd.concat([old, new], ignore_index=True)
    new = new[new["time_utc"].dt.year == end.year]
    new = new.drop_duplicates(["codigo", "time_utc"], keep="last").sort_values(["codigo", "time_utc"])
    new.to_csv(out, index=False)
    log.info("INMET API: %s atualizado até %s", out, new["time_utc"].max())
    return out
