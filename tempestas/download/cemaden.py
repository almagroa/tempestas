"""Pluviômetros automáticos do Cemaden (rede 11) pela API PED.

Credenciais ficam FORA do repositório, na pasta do usuário:
- ``%USERPROFILE%/.tempestas/cemaden_token.txt``: token (JWT) já gerado; ou
- ``%USERPROFILE%/.tempestas/cemaden.json``: {"email": ..., "password": ...},
  usado para gerar o token automaticamente no SGAA.

Endpoints:
- POST https://sgaa.cemaden.gov.br/SGAA/rest/controle-token/tokens  (gera token)
- GET  https://sws.cemaden.gov.br/PED/rest/pcds/dados_rede?uf=RS&inicio=AAAAMMDDHHMM&fim=...&rede=11
  com o token no header ``token``.
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

SGAA_TOKEN = "https://sgaa.cemaden.gov.br/SGAA/rest/controle-token/tokens"
PED = "https://sws.cemaden.gov.br/PED/rest"
CRED_DIR = Path.home() / ".tempestas"

# nomes possíveis dos campos na resposta (a API não documenta o esquema)
_FIELD_CANDIDATES = {
    "codigo": ["codestacao", "cod_estacao", "codigo", "id_estacao", "codEstacao"],
    "estacao": ["nomeestacao", "nome", "estacao", "nomeEstacao"],
    "lat": ["latitude", "lat"],
    "lon": ["longitude", "lon", "long"],
    "time": ["datahora", "data_hora", "dataHora", "datahoraleitura", "data"],
    "precip_mm": ["valormedida", "valor", "chuva", "valorMedida", "medida"],
    "sensor": ["id_sensor", "sensor", "idsensor"],
    "municipio": ["municipio", "cidade"],
}


def get_token(timeout: int = 60) -> str:
    tok_file = CRED_DIR / "cemaden_token.txt"
    cred_file = CRED_DIR / "cemaden.json"
    if cred_file.exists():
        import requests

        cred = json.loads(cred_file.read_text(encoding="utf-8"))
        r = requests.post(SGAA_TOKEN, json={"email": cred["email"], "password": cred["password"]}, timeout=timeout)
        r.raise_for_status()
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
        token = body.get("token") if isinstance(body, dict) else str(body).strip().strip('"')
        if not token:
            raise RuntimeError("SGAA não devolveu token")
        return token
    if tok_file.exists():
        return tok_file.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"sem credenciais do Cemaden em {CRED_DIR} (cemaden_token.txt ou cemaden.json)")


def _pick(df: pd.DataFrame, key: str) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in _FIELD_CANDIDATES[key]:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def normalize(records, tz_offset_hours: float = 0.0) -> pd.DataFrame:
    """Converte a resposta da API para o formato do Tempestas (time_utc, precip_mm, codigo, estacao, lat, lon).

    ``tz_offset_hours``: deslocamento do horário da API em relação ao UTC
    (0 se já vier em UTC; −3 se vier em horário de Brasília).
    """
    df = pd.json_normalize(records) if not isinstance(records, pd.DataFrame) else records
    if df.empty:
        return df
    m = {k: _pick(df, k) for k in _FIELD_CANDIDATES}
    missing = [k for k in ("codigo", "time", "precip_mm") if m[k] is None]
    if missing:
        raise KeyError(f"campos não encontrados na resposta do Cemaden: {missing}; colunas: {list(df.columns)}")
    if m["sensor"] is not None:
        df = df[df[m["sensor"]].astype(str).str.strip().isin(["10", "10.0"])]  # 10 = chuva
        if df.empty:
            return pd.DataFrame()
    t = pd.to_datetime(df[m["time"]], errors="coerce")
    if getattr(t.dt, "tz", None) is None:
        t = (t - pd.Timedelta(hours=tz_offset_hours)).dt.tz_localize("UTC")
    else:
        t = t.dt.tz_convert("UTC")
    out = pd.DataFrame({
        "time_utc": t,
        "precip_mm": pd.to_numeric(df[m["precip_mm"]].astype(str).str.replace(",", "."), errors="coerce"),
        "codigo": df[m["codigo"]].astype(str),
        "estacao": (df[m["estacao"]] if m["estacao"] else df[m["municipio"]] if m["municipio"] else df[m["codigo"]]).astype(str).str.upper(),
        "lat": pd.to_numeric(df[m["lat"]].astype(str).str.replace(",", "."), errors="coerce") if m["lat"] else None,
        "lon": pd.to_numeric(df[m["lon"]].astype(str).str.replace(",", "."), errors="coerce") if m["lon"] else None,
    })
    return out.dropna(subset=["time_utc"])


def fetch(start: date, end: date, uf: str = "RS", token: str | None = None, timeout: int = 300):
    """Registros crus da rede Cemaden na UF, de start 00:00 a end+1 00:00."""
    import requests

    token = token or get_token()
    params = {
        "uf": uf, "rede": "11", "sensor": "10",  # 10 = chuva
        "inicio": f"{start:%Y%m%d}0000",
        "fim": f"{end + timedelta(days=1):%Y%m%d}0000",
    }
    r = requests.get(f"{PED}/pcds/dados_rede", params=params, headers={"token": token}, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return parse_body(r.text)


def parse_body(text: str):
    """A API devolve 'string': pode ser JSON ou texto delimitado (CSV). Aceita os dois."""
    import io

    text = text.strip()
    if not text:
        return pd.DataFrame()
    if text[0] in "[{":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("data", data.get("dados", [data]))
        return pd.json_normalize(data)
    lines = text.splitlines()
    sep = ";" if lines[0].count(";") >= lines[0].count(",") else ","
    try:
        return pd.read_csv(io.StringIO(text), sep=sep, dtype=str)
    except Exception:
        raise RuntimeError(f"resposta não reconhecida: {text[:300]!r}") from None


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Leituras sub-horárias → total horário (carimbo no fim da hora)."""
    if df.empty:
        return df
    d = df.copy()
    d["time_utc"] = d["time_utc"].dt.ceil("h")
    keys = ["codigo", "estacao", "lat", "lon", "time_utc"]
    return d.groupby(keys, dropna=False)["precip_mm"].sum(min_count=1).reset_index()


def update_recent(out_root, days: int = 35, uf: str = "RS", tz_offset_hours: float = 0.0) -> Path:
    """Baixa os últimos ``days`` dias, um pedido por dia (limite da API), e salva/une por ano."""
    end = date.today()
    token = get_token()
    frames = []
    s = end - timedelta(days=days)
    while s <= end:
        e = s  # dados_rede aceita no máximo 1 dia por pedido
        try:
            frames.append(to_hourly(normalize(fetch(s, e, uf, token), tz_offset_hours)))
        except Exception as exc:
            log.warning("Cemaden %s–%s: %s", s, e, exc)
        s = e + timedelta(days=1)
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError("nenhum dado retornado pelo Cemaden")
    new = pd.concat(frames, ignore_index=True)
    new["uf"] = uf
    out = Path(out_root) / "CEMADEN" / f"CEMADEN_{uf}_{end.year}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = pd.read_csv(out, dtype={"codigo": str})
        old["time_utc"] = pd.to_datetime(old["time_utc"], utc=True)
        new = pd.concat([old, new], ignore_index=True)
    new = new[new["time_utc"].dt.year == end.year]
    new = new.drop_duplicates(["codigo", "time_utc"], keep="last").sort_values(["codigo", "time_utc"])
    new.to_csv(out, index=False)
    log.info("Cemaden: %s, %d estações, até %s", out, new["codigo"].nunique(), new["time_utc"].max())
    return out
