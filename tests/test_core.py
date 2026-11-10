from datetime import date

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tempestas.config import Domain, load_config
from tempestas.io.forecasts import forecast_for_period
from tempestas.io.standardize import regrid, standardize, target_grid
from tempestas.verify.metrics import categorical, continuous

DOM = Domain("teste", -34.0, -27.0, -58.0, -49.0)


def _field(lat, lon, times, value, lon360=False, lat_desc=False, units="mm"):
    lat = np.array(lat)[::-1] if lat_desc else np.array(lat)
    lon = np.array(lon) % 360 if lon360 else np.array(lon)
    data = np.full((len(times), lat.size, lon.size), value, dtype="float32")
    return xr.DataArray(data, coords={"time": times, "latitude": lat, "longitude": lon},
                        dims=("time", "latitude", "longitude"), attrs={"units": units})


def test_standardize_conventions():
    times = pd.date_range("2026-11-03 23:59:59", periods=2, freq="D")
    da = _field(np.arange(-36, -25, 0.25), np.arange(-60, -47, 0.25), times, 0.01, lon360=True, lat_desc=True, units="m")
    out = standardize(da, domain=DOM)
    assert out.dims == ("time", "lat", "lon")
    assert float(out.lon.min()) >= -58 and float(out.lon.max()) <= -49
    assert np.all(np.diff(out.lat) > 0)
    assert np.allclose(out.values, 10.0)  # m -> mm
    assert pd.Timestamp(out.time.values[0]) == pd.Timestamp("2026-11-03")


def test_regrid_coarsen_preserves_mean():
    lats, lons = target_grid(DOM, 0.1)
    fine = xr.DataArray(np.random.default_rng(0).gamma(1, 10, (200, 260)).astype("float32"),
                        coords={"lat": np.linspace(-34.5, -26.5, 200), "lon": np.linspace(-58.5, -48.5, 260)},
                        dims=("lat", "lon"))
    out = regrid(fine, lats, lons)
    assert out.shape == (lats.size, lons.size)
    inner = fine.sel(lat=slice(-34, -27), lon=slice(-58, -49))
    assert abs(float(out.mean()) - float(inner.mean())) < 0.5


def test_metrics_perfect_and_known():
    ob = xr.DataArray(np.array([[0, 30], [60, 120]], dtype=float), dims=("lat", "lon"))
    c = continuous(ob, ob)
    assert c["bias_mm"] == 0 and c["rmse"] == 0
    fc = xr.DataArray(np.array([[30, 0], [60, 10]], dtype=float), dims=("lat", "lon"))
    cat = categorical(fc, ob, 25)
    # obs>=25: (0,1),(1,0),(1,1); fc>=25: (0,0),(1,0)
    assert (cat["hits"], cat["misses"], cat["false_alarms"]) == (1, 2, 1)
    assert cat["pod"] == pytest.approx(1 / 3)
    assert cat["far"] == pytest.approx(0.5)
    assert cat["csi"] == pytest.approx(0.25)


def test_forecast_for_period(tmp_path):
    cfg = load_config(overrides={"paths": {"forecast_root": str(tmp_path)}})
    lat, lon = np.arange(-35, -26, 0.25), np.arange(-59, -48, 0.25)
    for init in ("2026-11-03", "2026-11-04", "2026-11-06"):
        times = pd.date_range(f"{init} 23:59:59", periods=7, freq="D")
        d = tmp_path / "ICON" / init
        d.mkdir(parents=True)
        (_field(lat, lon, times, 5.0).cumsum("time")).to_dataset(name="tp").to_netcdf(d / "icon_forecast.nc")
    out = forecast_for_period(cfg, "ICON", date(2026, 11, 6), date(2026, 11, 8), max_lead=7)
    assert list(out.lead_start.values) == [4, 3, 1]
    assert np.allclose(out.values, 15.0)


def test_deaccumulate():
    from tempestas.io.forecasts import deaccumulate
    times = pd.date_range("2026-11-03", periods=3, freq="D")
    da = xr.DataArray(np.array([2.0, 5.0, 5.0])[:, None, None] * np.ones((3, 2, 2)),
                      coords={"time": times, "lat": [0, 1], "lon": [0, 1]}, dims=("time", "lat", "lon"))
    out = deaccumulate(da)
    assert np.allclose(out.isel(lat=0, lon=0).values, [2, 3, 0])


def test_inmet_parser_and_daily():
    from pathlib import Path
    from tempestas.download.inmet import daily_precip, parse_csv
    raw = (Path(__file__).parent / "dados" / "inmet_A801_amostra.csv").read_bytes()
    meta, df = parse_csv(raw)
    assert meta["codigo"] == "A801" and meta["uf"] == "RS"
    assert len(df) == 72 and df["time_utc"].dt.tz is not None
    d_utc = daily_precip(df, day_start_hour_utc=0)
    # 00 UTC do dia 1 pertence ao dia 31/12 (incompleto); 02/01 completo
    assert d_utc.set_index("dia").loc["2025-01-02", "n_horas"] == 24


def test_point_table_nearest():
    from tempestas.verify.metrics import point_table
    lats, lons = np.array([-30.05, -29.95]), np.array([-51.15, -51.05])
    fld = xr.DataArray([[10.0, 20.0], [30.0, 60.0]], coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))
    st = pd.DataFrame({"lat": [-30.04, -29.96], "lon": [-51.16, -51.04], "obs_mm": [10.0, 50.0]})
    t = point_table(st, {"IMERG": fld}, [25])
    row = t.iloc[0]
    assert row.n_stations == 2 and row.bias_mm == pytest.approx(5.0)
    assert (row.hits, row.misses, row.false_alarms) == (1, 0, 0)


def test_ana_xml_and_hourly():
    from tempestas.download.ana import _rows, to_hourly
    xml = """<?xml version="1.0"?><DataTable><diffgr:diffgram xmlns:diffgr="urn:x"><DocumentElement>
    <DadosHidrometereologicos><CodEstacao>1</CodEstacao><DataHora>2026-11-03 10:15:00</DataHora><Chuva>1,5</Chuva></DadosHidrometereologicos>
    <DadosHidrometereologicos><CodEstacao>1</CodEstacao><DataHora>2026-11-03 10:30:00</DataHora><Chuva>0.5</Chuva></DadosHidrometereologicos>
    </DocumentElement></diffgr:diffgram></DataTable>"""
    rows = _rows(xml, "DadosHidrometereologicos")
    assert len(rows) == 2 and rows[0]["Chuva"] == "1,5"
    local = pd.to_datetime([r["DataHora"] for r in rows])
    df = pd.DataFrame({"time_utc": (local + pd.Timedelta(hours=3)).tz_localize("UTC"),
                       "precip_mm": [1.5, 0.5], "codigo": "1"})
    h = to_hourly(df)
    assert len(h) == 1 and pd.Timestamp(h.time_utc.iloc[0]) == pd.Timestamp("2026-11-03 14:00", tz="UTC")
    assert np.isnan(h.precip_mm.iloc[0])  # só 2 registros de 15 min: hora incompleta


def test_cemaden_normalize():
    from tempestas.download.cemaden import normalize, to_hourly
    recs = [{"codestacao": "431490201A", "nomeestacao": "Centro", "latitude": "-30.0", "longitude": "-51.2",
             "datahora": "2026-11-03 10:10:00", "valormedida": "1.2"},
            {"codestacao": "431490201A", "nomeestacao": "Centro", "latitude": "-30.0", "longitude": "-51.2",
             "datahora": "2026-11-03 10:20:00", "valormedida": "0.8"}]
    df = normalize(recs)
    assert list(df.columns[:3]) == ["time_utc", "precip_mm", "codigo"]
    h = to_hourly(df)
    assert len(h) == 1 and h.precip_mm.iloc[0] == pytest.approx(2.0)
