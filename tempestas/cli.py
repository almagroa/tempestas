"""Linha de comando do Tempestas.

Exemplo:
    python -m tempestas evento --inicio 2026-11-03 --fim 2026-11-07
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from tempestas.config import load_config
from tempestas.verify.event import verify_event


def _cmd_evento(args) -> None:
    cfg = load_config(args.config)
    start, end = date.fromisoformat(args.inicio), date.fromisoformat(args.fim)
    result = verify_event(cfg, start, end, models=args.modelos)
    out_dir = Path(args.saida or cfg.paths["work_dir"] / "eventos" / f"{start}_{end}")
    out_dir.mkdir(parents=True, exist_ok=True)
    from tempestas.report.docx_report import build_event_report
    from tempestas.verify.wind import characterize_wind

    from tempestas.verify.daily import daily_verification

    wind = characterize_wind(cfg, start, end)
    daily = daily_verification(cfg, start, end)

    report = build_event_report(
        result, out_dir / f"Tempestas_evento_{start:%Y%m%d}_{end:%Y%m%d}.docx",
        lead=args.antecedencia, threshold=args.limiar, wind=wind, daily=daily,
    )
    print(f"Relatório: {report}\nMétricas: {report.with_suffix('.csv')}")
    if result.skipped:
        print("Fontes ignoradas:")
        for k, v in result.skipped.items():
            print(f"  {k}: {v}")


def _cmd_mensal(args) -> None:
    cfg = load_config(args.config)
    year, month = (int(x) for x in args.mes.split("-"))
    from tempestas.report.monthly_report import build_monthly_report
    from tempestas.verify.monthly import verify_month

    res = verify_month(cfg, year, month, models=args.modelos)
    out_dir = Path(args.saida or cfg.paths["work_dir"] / "mensal" / f"{year}-{month:02d}")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_monthly_report(
        res, out_dir / f"Tempestas_mensal_{year}{month:02d}.docx",
        daily_threshold=args.limiar_diario, event_threshold=args.limiar_evento,
    )
    print(f"Relatório: {report}")
    print(f"Eventos detectados: {', '.join(f'{s:%d/%m}–{e:%d/%m}' for s, e in res.events) or 'nenhum'}")
    if res.skipped:
        print("Fontes ignoradas:")
        for k, v in res.skipped.items():
            print(f"  {k}: {v}")


def _cmd_ponto(args) -> None:
    cfg = load_config(args.config)
    if args.mes:
        y, m = (int(v) for v in args.mes.split("-"))
        start = date(y, m, 1)
        end = (pd.Timestamp(start) + pd.offsets.MonthEnd(0)).date()
    else:
        start, end = date.fromisoformat(args.inicio), date.fromisoformat(args.fim)
    from tempestas.report.maps import point_chart
    from tempestas.verify.point import point_analysis

    res = point_analysis(cfg, args.nome, start, end, lat=args.lat, lon=args.lon, radius_km=args.raio, models=args.modelos)
    slug = args.nome.lower().replace(" ", "_")
    out_dir = Path(args.saida or cfg.paths["work_dir"] / "pontos" / f"{slug}_{start:%Y%m%d}_{end:%Y%m%d}")
    out_dir.mkdir(parents=True, exist_ok=True)
    png = point_chart(res, out_dir / f"{slug}_leque.png")
    summ = res.summary()
    summ.round(1).to_csv(out_dir / f"{slug}_diario.csv", index_label="dia")
    res.members.round(1).to_csv(out_dir / f"{slug}_previsoes.csv", index_label="dia")
    if not res.stations.empty:
        res.stations.to_csv(out_dir / f"{slug}_estacoes.csv", index=False)
        st = res.stations.groupby(["rede", "codigo", "estacao"]).agg(dist_km=("dist_km", "first"), dias=("dia", "nunique"),
                                                                    total_mm=("precip_mm", "sum"))
        print("Estações usadas:\n" + st.round(1).to_string())
    print(f"Gráfico: {png}\nTabela diária: {out_dir / (slug + '_diario.csv')}")
    for col in [c for c in summ.columns if c.startswith("pos_")]:
        counts = summ[col][summ[col] != ""].value_counts()
        print(f"{col[4:]}: " + ", ".join(f"{k} {v} dias" for k, v in counts.items()))
    if res.skipped:
        print("Fontes ignoradas:")
        for k, v in res.skipped.items():
            print(f"  {k}: {v}")


def _cmd_baixar(args) -> None:
    cfg = load_config(args.config)
    root = Path(args.saida or cfg.paths["work_dir"] / "dados")
    today = date.today()
    desde = date.fromisoformat(args.desde) if args.desde else None
    recent_days = (today - desde).days + 1 if desde else 35
    if "era5" in args.fontes:
        from tempestas.download import era5
        lag = 5
        dias = max(args.dias, (today - timedelta(days=lag) - desde).days + 1) if desde else args.dias
        era5.update(root, days_back=dias, lag_days=lag)
    if "metar" in args.fontes:
        from tempestas.download import metar
        first = desde or (today.replace(day=1) - timedelta(days=1) if today.day <= 7 else today)
        for p in pd.period_range(first.replace(day=1), today, freq="M"):
            try:
                metar.update_month(root, p.year, p.month)
            except Exception as exc:
                logging.warning("METAR %s: %s", p, exc)
    if "inmet" in args.fontes:
        from tempestas.download import inmet
        try:
            inmet.download_year(today.year, root)
        except Exception as exc:
            logging.warning("INMET portal: %s", exc)
        try:
            inmet.update_recent(root, days=recent_days)  # completa os dias recentes pela API
        except Exception as exc:
            logging.warning("INMET API: %s", exc)
    if "ana" in args.fontes:
        from tempestas.download import ana
        try:
            ana.update_recent(root, days=recent_days)
        except Exception as exc:
            logging.warning("ANA: %s", exc)
    if "cemaden" in args.fontes:
        from tempestas.download import cemaden
        try:
            cemaden.update_recent(root, days=recent_days)
        except Exception as exc:
            logging.warning("Cemaden: %s", exc)
    print(f"Downloads concluídos em {root}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tempestas", description="Verificação de eventos extremos de chuva")
    parser.add_argument("--config", help="arquivo YAML de configuração")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("evento", help="verifica previsto × observado num período")
    ev.add_argument("--inicio", required=True, help="primeiro dia (AAAA-MM-DD)")
    ev.add_argument("--fim", required=True, help="último dia (AAAA-MM-DD)")
    ev.add_argument("--modelos", nargs="*", help="subconjunto de modelos")
    ev.add_argument("--saida", help="pasta de saída")
    ev.add_argument("--antecedencia", type=int, default=1, help="antecedência (dias) usada nos mapas e tabelas")
    ev.add_argument("--limiar", type=float, default=50.0, help="limiar (mm no período) das métricas categóricas")
    ev.set_defaults(func=_cmd_evento)

    me = sub.add_parser("mensal", help="relatório mensal: chuva observada, eventos, desempenho diário e vento")
    me.add_argument("--mes", required=True, help="mês (AAAA-MM)")
    me.add_argument("--modelos", nargs="*", help="subconjunto de modelos")
    me.add_argument("--saida", help="pasta de saída")
    me.add_argument("--limiar-diario", type=float, default=25.0, help="limiar diário (mm) das métricas categóricas")
    me.add_argument("--limiar-evento", type=float, default=50.0, help="limiar (mm no evento) das métricas categóricas")
    me.set_defaults(func=_cmd_mensal)

    po = sub.add_parser("ponto", help="leque das previsões × chuva observada num município/ponto")
    po.add_argument("--nome", default="Porto Alegre")
    po.add_argument("--mes", help="mês (AAAA-MM); ou use --inicio/--fim")
    po.add_argument("--inicio")
    po.add_argument("--fim")
    po.add_argument("--lat", type=float)
    po.add_argument("--lon", type=float)
    po.add_argument("--raio", type=float, default=15.0, help="raio (km) para as estações")
    po.add_argument("--modelos", nargs="*")
    po.add_argument("--saida")
    po.set_defaults(func=_cmd_ponto)

    bx = sub.add_parser("baixar", help="atualiza ERA5/ERA5-Land, METAR e INMET")
    bx.add_argument("--fontes", nargs="*", default=["era5", "metar", "inmet", "ana", "cemaden"], choices=["era5", "metar", "inmet", "ana", "cemaden"])
    bx.add_argument("--dias", type=int, default=15, help="janela de dias do ERA5 a garantir")
    bx.add_argument("--desde", help="baixar desde esta data (AAAA-MM-DD), para preencher eventos passados")
    bx.add_argument("--saida", help="pasta de dados (padrão: <work_dir>/dados)")
    bx.set_defaults(func=_cmd_baixar)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
