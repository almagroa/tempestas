"""Linha de comando do Tempestas.

Exemplo:
    python -m tempestas evento --inicio 2026-11-03 --fim 2026-11-07
"""
from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from tempestas.config import load_config
from tempestas.verify.event import verify_event


def _cmd_evento(args) -> None:
    cfg = load_config(args.config)
    start, end = date.fromisoformat(args.inicio), date.fromisoformat(args.fim)
    result = verify_event(cfg, start, end, models=args.modelos)
    out_dir = Path(args.saida or cfg.paths["work_dir"] / "eventos" / f"{start}_{end}")
    out_dir.mkdir(parents=True, exist_ok=True)
    result.scores.to_csv(out_dir / "metricas.csv", index=False, encoding="utf-8")
    print(f"Métricas salvas em {out_dir / 'metricas.csv'}")
    if result.skipped:
        print("Fontes ignoradas:")
        for k, v in result.skipped.items():
            print(f"  {k}: {v}")


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
    ev.set_defaults(func=_cmd_evento)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
