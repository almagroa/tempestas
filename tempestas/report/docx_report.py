"""Relatório técnico (.docx) de um evento ou período: previsto × observado.

O texto é descritivo e técnico. Não menciona a finalidade do uso dos dados.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

from tempestas.mask import DEFAULT_REGIONS
from tempestas.report.maps import event_maps, skill_by_lead, wind_maps
from tempestas.verify.event import EventResult
from tempestas.verify.wind import WindResult

MODEL_INFO = {
    "GEFS": "NOAA/NCEP – ensemble GEFS (média), 0,25°, até 16 dias",
    "IFS": "ECMWF – modelo determinístico IFS, 0,25°, até 15 dias",
    "AIFS": "ECMWF – modelo de inteligência artificial AIFS, 0,25°, até 15 dias",
    "ICON": "DWD – modelo global ICON, 0,25°, até 7 dias",
    "CPT-WRF": "WRF regional (CPTEC), ~7 km, até 7 dias",
    "CHIRPS-GEFS": "UCSB/CHC – GEFS corrigido por CHIRPS, 0,05°, até 16 dias",
}
OBS_INFO = {
    "IMERG": "NASA GPM IMERG Early v07, estimativa por satélite, 0,1°, diário",
    "INMET": "Estações automáticas do INMET (pluviômetro), horário",
}

BLUE = RGBColor(0x1F, 0x3A, 0x5F)


def _fmt_date(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def _num(x, nd=0, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return s + ("%" if pct else "")


def _heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        r.font.color.rgb = BLUE
    return h


def _table(doc, header, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = h
        cell.paragraphs[0].runs[0].font.bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
    for row in t.rows:
        for i, cell in enumerate(row.cells):
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.font.size = Pt(8.5)
            if widths:
                cell.width = Cm(widths[i])
    return t


def _best_lead_rows(scores: pd.DataFrame, lead: int) -> pd.DataFrame:
    """Para cada modelo, a rodada com a menor antecedência ≥ lead."""
    s = scores[scores.lead_start >= lead]
    idx = s.groupby("model")["lead_start"].transform("min") == s["lead_start"]
    return s[idx]


def _synthesis(scores: pd.DataFrame, obs: str, thr: float, lead: int) -> list[str]:
    s = _best_lead_rows(scores, lead)
    s = s[(s.region == "RS") & (s.observed == obs) & (s.threshold_mm == thr)]
    if s.empty:
        return []
    obs_mean = s["obs_mean"].iloc[0]
    out = [f"A chuva média observada ({obs}) no RS no período foi de {_num(obs_mean, 0)} mm."]
    under = s[s.bias_pct <= -20]
    over = s[s.bias_pct >= 20]
    ok = s[(s.bias_pct > -20) & (s.bias_pct < 20)]
    if len(ok):
        out.append("Previram volume médio próximo ao observado (diferença inferior a 20%): " + ", ".join(ok.model) + ".")
    if len(under):
        out.append("Subestimaram o volume em 20% ou mais: " + ", ".join(f"{r.model} ({_num(r.bias_pct, 0, True)})" for r in under.itertuples()) + ".")
    if len(over):
        out.append("Superestimaram o volume em 20% ou mais: " + ", ".join(f"{r.model} (+{_num(r.bias_pct, 0, True)})" for r in over.itertuples()) + ".")
    best = s.sort_values("csi", ascending=False).iloc[0]
    out.append(
        f"Para acumulados de pelo menos {thr:g} mm, o melhor índice CSI foi do {best.model} "
        f"({_num(best.csi, 2)}), com POD de {_num(best.pod, 2)} e FAR de {_num(best.far, 2)}."
    )
    return out


def build_event_report(
    result: EventResult,
    out_path: str | Path,
    client: str = "Corsan – Companhia Riograndense de Saneamento",
    lead: int = 1,
    threshold: float = 50.0,
    title: str | None = None,
    wind: WindResult | None = None,
    responsible: str = "André Almagro – Eng. Ambiental, Dr. em Hidrologia – CREA 20072-MS",
) -> Path:
    out_path = Path(out_path)
    fig_dir = out_path.parent / "figuras"
    fig_dir.mkdir(parents=True, exist_ok=True)
    obs_ref = next(iter(result.observed))
    period = f"{_fmt_date(result.start)} a {_fmt_date(result.end)}"
    n_days = (result.end - result.start).days + 1

    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.0)
        s.top_margin = s.bottom_margin = Cm(1.8)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)

    t = doc.add_paragraph()
    r = t.add_run(title or "Análise de evento de chuva – previsto × observado")
    r.bold, r.font.size, r.font.color.rgb = True, Pt(16), BLUE
    p = doc.add_paragraph()
    p.add_run("Tempestas · HidroNexus Engenharia").italic = True
    doc.add_paragraph(f"Cliente: {client}\nÁrea: Rio Grande do Sul\nPeríodo analisado: {period} ({n_days} dias)\n"
                      f"Emissão: {_fmt_date(date.today())}")

    _heading(doc, "1. Objetivo")
    doc.add_paragraph(
        "Comparar a precipitação prevista pelos modelos numéricos de previsão do tempo com a precipitação "
        "observada no Rio Grande do Sul no período indicado, quantificando a diferença entre o previsto e o "
        "ocorrido para o estado e para cada região hidrográfica (Guaíba, Litoral e Uruguai)."
    )

    _heading(doc, "2. Dados e método")
    doc.add_paragraph("Previsões utilizadas (rodadas diárias arquivadas pela HidroNexus):")
    for m in result.forecasts:
        doc.add_paragraph(f"{m}: {MODEL_INFO.get(m, '')}", style="List Bullet")
    doc.add_paragraph("Precipitação observada (cada produto é avaliado separadamente, sem fusão):")
    for o in result.observed:
        doc.add_paragraph(f"{o}: {OBS_INFO.get(o, '')}", style="List Bullet")
    doc.add_paragraph(
        "Todos os campos foram convertidos para totais diários (dia UTC) e reamostrados para uma grade comum "
        "de 0,1°. Para cada modelo, comparou-se o total previsto para o período inteiro com o total observado. "
        "A antecedência indica quantos dias antes do início do período a rodada foi emitida (1 = rodada do "
        "próprio dia de início). Métricas: viés (média de previsto − observado, em mm e em %), erro quadrático "
        "médio (RMSE), correlação espacial e, para limiares de acumulado, probabilidade de detecção (POD), "
        "razão de alarmes falsos (FAR) e índice de sucesso crítico (CSI)."
    )
    if result.stations is not None:
        doc.add_paragraph(f"INMET: {OBS_INFO['INMET']}; {len(result.stations)} estações com registro completo.", style="List Bullet")
    if result.skipped:
        doc.add_paragraph("Fontes sem dados para o período: " + "; ".join(result.skipped) + ".")

    _heading(doc, "3. Síntese")
    for line in _synthesis(result.scores, obs_ref, threshold, lead):
        doc.add_paragraph(line, style="List Bullet")

    _heading(doc, "4. Mapas")
    map_png = event_maps(
        result.observed, result.forecasts, fig_dir / "mapas.png", lead_start=lead, reference=obs_ref,
        outline_path=DEFAULT_REGIONS, mask=result.masks["RS"], suptitle=f"Chuva acumulada de {period}",
    )
    doc.add_picture(str(map_png), width=Cm(17))
    doc.add_paragraph(
        f"Linha 1: chuva observada. Linha 2: chuva prevista por modelo. Linha 3: diferença previsto − {obs_ref} "
        "(marrom = previu menos que o observado; verde = previu mais)."
    ).runs[0].italic = True

    _heading(doc, "5. Desempenho por região")
    s = _best_lead_rows(result.scores, lead)
    for obs in result.observed:
        sub = s[(s.observed == obs) & (s.threshold_mm == threshold)]
        doc.add_paragraph(f"Referência observada: {obs}. Limiar das métricas categóricas: {threshold:g} mm no período.")
        rows = [
            [r.region, r.model, r.lead_start, _num(r.obs_mean, 0), _num(r.fc_mean, 0), _num(r.bias_pct, 0, True),
             _num(r.rmse, 0), _num(r.corr, 2), _num(r.pod, 2), _num(r.far, 2), _num(r.csi, 2)]
            for r in sub.itertuples()
        ]
        _table(doc, ["Região", "Modelo", "Antec. (d)", "Obs. (mm)", "Prev. (mm)", "Viés", "RMSE", "Corr.", "POD", "FAR", "CSI"],
               rows, widths=[1.8, 2.0, 1.3, 1.5, 1.5, 1.4, 1.3, 1.2, 1.1, 1.1, 1.1])
        doc.add_paragraph()

    sec = 6
    if result.station_scores is not None and not result.station_scores.empty:
        _heading(doc, f"{sec}. Comparação com estações (INMET)")
        sec += 1
        ss = result.station_scores
        n_st = int(ss["n_stations"].iloc[0])
        doc.add_paragraph(
            f"Comparação pontual com {n_st} estações automáticas do INMET com registro completo no período "
            "(dia UTC). Para cada estação usou-se a célula da grade mais próxima. A linha do produto em grade "
            "(ex.: IMERG) mostra quanto a própria estimativa por satélite difere dos pluviômetros."
        )
        sub = ss[ss.threshold_mm == threshold]
        if "lead_start" in sub:
            fc_rows = sub[sub["lead_start"].notna()]
            fc_rows = fc_rows[fc_rows.groupby("source")["lead_start"].transform(lambda x: x[x >= lead].min() if (x >= lead).any() else -1) == fc_rows["lead_start"]]
            sub = pd.concat([sub[sub["lead_start"].isna()], fc_rows])
        rows = [
            [r.source, "–" if pd.isna(getattr(r, "lead_start", np.nan)) else int(r.lead_start), _num(r.obs_mean, 0),
             _num(r.fc_mean, 0), _num(r.bias_pct, 0, True), _num(r.rmse, 0), _num(r.corr, 2),
             _num(r.pod, 2), _num(r.far, 2), _num(r.csi, 2)]
            for r in sub.itertuples()
        ]
        _table(doc, ["Fonte", "Antec. (d)", "Estações (mm)", "Fonte (mm)", "Viés", "RMSE", "Corr.", "POD", "FAR", "CSI"],
               rows, widths=[2.6, 1.3, 1.7, 1.6, 1.4, 1.3, 1.2, 1.1, 1.1, 1.1])
        top = result.stations.sort_values("obs_mm", ascending=False).head(10)
        doc.add_paragraph()
        doc.add_paragraph("Maiores acumulados observados nas estações:")
        _table(doc, ["Estação", "Código", "Chuva (mm)"],
               [[r.estacao.title(), r.codigo, _num(r.obs_mm, 1)] for r in top.itertuples()], widths=[7, 2, 3])

    multi = result.scores.groupby("model")["init"].nunique().max() > 1
    if multi:
        _heading(doc, f"{sec}. Desempenho por antecedência")
        png = skill_by_lead(result.scores[result.scores.region == "RS"], threshold, obs_ref, "csi", fig_dir / "csi_antecedencia.png")
        doc.add_picture(str(png), width=Cm(15))

    if wind is not None and (wind.grids or wind.stations is not None):
        _heading(doc, f"{sec}. Vento observado")
        sec += 1
        doc.add_paragraph(
            "Caracterização do vento no período a partir de dados observados e de reanálise, sem comparação "
            "com previsão. ERA5: rajada a 10 m e vento médio horário (0,25°). ERA5-Land: vento médio horário a "
            "10 m (0,1°). Estações: velocidade horária e rajada máxima do INMET e dos aeroportos (METAR)."
        )
        if wind.grids:
            png = wind_maps(wind.grids, fig_dir / "vento.png", outline_path=DEFAULT_REGIONS,
                            suptitle=f"Vento máximo de {period}")
            doc.add_picture(str(png), width=Cm(17))
        if wind.stations is not None and len(wind.stations):
            top = wind.stations.head(15)
            doc.add_paragraph("Maiores rajadas registradas em estações e aeroportos:")
            _table(doc, ["Rede", "Estação", "Rajada máx. (m/s)", "Rajada (km/h)", "Vento médio máx. (m/s)"],
                   [[r.rede, str(r.estacao).title(), _num(r.rajada_max_ms, 1), _num(r.rajada_max_ms * 3.6, 0),
                     _num(r.vento_max_ms, 1)] for r in top.itertuples()],
                   widths=[1.8, 6.0, 2.8, 2.5, 3.2])
        if wind.skipped:
            doc.add_paragraph("Fontes de vento sem dados: " + "; ".join(wind.skipped) + ".")

    _heading(doc, "Responsável técnico")
    doc.add_paragraph(f"{responsible}\nHidroNexus Engenharia LTDA – CNPJ 66.628.750/0001-10")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    result.scores.to_csv(out_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    if result.station_scores is not None:
        result.station_scores.to_csv(out_path.with_name(out_path.stem + "_estacoes.csv"), index=False, encoding="utf-8-sig")
        result.stations.to_csv(out_path.with_name(out_path.stem + "_estacoes_obs.csv"), index=False, encoding="utf-8-sig")
    return out_path
