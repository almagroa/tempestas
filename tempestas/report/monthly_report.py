"""Relatório mensal (.docx) do Tempestas."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from docx import Document  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402

from tempestas.mask import DEFAULT_REGIONS  # noqa: E402
from tempestas.report.docx_report import MODEL_INFO, OBS_INFO, _heading, _num, _synthesis, _table  # noqa: E402
from tempestas.verify.wind import ERA5_NOTE, WIND_QC_TEXT
from tempestas.report.maps import (  # noqa: E402
    INK, MUTED, event_maps, network_map, probability_maps, reliability_rank, skill_by_lead, wind_maps,
)
from tempestas.verify.probabilistic import event_summary  # noqa: E402
from tempestas.verify.daily import summarize_by_lead  # noqa: E402
from tempestas.verify.monthly import MonthResult  # noqa: E402

BLUE = RGBColor(0x1F, 0x3A, 0x5F)
MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro",
         "outubro", "novembro", "dezembro"]


def _daily_bars(res: MonthResult, out: Path) -> Path:
    ref = next(iter(res.obs_daily))
    d = res.regional_daily[(res.regional_daily.produto == ref)]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    rs = d[d.regiao == "RS"].sort_values("dia")
    ax.bar(rs["dia"], rs["media_mm"], color="#2a6fdb", width=0.8, label=f"RS ({ref})")
    for s, e in res.events:
        ax.axvspan(pd.Timestamp(s) - pd.Timedelta(hours=12), pd.Timestamp(e) + pd.Timedelta(hours=12),
                   color="#f2c94c", alpha=0.25, lw=0)
    ax.set_ylabel("Chuva média no RS (mm/dia)", color=MUTED, fontsize=8)
    ax.set_title("Chuva diária média no RS (faixas amarelas = eventos detectados)", loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color="#eaeef2", lw=0.6)
    ax.tick_params(labelsize=7, colors=MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.autofmt_xdate()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


PROB_METHOD = (
    "Os modelos arquivados são determinísticos (o GEFS guardado é a média do seu ensemble). A incerteza da "
    "previsão é estimada juntando, para o mesmo período, todos os modelos e as rodadas feitas de 1 a 5 dias antes "
    "(ensemble multimodelo defasado). A probabilidade de um limiar é a fração dessas previsões que o ultrapassou. "
    "O leque mede a discordância entre modelos e rodadas; não é a incerteza calibrada de um ensemble de perturbações."
)


def _event_probabilistic(doc, ens, ev, ref, k, per, fig_dir) -> None:
    obs = ev.observed[ref]
    mask = ev.masks["RS"]
    thr_list = [50, 100]
    sm = event_summary(ens, obs, mask, thr_list)
    doc.add_paragraph(f"Previsão probabilística do evento ({sm['n']} previsões: modelos × antecedências de 1 a 5 dias)",
                      style="Heading 3")
    lines = []
    if "p10" in sm:
        pos = sm["pct_obs"]
        where = ("abaixo de todas as previsões" if sm["obs_media"] < sm["minimo"] else
                 "acima de todas as previsões" if sm["obs_media"] > sm["maximo"] else
                 f"no percentil {pos * 100:.0f} das previsões")
        lines.append(f"Chuva média no RS: observado {sm['obs_media']:.0f} mm; previsões entre {sm['minimo']:.0f} e "
                     f"{sm['maximo']:.0f} mm (P10 {sm['p10']:.0f}, mediana {sm['p50']:.0f}, P90 {sm['p90']:.0f} mm). "
                     f"O observado ficou {where}.")
    ap = sm["area_posicao"]
    lines.append(f"Ponto a ponto, {ap['dentro do leque'] * 100:.0f}% da área do RS teve chuva dentro do leque das "
                 f"previsões, {ap['acima de todas as previsões'] * 100:.0f}% acima de todas e "
                 f"{ap['abaixo de todas as previsões'] * 100:.0f}% abaixo de todas.")
    for t in sm["limiares"]:
        if t["area_obs"] > 0:
            lines.append(f"≥ {t['limiar_mm']} mm: observado em {t['area_obs'] * 100:.0f}% da área; ali a probabilidade "
                         f"prevista média era {t['prob_media_onde_ocorreu'] * 100:.0f}% (máxima no estado "
                         f"{t['prob_max'] * 100:.0f}%).")
        else:
            lines.append(f"≥ {t['limiar_mm']} mm: não observado; a probabilidade prevista chegou a "
                         f"{t['prob_max'] * 100:.0f}% e passou de 50% em {t['area_prob_ge_50pct'] * 100:.0f}% da área.")
    for ln in lines:
        doc.add_paragraph(ln, style="List Bullet")
    from tempestas.verify.probabilistic import probability

    thr = next((t["limiar_mm"] for t in reversed(sm["limiares"]) if t["area_obs"] >= 0.02), 50)
    png = probability_maps(probability(ens, thr), obs, thr, sm["cat"], sm["pct"],
                           fig_dir / f"evento{k}_probabilidade.png", outline_path=DEFAULT_REGIONS, mask=mask,
                           n_members=sm["n"], suptitle=f"Evento {k}: {per} – previsão probabilística")
    doc.add_picture(str(png), width=Cm(16.5))


def _monthly_probabilistic(doc, prob, daily_threshold, fig_dir) -> None:
    doc.add_paragraph("Previsão probabilística no mês (chuva diária)", style="Heading 2")
    doc.add_paragraph(PROB_METHOD + f" Neste mês: até {prob['n_members']} previsões por dia.")
    pos = prob["position"]
    doc.add_paragraph(
        f"Em todos os dias e pontos do RS com chuva observada ou prevista (≥ 1 mm), a chuva observada ficou dentro do leque das previsões em "
        f"{pos['dentro do leque'] * 100:.0f}% dos casos, acima de todas em {pos['acima de todas as previsões'] * 100:.0f}% "
        f"e abaixo de todas em {pos['abaixo de todas as previsões'] * 100:.0f}%."
    )
    b = prob["brier"]
    _table(doc, ["Limiar (mm/dia)", "Frequência observada", "Brier", "Brier da climatologia", "Ganho (BSS)"],
           [[f"≥ {r.limiar_mm:g}", f"{r.freq_obs * 100:.1f}%", _num(r.brier, 3), _num(r.brier_clim, 3),
             _num(r.bss, 2)] for r in b.itertuples()], widths=[3, 3.4, 2.2, 3.6, 2.6])
    doc.add_paragraph(
        "Brier: erro quadrático médio da probabilidade (0 = perfeito). Ganho (BSS) > 0 indica que as probabilidades "
        "foram melhores do que usar sempre a frequência observada no mês; ≤ 0, que não acrescentaram informação."
    )
    png = reliability_rank(prob["reliability"], prob["rank_hist"], fig_dir / "confiabilidade_mes.png")
    doc.add_picture(str(png), width=Cm(15.5))
    doc.add_paragraph(
        "Confiabilidade: quando as previsões indicaram uma probabilidade, com que frequência a chuva ocorreu "
        "(pontos sobre a diagonal = probabilidades confiáveis; abaixo = superestimam a chance). Posição do observado: "
        "barras altas nas pontas indicam que o leque é estreito demais (o observado cai fora com frequência)."
    )


def build_monthly_report(
    res: MonthResult,
    out_path: str | Path,
    client: str = "Corsan – Companhia Riograndense de Saneamento",
    daily_threshold: float = 25.0,
    event_threshold: float = 50.0,
    responsible: str = "André Almagro – Eng. Ambiental, Dr. em Hidrologia – CREA 20072-MS",
) -> Path:
    out_path = Path(out_path)
    fig_dir = out_path.parent / "figuras"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ref = next(iter(res.obs_daily))
    mes = f"{MESES[res.month - 1]} de {res.year}"

    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.0)
        s.top_margin = s.bottom_margin = Cm(1.8)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)

    t = doc.add_paragraph()
    r = t.add_run(f"Relatório mensal de verificação de chuva – {mes.capitalize()}")
    r.bold, r.font.size, r.font.color.rgb = True, Pt(16), BLUE
    doc.add_paragraph().add_run("Tempestas · HidroNexus Engenharia").italic = True
    doc.add_paragraph(f"Cliente: {client}\nÁrea: Rio Grande do Sul\n"
                      f"Período: {res.start:%d/%m/%Y} a {res.end:%d/%m/%Y}\nEmissão: {date.today():%d/%m/%Y}")

    # 1. Resumo
    _heading(doc, "1. Resumo do mês")
    rs = res.regional_daily[(res.regional_daily.produto == ref) & (res.regional_daily.regiao == "RS")]
    tot_rs = rs["media_mm"].sum()
    wettest = rs.sort_values("media_mm", ascending=False).iloc[0]
    lines = [f"Chuva média acumulada no RS ({ref}): {_num(tot_rs, 0)} mm no mês.",
             f"Dia mais chuvoso na média do estado: {wettest['dia']:%d/%m} ({_num(wettest['media_mm'], 1)} mm)."]
    if res.events:
        ev = "; ".join(f"{s:%d/%m}" if s == e else f"{s:%d/%m}–{e:%d/%m}" for s, e in res.events)
        lines.append(f"Eventos de chuva forte detectados: {len(res.events)} ({ev}).")
    else:
        lines.append("Nenhum evento de chuva forte detectado no mês pelos critérios adotados.")
    summ = summarize_by_lead(res.daily, "RS", daily_threshold)
    if len(summ):
        s1 = summ[(summ.observed == ref) & (summ.lead == 1)].sort_values("csi", ascending=False)
        if len(s1):
            b = s1.iloc[0]
            lines.append(f"Melhor desempenho diário com 1 dia de antecedência (≥ {daily_threshold:g} mm/dia): "
                         f"{b.model} (CSI {_num(b.csi, 2)}).")
    for ln in lines:
        doc.add_paragraph(ln, style="List Bullet")

    # 2. Dados e método
    _heading(doc, "2. Dados e método")
    doc.add_paragraph("Previsões (rodadas diárias arquivadas): " + "; ".join(f"{m} ({v})" for m, v in MODEL_INFO.items()) + ".")
    doc.add_paragraph("Observado (cada fonte avaliada separadamente, sem fusão): " +
                      "; ".join(f"{k}: {v}" for k, v in OBS_INFO.items()) + ".")
    doc.add_paragraph(
        "Todos os campos foram convertidos para totais diários (dia UTC) e reamostrados para uma grade comum de 0,1°. "
        "Um dia é classificado como de chuva forte quando pelo menos 5% da área do RS recebe 50 mm ou mais, ou 25% "
        "da área recebe 20 mm ou mais; dias consecutivos formam um evento. Métricas: viés, POD (probabilidade de "
        "detecção), FAR (razão de alarmes falsos) e CSI (índice de sucesso crítico)."
    )

    # 3. Chuva observada
    _heading(doc, "3. Chuva observada")
    doc.add_picture(str(_daily_bars(res, fig_dir / "chuva_diaria_rs.png")), width=Cm(16))
    if res.inventory is not None:
        png = network_map(res.inventory, res.obs_total[ref], fig_dir / "chuva_mes_estacoes.png", bg_name=ref,
                          outline_path=DEFAULT_REGIONS, mask=res.masks["RS"], suptitle=f"Chuva acumulada – {mes}")
        doc.add_picture(str(png), width=Cm(15))
    reg = res.regional_daily[res.regional_daily.produto == ref].groupby("regiao")["media_mm"].agg(["sum", "max"])
    _table(doc, ["Região", "Total no mês (mm)", "Maior chuva diária média (mm)"],
           [[k, _num(v["sum"], 0), _num(v["max"], 1)] for k, v in reg.iterrows()], widths=[4, 5, 6])
    if res.stations is not None and len(res.stations):
        doc.add_paragraph()
        top = res.stations.sort_values("obs_mm", ascending=False).head(10)
        doc.add_paragraph("Maiores acumulados do mês em estações (dias faltantes não preenchidos):")
        _table(doc, ["Rede", "Estação", "Código", "Chuva (mm)", "Dias faltando"],
               [[r.rede, str(r.estacao).title(), r.codigo, _num(r.obs_mm, 1), int(getattr(r, "dias_faltando", 0))]
                for r in top.itertuples()], widths=[2, 6.5, 2.5, 2.5, 2.5])

    # 4. Eventos
    sec = 4
    _heading(doc, f"{sec}. Eventos de chuva forte")
    if not res.event_results:
        doc.add_paragraph("Não houve evento de chuva forte no mês pelos critérios adotados.")
    for k, ev in enumerate(res.event_results, 1):
        per = f"{ev.start:%d/%m}" if ev.start == ev.end else f"{ev.start:%d/%m} a {ev.end:%d/%m}"
        _heading(doc, f"{sec}.{k} Evento {k}: {per}", level=2)
        for ln in _synthesis(ev.scores, ref, event_threshold, 1):
            doc.add_paragraph(ln, style="List Bullet")
        png = event_maps(ev.observed, ev.forecasts, fig_dir / f"evento{k}_mapas.png", lead_start=1, reference=ref,
                         outline_path=DEFAULT_REGIONS, mask=ev.masks["RS"], suptitle=f"Evento {k}: {per}")
        doc.add_picture(str(png), width=Cm(16))
        s = ev.scores[(ev.scores.region == "RS") & (ev.scores.observed == ref) & (ev.scores.threshold_mm == event_threshold)]
        s = s.sort_values(["model", "lead_start"])
        _table(doc, ["Modelo", "Antec. (d)", "Obs. (mm)", "Prev. (mm)", "Viés", "POD", "FAR", "CSI"],
               [[r.model, r.lead_start, _num(r.obs_mean, 0), _num(r.fc_mean, 0), _num(r.bias_pct, 0, True),
                 _num(r.pod, 2), _num(r.far, 2), _num(r.csi, 2)] for r in s.itertuples()],
               widths=[2.6, 1.6, 1.8, 1.8, 1.6, 1.4, 1.4, 1.4])
        ens = res.event_ens[k - 1] if k - 1 < len(res.event_ens) else None
        if ens is not None:
            _event_probabilistic(doc, ens, ev, ref, k, per, fig_dir)
        if ev.station_scores is not None and len(ev.station_scores):
            ss = ev.station_scores[(ev.station_scores.threshold_mm == event_threshold)]
            ss = ss[(ss["lead_start"].isna()) | (ss["lead_start"] == ss.groupby("source")["lead_start"].transform("min"))]
            doc.add_paragraph(f"Comparação com estações (menor antecedência disponível, ≥ {event_threshold:g} mm):")
            _table(doc, ["Rede", "Fonte", "Estações", "Obs. (mm)", "Prev. (mm)", "Viés", "CSI"],
                   [[r.rede, r.source, r.n_stations, _num(r.obs_mean, 0), _num(r.fc_mean, 0),
                     _num(r.bias_pct, 0, True), _num(r.csi, 2)] for r in ss.itertuples()],
                   widths=[1.8, 3, 1.6, 1.8, 1.8, 1.6, 1.4])
        doc.add_paragraph()

    # 5. Desempenho diário
    sec += 1
    _heading(doc, f"{sec}. Desempenho dos modelos no mês (verificação dia a dia)")
    doc.add_paragraph(
        f"Todos os dias do mês, comparando cada modelo com o {ref} com 1, 2, 3, 5 e 7 dias de antecedência. "
        f"Acertos, perdas e alarmes falsos (≥ {daily_threshold:g} mm/dia) somados sobre os dias e a área."
    )
    if len(summ):
        sref = summ[summ.observed == ref].copy()
        plot_df = sref.rename(columns={"lead": "lead_start"}).assign(threshold_mm=daily_threshold, observed=ref)
        png = skill_by_lead(plot_df, daily_threshold, ref, "csi", fig_dir / "csi_mes_antecedencia.png")
        doc.add_picture(str(png), width=Cm(14))
        _table(doc, ["Modelo", "Antec. (d)", "Dias", "Viés", "POD", "FAR", "CSI"],
               [[r.model, r.lead, r.dias, _num(r.bias_pct, 0, True), _num(r.pod, 2), _num(r.far, 2), _num(r.csi, 2)]
                for r in sref.sort_values(["model", "lead"]).itertuples()],
               widths=[2.8, 1.8, 1.4, 1.8, 1.6, 1.6, 1.6])
        for region in [r for r in res.masks if r != "RS"]:
            sr = summarize_by_lead(res.daily, region, daily_threshold)
            sr = sr[(sr.observed == ref) & (sr.lead == 1)]
            if len(sr):
                doc.add_paragraph(f"{region} – 1 dia de antecedência: " + "; ".join(
                    f"{r.model} CSI {_num(r.csi, 2)} (viés {_num(r.bias_pct, 0, True)})" for r in sr.itertuples()) + ".")

    if res.prob is not None:
        _monthly_probabilistic(doc, res.prob, daily_threshold, fig_dir)

    # 6. Vento
    w = res.wind
    if w is not None and (w.grids or w.stations is not None or w.metar_weather is not None):
        sec += 1
        _heading(doc, f"{sec}. Vento observado no mês")
        doc.add_paragraph(ERA5_NOTE)
        doc.add_paragraph(WIND_QC_TEXT)
        if w.grids:
            doc.add_picture(str(wind_maps(w.grids, fig_dir / "vento_mes.png", outline_path=DEFAULT_REGIONS,
                                          suptitle=f"Vento máximo – {mes}")), width=Cm(16))
        if w.stations is not None and len(w.stations):
            top = w.stations.head(15)
            _table(doc, ["Rede", "Estação", "Rajada máx. (m/s)", "(km/h)", "Vento médio máx. (m/s)"],
                   [[r.rede, str(r.estacao).title(), _num(r.rajada_max_ms, 1), _num(r.rajada_max_ms * 3.6, 0),
                     _num(r.vento_max_ms, 1)] for r in top.itertuples()], widths=[1.8, 6, 2.8, 2, 3.2])
            st = w.stations
            if "rajada_suspeita" in st and st["rajada_suspeita"].fillna(False).astype(bool).any():
                sus = st.loc[st["rajada_suspeita"].fillna(False).astype(bool), "estacao"]
                doc.add_paragraph("Rajada desconsiderada por suspeita de defeito do anemômetro de rajada (mantidas só as "
                                  "rajadas coerentes com o vento médio): " + ", ".join(s.title() for s in sus) + ".")
        if w.metar_weather is not None and len(w.metar_weather):
            doc.add_paragraph()
            doc.add_paragraph("Tempo presente nos aeroportos (horas com o fenômeno no mês):")
            mw = w.metar_weather.sort_values("horas_chuva", ascending=False)
            _table(doc, ["Aeroporto", "Chuva", "Chuva forte", "Trovoada"],
                   [[f"{r.nome} ({r.icao})", int(r.horas_chuva), int(r.horas_chuva_forte), int(r.horas_trovoada)]
                    for r in mw.itertuples()], widths=[6, 3, 3, 3])

    # 7. Disponibilidade
    sec += 1
    _heading(doc, f"{sec}. Disponibilidade de dados")
    _table(doc, ["Modelo", "Rodadas", "Dias no mês", "Datas sem rodada"],
           [[r.modelo, r.rodadas, r.dias_no_mes, r.faltando] for r in res.availability.itertuples()],
           widths=[3, 2, 2.2, 9])
    if res.inventory is not None:
        cnt = res.inventory.groupby(["rede", "situacao"]).size().unstack(fill_value=0)
        doc.add_paragraph()
        _table(doc, ["Rede", "Mês completo", "Parcial (até 3 dias faltando)", "Sem dados suficientes", "Aeroportos"],
               [[k, int(cnt.loc[k].get("completa", 0)), int(cnt.loc[k].get("parcial", 0)),
                 int(cnt.loc[k].get("incompleta/sem dados", 0)), int(cnt.loc[k].get("aeroporto", 0))]
                for k in cnt.index], widths=[2.5, 3, 4, 4, 2.5])
    if res.skipped:
        doc.add_paragraph("Observações: " + "; ".join(f"{k}: {v}" for k, v in res.skipped.items()))

    _heading(doc, "Responsável técnico")
    doc.add_paragraph(f"{responsible}\nHidroNexus Engenharia LTDA – CNPJ 66.628.750/0001-10")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    res.daily.to_csv(out_path.with_name(out_path.stem + "_diario.csv"), index=False, encoding="utf-8-sig")
    return out_path
