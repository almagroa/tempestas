# Tempestas

Verificação retrospectiva de eventos extremos de chuva no Rio Grande do Sul:
compara a chuva **prevista** por vários modelos com a **observada** por vários
produtos, sem fundi-los, e gera métricas e relatórios técnicos.

Desenvolvido pela HidroNexus Engenharia. Responsável técnico: André Almagro.

## Fontes

| Tipo | Fonte | Observação |
|---|---|---|
| Previsão | GEFS, IFS, AIFS, ICON, CPT-WRF, CHIRPS-GEFS | rodadas diárias arquivadas em `E:/<modelo>/<AAAA-MM-DD>/` |
| Observado | IMERG Early (diário, 0,1°) | `E:/IMERG/IMERG_BR_AAAA-MM.nc` |
| Observado | Estações INMET, ANA, Cemaden | em desenvolvimento |
| Vento | ERA5-Land (10 m), ERA5 (rajada), INMET, METAR | em desenvolvimento |

As pastas de origem são **somente leitura**. Tudo o que o Tempestas grava vai
para `paths.work_dir` (padrão `E:/tempestas`).

Cada fonte é padronizada na leitura (`tempestas/io/standardize.py`): longitude
em −180..180, latitude crescente, mm/dia, e o acumulado desde o início da
rodada (GEFS, IFS, AIFS, ICON, CPT-WRF) é convertido em totais diários.

## Uso

```bash
pip install -e .[dev]
python -m tempestas evento --inicio 2026-11-03 --fim 2026-11-07
```

Saída: `metricas.csv` com uma linha por modelo × rodada × produto observado ×
limiar, contendo viés, MAE, RMSE, correlação, POD, FAR, CSI e FBI.

## Métricas

- **Viés** = média(previsto − observado), em mm e %.
- **POD** (probabilidade de detecção) = acertos / (acertos + perdas).
- **FAR** (razão de alarmes falsos) = alarmes falsos / (acertos + alarmes falsos).
- **CSI** (índice de sucesso crítico) = acertos / (acertos + perdas + alarmes falsos).
- **FBI** (viés de frequência) = (acertos + alarmes falsos) / (acertos + perdas).

## Testes

```bash
python -m pytest
```
