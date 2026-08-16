# Análise de suficiência — docs/ + data/ vs desafio da distribuidora

**Data:** 2026-08-08 | **Autor:** grupo MAPEN (análise interna)
**Referência:** `projeto4_desafio.md` (desafio do cliente)

> Status: análise interna, feita após leitura dos PDFs da disciplina (Projeto 4 - DADOS)
> e do desafio da distribuidora de energia elétrica. Nada ainda discutido com o grupo.

---

## Resumo

O desafio da distribuidora tem **duas camadas de visão computacional**. A infraestrutura de
dados/docs do MAPEN cobre **bem** a primeira e **não cobre** a segunda (que é o coração
do desafio):

| Camada do desafio | Temos? | Onde |
|---|---|---|
| 1. Leitura do display do medidor (OCR do número) | ✅ suficiente (folgado) | `data/ufpr_amr/`, `goodcoffee_meter_reading/`, `utility_meters_*`, `henrik_energy_meter/`, `praekelt_meter_readings/`, `roi_*` |
| 2. Validação foto ↔ nota do leiturista (I100 casa fechada, etc.) | ❌ nenhum dataset | gap crítico |
| 3. Camada de perdas (contexto p/ priorizar fiscalização) | ✅ forte | `data/consumo/` (EPE, SAMP, SAMP-Balanço) + docs de perdas |

---

## Camada 1 — OCR do display: suficiente

- **UFPR-AMR** (2.000 imgs, 1.400/300/300 train/test/valid) — medidores **brasileiros de
  verdade** (família Laroca IJCNN 2019, UFPR/Copel). Base real para o baseline.
- **goodcoffee/Meter_Reading** (1.500 png + JSON VQA train/test, Apache-2.0) — treino
  OCR com rótulos em formato VQA.
- **Complementos:** utility_meters LCD/mecânicos, henrik (168), praekelt (165), crops
  `roi_donut`/`roi_counters`, `meter_reading_sample`.
- Total aprox.: **~3.840+ imgs de medidores elétricos** (sem contar duplicatas grdf).

Conclusão: **dá para começar baseline e primeiros experimentos (SR1) sem baixar nada
novo.** O gap aqui não é volume, é o rótulo de *qual foto corresponde a qual cliente*
(tarefa de validação da Camada 2).

## Camada 2 — Validação foto ↔ nota do leiturista: GAP CRÍTICO

O problema real do cliente é **duplo**:

1. a foto corresponde **ao medidor do cliente**? (identidade)
2. a foto está **coerente com a nota/ocorrência** aplicada pelo leiturista?
   (ex.: I100 – Casa fechada → foto deve mostrar portão/fachada fechada)

Isso é **classificação de cena/contexto**, não OCR de display:

- Nenhum dataset do MAPEN tem rótulo de **tipo de ocorrência/nota** (I100, I300, …).
- Todos os datasets são **foto do display de perto** — nenhum traz medidor no contexto
  (fachada, portão, caixa de padrão, instalação externa).
- Não há dataset público equivalente identificado até aqui (ver `plano_subprojeto_cv.md`
  §"Dados" para os candidatos e a estratégia).
- **Ação recomendada:** confirmar no Kickoff (12/09) se a distribuidora disponibiliza um
  lote real de fotos de campo com as notas aplicadas. Sem isso, só treinaremos em dados
  sintéticos/abertos ou com um piloto montado à mão.

## Camada 3 — Perdas (contexto estratégico): suficiente

- EPE (UF, `perda_total_GWh`), SAMP (distribuidora/classe), SAMP-Balanço
  (injetada vs vendida) cobrem o diff por UF/distribuidora mês a mês
  (`perdas_energia_brasil.md`).
- **Uso no desafio:** indireto — perda alta numa área pode priorizar fiscalização de
  fotos naquela região (redflag → inspeção), mas **não treina o modelo de visão**.

## Lacunas de docs

- Não existe **plano do subprojeto CV** (arquitetura, pipeline, métricas, divisão de
  dados) — criado em `plano_subprojeto_cv.md`.
- `docs/` cobre bem "origem dos dados" e "perdas"; falta plano de modelagem/MVP.

## Prioridades (o que falta)

1. Dataset de fotos de campo com rótulo de ocorrência (buscar público / negociar com
   cliente / piloto manual).
2. Doc de plano do subprojeto CV (arquitetura + pipeline + métricas).
3. Confirmação no Kickoff sobre acesso aos dados reais da distribuidora.
