# Plano — Subprojeto Visão Computacional (leitura + validação de fotos de leitura)

**Data:** 2026-08-08 | **Status:** RASCUNHO — nada ainda discutido com o grupo
**Referência:** `projeto4_neoenergia.md` (desafio), `analise_suficiencia.md` (gap)
**Contexto completo:** `contexto_mapen.md`

> Este é um ponto de partida simples. Validar/alterar com o grupo antes de virar
> plano oficial.

---

## 1. Problema (recorte do desafio)

A Neoenergia PE quer fiscalizar as **fotos tiradas pelos leituristas em campo**. O
problema tem duas tarefas de visão:

1. **Leitura**: extrair o número do display do medidor da foto (OCR).
2. **Validação/coerência**: a foto corresponde ao medidor do cliente? Está legível?
   A foto é coerente com a **nota/ocorrência** aplicada pelo leiturista
   (ex.: I100 – Casa fechada → deve mostrar portão/fachada fechada)?

**Escopo deste plano:** as duas tarefas, com prioridade na tarefa 1 (dados disponíveis)
e um **piloto da tarefa 2** (depende de dados).

## 2. Dados

### Tarefa 1 (leitura) — dados suficientes ✅

Já baixados no MAPEN (~3.840+ imgs de medidor elétrico BR):

- `ufpr_amr/` (2.000 imgs, 1.400/300/300) — **base principal**: display de medidor BR.
- `goodcoffee_meter_reading/` (1.500 png + JSON VQA train/test).
- `utility_meters_lcd|mechanical/`, `henrik_energy_meter/` (168), `praekelt_meter_readings/` (165),
  `roi_donut|counters/`, `meter_reading_sample/`.

### Tarefa 2 (validação de cena/coerência) — GAP, sem dado público com nota

**Não existe dataset público com as notas do leiturista** (códigos tipo I100 são
proprietários de cada distribuidora). Candidatos mais próximos (foto de campo real,
cenário não controlado):

| Candidato | O quê | Licença/acesso | Status |
|---|---|---|---|
| **Copel-AMR** (UFPR) | 12.500 fotos de campo (Copel/PR), 20% ilegíveis/oclusão — cenário real | verificar (acadêmico?) | não baixado |
| **UFPR-ADMR-v2** (Copel) | 5.000 imgs de campo, dials | licença acadêmica (assinar) | pendente |
| **IEEE DataPort** (IIT BHU) | 570 imgs, condições diversas | download aberto | não baixado |
| **SIVAL / autoleitura** (UFMA, CEMAR/CELPA) | papers com base de campo classificada (boas/regulares/ruins) | base **não pública** | apenas referência |

**Estratégia (sem dados do cliente):**
1. Piloto da tarefa 2 com Copel-AMR/UFPR-ADMR-v2 (foto real de campo BR): classificar
   "foto válida de medidor" e, se possível, "acesso/imóvel fechado" — proxies das notas.
2. **Pedir à Neoenergia um lote real de fotos com as notas aplicadas no Kickoff (12/09).**
   Sem isso, a validação da nota específica fica limitada a piloto/proxy.

## 3. Modelos de partida (leves, preferencialmente já relacionados)

Pipeline em **2 estágios** (separar cena de OCR evita erro propagado e mantém cada
modelo pequeno):

### Estágio A — Validação de cena / coerência (tarefa 2)
- **Moondream2** (~1.9B, Apache-2.0, roda em CPU/MPS) — VLM pequeno p/ VQA/classificação
  ("tem medidor na foto?", "a casa está fechada?"). Bom ponto inicial, já relacionado a
  visão-leve.
- Alternativa menor: **MobileNetV3 / ResNet18** fine-tune (poucos M params) — se o
  objetivo for só classificação binária por tipo de foto.

### Estágio B — OCR do display (tarefa 1)
- **PP-OCRv6_tiny_rec** (1,1M params, ~4 MB, CPU ~1–3 ms) — mais leve; PP-OCRv6_small
  (20 MB, 81,3% acc) como upgrade. Ideal p/ dígitos de display. ✅ baixado em
  `models/pp_ocr_v6_tiny_rec_onnx/` (ONNX, Apache-2.0).
- **TrOCR-Small** (62M, transformers) — fine-tune nos dados UFPR-AMR/goodcoffee p/ leitura
  end-to-end do número (formato "8430.6"), integrado ao HF. ✅ baixado em
  `models/trocr-small-printed/` (baseline imediato, MIT) e `models/trocr-small-stage1/`
  (base p/ fine-tune, MIT).
- Referência (NÃO leve): Word2Li/Electricity-Meter-OCR-7B (Qwen2.5-VL-7B) — só se virar
  modelo principal.

> Origem/licença dos modelos: `origem_dos_dados.md` §15.

**Sugestão de baseline (SR1):** PP-OCRv6_tiny no crop do display (UFPR-AMR)
→ comparar com TrOCR-small fine-tune; Moondream2 para o pipeline de cena.

## 4. Métricas

- **Tarefa 1**: acurácia do dígito; CER/word-accuracy da leitura inteira; erro absoluto
  médio em kWh lidos vs rótulo.
- **Tarefa 2**: acurácia/F1 por classe de foto (válida, ilegível, casa fechada, outro
  medidor); taxa de **falso negativo em I100** (foto incoerente que passa) — o que mais
  custa para o cliente.

## 5. Plano de trabalho (alinhado a CRISP-DM e marcos)

1. **Entendimento + dados** (→ Kickoff 12/09): fechar recorte com o grupo; pedir lote
   real à Neoenergia; baixar Copel-AMR/UFPR-ADMR-v2/IEEE DataPort (negociar licenças).
2. **Preparação** (→ SR1 03/10): extrair crops de display (UFPR-AMR); montar splits;
   criar rótulos do piloto de cena; pipeline reutilizável em `src/`.
3. **Modelagem + baseline** (→ SR1): PP-OCRv6_tiny + TrOCR-small fine-tune; Moondream2
   zero-shot na tarefa de cena como baseline antes de qualquer fine-tune.
4. **Avaliação**: métricas da §4; comparar modelos; documentar em `docs/`.
5. **MVP** (→ SR2 05/12): inferência em lote (foto → {leitura, flags de coerência}).

## 6. Open items para o grupo

- [ ] Validar este recorte (duas tarefas, prioridade 1, piloto da 2).
- [ ] Quem fica com o quê (papéis da disciplina).
- [ ] Baixar/assinar Copel-AMR + UFPR-ADMR-v2 (negociar com o professor/grupo UFPR).
- [ ] Pergunta oficial à Neoenergia no Kickoff sobre o lote real de fotos+notas.
