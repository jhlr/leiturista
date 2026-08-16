# Origem dos dados baixados — LEITURISTA (imagens de medidor + modelos)

**Levantamento:** 2026-08-08 | **última atualização:** 2026-08-10 (separado do repo MAPEN)

Objetivo do projeto: **leitura automática de medidores de energia elétrica** por
CV/OCR (imagem do display → número) + extração do **serial** + validação de cena
(desafio de distribuidoras de energia elétrica, Projeto 4). Este doc cobre os datasets de **imagem** e os
**modelos de visão**. Séries de consumo/perdas/redes ficam no repo irmão `mapen`.

---

## Resumo executivo

| Dataset | Origem (autor/repo) | Imagens | Foco | Estado | Em |
|---|---|---|---|---|---|
| `Chaymaa/meter_reading` | HF, Chaymaa | 6 | Medidor elétrico BR (Elster/INMETRO) | ✅ completo | `meter_reading_sample/` |
| `Chaymaa/UFPR-AMR` | HF, Chaymaa (base UFPR, Laroca IJCNN 2019) | 2.000 (1.400/300/300) | Medidor elétrico BR | ✅ completo | `ufpr_amr/` |
| `Chaymaa/roi_donut` | HF, Chaymaa | crops display | OCR display | ❌ corrompido — removido (parquet truncado) | re-baixar via `hf download Chaymaa/roi_donut` |
| `Chaymaa/roi_counters` | HF, Chaymaa | crops contador | OCR contador | ✅ baixado | `roi_counters/` |
| `Chaymaa/ufpr-amr-donut` | HF, Chaymaa | 15 | Amostra Donut | ✅ baixado | `ufpr_amr_donut/` |
| `Praekelt/ElectricityMeterReadings1o4` | HF, org Helm (África do Sul) | 165 | Medidor elétrico | ✅ completo | `praekelt_meter_readings/` |
| `utilitimetersai/Utility-Meters-LCD-Electricity-v1` | HF | <1K | Medidor elétrico LCD | ✅ baixado | `utility_meters_lcd/` |
| `utilitimetersai/Raw-Mechanical-Utility-Meters-Dials-Images` | HF | <1K | Dials mecânicos | ✅ baixado | `utility_meters_mechanical/` |
| `goodcoffee/Meter_Reading` | HF | 1K–10K | Medidor (imagens+JSON VQA) | ✅ completo | `goodcoffee_meter_reading/` |
| `henrik-dra/energy-meter` | HF | 168 | Medidor de energia | ✅ baixado | `henrik_energy_meter/` |

**Totais:** ~3.840+ imgs de medidores elétricos (base real de dados de imagem do projeto).

---

## Detalhamento por dataset

### 1. Chaymaa/meter_reading — AMOSTRA
- **URL:** https://huggingface.co/datasets/Chaymaa/meter_reading (subido 2023-04-28)
- **Licença:** não declarada
- **Conteúdo:** 6 imagens JPEG (2 train / 2 valid / 2 test) em parquet (colunas
  `image` STRUCT(bytes,path) + `ground_truth` JSON).
- **Origem real:** mini-amostra do `Chaymaa/UFPR-AMR`; `gt_parse` com `supplier: elster`,
  leitura de display (ex.: `00495`), placa metrológica (3 EL., 4 FIOS, 120V, 60 Hz,
  INMETRO/DIMEL, Portaria 333/2007, Classe B) — medidores brasileiros.

### 2. Chaymaa/UFPR-AMR — DATASET REAL DE ENERGIA ELÉTRICA BR (2.000 imgs) ✅
- **URL:** https://huggingface.co/datasets/Chaymaa/UFPR-AMR (2023-08-02)
- **Base:** UFPR-AMR (Laroca et al., IJCNN 2019) — "10.000 dígitos em 2.000 imagens de
  display (5 dígitos cada)", grupo UFPR (PR), mesma família do UFPR-ADMR (Copel).
- **Licença:** não declarada no card.
- **Conteúdo:** 3 parquets em `data/ufpr_amr/`: train 1.400 / test 300 / valid 300
  (test 41 MB, train 201 MB, valid 42 MB). Mesma estrutura do #1.

### 3. Chaymaa/roi_donut, roi_counters, ufpr-amr-donut — derivados
- **URLs:**
  - https://huggingface.co/datasets/Chaymaa/roi_donut
  - https://huggingface.co/datasets/Chaymaa/roi_counters
  - https://huggingface.co/datasets/Chaymaa/ufpr-amr-donut
- **Conteúdo:** crops do display/contador do UFPR-AMR no formato Donut (treino OCR).
  `roi_donut` 800 imgs, `roi_counters` 117, `ufpr-amr-donut` 15 (amostra).
- **Download:** `HF_HUB_DISABLE_XET=1 .venv/bin/hf download Chaymaa/<repo> --repo-type dataset`
  → copiar para `data/<nome>/`.

### 4. Praekelt/ElectricityMeterReadings1o4 — ✅ completo
- **URL:** https://huggingface.co/datasets/Praekelt/ElectricityMeterReadings1o4
- **Origem:** org **Helm** (helm.africa; conta "Praekelt" no HF = Helm, ex-Praekelt
  Foundation/África do Sul). Card vazio — **sem licença declarada**.
- **Conteúdo:** 165 imagens + `reading` (string; leituras tipo `93809.4` e muitos
  `invalid`). Dataset completo — não é amostra (não há partes 2-4 públicas; só a cópia
  idêntica `Manthan7507/ElectricityMeterReadings1o4`).
- **Download:** `HF_HUB_DISABLE_XET=1 .venv/bin/hf download Praekelt/ElectricityMeterReadings1o4 --repo-type dataset`
  → parquet 44.2 MB em `data/praekelt_meter_readings/`.

### 5. utilitimetersai/Utility-Meters-LCD-Electricity-v1 e Raw-Mechanical-...
- **URLs:**
  - https://huggingface.co/datasets/utilitimetersai/Utility-Meters-LCD-Electricity-v1
  - https://huggingface.co/datasets/utilitimetersai/Raw-Mechanical-Utility-Meters-Dials-Images
- **Licença:** CC BY-NC 4.0
- **Conteúdo:** imagefolder de medidores elétricos — LCD (em `utility_meters_lcd/`) e
  mecânicos/dials (em `utility_meters_mechanical/`).
- **Download:**
  - `HF_HUB_DISABLE_XET=1 .venv/bin/hf download utilitimetersai/Utility-Meters-LCD-Electricity-v1 --repo-type dataset`
  - `HF_HUB_DISABLE_XET=1 .venv/bin/hf download utilitimetersai/Raw-Mechanical-Utility-Meters-Dials-Images --repo-type dataset`

### 6. goodcoffee/Meter_Reading — ✅ completo
- **URL:** https://huggingface.co/datasets/goodcoffee/Meter_Reading | **Licença:** Apache-2.0
- **Conteúdo:** 3.004 arquivos em `data/goodcoffee_meter_reading/` (1.500 png
  `v_XXXX_f_0000_rgba[__ds6].png` + 1.502 json + README): imagens de display + JSON de
  VQA (`train__vqa_dataset.json` / `test__vqa_dataset.json`) para treino OCR.
- **Status:** completo, conferido contra o repo (3004 = 3004). 2.3 GB.
- **Download:** `HF_HUB_DISABLE_XET=1 .venv/bin/hf download goodcoffee/Meter_Reading --repo-type dataset`

### 7. henrik-dra/energy-meter — ✅ completo
- **URL:** https://huggingface.co/datasets/henrik-dra/energy-meter
- **Conteúdo:** 344 MB em `data/henrik_energy_meter/`: parquets `train` (286 MB) e
  `test` (58 MB) de imagens de medidor de energia (imagem+texto).
- **Download:** `HF_HUB_DISABLE_XET=1 .venv/bin/hf download henrik-dra/energy-meter --repo-type dataset`

### 8. finetune_ufpramr/ — extração local (dados de treino do pipeline)
- Gerado por `leiturista extract` a partir dos parquets UFPR-AMR: 2.000 imagens
  (test 300 / train 1400 / valid 300) + `labels.csv` (`split,image,label`).
- Rótulo = leitura do display (inteiro 0–99999). Sem decimais/negativos.
- Armadilhas do formato: ver `origem_dos_dados.md` §2 (UFPR-AMR) e
  `docs/finetune_trocr_ufpramr.md`.

## Modelos de visão baixados

Baixados em 2026-08-08 para `models/` via `.venv/bin/hf download` (HF Hub,
`HF_HUB_DISABLE_XET=1`). Uso no `docs/plano_subprojeto_cv.md` §3 e na demo Streamlit
(`app/app.py`).

| Modelo | Repo HF | Tamanho | Uso |
|---|---|---|---|
| **PP-OCRv6_tiny_rec** (ONNX) | `PaddlePaddle/PP-OCRv6_tiny_rec_onnx` | 4,4 MB | OCR leve (1,1M params, ~1–3 ms CPU) — reconhece a linha de texto (leitura do display) |
| **PP-OCRv5_mobile_det** (ONNX) | `PaddlePaddle/PP-OCRv5_mobile_det_onnx` | ~2 MB | **Detecção** de caixas de texto na foto (display, placa/serial) |
| **TrOCR-small-printed** | `microsoft/trocr-small-printed` | 493 MB | OCR end-to-end, já fine-tuned p/ texto impresso (baseline imediato) |
| **TrOCR-small-stage1** | `microsoft/trocr-small-stage1` | 247 MB | Pré-treinado (sem fine-tune) — base p/ fine-tune em dígitos de display |

- `PP-OCRv6_tiny_rec_onnx`: **Apache-2.0** | `PP-OCRv5_mobile_det_onnx`: **Apache-2.0** |
  `trocr-small-*`: **MIT**
- TrOCR: Li et al., "TrOCR: Transformer-based OCR with Pre-trained Models", AAAI 2023.
- Baseline avaliado: PP-OCRv6_tiny_rec exact-match **0.357** / digit-acc **0.846**
  (test UFPR-AMR, 300 imgs) — ver `scripts/ppocr_baseline.py`.

---

## Notas de troubleshooting

- **CDN xet do HF instável:** downloads grandes falhando com
  `RuntimeError: File reconstruction error: CAS Client Error: ... Request middleware error`
  (o CDN `xorbs/xet` está com problemas; **não é a rede local**). Mitigações:
  re-tentar com intervalos, 1 arquivo por vez, menos concorrência.
- **`HF_HUB_DISABLE_XET=1`** resolveu o travamento do UFPR-AMR (`.incomplete` de 0 bytes
  em toda tentativa → concluído em ~10 min). Usar em downloads grandes no HF.
- **`hf download`** (huggingface_hub) tem resume/retry nativo e lida melhor com o xet —
  preferir para arquivos grandes; `curl` para S3/parquet de fonte direta.
- **Download sempre em background sem timeout** (`nohup ... &` + log) — ver AGENTS.md global.
- **Datasets que exigem conta/chave/licença** (Roboflow, UFPR-ADMR-v2, Copel-AMR,
  NRC-GAMMA, Pointer-10K, IEEE DataPort): ver `candidatos_nao_baixados.md`.

---

## Docs irmãos

| Doc | Conteúdo |
|---|---|
| `finetune_trocr_ufpramr.md` | Fine-tune TrOCR-small em UFPR-AMR: biblioteca `leiturista`, CLI, MLflow |
| `plano_subprojeto_cv.md` | Plano do subprojeto de visão computacional (2 tarefas) |
| `copel_amr.md` | Copel-AMR (12.5k fotos de campo): o que é, licença, trâmite |
| `artigos.md` | Papers relevantes (UFPR-AMR = Laroca IJCNN 2020, DOI 10.1109/IJCNN48605.2020.9207318) |
| `projetos_notaveis.md` | Projetos/open-source de leitura de medidores |
| `candidatos_nao_baixados.md` | Datasets com acesso restrito (conta/licença) |
| `projeto4_desafio.md` | Disciplina Projeto 4 (BD2026.2) + desafio de distribuidora |
| `analise_suficiencia.md` | docs/+data/ vs desafio: o que temos e o que falta |
