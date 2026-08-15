# Fine-tune TrOCR-small (UFPR-AMR) — biblioteca `leiturista` + MLflow

**Data:** 2026-08-08 | **Status:** pipeline montado; **treino ainda NÃO executado**
**Base:** `models/trocr-small-stage1` (62M params, MIT) | **Dados:** UFPR-AMR (2.000 imgs, 1.400/300/300)
**Contexto:** `plano_subprojeto_cv.md` §3 (Estágio B)

---

## Por que TrOCR-small-stage1

Fine-tune de OCR end-to-end (imagem do display → leitura, ex.: "95971"). O `stage1` é o
checkpoint pré-treinado (sem fine-tune), base ideal para adaptar aos dígitos de display.
TrOCR-small (62M) roda no M1 (MPS) sem GPU dedicada.

## Biblioteca reutilizável (`src/leiturista/`, pacote `leiturista`)

Toda a lógica de dados/modelo/treino/avaliação vive na biblioteca (instalada editable:
`pip install -e .`). Nada de script solto.

| Módulo | Responsabilidade |
|---|---|
| `leiturista.paths` | Caminhos canônicos (dados/modelos/mlflow.db) — fonte única |
| `leiturista.data` | `extract_ufpr_amr()` (parquets → imgs + labels.csv), `build_dataset()` |
| `leiturista.models` | `load_processor`/`load_trocr`/`configure_model`/`save_model` |
| `leiturista.train` | `train()` — fine-tune com tracking MLflow |
| `leiturista.eval` | `evaluate()` — exact-match no test + report CSV |
| `leiturista.artifacts` | Blobs no mlflow.db (`save_text`/`save_blob`/`save_zip`/`read_blob`/`list_names`) |
| `leiturista.cli` | CLI `leiturista extract/train/eval/artifacts/restore` |

## Como rodar

```bash
# 1) dados (idempotente; já feito → data/finetune_ufpramr/)
.venv/bin/leiturista extract

# 2) treino (device auto: cuda > mps > cpu)
.venv/bin/leiturista train --epochs 8 --batch 4 --grad-accum 8 --lr 2e-5

# 3) avaliação no test (gera models/<out>/eval_test.csv + run de eval)
.venv/bin/leiturista eval

# debug rápido (subset pequeno):
.venv/bin/leiturista train --max-samples 40
```

Checkpoint default: `models/trocr-small-finetuned-ufpramr/`.

## MLflow

- **Tracking:** local em **SQLite** (`sqlite:///mlflow.db`, gitignored) — o file store
  (`file:mlruns`) caiu em maintenance mode no mlflow 3.15. Experiment `trocr-ufpramr`.
- **Artefatos TAMBÉM no `.db`:** o MLflow não tem backend de artefato em sqlite (só
  filesystem/nuvem, que criaria o diretório `mlruns/`). Para concentrar tudo num arquivo,
  os artefatos viram blobs numa tabela própria (`mapen_artifact_blobs`) dentro do mesmo
  `mlflow.db`, chaveada por run_id:
  - `train` grava `run_summary.txt` + **`checkpoint.zip`** (o modelo treinado, empacotado).
  - `eval` grava `eval_test.csv` + `eval_summary.txt`.
  - O checkpoint também fica em `models/` (working copy; path logado como param).
- Cada `train` loga: params (epochs, batch, grad-accum, lr, max_len, base), métricas por
  passo (eval_loss, eval_exact_match via callback), best metric e caminho do checkpoint.
- Cada `eval` loga: `test_exact_match`, `test_digit_acc`.
- **Não há Model Registry** (`--log-model` foi removido — exigiria diretório de artefato);
  o artefato de modelo é o `checkpoint.zip` no db + o diretório em `models/`.

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000

# listar/dump blobs de um run e restaurar o checkpoint:
.venv/bin/leiturista artifacts <run_id>
.venv/bin/leiturista artifacts <run_id> run_summary.txt
.venv/bin/leiturista restore <run_id>                      # extrai checkpoint.zip p/ models/
```

## Dados

- `data/finetune_ufpramr/` — 2.000 imagens (test 300 / train 1400 / valid 300) +
  `labels.csv` (`split,image,label`), extraído dos parquets UFPR-AMR.
- Rótulo = leitura do display (inteiro 0–99999). Sem decimais/negativos.
- Armadilhas do formato: ver `origem_dos_dados.md` §2 e notas de transformers v5 abaixo.

## Notas técnicas (transformers 5.14 / MPS)

- **TrOCRProcessor auto-load quebra** na v5.14 quando o repo só tem
  `sentencepiece.bpe.model` (sem `tokenizer.json`) — o loader foi construído
  manualmente em `leiturista.models.load_processor` (workaround permanente na lib).
  ⚠️ **Bug corrigido 2026-08-08:** o código usava `RobertaTokenizer` fixo, mas o
  `tokenizer_config.json` declara `XLMRobertaTokenizer` — vocab desalinhado (tudo vira
  UNK). Agora o loader lê a classe do config. Por isso a 1ª eval do TrOCR deu
  exact-match 1.0 vacuoso (pred/label vazios).
- `Seq2SeqTrainer` na v5: `tokenizer=` virou `processing_class=`; `compute_metrics`
  segue suportado. (`leiturista.train` já usa a API nova.)
- **MPS: fp16 não é confiável no M1** — treino roda com fp16/bf16 desligados.
- Instalados no `.venv`: torch 2.13 (MPS), transformers 5.14, datasets 5.0, mlflow 3.15.

## Baselines avaliados (test UFPR-AMR, 300 imgs)

| Baseline | exact-match | digit-acc | run MLflow | obs |
|---|---|---|---|---|
| **PP-OCRv6_tiny_rec** (off-the-shelf) | **0.357** | **0.846** | `baseline-ppocrv6-tiny` | `src/ppocr_baseline.py`, dict 6903 chars, blank=0 |
| **TrOCR-small-printed** (off-the-shelf) | **0.160** | **0.642** | `eval-test` (2026-08-08) | OCR genérico; não viu display |

O TrOCR off-the-shelf (texto impresso genérico) fica **atrás** do PP-OCRv6 no display —
o fine-tune no UFPR-AMR é quem precisa justificar o custo. O TrOCR fine-tuned precisa
superar **0.357 | 0.846**.

## Próximo passo

1. Revisar hiperparâmetros (8 epochs, batch 4, grad-accum 8, lr 2e-5) — valores iniciais.
2. Rodar `leiturista train` (treino só o usuário dispara).
3. Comparar com os baselines acima.

> ⚠️ O run `train-8ep` (2026-08-08) ficou **incompleto**: params logados (stage1, 8ep,
> lr 2e-5) mas **sem checkpoint blob** (caiu antes de salvar). **Causa dupla, já
> corrigida (2026-08-08):** (1) tokenizer era `RobertaTokenizer` fixo (labels → UNK;
> fix em `leiturista.models`); (2) `DataCollatorForSeq2Seq` exige `input_ids` de texto e
> quebrava no batch vision-encoder-decoder (fix: collate customizado `_collate` em
> `leiturista.train` — stack pixel_values + pad→-100). **Rerodar `leiturista train`.**
