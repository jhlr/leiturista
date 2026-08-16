# AGENTS.md — LEITURISTA

Regras deste projeto (somam-se às regras globais de `~/.config/opencode/AGENTS.md`).

## Contexto

- **LEITURISTA** — visão computacional de medidores de energia elétrica: leitura OCR do
  display + extração do serial + validação de cena/coerência.
- Disciplina **Projeto 4 - DADOS (Cesar School, BD2026.2)**, Grupo 3, cliente
  **distribuidora de energia elétrica**.
- Desmembrado do repositório `mapen` em 2026-08-10 (o `mapen` ficou só com a camada de
  dados de consumo/perdas/redes). Contexto do desafio em `docs/projeto4_desafio.md` e
  `docs/plano_subprojeto_cv.md`.

## Objetivo

Leitura automática de medidor a partir de foto (imagem → número) **e** validação de
coerência foto ↔ ocorrência do leiturista (ex.: I100 = casa fechada). Duas tarefas:

1. **Leitura (Tarefa 1):** extrair o número do display (OCR) — dados suficientes
   (~3.840+ imgs baixadas, base UFPR-AMR 2.000).
2. **Validação (Tarefa 2):** a foto corresponde ao medidor/cliente? Coerente com a nota?
   — **GAP**: sem dataset público; aguardando lote real da distribuidora no Kickoff (12/09).

## Estrutura

```
leiturista/
├── data/                  # datasets de imagem (ufpr_amr, finetune_ufpramr, goodcoffee...)
├── models/                # PP-OCRv6_tiny_rec (ONNX), PP-OCRv5_mobile_det (ONNX), TrOCR-small*
├── src/leiturista/        # LIBRARY `leiturista` (data/models/train/eval/artifacts/cli/inference)
├── scripts/               # ferramentas avulsas (ppocr_baseline.py + dict)
├── app/                   # demo Streamlit (upload foto → leitura + serial + flags)
├── mlflow.db              # MLflow tudo num arquivo (tracking + artefatos) (gitignored)
├── pyproject.toml         # pacote `leiturista` (instalado editable no .venv)
└── docs/                  # toda documentação
```

## Regras duras do projeto

1. **Documentar tudo em `docs/`** — descoberta/decisão/origem vira doc commitável com
   data. Doc novo no chat SEM esperar pedido.
2. **Código reutilizável vai na biblioteca `leiturista`** (instalada editable:
   `pip install -e .`). Pipeline = CLI `leiturista` (extract/train/eval, com MLflow).
   Ferramentas avulsas em `scripts/`. Rodar com `.venv/bin/python`.
3. **Datasets em `data/`** — 1 pasta por dataset, arquivos flat.
4. **Download sempre em background sem timeout** (`nohup ... > log 2>&1 &`),
   `HF_HUB_DISABLE_XET=1` p/ downloads grandes no HF.
5. **Treino só o usuário roda.** Eu NUNCA executo `leiturista train` — preparo a
   pipeline e o usuário dispara e avalia.
6. **Escopo: medidores de energia elétrica.** Não expandir p/ gás/água sem pedido.

## Rotinas

- **Pipeline CV:** CLI `leiturista` → `extract` (extrai UFPR-AMR + labels.csv), `train`
  (fine-tune TrOCR-small-stage1, MLflow em `mlflow.db`), `eval` (avalia no test).
  `leiturista artifacts <run_id>` lista/dump blobs, `leiturista restore` extrai o
  checkpoint. Ver `docs/finetune_trocr_ufpramr.md`.
- **Demo Streamlit:** `.venv/bin/streamlit run app/app.py` → upload de foto → leitura +
  serial + flags de coerência (det PP-OCRv5 + rec PP-OCRv6_tiny). Ver
  `docs/app_demo_streamlit.md`.
- **Baseline:** `scripts/ppocr_baseline.py` (PP-OCRv6_tiny_rec no test UFPR-AMR).
- **MLflow UI:** `.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Referências

- Desafio: `docs/projeto4_desafio.md` + `docs/pedido_kickoff.md`.
- Dados: `docs/origem_dos_dados.md` (imagens + modelos), `docs/candidatos_nao_baixados.md`.
- Papers: `docs/artigos.md` (UFPR-AMR = Laroca IJCNN 2020, DOI 10.1109/IJCNN48605.2020.9207318).
- Projeto irmão (camada de dados): `~/Developer/mapen` (ONS/EPE/ANEEL, perdas, mapas).
