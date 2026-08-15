# Compartilhando modelos com um colega (para testar o app)

Data: 2026-08-15

Os pesos/dados não entram no git (regra do projeto: `data/`, `models/`,
`mlflow.db` e `.model_cache/` são gitignored/regeráveis). Para o colega testar,
os artefatos de inferência são distribuídos via **GitHub Release** do repo
privado `jhlr/leiturista`.

## Conteúdo do release

Tag: `modelos-1.0` (link: `https://github.com/jhlr/leiturista/releases/tag/modelos-1.0`)

- `leiturista-models.tar.gz` (147M, SHA256 `f02c13ee5a...63e10`) com:
  1. `models/pp_ocr_v5_mobile_det_onnx/` — detector (PP-OCRv5_mobile_det, ONNX).
  2. `models/pp_ocr_v6_tiny_rec_onnx/` — recognizer fallback (PP-OCRv6_tiny, ONNX).
  3. `.model_cache/trocr-small-printed/` — TrOCR-small-printed fine-tuned,
     materializado do run MLflow `9c14db62` (blobs), fonte de verdade que o
     `MeterOCR` carrega (`inference.py:_load_trocr`).

O `scripts/ppocr_v6_dict.json` (dict do recognizer) já está no repo.

## Passos para o colega

1. **Acesso:** ser convidado como colaborador do repo privado `jhlr/leiturista`
   no GitHub (Settings → Collaborators). Clonar:
   ```bash
   git clone git@github.com:jhlr/leiturista.git && cd leiturista
   ```
2. **Ambiente:**
   ```bash
   python3 -m venv .venv && .venv/bin/pip install -e .
   ```
3. **Baixar os modelos** (Release → `leiturista-models.tar.gz`):
   ```bash
   tar -xzf leiturista-models.tar.gz
   ```
   (extrair na raiz do repo; cria `models/` e `.model_cache/`)
4. **Rodar a demo:**
   ```bash
   .venv/bin/streamlit run app/app.py --server.headless true
   ```
   Upload de foto → leitura + serial + assinatura/identidade do medidor.

## Verificação

Sem os modelos, o app não sobe limpo (sem blobs no `mlflow.db` a materialização
falha). Com o tarball extraído na raiz, deve rodar direto — os caminhos são
relativos a `paths.ROOT` (`src/leiturista/paths.py`).
