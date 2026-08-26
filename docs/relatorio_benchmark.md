# Relatório / Benchmark — Leiturista (Projeto 4 · distribuidora de energia)

**Data:** 2026-08-15 (rev. 2026-08-16) | **Escopo:** achados e performance do que JÁ temos (modelos off-the-shelf + pipeline)
**Disciplina:** Projeto 4 - DADOS (Cesar School, BD2026.2) | **Grupo 3**
**Repo:** `jhlr/leiturista` (privado)

---

## 1. Problema

Uma **distribuidora de energia elétrica** quer automatizar a fiscalização das **fotos tiradas
pelos leituristas**:

1. **Tarefa 1 — Leitura:** extrair o número do display do medidor de energia a partir da foto (OCR).
2. **Tarefa 2 — Validação/coerência:** a foto corresponde ao medidor/cliente? É coerente com a
   nota/ocorrência do leiturista (ex.: I100 – casa fechada)?

Este relatório cobre o **estado atual**: baselines e pipeline **off-the-shelf** (sem treino próprio),
com números medidos no test set público UFPR-AMR e achados de campo de projeto.

## 2. Dados e metodologia

| Item | Valor |
|---|---|
| Dataset | **UFPR-AMR** (Laroca, IJCNN 2020 — 2.000 imgs de display de medidor BR; 1.400/300/300) |
| Split avaliado | **test** (300 imagens) |
| Imagens | recorte do display (text-line), prontas para o recognizer |
| Comprimento médio da leitura | ~4,5 dígitos |
| Modelos | off-the-shelf, sem qualquer fine-tune |
| Origem | mirror HF `Chaymaa/UFPR-AMR` (base: dataset da UFPR, Laroca IJCNN 2020) |
| Download | `hf download Chaymaa/UFPR-AMR --local-dir data/ufpr_amr` (3 parquets) |
| Extração | `leiturista extract` → imagens + `labels.csv` (`split,image,label`) |
| Fotos de campo p/ achados | `Praekelt/ElectricityMeterReadings1o4` (165 fotos reais — demo/inspeção §5) |

**Preparação (detalhe):** o mirror HF guarda 3 parquets (`train`/`test`/`valid`) com a imagem e
a leitura (`gt_parse`). `leiturista extract` lê os parquets e grava as imagens em arquivos flat +
`labels.csv` em `data/finetune_ufpramr/` — a fonte que o baseline e o `leiturista eval` leem. O
split **test (300 imagens)** é o avaliado aqui.

**Métricas** (definidas em `scripts/ppocr_baseline.py` e `src/leiturista/eval.py`):
- **exact-match:** fração de imagens com leitura 100% correta (idêntica ao rótulo).
- **digit-acc:** acurácia por dígito, alinhando a predição **da direita para a esquerda**
  (`pred.rjust(len(label))`) — robusta a variação de comprimento.

> Nota de transparência: o TrOCR (beam search) emite tokens separados por espaço. Reportamos as
> métricas **cruas** (como logadas no MLflow, contando o espaço) **e limpas** (removendo os espaços,
> comparável ao PP-OCR, que não emite espaço). Ver §3 (linhas "cru"/"limpo").

## 3. Benchmark — Tarefa 1 (leitura do display)

| Modelo (off-the-shelf) | exact-match | digit-acc | Definição | MLflow run |
|---|---|---|---|---|
| **PP-OCRv6_tiny_rec** (1,1M params) | **0.357** | **0.846** | decode CTC, sem espaço — "limpo" por natureza | `baseline-ppocrv6-tiny` |
| **TrOCR-small-printed** (62M params, MIT) | 0.160 | 0.642 | cru (beam, conta espaços) | `trocr-small-printed-offtheshelf` |
| **TrOCR-small-printed** (idem) | **0.253** | **0.776** | limpo (remove espaços do beam) | — (recomputado do `eval_test.csv`) |

**Leitura:**
1. O **PP-OCRv6_tiny** (reconhecedor leve da família PaddleOCR) é o melhor baseline off-the-shelf
   no display: **35,7%** de leituras exatas, **84,6%** de dígitos certos.
2. O **TrOCR-small-printed** (OCR de texto impresso genérico, AAAI 2023) **não foi treinado para
   display de medidor** — é competitivo por dígito quando limpo (77,6%), mas erra mais leituras
   inteiras (25,3% limpo). A métrica crua do MLflow (0.642) é **otimista**: conta o espaço de
   separação do beam como caractere.
3. Conclusão de benchmark: **para Tarefa 1 com modelos prontos, o PP-OCRv6 é a melhor leitura
   individual**; o TrOCR tem leitura "mais fiel" em amostras inspecionadas, mas acerta menos no
   agregado.

## 4. Pipeline atual (em produção na demo)

`src/leiturista/inference.py` (`MeterOCR`) — det → rec → classificação → seleção:

1. **Detecção:** PP-OCRv5_mobile_det (ONNX) — caixas de texto (display, placa, serial).
   Post-process DB (thresh 0.3, box_thresh 0.6, unclip 1.5); ONNX exige dimensões múltiplas de 32
   (padding + remapeamento das caixas).
2. **Reconhecimento:** **TrOCR-small-printed** (leitor principal de cada linha) com **fallback
   PP-OCRv6_tiny** quando o TrOCR devolve vazio — combina a fidelidade do TrOCR com a robustez do
   PP-OCRv6.
3. **Classificação por campo:**
   - `leitura`: só dígitos (aceita `.`/`,`/espaço como separadores), 1–8 dígitos.
   - `serial`: alfanumérico, 6–24 chars, excluindo símbolos de placa (`= ° ( ) % / _ ~ "`…).
4. **Seleção da leitura:** prioriza texto **só-dígitos**, depois o de **mais dígitos**; nunca junta
   caixas soltas (vira lixo em foto completa).

## 5. Achados (findings) e performance deles

### 5.1 Rotação da foto — "rotação-primeiro"
Fotos de campo vêm em qualquer orientação. Em vez de testar N rotações (caro e propenso a
curto-circuito), o pipeline **detecta a orientação antes** (`_detect_rotation`): roda o det em
imagem reduzida (640px) e pontua a fração de área **horizontal** das linhas de texto mescladas.
- Performance: correto nos 6 casos de teste; display de odômetro só legível invertido (§5.3).

### 5.2 Duas fases de inferência (sempre)
1. **Fase 1:** imagem normal, caixas mescladas por linha.
2. **Fase 2:** imagem invertida, quads individuais (sem merge).
Display **branco-no-preto** (odômetro mecânico) só aparece na fase 2. Cada caixa carrega a origem
(`normal`/`inverted`).

### 5.3 Retry invertido (display claro-em-escuro)
O PP-OCRv5 foi treinado em texto **escuro-em-claro**; o odômetro (dígitos claros em roda escura) é
invisível pro det. Quando a leitura não é encontrada, o pipeline **inverte a imagem** e refaz det+rec
nos quads individuais (sem merge, para não juntar display e texto ao lado).
- **Caso real:** `praekelt_02_41365` (odômetro Landis+Gyr) → sem inversão `None`; com inversão lê
  **41365**.

### 5.4 Seleção da leitura (anti-gambiarra)
Preferir **texto só-dígitos** primeiro, depois **mais dígitos**. Preferir fonte invertida quebrou
display preto-no-branco (ex.: `6768746` virou `30`). O concorrente de placa (`83415801:` do caso
41365) tem `:` e cai sozinho no filtro de só-dígitos.

### 5.5 Serial vertical
Caixas verticais (w < 1.5h) com dígitos e ≥ 6 chars são **rebaixadas a "serial"** (serial estampado
na vertical lê como só-dígitos e disputaria com a leitura do display).

### 5.6 Assinatura / identidade do medidor
Tudo que **não é leitura** (serial + texto da placa) vira a **assinatura** do medidor: tokens
normalizados (maiúsculo, alfanumérico, ≥2 chars) ordenados por posição (y, x) → hash sha1 de 12
chars. `meter_similarity` (Jaccard dos tokens) re-identifica o mesmo medidor entre fotos mesmo com o
serial ilegível: mesmo medidor ≈ 1.0, distintos ≈ 0.0.

### 5.7 Flags de coerência (Tarefa 2 — sementes)
O app já emite sinais usáveis na validação: legibilidade do crop (var. do Laplacian < 25 →
"display borrado"), leitura não detectada, múltiplas leituras, serial ausente.

## 6. Tarefa 2 — Validação de cena/coerência (status)

**ATUALIZADO 2026-08-25 — gap resolvido:** o grupo obteve uma amostra real de **uma distribuidora parceira**
(`data/distribuidora_campo/`, gitignored): ~12.343 imagens de campo em 4 lotes diários, os CSVs
`BaseExtracao_<data>_Dia.csv` com a nota/ocorrência aplicada pelo leiturista por foto, e o
catálogo completo de 61 códigos de ocorrência (`DESCRIÇÃO NOTAS LEITURISTAS X SOLICITAÇÃO DE
FOTO.xlsx`, 32 exigem foto / 29 não exigem). Falta apenas o rótulo de decisão da fiscalização
(foto aceita/rejeitada) — ver `docs/pedido_kickoff.md` para o detalhamento e as perguntas ainda
em aberto para o Kickoff (12/09).

<details><summary>Texto original (2026-08-16, histórico)</summary>

**GAP confirmado** (`docs/analise_suficiencia.md`): **não existe dataset público com as notas do
leiturista** (códigos tipo I100 são proprietários de cada distribuidora). Candidatos mapeados sem
download (pendência de licença): Copel-AMR (12,5k fotos de campo BR), UFPR-ADMR-v2 (5k dials),
IEEE DataPort (570). A estratégia é um **piloto** com esses dados e, sobretudo, o **lote real da
distribuidora no Kickoff (12/09)**.

</details>

## 7. Limitações honestas (o que o relatório NÃO esconde)

1. Todos os números acima são de modelos **off-the-shelf** — a taxa real de leitura no agregado é
   ~25–36% de leituras exatas (melhor por dígito: 77–85%).
2. **Serial é heurística** (não há detector específico); ruído da placa pode vazar como "serial".
3. **Legibilidade é sinal, não rótulo**: threshold simples de Laplacian.
4. Limite assumido: `praekelt_93809` — serial ganha; odômetro pequeno não é segmentado pelo det nem
   invertido, e o TrOCR alucina no crop. Não há localização manual do display (o modelo não terá
   essa opção em produção).

## 8. Reprodução completa (passo a passo)

Todo o pipeline vive na biblioteca `leiturista` (CLI instalada via `pip install -e .`).
Números verificáveis: **0.357 / 0.846** (PP-OCRv6) e **0.160 / 0.642** (TrOCR cru),
**0.253 / 0.776** (TrOCR limpo).

### 8.1 Ambiente

```bash
git clone git@github.com:jhlr/leiturista.git && cd leiturista
python3 -m venv .venv
.venv/bin/pip install -e .        # instala o CLI `leiturista` + deps (torch, transformers, paddle…)
```

### 8.2 Dados (UFPR-AMR)

```bash
# 1) download do mirror HF (3 parquets; ~10 min, usar em background)
HF_HUB_DISABLE_XET=1 .venv/bin/hf download Chaymaa/UFPR-AMR --local-dir data/ufpr_amr

# 2) extrai imagens + labels.csv para data/finetune_ufpramr/
.venv/bin/leiturista extract
```

`extract` grava `labels.csv` com colunas `split,image,label` (leitura) — é o que os baselines leem.

### 8.3 Baselines (reproduzir os números do §3)

```bash
# PP-OCRv6_tiny_rec — baseline off-the-shelf no test (300 imgs)
.venv/bin/python scripts/ppocr_baseline.py

# TrOCR-small-printed — avalia o checkpoint (modelo) no test
.venv/bin/leiturista eval --data data/finetune_ufpramr --out models/trocr-small-printed

# TrOCR "limpo": remove os espaços do beam no eval_test.csv gerado e recomputa
.venv/bin/python - <<'EOF'
import pandas as pd
df = pd.read_csv("models/trocr-small-printed/eval_test.csv")
df["pred"] = df["pred"].str.replace(" ", "")
exact = (df["pred"] == df["label"]).mean()
digit = df.apply(lambda r: sum(p == l for p, l in zip(r["pred"].rjust(len(r["label"]))[-len(r["label"]):], r["label"])) / len(r["label"]) if r["label"] else 0.0, axis=1).mean()
print(f"exact-match={exact:.4f} digit-acc={digit:.4f}")
EOF
```

### 8.4 Tracking e inspeção (MLflow, tudo num `mlflow.db`)

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
.venv/bin/leiturista artifacts <run_id>            # lista blobs (eval_test.csv, eval_summary.txt…)
.venv/bin/leiturista artifacts <run_id> eval_test.csv -o eval_test.csv
.venv/bin/leiturista restore <run_id> checkpoint.zip -o models/   # extrai checkpoint
```

Runs registrados (experimento `trocr-ufpramr`):

| Run id | Nome | Conteúdo |
|---|---|---|
| `eff9834d` | `baseline-ppocrv6-tiny` | PP-OCRv6: exact 0.357 / digit 0.846 + `eval_test.csv` |
| `9c14db62` | `trocr-small-printed-offtheshelf` | TrOCR cru: exact 0.160 / digit 0.642 + blobs do modelo |
| `e2c212ee` | `eval-test` | re-avaliação `models/trocr-small-printed` (mesmos 0.160 / 0.642) |
| `4021ef26` | `train-8ep` | fine-tune **incompleto** (sem checkpoint) — rerodar `leiturista train` |

### 8.5 Modelos utilizados (off-the-shelf)

| Modelo | Repo HF | Tamanho | Licença | Papel |
|---|---|---|---|---|
| PP-OCRv5_mobile_det (ONNX) | `PaddlePaddle/PP-OCRv5_mobile_det_onnx` | ~2 MB | Apache-2.0 | detecção de caixas de texto |
| PP-OCRv6_tiny_rec (ONNX) | `PaddlePaddle/PP-OCRv6_tiny_rec_onnx` | 4,4 MB | Apache-2.0 | reconhecimento (1,1M params) |
| TrOCR-small-printed | `microsoft/trocr-small-printed` | 493 MB | MIT | OCR end-to-end (62M params) |
| TrOCR-small-stage1 | `microsoft/trocr-small-stage1` | 247 MB | MIT | base p/ fine-tune (`leiturista train`) |

### 8.6 Artefatos pré-empacotados

Os modelos já materializados estão na **GitHub Release `modelos-1.0`** do repo:
`leiturista-models.tar.gz` (147 MB, SHA256 `f02c13ee5a…63e10`) com o detector ONNX, o recognizer
ONNX e o TrOCR em `.model_cache/`. Extrair na raiz do repo habilita a demo imediatamente.

```bash
tar -xzf leiturista-models.tar.gz
.venv/bin/streamlit run app/app.py --server.headless true   # demo (leitura + serial + flags)
```

## 9. Referências

1. **UFPR-AMR (dataset + baselines).** Laroca, R., et al., *Deep Learning for Image-based Automatic
   Dial Meter Reading: Dataset and Baselines*, IJCNN 2020. DOI: `10.1109/IJCNN48605.2020.9207318`.
   Mirror HF: `Chaymaa/UFPR-AMR`.
2. **TrOCR.** Li, M., et al., *TrOCR: Transformer-based Optical Character Recognition with
   Pre-trained Models*, AAAI 2023. arXiv: `2109.10282` (checkpoint `microsoft/trocr-small-printed`).
3. **PP-OCR (família PaddleOCR).** PaddlePaddle — detector/reconhecedor leves para cenas reais.
   Checkpoints ONNX: `PaddlePaddle/PP-OCRv5_mobile_det_onnx`, `PaddlePaddle/PP-OCRv6_tiny_rec_onnx`.
4. **Dados de campo p/ os achados (§5).** `Praekelt/ElectricityMeterReadings1o4` (HF, fotos reais).
5. **Copel-AMR / UFPR-ADMR-v2 (candidatos da Tarefa 2).** datasets BR de campo sob licença
   acadêmica — contato `menotti@inf.ufpr.br` (ver §6).
6. **Contexto interno do projeto:** `docs/projeto4_desafio.md`, `docs/plano_subprojeto_cv.md`,
   `docs/artigos.md`, `docs/origem_dos_dados.md` (não necessários para reproduzir o §3).
