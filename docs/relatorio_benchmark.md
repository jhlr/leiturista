# Relatório / Benchmark — Leiturista (Projeto 4 · Neoenergia PE)

**Data:** 2026-08-15 | **Escopo:** achados e performance do que JÁ temos (modelos off-the-shelf + pipeline)
**Disciplina:** Projeto 4 - DADOS (Cesar School, BD2026.2) | **Grupo 3** | **Professor:** Erick Simões
**Repo:** `jhlr/leiturista` (privado)

---

## 1. Problema

A Neoenergia PE quer automatizar a fiscalização das **fotos tiradas pelos leituristas**:

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

**Métricas** (definidas em `scripts/ppocr_baseline.py` e `src/leiturista/eval.py`):
- **exact-match:** fração de imagens com leitura 100% correta (idêntica ao rótulo).
- **digit-acc:** acurácia por dígito, alinhando a predição **da direita para a esquerda**
  (`pred.rjust(len(label))`) — robusta a variação de comprimento.

> Nota de transparência: o TrOCR (beam search) emite tokens separados por espaço. Reportamos as
> métricas **cruas** (como logadas no MLflow, contando o espaço) **e limpas** (removendo os espaços,
> comparável ao PP-OCR, que não emite espaço). Ver §4.

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

**GAP confirmado** (`docs/analise_suficiencia.md`): **não existe dataset público com as notas do
leiturista** (códigos tipo I100 são proprietários de cada distribuidora). Candidatos mapeados sem
download (pendência de licença): Copel-AMR (12,5k fotos de campo BR), UFPR-ADMR-v2 (5k dials),
IEEE DataPort (570). A estratégia é um **piloto** com esses dados e, sobretudo, o **lote real da
Neoenergia no Kickoff (12/09)**.

## 7. Limitações honestas (o que o relatório NÃO esconde)

1. Todos os números acima são de modelos **off-the-shelf** — a taxa real de leitura no agregado é
   ~25–36% de leituras exatas (melhor por dígito: 77–85%).
2. **Serial é heurística** (não há detector específico); ruído da placa pode vazar como "serial".
3. **Legibilidade é sinal, não rótulo**: threshold simples de Laplacian.
4. Limite assumido: `praekelt_93809` — serial ganha; odômetro pequeno não é segmentado pelo det nem
   invertido, e o TrOCR alucina no crop. Não há localização manual do display (o modelo não terá
   essa opção em produção).

## 8. Artefatos e reprodução

- Código: biblioteca `leiturista` (`src/leiturista/`), demo `app/app.py`.
- Baselines: `scripts/ppocr_baseline.py` (PP-OCRv6 no test UFPR-AMR, registra MLflow).
- Tracking: `mlflow.db` (runs `baseline-ppocrv6-tiny`, `trocr-small-printed-offtheshelf`; CSVs de
  predição como blobs).
- Modelos distribuídos via GitHub Release `modelos-1.0` (`docs/compartilhar_modelo_colega.md`).
- Docs de referência: `docs/projeto4_neoenergia.md`, `docs/plano_subprojeto_cv.md`,
  `docs/artigos.md`, `docs/origem_dos_dados.md`.
