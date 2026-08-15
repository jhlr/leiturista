# Demo Streamlit — Leiturista (leitura + serial do medidor)

**Data:** 2026-08-10 | **Status:** funcional (modelos off-the-shelf)

App de demonstração do pipeline de leitura de medidores (Projeto 4 / Neoenergia PE):
upload de foto → **leitura do display (OCR)** + **serial** + **flags de coerência**.
Serve de MVP visual para SR2 e de ferramenta de inspeção das caixas detectadas.

## Rodar

```bash
.venv/bin/streamlit run app/app.py   # abre em http://localhost:8501
```

## Pipeline (biblioteca `leiturista.inference`)

1. **Detecção (det):** `PP-OCRv5_mobile_det_onnx` (ONNX, Apache-2.0) — acha as caixas de
   texto na foto (display, placa/serial). Post-process DB (thresh 0.3, box_thresh 0.6,
   unclip 1.5). ONNX exige dims múltiplas de 32 (padding + mapeamento das caixas de volta).
2. **Reconhecimento (rec):** `TrOCR-small-printed` (`microsoft/trocr-small-printed`, MIT,
   62M params, AAAI 2023) — leitor principal de cada linha. Materializado do `mlflow.db`
   (blobs `trocr-small-printed/*`, run `9c14db62`) com cache em `.model_cache/`;
   fallback `PP-OCRv6_tiny_rec_onnx` (1,1M params) quando o TrOCR devolve vazio.
   Test UFPR-AMR: TrOCR exact-match 0.160 / digit-acc 0.642 (leitura mais fiel que o
   PP-OCRv6 em amostras); PP-OCRv6: 0.357 / 0.846.
3. **Classificação por campo:**
   - `leitura`: só dígitos (`.`/`,`/espaço como separadores), 1–8 dígitos.
   - `serial`: alfanumérico (letra + dígito), 6–24 chars, sem símbolos de especificação
     (`= ° ( ) % / _ ~ "` ou `..`) — exclui specs de placa como "220V5(60)A50Hz".
4. **Seleção da leitura:** caixa de leitura com mais dígitos **e texto só-dígitos**
   (preferido) — nunca a junção de caixas soltas (vira lixo em foto completa).
5. **Flags de coerência:** legibilidade do crop do display (var. do Laplacian < 25 →
   "display borrado"), leitura não detectada, múltiplas leituras, serial ausente.

## Retry com imagem invertida (display claro-em-escuro)

Display de **odômetro mecânico** (dígitos claros em roda escura) é invisível pro det —
o PP-OCRv5 foi treinado em texto escuro-em-claro. Quando a leitura **não é encontrada ou
a fidelidade é baixa** (Laplacian), o pipeline inverte a imagem (fase 2) e refaz det+rec
nos quads individuais (sem merge — a linha mesclada juntaria o display ao texto ao lado):

1. `reading is None` → tenta TrOCR na imagem invertida inteira.
2. Caixa de leitura invertida só **substitui** a da fase 1 se tiver mais dígitos.
3. Caso real: `praekelt_02_41365.5` (odômetro Landis+Gyr) → sem inversão retorna `None`;
   com inversão lê `41365` (truncado do decimal).

## Saída

- Imagem anotada (verde = leitura, laranja = serial, cinza = outro) com o texto por caixa.
- Métricas: leitura (kWh), serial, legibilidade.
- Tabela de caixas (campo, texto, confiança do box).

## Identidade do medidor (assinatura)

Tudo que **não** é leitura (serial + texto da placa) vira a **assinatura/identidade**
do medidor: tokens normalizados (maiúsculo, alfanumérico, ≥2 chars) ordenados por
posição (y, x), com hash sha1 determinístico. Permite re-identificar o mesmo medidor
entre fotos mesmo com o serial ilegível. Comparação fuzzy via
`leiturista.inference.meter_similarity` (Jaccard dos tokens) — mesmo medidor ≈ 1.0,
medidores distintos ≈ 0.0.

## Limitações (honestas)

- Modelos **off-the-shelf** (sem fine-tune): no test UFPR-AMR o TrOCR-small-printed tem
  exact-match 0.160 / digit-acc 0.642; o PP-OCRv6_tiny_rec 0.357 / 0.846. Leituras podem
  sair erradas; o app mostra as caixas para conferência.
- Serial é heurística (não há detector específico); ruído da placa pode vazar como
  "serial".
- Legibilidade usa threshold simples de Laplacian — sinal, não rótulo de qualidade.

## Próximos passos

1. `leiturista train` (fine-tune TrOCR-small no UFPR-AMR) → comparar no app.
2. Copel-AMR / lote real da Neoenergia (Kickoff 12/09) → piloto da tarefa de cena.
3. Flags de cena (Moondream2 zero-shot) no app: "a casa está fechada?" (I100).
