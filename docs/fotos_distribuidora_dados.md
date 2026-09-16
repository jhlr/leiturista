# Fotos da distribuidora — lote real (piloto da disciplina) (dataset de fine-tune)

**Data:** 2026-09-14 | **Status:** import e QA concluídos | **Origem:** pap. da
distribuidora (Kickoff). Complementa a descoberta em `origem_dos_dados.md`.

## O que chegou

Pasta `FotosDistribuidora/` no repo (não versionada — ver .gitignore):

- 4 lotes `PSP_EXTRATLEITIMPL_{data}_{hora}/`, cada um com:
  - `BaseExtracao_{data}_Dia.csv` — 5 colunas:
    `Numero do medidor ; Posicao do medidor lida ; Nota de Leitura Atual ; Foto do medidor`
  - fotos `*.jpg` 360x480 (foto do **medidor inteiro**, display eletrônico)
- `DESCRIÇÃO NOTAS LEITURISTAS X SOLICITAÇÃO DE FOTO.xlsx` — catálogo de notas
  (ex.: `NA`=normal, `I100`=casa fechada, `L101`=leitura informada pelo cliente,
  `T181`=função não existe no sistema) com flag "exige foto?".

## Números

- 11.474 fotos (1 por medidor, sem duplicatas) com leitura numérica no CSV.
- Nota `NA` (leitura normal): **7.265** fotos — usadas como base do fine-tune.
- Leitura registrada: inteiro, tipicamente 4–6 dígitos (até 999999).

## Decisão da sessão

Treinar **só nota `NA`** (rótulo = leitura registrada no display) a partir do
checkpoint **`.model_cache/trocr-small-printed`** (já em disco; NÃO baixar stage1).

## Pipeline (`leiturista import-distribuidora`)

Módulo novo `src/leiturista/distribuidora.py` (+ subcomando CLI `import-distribuidora`):

1. `load_rows()` — parseia os CSVs, filtra foto existente, normaliza leitura p/ dígitos.
2. `_reading_candidate()` — det (PP-OCRv5) normal+invertido → merge/quads → crop →
   rec (PP-OCRv6) p/ classificar `leitura`; escolhe o crop com maior alinhamento de
   dígitos vs rótulo (`_digit_match`, alinhado à direita).
3. Filtros de qualidade (calibrados com QA visual do usuário):
   - só 3..8 dígitos (mata placa de identidade ~10 dígitos);
   - nº de dígitos >= dígitos do rótulo;
   - dígitos extras à esquerda **só zeros** (mata voltagem colada `12038034`,
     placa `83015067`, rótulo incompatível `300085` vs `85`);
   - rejeita texto de unidade (`120V`, `240V`, `60Hz`, `kW`, `kWh`, `A`);
   - match vs rótulo >= 0.5;
   - nitidez (Laplaciano) >= 10 (gate frouxo — o match é o discriminante).
4. `build_dataset()` — grava crops `.png` + `labels.csv` (`split,image,label`,
   split **sobre o pool de aceitos**, 80/10/10, seed 42) + `manifest.csv` (auditoria:
   lote/foto/medidor/leitura/rec/match/nitidez/source) + mosaicos em `_qa/`.

### Uso

```bash
.venv\Scripts\python.exe -m leiturista.cli import-distribuidora -o data/distribuidora_amr
# debug:  -n 150   |  trocar notas:  --notes NA,L101   |  limiar:  --min-sharpness 10
```

## Resultado do import (7.265 fotos NA)

| status | qtd | obs |
|---|---|---|
| aceito | **1.087** | crops de display com leitura validada |
| borrado (nitidez < 10) | 6 | |
| sem crop viável | 6.172 | display não localizado/ilegível na foto |

- Split: train **869** / valid **108** / test **110**.
- `match` mediano 1.00 (≥0.8 em 838) — rec confirma o dígito do rótulo no crop.
- source: normal 634 / inverted 453 (crop da imagem invertida p/ display claro-escuro).
- crops 360x480-fonte; larguras típicas mediana ~145px original, ratio w/h ≥ 2 (tira de display).

## Tratar como dado curado, ainda assim conferir

O `manifest.csv` lista `match` por amostra — use para auditoria. Amostras com
`match < 0.8` (249) são as de leitura parcial/rec fraca; avaliar se mantê-las no
treino após a 1ª rodada.

## Próximo passo

Fine-tune TrOCR (SÓ O USUÁRIO DISPARA):

```bash
.venv\Scripts\python.exe -m leiturista.cli train ^
  --model .model_cache/trocr-small-printed ^
  --data data/distribuidora_amr ^
  --out models/trocr-small-finetuned-distribuidora ^
  --epochs 8 --batch 4 --grad-accum 8 --lr 2e-5 ^
  --experiment trocr-distribuidora
.venv\Scripts\python.exe -m leiturista.cli eval ^
  --data data/distribuidora_amr --out models/trocr-small-finetuned-distribuidora
```

Comparar com baselines UFPR-AMR (doc `finetune_trocr_ufpramr.md`):
PP-OCRv6_tiny EM 0.357 / digit-acc 0.846 (~o rec usado para gerar estes crops).

## Nota importante (insight)

Nos crops corretamente localizados o rec PP-OCRv6 já casa com o rótulo (match ~1.0).
O gargalo nos dados reais é a **localização do display na foto** (6.172/7.265 sem
crop), não a leitura. O fine-tune do TrOCR melhora o leitor principal do pipeline,
mas a detecção/classificação de campo (`leitura` vs serial/placa) permanece o topo
de melhoria a atacar em seguida.