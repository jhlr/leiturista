# Classificador de cena, detector de visor fine-tunado e retest do amarelo

**Data:** 2026-10-08. Treino dos dois modelos disparado a pedido do usuário. Amostra: as mesmas 500
fotos (seed 0) da baseline de 2026-10-01, que precisam de OCR. Números por foto vêm dos JSONs
`docs/2026-10-08_triagem_*` (população = R1/R2 fixos + amostra extrapolada, IC de Wilson 95%).

## 1. Classificador de cena (MobileNetV3-Small, "tem medidor?")

`leiturista train-scene` (10 épocas, ~50 min em CPU), `models/scene_medidor.pt`. Rótulos fracos do
LLM; só 26 (valid) e 28 (test) fotos "sem medidor".

| Split | AUROC | "sem medidor": precisão / recall / F1 |
|---|---|---|
| valid | 0,982 | 0,864 / 0,731 / 0,792 |
| test | n/d | 0,550 / 0,393 / 0,458 |

O F1 no test é bem pior que no valid; com 28 negativos e rótulo fraco, o número tem margem larga.
Efeito no amarelo: converte ~1-3 pontos (28 das 500 fotos viram vermelho), ver tabela.

## 2. Detector de visor (PP-OCRv5 mobile det fine-tunado)

Só havia ONNX (sem Paddle): ONNX -> PyTorch (`onnx2torch`, saída idêntica ao ORT, 1e-7) -> loss DB
só no mapa de probabilidade (BCE com hard negative mining + dice, shrink 0,4) -> ONNX no mesmo
contrato (`src/leiturista/det.py`, `leiturista train-det`, `models/pp_ocr_v5_mobile_det_visor_onnx/`).
Treino: 872 quads do visor (`boxes_visor.csv`), 8 épocas, ~7 min/época em CPU, augment de escala,
brilho e inversão. Split do CSV é aleatório entre os 4 lotes (não por lote).

Recall do visor (IoU >= 0,5, foto original ou invertida) e caixas por foto:

| Detector | valid | test | caixas/foto |
|---|---|---|---|
| Original | 0,936 | 0,891 | ~35 |
| Fine-tunado, limiar padrão (0,3 / 0,6) | 0,688 | 0,827 | ~1,9 |
| Fine-tunado, limiar calibrado (0,2 / 0,2) | 0,84 (valid+test) | | ~2,3 |

Ressalvas: (a) os positivos só existem em fotos onde o det original já achava algo, então o recall
do original é otimista e o do fine-tunado é medido num conjunto viesado; (b) o limiar foi calibrado
em valid+test juntos (vazamento leve); (c) o recall oscilou entre épocas (0,53-0,69 no valid), 8
épocas é pouco.

## 3. Retest do amarelo (população de 13.669 registros)

| Configuração | Verde | Amarelo | Vermelho | Amarelo com cena |
|---|---|---|---|---|
| Baseline 2026-10-01, PP-OCR | 5,4% | 88,7% (86,8-90,1) | 5,8% | 85,4% |
| Baseline, CRNN do professor | 4,5% | 89,7% | 5,8% | n/d |
| Det fine-tunado, limiar padrão, PP-OCR | 7,2% | 59,2% | 33,6% | 58,7% |
| **Det fine-tunado, limiar calibrado, PP-OCR** | 6,9% | **76,6% (73,6-79,3)** | 16,5% (14,0-19,4) | **75,3%** |
| Det fine-tunado calibrado, CRNN novo (`crnn_bn128_mix500_cur30`) | 4,5% | 79,0% | 16,5% | 77,7% |

**Leitura honesta:**
1. O amarelo cai de 88,7% para 76,6% (75,3% com a cena), mas **quase tudo vira vermelho, não
   verde**: o vermelho sobe de 5,8% para 16,5%. O vermelho aqui é "nenhuma caixa detectada" (R3); o
   detector novo não acha nada em ~27% das fotos (134/500) mesmo com limiar baixo. Parte disso é
   foto sem medidor, mas o classificador de cena só aponta 28 fotos sem medidor, então boa parte é
   **vermelho falso** (visor não localizado). Não é ganho de qualidade; é troca de erro.
2. Com o limiar padrão (0,3/0,6) o detector novo "resolvia" o amarelo para 59% e gerava 34% de
   vermelho: engano puro. Reportado só para documentar.
3. O verde mal mexe (5,4% -> 6,9%): sem leitura certa nada vira verde. O CRNN (UFPR-AMR) não
   acerta nenhum recorte de campo da amostra (0 verdes além dos 4,5% de R1), confirmando o gap de
   domínio já registrado; trocar para o CRNN novo (0,873 no UFPR-AMR) não mudou isso.
4. Latência média cai de 4,76 s para 0,91 s por foto (poucas caixas, pouco TrOCR).
5. Meta (< 40%) segue longe. O caminho que sobra: treinar o leitor em recortes da distribuidora
   (alavanca 3 do doc `2026-10-01_gargalo_amarelo_e_plano_visor.md`) e usar o det novo como
   *complemento* do original (união de candidatos), não substituto: o fine-tune hoje troca recall
   por precisão.
