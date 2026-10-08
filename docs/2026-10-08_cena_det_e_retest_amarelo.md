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
| **União** det original + visor-ft (`LEITURISTA_DET2_ONNX`), PP-OCR | 5,4% | 90,0% (88,3-91,2) | 4,6% | 86,7% |
| União, CRNN novo | 4,6% | 90,8% | 4,6% | 87,5% |

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

## 4. União dos detectores (testada em seguida)

O 2º detector é opt-in (`LEITURISTA_DET2_ONNX`, limiar próprio 0,2/0,2): os candidatos passam a ser a
união das caixas do det original e do visor-ft. **Resultado: nada ganho.** Vermelho volta a 4,6%
(sem vermelho falso), mas o amarelo vai para 90,0% e o verde fica em 5,4%, igual à baseline. A
seleção do candidato (`_best_reading`, "mais dígitos") continua escolhendo serial/placa em vez do
visor, então ter a caixa do visor entre os candidatos não muda a leitura. Conclusão: o gargalo
deixa de ser *achar* o visor e passa a ser *escolher* entre candidatos e *ler* o recorte, ou seja, o
passo A do plano (reranker "é visor?") mais fine-tune do leitor em recorte da distribuidora. O
detector fine-tunado sozinho só troca amarelo por vermelho falso; a união não vale o custo de
latência (4,6 s/foto contra 0,9 s do det novo sozinho).

## 5. Regra das notas que não exigem foto (planilha do cliente)

`DESCRIÇÃO NOTAS LEITURISTAS X SOLICITAÇÃO DE FOTO.xlsx` lista 61 códigos com "exige foto" SIM/NÃO.
Os "NÃO" (ex.: T111, L131, T161, M141, R111) já vão direto para **verde (R1, 611 registros, 4,5%)**
sem olhar a imagem; os "SIM" sem foto vão para vermelho (R2, 296). **A planilha não tem a nota
`NA`** (9.109 registros: 7.265 com foto, 1.844 sem) nem `V100` (6). Por instrução do usuário
(2026-10-08: "códigos que não exigem foto não vão para o amarelo"), `NA` sem foto passou a **verde
(R1b)**, em vez de amarelo (R2b, regra antiga); `V100` sem foto segue amarelo. É suposição fora da
planilha: `--na-sem-foto amarelo` volta à regra antiga, e as tabelas abaixo trazem as duas. R1b move
13,5 pontos de amarelo para verde sem modelo nenhum.

## 6. CRNN fine-tunado em recortes da distribuidora

`leiturista train-crnn --data data/distribuidora_amr --data data/finetune_ufpramr --init
models/crnn_bn128_mix500_cur30.pt` (30 épocas, lr 3e-4, inversão 0,4, jitter 0,07), saída
`models/crnn_dist_ft.pt`. Rótulos dos recortes = leitura digitada confirmada pelo OCR (viés de
recortes fáceis).

| Leitura exata | Antes (zero-shot) | Depois |
|---|---|---|
| Distribuidora test (n=110) | 0,036 | **0,609** (0,873 por dígito) |
| UFPR-AMR test (n=300) | 0,873 | 0,890 (não piorou) |

Retest ponta a ponta. A amostra exclui 46 fotos cujo recorte treinou/validou o CRNN (454 fotos), e as
colunas de política `NA sem foto`: verde (R1b) × amarelo (regra antiga):

| Detector + leitor | Verde | Amarelo (NA=verde) | Amarelo (NA=amarelo) | Vermelho |
|---|---|---|---|---|
| Baseline, PP-OCR | 18,9% | 75,2% | 88,7% | 5,8% |
| União, CRNN fine-tunado | 22,0% | 73,2% (69,5% com cena) | 86,7% | 4,8% |
| Det visor-ft, CRNN fine-tunado | 24,1% | 58,1% (56,7% com cena) | 71,6% | 17,8% |

Leitura: o ganho real do CRNN fine-tunado é no verde (18,9 -> 22,0% com a união, e +3 pontos sobre o
mesmo detector com o CRNN antigo), sem trocar erro por vermelho. O amarelo continua alto (~70%)
porque ~45% das fotos amostradas seguem sem leitura detectada ou diverge da digitada. Ressalvas:
rótulos dos recortes enviesados para fotos fáceis, amostra de 454 fotos, 28 "sem medidor" da cena
contados na amostra inteira.
