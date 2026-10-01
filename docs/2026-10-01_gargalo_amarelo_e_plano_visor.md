# Gargalo do amarelo e plano do detector de visor

**Data:** 2026-10-01 · **Meta:** amarelo < 40% (baseline hoje ≈ 89%, ver
`2026-10-01_baseline_triagem_{ppocr,crnn}.json` e `scripts/baseline_triagem.py`).
Validação **cross-dataset é intencional**: modelo treinado no UFPR-AMR, avaliado em fotos de campo
da distribuidora (item L2 do enunciado).

## Onde o amarelo nasce (amostra de 500 fotos que precisam de OCR, 79,8% da base)

| Estágio | Medida | Leitura |
|---|---|---|
| Foto tem o dado? | Rótulos fracos do LLM (3.247 fotos): ~80% close-up, 72% digital, 62% visor legível, ~8% sem medidor | O gargalo **não** é a qualidade da foto |
| 2 · Localizar o visor | Só 252/500 fotos geram um recorte de "leitura". Em 48 recortes olhados à mão: **22 (46%) são o visor**, 10 são serial de 10 dígitos, 10 são placa/código de material (`802xxx`), 6 lixo | ≈ 23% do pool com o visor localizado (n=48, margem larga) |
| 3 · Ler os dígitos | Nos ~22 recortes que SÃO visor, o CRNN do professor acerta **0**. PP-OCR acerta 7 de 314 recortes com leitura | O CRNN falha também em visor limpo (ex.: `002042` lido como `1081`): é gap de domínio, falta fine-tune em recorte da distribuidora |
| 4 · Decisão | Verde só se leitura == digitada e nitidez ok | Regra correta, mas sem leitura certa nada vira verde |

Os dois estágios (2 e 3) são gargalo; o 2 vem primeiro porque limita o que o 3 recebe.

## Alavancas, em ordem de pontos de amarelo

1. **Política da nota NA sem foto: 13,5 pontos, zero modelo.** 1.850 registros. Pergunta ao
   cliente: NA exige foto? (sim → vermelho, não → verde).
2. **Visor localizado** (estágio 2): de ~23% para ≥ 70% do pool.
3. **Leitura no visor** (estágio 3): fine-tune do CRNN em recortes da distribuidora, com
   augmentation de inversão de polaridade (41,7% dos recortes vêm da fase invertida).
4. Triagem de cena (estágio 1): converte "sem medidor" (~8%) de amarelo em vermelho. Ganho pequeno.

Conta para < 40%: 13,5 pts fixos de R2b saem com a alavanca 1; o resto exige decidir ≥ ~60% do
pool com verde/vermelho confiável, o que só acontece com 2 + 3.

## Plano do detector de visor (estágio 2)

Não reinventar o detector antes de saber se o problema é **achar** o visor ou **escolher** entre os
candidatos que o PP-OCRv5 já acha.

**Passo 0 — medir o recall dos candidatos (barato, sem treino).** Em ~50 fotos inteiras, desenhar
TODAS as caixas do det e contar em quantas existe alguma caixa sobre o visor. Se recall alto, o
problema é a escolha (passo A). Se baixo, é detecção (passo B).

**Passo A — reranker "é visor?" (se o recall for alto).** Classificador binário de recorte
(visor × serial/placa/outro), MobileNetV3-Small por transferência. Entra no lugar de
`_best_reading` (hoje: "mais dígitos"), que é justamente o que escolhe serial/placa no lugar do
visor. Positivos: recortes aceitos do `leiturista import-distribuidora` (1.087; dataset **não
existe neste Mac**, reconstruir com `LEITURISTA_FOTOS_DIR=data/distribuidora_campo`). Negativos:
os demais candidatos do det nas mesmas fotos.

**Passo B — detector de visor (se o recall for baixo).** Fine-tune de detector leve com 1 classe.
Licença importa: evitar YOLO/ultralytics (AGPL, o professor já apontou isso ao Nortdata); usar
`torchvision` SSDLite/Faster R-CNN MobileNet (BSD). Caixas de treino: os quads aceitos do import.

**Em ambos:** partição por lote, avaliação = % de fotos com visor correto localizado (IoU ou
inspeção) e, ponta a ponta, queda do amarelo no `scripts/baseline_triagem.py`. Treino só o usuário
dispara (exceto autorização explícita).

## Em aberto

1. NA exige foto? (alavanca 1)
2. Recall do det (passo 0) decide A × B.
3. Medida de "visor correto" nos recortes foi visual, n=48; precisa de ~200 rotulados à mão
   (também serve ao item L2).
