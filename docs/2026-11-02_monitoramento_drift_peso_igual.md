# Monitoramento: histogramas agregados com peso igual por imagem

Data: 2026-11-02 | Dono da regra: `aggregate_histograms` em `src/leiturista/monitoring/drift.py` | Decisão registrada na seção 9 do SDD
(`docs/2026-11-01_sdd_monitoramento_mlops.md`).

## O que mudou e por quê

Antes, o KS por canal (e o PSI dos histogramas de pixels) comparava histogramas agregados somando a contagem crua de pixels de cada
imagem; uma foto de 4608 px de largura pesava duas ordens de grandeza a mais que uma de 450 px. Agora cada histograma de 256 bins é
normalizado para somar 1 e só depois somado: o total do agregado é o número de imagens. Isso vale para a janela, para
`artifacts/reference_hist.csv` (agora com números reais, cada coluna soma 132), para `/data/summary`, para os gauges de drift do
`monitor` e para `scripts/analyze_log.py`. Limiares (D 0,05 e 0,10, faixas de PSI) não mudaram. O p-valor já usava n em imagens.

## Experimento offline (sem drift)

Referência de 132 fotos Praekelt, canal L: sorteia 19 imagens como "janela" e usa as outras 113 como referência, 2000 sorteios
(seed 0). Nenhum deslocamento aplicado, então qualquer D alto é ruído de amostragem.

| Agregação | Sorteios com D >= 0,10 | D mediano |
|---|---|---|
| peso por pixel (antes) | 30% | 0,076 a 0,079 |
| peso igual por imagem (agora) | 9% a 10% | 0,059 |

O primeiro conjunto de números (30%, 9%, 0,076, 0,059) veio de uma medição offline feita antes da mudança; a repetição acima
(script avulso, não versionado) deu 30,9%, 9,6%, 0,079 e 0,059. Com 19 contra 113 imagens ainda sobra ruído: o peso igual reduz o falso
vermelho, não o elimina.

## Resultados recalculados (mesmas janelas de `2026-11-02_monitoramento_drift_resultados.md`)

Mesmo log (`serving-logs/`), mesmos intervalos de tempo de cada carga (19 imagens, 20 em `lcd`), referência de 132 fotos.
`artifacts/reference_hist.csv` regravado com `build_reference.py --hist-only` (sem rodar o MeterOCR).

| Carga | Imagens | D L | D R | D G | D B | Nível (L, R, G, B) | p (L, R, G, B) |
|---|---|---|---|---|---|---|---|
| holdout, `strong` | 19 | 0,629 | 0,767 | 0,584 | 0,427 | vermelho nos 4 | 0,0000, 0,0000, 0,0000, 0,0046 |
| holdout, `none` (controle) | 19 | 0,048 | 0,061 | 0,045 | 0,032 | verde, amarelo, verde, verde | 1,00 nos 4 |
| holdout, `mild` | 19 | 0,245 | 0,214 | 0,259 | 0,215 | vermelho nos 4 | 0,27, 0,43, 0,21, 0,43 |
| `henrik` | 19 | 0,225 | 0,211 | 0,164 | 0,326 | vermelho nos 4 | 0,37, 0,45, 0,76, 0,058 |
| `lcd` | 20 | 0,118 | 0,091 | 0,117 | 0,104 | vermelho, amarelo, vermelho, vermelho | 0,97, 1,00, 0,97, 0,99 |

Antes (peso por pixel): `strong` 0,69/0,82/0,65/0,51; `none` 0,19 nos 4 (vermelho); `mild` 0,35/0,31/0,36/0,35; `henrik`
0,20/0,21/0,14/0,33; `lcd` 0,10/0,08/0,10/0,07 (amarelo).

PSI dos histogramas de pixels agregados (16 bins) com peso igual:

| Carga | PSI L | PSI R | PSI G | PSI B |
|---|---|---|---|---|
| `strong` | 6,720 | 8,958 | 6,013 | 4,034 |
| `none` | 0,079 | 0,083 | 0,076 | 0,061 |
| `mild` | 1,715 | 1,441 | 1,944 | 1,676 |
| `henrik` | 0,414 | 0,461 | 0,242 | 0,813 |
| `lcd` | 0,159 | 0,135 | 0,158 | 0,131 |

O PSI de `luminance_mean` (variável por imagem) não depende da agregação e ficou igual ao do documento anterior.

## Leitura

1. **O controle `none` deixou de dar vermelho:** D de 0,03 a 0,06 (um canal amarelo, três verdes) contra 0,18 a 0,20 antes. O PSI do
   controle caiu de 0,25 para 0,06 a 0,08 (estável). A explicação de que fotos grandes dominavam a soma, dada como provável no
   documento anterior, se confirma no sentido de que removê-la derruba o controle.
2. `strong` continua vermelho nos 4 canais, com D alto e p < 0,01 (B: 0,0046).
3. `mild` e `henrik` seguem vermelhos nos 4 canais; os D de `mild` caíram (0,21 a 0,26) e ainda ficam bem acima do controle.
4. **`lcd` piorou:** de amarelo (D 0,07 a 0,10) para vermelho em L, G e B (D 0,10 a 0,12), R amarelo. Reportado como medido; com 20
   imagens e p perto de 1 isso é compatível com ruído, e `lcd` é outro tipo de medidor, então drift real é esperado.
5. A mudança de agregação não ajustou nenhum limiar. A referência continua pequena (132 imagens) e as janelas têm 19 ou 20 imagens:
   leia cada número com essa ressalva.
