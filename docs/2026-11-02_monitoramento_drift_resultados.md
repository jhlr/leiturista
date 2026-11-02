# Monitoramento: resultados da análise de drift (cargas reais)

**Registro histórico: todos os D e PSI de pixels abaixo foram calculados com histogramas agregados por CONTAGEM DE PIXELS (fotos grandes dominam).** Desde 2026-11-02 o agregado dá peso igual a cada imagem; os números recalculados estão em `docs/2026-11-02_monitoramento_drift_peso_igual.md`. Os valores por imagem (`luminance_mean` etc.) não mudam.

Data: 2026-11-02 | Script: `scripts/analyze_log.py` (`just analyze`), que só chama `leiturista.monitoring.drift` para KS, PSI e
Benjamini-Hochberg. Referência: 132 fotos Praekelt (pequena: leia cada número com essa ressalva). Cada linha abaixo é uma carga
de `scripts/load_gen.py --n 20` (uma chamada por vez, serviço no Docker, CPU), analisada com `--since`/`--until` da própria carga,
no relatório `reports/<carga>/` (ignorado pelo git).

**Desvio do SDD:** o critério pede `--n 100`; foram usadas 20 chamadas por carga (as fotos levam 10 a 40 s cada em CPU no Docker).
Com `--invalid-pct 5` cada carga tem 1 chamada inválida, então a janela tem 19 ou 20 imagens com variáveis de entrada.

## KS por canal (D sobre os histogramas 0-255 agregados; níveis: amarelo D >= 0,05, vermelho D >= 0,10)

| Carga | Imagens | D L | D R | D G | D B | Nível (L, R, G, B) | p (L, R, G, B) |
|---|---|---|---|---|---|---|---|
| holdout, `strong` | 19 | 0,693 | 0,819 | 0,650 | 0,506 | vermelho nos 4 | 0,0000, 0,0000, 0,0000, 0,0004 |
| holdout, `none` (controle) | 19 | 0,192 | 0,183 | 0,193 | 0,195 | vermelho nos 4 | 0,57, 0,63, 0,57, 0,55 |
| holdout, `mild` | 19 | 0,351 | 0,307 | 0,363 | 0,348 | vermelho nos 4 | 0,033, 0,088, 0,025, 0,036 |
| `henrik` | 19 | 0,203 | 0,209 | 0,144 | 0,331 | vermelho nos 4 | 0,50, 0,46, 0,88, 0,053 |
| `lcd` | 20 | 0,099 | 0,079 | 0,095 | 0,069 | amarelo nos 4 | 0,996, 1,000, 0,997, 1,000 |
| últimas 5 imagens | 5 | 0,190 | 0,308 | 0,122 | 0,368 | inconclusivo (menos de 8 imagens) | |

## PSI dos histogramas de pixels agregados (16 bins) e de duas variáveis por imagem

| Carga | PSI L | PSI R | PSI G | PSI B | `luminance_mean` KS D (p) | `luminance_mean` PSI |
|---|---|---|---|---|---|---|
| `strong` | 7,314 | 9,854 | 6,502 | 4,374 | 0,932 (0,0000) | 12,29 |
| `none` | 0,265 | 0,257 | 0,258 | 0,250 | 0,237 (0,26) | 0,229 |
| `mild` | 1,780 | 1,528 | 2,084 | 1,756 | 0,585 (0,0000) | 5,31 |
| `henrik` | 0,390 | 0,482 | 0,265 | 0,894 | 0,456 (0,0012) | 3,88 |
| `lcd` | 0,102 | 0,081 | 0,100 | 0,050 | 0,261 (0,16) | 2,44 |

## O que isso diz (sem ajustar limiares)

1. **`strong` dá vermelho em L, R, G e B**, como esperado (L e R e B pedidos pelo critério, os quatro vermelhos).
2. **O controle `none` NÃO deu verde: deu vermelho nos 4 canais (D 0,18 a 0,20) com p entre 0,55 e 0,63.** Ou seja, D acima de 0,10 é
   compatível com ruído de amostragem aqui: 19 imagens contra 132, e o histograma agregado pesa por pixel, então fotos grandes tendem a dominar a soma (explicação provável, não testada).
   O PSI do controle também fica em "significativo" (0,25 a 0,27 nos pixels). Resultado reportado como medido; o limiar de 0,10 do curso
   não separa o controle do `mild` com tão poucas imagens (D 0,19 contra 0,35).
3. `mild` fica acima do controle em D e muito acima em PSI (1,5 a 2,1 contra 0,26), mas o nível é o mesmo (vermelho).
4. `henrik` (outras câmeras e medidores) tem D parecido com o controle nos canais L, R e G e maior em B; o PSI das médias por imagem
   (`luminance_mean` 3,9) mostra o drift real melhor do que o D dos pixels. `lcd` dá amarelo, perto do controle em D.
5. Os p-valores usam n em imagens (19 ou 20 contra 132); com esse n o KS só rejeita para D grande. O PSI com 10 bins de quantis e
   19 imagens (menos de 2 por bin) é inflado pelo acaso, como o controle do curso já mostrou.
6. Antes de qualquer uso operacional, vale uma referência maior e janelas de pelo menos 100 imagens.

Latência e estabilidade destas cargas (fora do escopo desta etapa): no Docker em CPU, 20 chamadas levaram de 6 a 16 minutos; houve 504
(timeout de 60 s do BentoML) em algumas cargas (`strong` 1, `lcd` 1) e a chamada de entrada inválida (arquivo que não é imagem) devolve
500, comportamento já documentado do BentoML.
