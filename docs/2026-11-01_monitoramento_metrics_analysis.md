# Como ler o `/metrics` do Leiturista (exemplo real)

Data: 2026-11-01 | Planilha com fórmulas vivas: `docs/2026-11-01_monitoramento_metrics_analysis.xlsx`
| Snapshot bruto usado aqui: `docs/2026-11-01_monitoramento_metrics_snapshot.txt` | Gerado por
`scripts/make_metrics_xlsx.py` a partir do snapshot.

O exemplo vem de uma execução real: serviço no ar (`just serve`, 1 worker, CPU do Mac do autor) e
`scripts/load_gen.py --n 33 --rps 0.07 --traffic holdout --invalid-pct 12`, mais 4 chamadas inválidas
(uma de cada tipo) feitas à mão. Total de 37 chamadas a `/predict`: 31 com status 200, 3 com 400
e 3 com 500. As imagens são as 33 fotos Praekelt do holdout (as mesmas dos 132 da referência não
entram). Os números abaixo são medidos, não estimados.

## 1. O que cada série significa

Nomes e rótulos reais do snapshot (BentoML 1.4.39):

| Série | Tipo | Rótulos | O que é |
|---|---|---|---|
| `bentoml_service_request_total` | counter | `endpoint`, `http_response_code`, `runner_name`, `service_name`, `service_version` | Quantas requisições terminaram, por endpoint e código HTTP. Só cresce (até o processo reiniciar). |
| `bentoml_service_request_duration_seconds_bucket` | histograma | os mesmos + `le` | Quantas requisições levaram **até** `le` segundos. É **cumulativo**: o bucket `le="10.0"` inclui os de `le="5.0"`. |
| `bentoml_service_request_duration_seconds_sum` / `_count` | counter | os mesmos | Soma das durações (s) e número de observações. |
| `bentoml_service_request_in_progress` | gauge | `endpoint`, ... | Requisições em andamento agora (sobe e desce). |
| `bentoml_service_last_request_timestamp_seconds` | gauge | `pid`, ... | Instante da última requisição. |
| `leiturista_input_luminance_mean_bucket/_sum/_count` | histograma | `le` | Luminância média (0-255) da foto de entrada, buckets de 16 em 16. |
| `leiturista_prediction_confidence_bucket/_sum/_count` | histograma | `le` | `confianca` da resposta, buckets de 0,1 em 0,1. |
| `leiturista_prediction_funcao_total` | counter | `funcao` (3 valores) | Predições por `leitura_normal`, `leitura_incerta`, `sem_leitura_detectada`. |

**Contador cumulativo:** o valor é o total desde que o processo subiu. Para saber "quantas por
segundo" usa-se a diferença entre duas leituras (`rate()`/`increase()` no Prometheus).
**Buckets `le`** ("menor ou igual"): `le="+Inf"` é igual a `_count`. Como são cumulativos, a
contagem de um bucket isolado é a diferença para o anterior.

Exemplo, `/predict` com status 200 (snapshot): `_count = 31`, `_sum = 438,9015 s`, logo a
média é `_sum / _count = 14,158 s`.

Série customizada no mesmo snapshot: `leiturista_prediction_funcao_total` = 23 normal, 5 incerta, 3 sem
leitura (soma 31, igual ao número de respostas 200). `leiturista_input_luminance_mean_count = 31`,
`_sum = 4066,3`, média 131,2.

## 2. Histograma de latência: de-acumular por diferença

Buckets finais (decisão, seção 6 abaixo): 0,25 / 0,5 / 1 / 2,5 / 5 / 7,5 / 10 / 12,5 / 15 / 20 / 30 / 60 s.

| `le` (s) | acumulado | contagem no bucket (diferença) | fração |
|---|---|---|---|
| 0,25 a 2,5 | 0 | 0 | 0% |
| 5 | 4 | 4 | 12,9% |
| 7,5 | 6 | 2 | 6,5% |
| 10 | 9 | 3 | 9,7% |
| 12,5 | 15 | 6 | 19,4% |
| 15 | 18 | 3 | 9,7% |
| 20 | 23 | 5 | 16,1% |
| 30 | 31 | 8 | 25,8% |
| 60 / +Inf | 31 | 0 | 0% |

Na planilha, aba `histogram`, cada contagem é `=raw!B{i}-raw!B{i-1}`. Para o gráfico de barras
usa-se essa coluna (nunca o acumulado). Maior fração por bucket: 25,8%, abaixo da meta de 50%.

## 3. "Boxplot" de latência a partir dos buckets (só uma aproximação)

O Prometheus não guarda observações, só contagens por bucket. Os quartis saem por interpolação
linear dentro do bucket, a mesma lógica de `histogram_quantile`: posição = `q x total`; acha-se o
primeiro bucket cujo acumulado alcança a posição; valor = `inferior + (superior - inferior) x
(posição - acumulado_antes) / (acumulado_no_bucket - acumulado_antes)`. A interpolação supõe
observações uniformes dentro do bucket.

| | Q1 | mediana | Q3 | whisker mín. | whisker máx. |
|---|---|---|---|---|---|
| Dos buckets (aba `quantiles`) | 8,96 s | 12,92 s | 20,31 s | 2,5 s | 30 s |
| Exato, de `latency_ms` do log | 9,72 s | 13,06 s | 19,72 s | 2,83 s | 28,12 s |

Whiskers dos buckets = limite inferior do primeiro bucket não vazio e limite superior do último
não vazio. A diferença para o exato mostra o erro da aproximação; para quartis exatos use o log.
Q1, mediana e Q3 da planilha foram recalculados em Python de forma independente
(`histogram_quantile` à mão sobre o snapshot) e batem (8,9583 / 12,9167 / 20,3125).

## 4. Taxa de erro

Com os nomes reais:

```promql
# acumulada desde o início do processo (5xx e 4xx separados)
sum(bentoml_service_request_total{http_response_code=~"5.."}) / sum(bentoml_service_request_total)
sum(bentoml_service_request_total{http_response_code=~"4.."}) / sum(bentoml_service_request_total)
# em janela de 5 minutos
sum(increase(bentoml_service_request_total{http_response_code=~"5.."}[5m])) / sum(increase(bentoml_service_request_total[5m]))
sum(rate(bentoml_service_request_total{http_response_code=~"5.."}[5m])) / sum(rate(bentoml_service_request_total[5m]))
```

No snapshot (`/predict`): 200 = 31, 400 = 3, 500 = 3, total 37, taxa 5xx = 3/37 = 8,1% e taxa 4xx = 3/37 = 8,1%
(aba `errors`). Atenção: `/readyz` e `/livez` também geram séries em `bentoml_service_request_total` (o
exemplo da planilha filtra `endpoint="/predict"`); no PromQL, filtre `endpoint="/predict"` se
não quiser contar as sondas de saúde.

### Códigos HTTP medidos para entrada inválida

| Entrada inválida | Código | Linha `response` do log (`error`) |
|---|---|---|
| Sem o campo `image` (multipart sem `image`) | 400 | `ValidationError/invalid_request` |
| Corpo vazio | 400 | `ValidationError/invalid_request` |
| Bytes que não são imagem (`image` com texto) | **500** | `HTTPError/http_500` |
| JPEG truncado | **500** | `OSError/invalid_image` |

O 500 para "bytes que não são imagem" vem do BentoML, não do Leiturista: o decodificador de
`PIL.Image` levanta `UnidentifiedImageError` dentro da validação da entrada e o BentoML a trata como
erro interno em vez de 4xx. Como a função da API nunca roda nesse caso, o log só sabe o status
(por isso `HTTPError/http_500`, sem a classe da exceção). Não alteramos esse comportamento
(o contrato da API não mudou). Para um 5xx real do servidor, o JPEG truncado passa pela função e
registra `OSError/invalid_image`.

## 5. Retenção, precisamente

- **Métricas** vivem na memória do processo de inferência e **zeram quando ele reinicia**; a
  aplicação não persiste nada.
- **Prometheus** (quando o servidor entra, camada P1) guarda no TSDB por 15 dias
  (`--storage.tsdb.retention.time=15d`, definido explicitamente).
- **Log em arquivo** (`serving-logs/inference.log`) dura até a rotação descartar o 5o backup:
  50 MB por arquivo, 5 backups, no máximo 300 MB. Com 7,4 KB por par requisição+resposta medido,
  um arquivo comporta cerca de 7.000 pares e os 6 arquivos cerca de 40.000 pares.

## 6. Medições registradas (SDD seção 11)

1. **Código HTTP de entrada inválida:** tabela da seção 4.
2. **Séries reais do `/metrics`:** tabela da seção 1 (todos os nomes acima vêm do snapshot).
3. **Distribuição de latência** (log, `latency_ms`, 31 chamadas 200 de fotos Praekelt do holdout, 33 fotos,
   `--rps 0.07`, sequencial, 1 worker): **partida fria (1a chamada, inclui carga lazy dos modelos): 21,8 s**
   (20,9 s em outra execução). Sem a 1a chamada (n = 30): **p50 = 12,62 s, p95 = 26,46 s**, mínimo 2,83 s, máximo 28,12 s.
   Com a 1a: p50 = 13,06 s, p95 = 26,30 s. As fotos Praekelt são grandes (até 4032x3024) e o pipeline
   (detecção em 4 rotações, det+rec normal e invertido, TrOCR de fallback na imagem inteira) roda em
   CPU; por isso a latência é de segundos e não de milissegundos. Imagens pequenas (as sintéticas de
   `samples/`, 480x360) levam de 0,5 s a 3 s. A `latency_ms` inclui o tempo de fila atrás do único
   worker e, na 1a chamada, a carga dos modelos.
   **Buckets:** a partida `(0,05 ... 30)` deixava 24 de 32 chamadas (75%) no bucket 10-30 s, acima da meta
   de 50%. Os buckets finais `(0,25; 0,5; 1; 2,5; 5; 7,5; 10; 12,5; 15; 20; 30; 60)` deixam no máximo 25,8% por
   bucket (distribuição `[0,0,0,0,4,2,3,6,3,5,8,0,0]` nos 13 intervalos). Calibrados com tráfego
   Praekelt; fotos pequenas caem nos buckets baixos (0,25 a 2,5 s), mantidos para isso.
4. **Bytes por par requisição+resposta no log real:** 7.505 bytes em média (linha `request` 6.842 B,
   porque carrega os 4 histogramas de 256 inteiros; linha `response` 553 B).
5. **Contagem de arquivos dos conjuntos de imagens:** referência 132, holdout 33 (165 `praekelt_*.jpg`),
   Henrik 168 PNG (recursivo em `train/` e `test/`), LCD 50 JPG. Verificado com `split_reference_images`
   e `traffic_images`. Se `just models` deixa `.model_cache/trocr-small-printed/model.safetensors` no
   lugar: **não verificado** (não baixei o release; o arquivo existe na máquina do autor).
6. **BentoML 1.4.39:** o hook de middleware existe (`Service.add_asgi_middleware`) e funciona;
   `ctx: bentoml.Context` na API é aceito e **não altera** o esquema de entrada (o OpenAPI continua
   só com `image`); `prometheus_client` direto aparece no mesmo `/metrics` com 1 worker, desde que os
   objetos sejam criados dentro do worker (o BentoML define `PROMETHEUS_MULTIPROC_DIR` depois do import do
   módulo); buckets de duração configuráveis por `@bentoml.service(metrics={"duration": {"buckets": [...]}})`
   (lista de floats). **A duração do BentoML e a `latency_ms` do log concordam:** soma de
   duração do BentoML 438,9015 s contra soma de `latency_ms` 438,9088 s em 31 chamadas (diferença de
   0,2 ms por chamada; médias 14,1581 s e 14,1583 s). GET via `@bentoml.api`: tier P1, não medido
   ainda.
7. **`RotatingFileHandler` criado de forma preguiçosa no worker rotaciona em processo servido:**
   sim, verificado (`tests/monitoring/test_rotation.py`, `SERVING_LOG_MAX_BYTES=20000`: 8 chamadas geram
   `inference.log.1` e as 16 linhas aparecem somadas sem perda).
