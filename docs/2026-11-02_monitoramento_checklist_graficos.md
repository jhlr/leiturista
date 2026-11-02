# Monitoramento: checklist dos gráficos do Grafana

Data: 2026-11-02 | Dashboards em `monitoring/grafana/provisioning/dashboards/*.json`, gerados por
`scripts/build_dashboards.py` (rode `uv run python scripts/build_dashboards.py` depois de mudar um painel).
Grafana 13.2.5, plugins `yesoreyeram-infinity-datasource` 4.1.5 e `volkovlabs-echarts-panel` 7.2.5.

## Como os dados chegam

| Fonte | O que traz | Quem lê |
|---|---|---|
| Prometheus (`inference:3000` e `monitor:3000`, a cada 5 s) | contadores e histogramas do BentoML, métricas `leiturista_*`, gauges `leiturista_drift_*` | séries de tempo |
| Loki (via Alloy) | as linhas do log | painéis de logs |
| Infinity (rotas `/data/*` do `monitor`) | valores por imagem, histogramas de 256 bins, quartis, drift atual | tudo que o Prometheus não guarda |

O Infinity usa o parser `backend`: o `root_selector` é uma expressão JSONata que transforma o JSON da rota em linhas.
O intervalo do dashboard vira `since`/`until` da rota (`${__from:date:iso}`).

## Box plots: o que funcionou

O Grafana 13.2.5 não tem box plot nativo. Foi usado o painel **Business Charts** (ECharts), plugin
`volkovlabs-echarts-panel` **7.2.5** (assinado, compatível com Grafana >= 12.3), instalado por `GF_PLUGINS_PREINSTALL_SYNC`.
O painel recebe de `/data/summary` os cinco números (min, Q1, mediana, Q3, max) da janela e da referência e desenha
duas caixas por variável com uma série `boxplot` do ECharts. O fallback do SDD (candlestick ou tabela de quartis) não foi
necessário. Cada box plot é um painel `volkovlabs-echarts-panel` cuja descrição diz como foi feito.

## Checklist (56 painéis, todos carregam sem erro e com dados)

Verificação: Grafana aberto no navegador automatizado, cinco dashboards com o intervalo "últimas 3 horas" depois das
cargas `strong`, `none`, `mild`, `henrik` e `lcd` (20 chamadas cada): 0 painéis com erro e 0 com "No data".
Captura de tela de cada dashboard lida durante a verificação (não versionadas).

### 1. Operação (`leiturista-operacao`, 12 painéis)

| Gráfico exigido | Painel | Como foi feito |
|---|---|---|
| requisições por segundo por status | série de tempo | `sum by (http_response_code)(rate(bentoml_service_request_total{endpoint="/predict"}[$__rate_interval]))` |
| p50, p95, p99 | série de tempo | `histogram_quantile` sobre `bentoml_service_request_duration_seconds_bucket`, janela de 5 min |
| heatmap de latência | heatmap | `increase()` dos buckets `le`, formato heatmap |
| histograma de latência (última janela) | gráfico de barras | cada barra é a diferença entre dois buckets `le` consecutivos em `$__range` (13 consultas instantâneas + transformação `reduce`) |
| taxa de erro 4xx e 5xx | série de tempo | razão entre `rate` dos códigos `4..`/`5..` e o total (com `or vector(0)`), janela de 5 min |
| requisições em andamento | stat | `bentoml_service_request_in_progress` |
| total de requisições e taxa de 5xx atual | 2 stats (mais 2 stats de taxa 4xx e 5xx) | `sum` do contador; `rate` de 5xx em 5 min |
| tabela das mais lentas | tabela | `/data/variables` via Infinity, ordenada por `latency_ms`, 20 linhas |
| extra | série de tempo | latência por requisição, do log |

### 2. Entrada (`leiturista-entrada`, 17 painéis)

| Gráfico exigido | Painel | Como foi feito |
|---|---|---|
| histogramas 0-255 janela x referência, L, R, G, B (4 painéis) | painel `trend`, barras (janela) e linha (referência) | `/data/summary`, fração dos pixels |
| médias dos 4 canais no tempo com a referência como faixa | ECharts | `/data/variables` mais `/data/summary`; faixa translúcida = Q1 a Q3 da referência e linha tracejada = média da referência |
| histogramas de `in_luminance_mean`, `luminance_std`, `sharpness` | painel Histogram nativo (3) | valores por imagem de `/data/variables` |
| box plots das médias por canal, `luminance_std`, `sharpness`, `width`, `height`, `file_bytes` | ECharts (6) | `/data/summary`, janela x referência |
| pizza de `dominant_channel` | pizza | `/data/summary`, contagens da janela |
| barras de `format` | gráfico de barras | idem |
| histograma de luminância da métrica do Prometheus | gráfico de barras | `leiturista_input_luminance_mean_bucket` com buckets consecutivos subtraídos |

### 3. Saída (`leiturista-saida`, 12 painéis)

| Gráfico exigido | Painel | Como foi feito |
|---|---|---|
| histograma, box plot e série de `confianca` | Histogram nativo, ECharts, série de tempo | `/data/variables` e `/data/summary` |
| proporções de `funcao` (barras empilhadas no tempo e pizza) | série de tempo em barras 100% e pizza | `increase(leiturista_prediction_funcao_total[...])` por `funcao`; pizza de `/data/summary` |
| taxas de `legivel`, `has_reading`, `has_serial` | série de tempo | média móvel de 10 imagens (transformação `calculateField` de janela) sobre 0/1 de `/data/variables` |
| histograma de `reading_digits` e de `n_flags` | Histogram nativo (bucket 1) | `/data/variables` |
| tabela das flags mais frequentes | tabela | contagem por frase em JSONata sobre `out_flags` |
| histograma e box plot de `candidate_sharpness` | Histogram nativo e ECharts | idem |
| histograma de confiança da métrica do Prometheus | gráfico de barras | `leiturista_prediction_confidence_bucket` com buckets subtraídos |

### 4. Drift (`leiturista-drift`, 11 painéis)

| Gráfico exigido | Painel | Como foi feito |
|---|---|---|
| KS D por canal no tempo com linhas em 0,05 e 0,10 | série de tempo com `thresholdsStyle: line` | `leiturista_drift_ks_d` |
| linha do tempo de estado | state timeline | `leiturista_drift_level` com mapeamento de valores |
| nível atual por canal (verde, amarelo, vermelho, cinza) | 4 stats com mapeamento | `leiturista_drift_level{channel=...}` |
| p-valor por canal | série de tempo | `leiturista_drift_ks_pvalue` |
| PSI por variável com limiares 0,1 e 0,25 | bar gauge | `leiturista_drift_psi`; limiares marcados como convenção da indústria; a barra vai até 5 |
| tabela com todos os números | 2 tabelas | `/data/drift` via Infinity (D, p, nível por canal; PSI, faixa, bins, motivo por variável) |
| tamanho da janela | stat | `leiturista_drift_window_images` |

### 5. Logs (`leiturista-logs`, 4 painéis)

| Gráfico exigido | Painel | Como foi feito |
|---|---|---|
| log ao vivo | logs | `{job="leiturista"}` |
| taxa por `event` | série de tempo | `sum by (event)(rate({job="leiturista"}[1m]))` |
| erros | logs | `{job="leiturista", event="response"} \| json \| status_code >= 500` |
| volume de log no tempo | barras empilhadas | `sum by (event)(count_over_time(...[$__interval]))` |

## Desvios e limites

1. A faixa da referência nas médias por canal é Q1 a Q3 mais a linha da média, não média mais ou menos um desvio: `/data/summary` não devolve o desvio-padrão.
2. Os histogramas nativos mostram só a janela (a referência tem 132 imagens e a janela algumas dezenas; as contagens não são comparáveis). A comparação com a referência está nos box plots e nos histogramas de pixels.
3. A barra do PSI vai até 5; valores maiores aparecem só no número.
4. Os plugins são baixados do grafana.com na primeira subida do Grafana (precisa de internet).
5. O intervalo padrão é "últimas 3 horas": a carga precisa cair nele ou o seletor de tempo precisa ser ajustado.
