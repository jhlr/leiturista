"""Gera os cinco dashboards provisionados do Grafana (SDD secao 7, Grafana).

  uv run python scripts/build_dashboards.py

Escreve `monitoring/grafana/provisioning/dashboards/*.json`; o Grafana os carrega sozinho (provider em
`dashboards.yml`). O JSON gerado e versionado: este script e so o dono dos padroes de painel (um box plot,
um histograma de pixels, um stat de nivel de drift... definidos uma vez e repetidos por variavel), para
nao manter 70 paineis a mao. Os baldes (buckets) vem de `leiturista.monitoring.metrics`, os nomes das
variaveis de `features.py`/`outputs.py`: nada e copiado.

Fontes: Prometheus (series de `inference:3000` e `monitor:3000`), Loki (log via Alloy) e Infinity
(rotas `/data/*` do `monitor`: valores por imagem, histogramas 0-255 e quartis). Box plots: o Grafana nao
tem box plot nativo; usam o painel Business Charts (ECharts, `volkovlabs-echarts-panel`) alimentado por
`/data/summary` (min, Q1, mediana, Q3, max).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from leiturista.monitoring.metrics import CONFIDENCE_BUCKETS, DURATION_BUCKETS, LUMINANCE_BUCKETS
from leiturista.paths import ROOT

Json = dict[str, Any]

OUT_DIR = ROOT / "monitoring" / "grafana" / "provisioning" / "dashboards"
PROM: Json = {"type": "prometheus", "uid": "prometheus"}
LOKI: Json = {"type": "loki", "uid": "loki"}
INF: Json = {"type": "yesoreyeram-infinity-datasource", "uid": "infinity"}
MONITOR = "http://monitor:3000"
ECHARTS = "volkovlabs-echarts-panel"
PRED = 'endpoint="/predict"'
REQ = "bentoml_service_request_total"
DUR = "bentoml_service_request_duration_seconds_bucket"
TIME_PARAMS: list[Json] = [
    {"key": "since", "value": "${__from:date:iso}"},
    {"key": "until", "value": "${__to:date:iso}"},
]
CHANNELS = {"L": "luminance_mean", "R": "r_mean", "G": "g_mean", "B": "b_mean"}
CHANNEL_COLORS = {"L": "#9aa0a6", "R": "#e24d42", "G": "#3ea65a", "B": "#3d85de"}
LEVEL_MAPPINGS: list[Json] = [
    {
        "type": "value",
        "options": {
            "-1": {"text": "inconclusivo", "color": "gray", "index": 0},
            "0": {"text": "verde", "color": "green", "index": 1},
            "1": {"text": "amarelo", "color": "yellow", "index": 2},
            "2": {"text": "vermelho", "color": "red", "index": 3},
        },
    }
]


# --- consultas -----------------------------------------------------------------------------------
def prom(expr: str, ref: str = "A", legend: str = "", instant: bool = False, fmt: str = "time_series") -> Json:
    target: Json = {"refId": ref, "datasource": PROM, "expr": expr, "legendFormat": legend or "__auto", "format": fmt}
    if instant:
        target.update({"instant": True, "range": False})
    return target


def loki(expr: str, ref: str = "A", legend: str = "", query_type: str = "range") -> Json:
    target: Json = {"refId": ref, "datasource": LOKI, "expr": expr, "queryType": query_type}
    if legend:
        target["legendFormat"] = legend
    return target


def col(selector: str, text: str, kind: str = "number") -> Json:
    return {"selector": selector, "text": text, "type": kind}


def infinity(
    path: str,
    root: str,
    columns: Sequence[Json],
    ref: str = "A",
    params: Sequence[Json] = (),
) -> Json:
    """Consulta Infinity (JSON, parser backend: o `root_selector` e uma expressao JSONata)."""
    return {
        "refId": ref,
        "datasource": INF,
        "type": "json",
        "source": "url",
        "format": "table",
        "parser": "backend",
        "url": f"{MONITOR}{path}",
        "url_options": {"method": "GET", "params": list(params)},
        "root_selector": root,
        "columns": list(columns),
    }


def variables(columns: Sequence[Json], root: str = "$[status_code=200]", ref: str = "A", limit: int = 10000) -> Json:
    """Uma linha por requisicao do intervalo do dashboard (`/data/variables`)."""
    params = [*TIME_PARAMS, {"key": "limit", "value": str(limit)}]
    return infinity("/data/variables", root, columns, ref, params)


def summary(root: str, columns: Sequence[Json], ref: str = "A") -> Json:
    return infinity("/data/summary", root, columns, ref, TIME_PARAMS)


def map_rows(source: str, fields: dict[str, str]) -> str:
    """JSONata: `$map(<source>, function($r){ {<campo>: <expressao>} })`."""
    body = ", ".join(f'"{name}": {expr}' for name, expr in fields.items())
    return f"$map({source}, function($r){{ {{{body}}} }})"


# --- paineis -------------------------------------------------------------------------------------
class Board:
    """Um dashboard: acumula paineis com posicao automatica (24 colunas) e escreve o JSON."""

    def __init__(self, uid: str, title: str, description: str, refresh: str = "30s", time_from: str = "now-3h") -> None:
        self.uid, self.title, self.description = uid, title, description
        self.refresh, self.time_from = refresh, time_from
        self.panels: list[Json] = []
        self._next_id = 1
        self._x = 0
        self._y = 0
        self._row_h = 0

    def row(self, title: str) -> None:
        self._newline()
        self.panels.append(
            {"id": self._take_id(), "type": "row", "title": title, "collapsed": False, "gridPos": {"h": 1, "w": 24, "x": 0, "y": self._y}, "panels": []}
        )
        self._y += 1

    def _take_id(self) -> int:
        self._next_id += 1
        return self._next_id - 1

    def _newline(self) -> None:
        if self._x:
            self._y += self._row_h
            self._x = 0
            self._row_h = 0

    def add(self, panel: Json, w: int = 12, h: int = 8) -> None:
        if self._x + w > 24:
            self._newline()
        panel["id"] = self._take_id()
        panel["gridPos"] = {"h": h, "w": w, "x": self._x, "y": self._y}
        self.panels.append(panel)
        self._x += w
        self._row_h = max(self._row_h, h)

    def dump(self) -> Json:
        return {
            "uid": self.uid,
            "title": self.title,
            "description": self.description,
            "tags": ["leiturista"],
            "timezone": "browser",
            "editable": False,
            "graphTooltip": 1,
            "schemaVersion": 41,
            "version": 1,
            "refresh": self.refresh,
            "time": {"from": self.time_from, "to": "now"},
            "timepicker": {"refresh_intervals": ["5s", "10s", "30s", "1m", "5m"]},
            "templating": {"list": []},
            "annotations": {"list": []},
            "links": [{"type": "dashboards", "title": "Leiturista", "tags": ["leiturista"], "asDropdown": True, "includeVars": False, "keepTime": True}],
            "panels": self.panels,
        }

    def write(self) -> Path:
        path = OUT_DIR / f"{self.uid}.json"
        path.write_text(json.dumps(self.dump(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path


def panel(kind: str, title: str, targets: Sequence[Json], description: str = "", **extra: Any) -> Json:
    base: Json = {"type": kind, "title": title, "targets": list(targets), "description": description}
    base["datasource"] = targets[0]["datasource"] if targets else PROM
    base.update(extra)
    return base


def timeseries(
    title: str,
    targets: Sequence[Json],
    description: str = "",
    unit: str = "",
    style: str = "line",
    stack: str = "none",
    minimum: float | None = None,
    maximum: float | None = None,
    thresholds: Sequence[Json] | None = None,
    transformations: Sequence[Json] = (),
    overrides: Sequence[Json] = (),
) -> Json:
    custom: Json = {
        "drawStyle": style,
        "lineWidth": 1,
        "fillOpacity": 60 if style == "bars" else 10,
        "showPoints": "auto",
        "spanNulls": True,
        "stacking": {"mode": stack, "group": "A"},
    }
    if thresholds:
        custom["thresholdsStyle"] = {"mode": "line"}
    defaults: Json = {"unit": unit, "custom": custom}
    if minimum is not None:
        defaults["min"] = minimum
    if maximum is not None:
        defaults["max"] = maximum
    if thresholds:
        defaults["thresholds"] = {"mode": "absolute", "steps": list(thresholds)}
    return panel(
        "timeseries",
        title,
        targets,
        description,
        fieldConfig={"defaults": defaults, "overrides": list(overrides)},
        options={"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True}, "tooltip": {"mode": "multi", "sort": "none"}},
        transformations=list(transformations),
    )


def stat(
    title: str,
    targets: Sequence[Json],
    description: str = "",
    unit: str = "",
    decimals: int | None = None,
    thresholds: Sequence[Json] | None = None,
    mappings: Sequence[Json] = (),
    color_mode: str = "value",
) -> Json:
    defaults: Json = {
        "unit": unit,
        "mappings": list(mappings),
        "thresholds": {"mode": "absolute", "steps": list(thresholds or [{"color": "blue", "value": None}])},
        "color": {"mode": "thresholds"},
    }
    if decimals is not None:
        defaults["decimals"] = decimals
    return panel(
        "stat",
        title,
        targets,
        description,
        fieldConfig={"defaults": defaults, "overrides": []},
        options={"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}, "colorMode": color_mode, "graphMode": "none", "textMode": "value", "justifyMode": "center"},
    )


def table(title: str, targets: Sequence[Json], description: str = "", transformations: Sequence[Json] = (), overrides: Sequence[Json] = ()) -> Json:
    return panel(
        "table",
        title,
        targets,
        description,
        fieldConfig={"defaults": {"custom": {"align": "auto", "cellOptions": {"type": "auto"}}}, "overrides": list(overrides)},
        options={"showHeader": True, "cellHeight": "sm"},
        transformations=list(transformations),
    )


def ordered(names: Sequence[str]) -> Json:
    """Ordem das colunas (o parser backend do Infinity as devolve em ordem alfabetica)."""
    return {"id": "organize", "options": {"indexByName": {name: index for index, name in enumerate(names)}}}


def histogram(title: str, targets: Sequence[Json], description: str = "", unit: str = "", bucket_size: float | None = None, bucket_count: int = 20) -> Json:
    options: Json = {"legend": {"showLegend": False, "displayMode": "list", "placement": "bottom"}, "combine": False, "fillOpacity": 80}
    if bucket_size is not None:
        options["bucketSize"] = bucket_size
    else:
        options["bucketCount"] = bucket_count
    return panel("histogram", title, targets, description, fieldConfig={"defaults": {"unit": unit, "custom": {"fillOpacity": 80, "lineWidth": 1}}, "overrides": []}, options=options)


def pie(title: str, targets: Sequence[Json], description: str = "", transformations: Sequence[Json] = ()) -> Json:
    return panel(
        "piechart",
        title,
        targets,
        description,
        fieldConfig={"defaults": {"unit": "short"}, "overrides": []},
        options={
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": True},
            "pieType": "pie",
            "displayLabels": ["name", "percent"],
            "legend": {"displayMode": "table", "placement": "right", "showLegend": True, "values": ["value", "percent"]},
            "tooltip": {"mode": "single"},
        },
        transformations=list(transformations),
    )


def barchart(title: str, targets: Sequence[Json], x_field: str, description: str = "", unit: str = "", transformations: Sequence[Json] = (), orientation: str = "auto") -> Json:
    return panel(
        "barchart",
        title,
        targets,
        description,
        fieldConfig={"defaults": {"unit": unit, "custom": {"fillOpacity": 80, "lineWidth": 1, "axisPlacement": "auto"}}, "overrides": []},
        options={"xField": x_field, "orientation": orientation, "showValue": "auto", "stacking": "none", "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}, "xTickLabelRotation": -45},
        transformations=list(transformations),
    )


def pixel_histogram(channel: str) -> Json:
    """Histograma 0-255 da janela (barras) com o da referencia sobreposto (linha); ambos em fracao dos pixels."""
    key = channel
    root = (
        f"( $w := histograms.{key}.window; $r := histograms.{key}.reference; $sw := $sum($w); $sr := $sum($r); "
        '$map([0..255], function($i){ {"bin": $i, "janela": $sw > 0 ? $w[$i]/$sw : 0, "referencia": $r[$i]/$sr} }) )'
    )
    target = summary(root, [col("bin", "Valor (0-255)"), col("janela", "Janela"), col("referencia", "Referência")])
    overrides: list[Json] = [
        {"matcher": {"id": "byName", "options": "Valor (0-255)"}, "properties": [{"id": "unit", "value": "none"}]},
        {"matcher": {"id": "byName", "options": "Janela"}, "properties": [
            {"id": "custom.drawStyle", "value": "bars"}, {"id": "custom.fillOpacity", "value": 70},
            {"id": "color", "value": {"mode": "fixed", "fixedColor": CHANNEL_COLORS[channel]}}]},
        {"matcher": {"id": "byName", "options": "Referência"}, "properties": [
            {"id": "custom.drawStyle", "value": "line"}, {"id": "custom.lineWidth", "value": 2},
            {"id": "color", "value": {"mode": "fixed", "fixedColor": "#5a5a5a"}}]},
    ]
    return panel(
        "trend",
        f"Histograma {channel}: janela x referência",
        [target],
        f"Histograma 0-255 do canal {channel} somado sobre as imagens do intervalo (barras) contra o da referência (linha), "
        "como fração dos pixels. Fonte: /data/summary via Infinity.",
        fieldConfig={"defaults": {"unit": "percentunit", "custom": {"lineWidth": 1, "fillOpacity": 30, "showPoints": "never"}}, "overrides": overrides},
        options={"xField": "Valor (0-255)", "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True}, "tooltip": {"mode": "multi"}},
    )


_JS_THEME = "const ink = (context.grafana && context.grafana.theme && context.grafana.theme.colors.text.primary) || '#aaa';"
BOXPLOT_JS = (
    "const s = context.panel.data.series[0];"
    "if (!s) { return {}; }"
    "const col = (n) => { const f = s.fields.find((x) => x.name === n); return f ? Array.from(f.values) : []; };"
    f"{_JS_THEME}"
    "const names = col('variavel');"
    "const box = (p) => names.map((_, i) => [col(p + '_min')[i], col(p + '_q1')[i], col(p + '_median')[i], col(p + '_q3')[i], col(p + '_max')[i]]);"
    "return {"
    " textStyle: { color: ink },"
    " tooltip: { trigger: 'item', confine: true },"
    " legend: { data: ['Janela', 'Referência'], textStyle: { color: ink } },"
    " grid: { left: 90, right: 20, top: 40, bottom: 30 },"
    " xAxis: { type: 'category', data: names, axisLabel: { color: ink } },"
    " yAxis: { type: 'value', scale: true, axisLabel: { color: ink }, splitLine: { lineStyle: { opacity: 0.2 } } },"
    " series: ["
    "  { name: 'Janela', type: 'boxplot', data: box('w'), itemStyle: { color: 'rgba(61,133,222,0.35)', borderColor: '#3d85de' } },"
    "  { name: 'Referência', type: 'boxplot', data: box('r'), itemStyle: { color: 'rgba(150,150,150,0.35)', borderColor: '#8a8a8a' } }"
    " ] };"
)
MEANS_BAND_JS = (
    "const A = context.panel.data.series.find((s) => s.refId === 'A');"
    "const B = context.panel.data.series.find((s) => s.refId === 'B');"
    "if (!A) { return {}; }"
    "const get = (f, n) => { const c = f.fields.find((x) => x.name === n); return c ? Array.from(c.values) : []; };"
    f"{_JS_THEME}"
    "const t = get(A, 'Time');"
    f"const colors = {json.dumps(CHANNEL_COLORS)};"
    "const names = B ? get(B, 'canal') : [];"
    "const series = Object.keys(colors).map((ch) => {"
    " const v = get(A, ch);"
    " const i = names.indexOf(ch);"
    " const s = { name: ch, type: 'line', showSymbol: true, symbolSize: 4, color: colors[ch], data: t.map((x, k) => [x, v[k]]) };"
    " if (i >= 0) {"
    "  s.markArea = { silent: true, itemStyle: { color: colors[ch], opacity: 0.12 }, data: [[{ yAxis: get(B, 'q1')[i] }, { yAxis: get(B, 'q3')[i] }]] };"
    "  s.markLine = { silent: true, symbol: 'none', lineStyle: { color: colors[ch], type: 'dashed', opacity: 0.6 }, data: [{ yAxis: get(B, 'media')[i] }] };"
    " }"
    " return s;"
    "});"
    "return {"
    " textStyle: { color: ink },"
    " tooltip: { trigger: 'axis' },"
    " legend: { data: Object.keys(colors), textStyle: { color: ink } },"
    " grid: { left: 50, right: 20, top: 40, bottom: 30 },"
    " xAxis: { type: 'time', axisLabel: { color: ink } },"
    " yAxis: { type: 'value', min: 0, max: 255, axisLabel: { color: ink }, splitLine: { lineStyle: { opacity: 0.2 } } },"
    " series: series };"
)


def echarts(title: str, targets: Sequence[Json], code: str, description: str) -> Json:
    return panel(
        ECHARTS,
        title,
        targets,
        description,
        options={"renderer": "canvas", "editorMode": "code", "getOption": code, "editor": {"language": "javascript", "format": "auto"}, "visualEditor": {"dataset": [], "series": [], "code": ""}, "map": "none", "themeEditor": {"name": "default", "config": "{}"}},
        fieldConfig={"defaults": {}, "overrides": []},
    )


def boxplot(title: str, names: Sequence[str], description: str = "") -> Json:
    """Box plot (janela x referência) das variáveis `names` a partir dos quartis de /data/summary."""
    fields = {"variavel": "$k"}
    for prefix, side in (("w", "window"), ("r", "reference")):
        for stat_name in ("min", "q1", "median", "q3", "max"):
            fields[f"{prefix}_{stat_name}"] = f"$n.{side}.{stat_name}"
    body = ", ".join(f'"{k}": {v}' for k, v in fields.items())
    root = f'$map({json.dumps(list(names))}, function($k){{ ( $n := $lookup($$.numeric, $k); {{{body}}} ) }})'
    columns = [col("variavel", "variavel", "string")] + [col(f"{p}_{s}", f"{p}_{s}") for p in ("w", "r") for s in ("min", "q1", "median", "q3", "max")]
    return echarts(
        title,
        [summary(root, columns)],
        BOXPLOT_JS,
        (description + " " if description else "")
        + "Box plot do intervalo do dashboard (azul) contra a referência (cinza): caixa Q1 a Q3, traço na mediana, bigodes no mínimo e no máximo. "
        "Painel Business Charts (ECharts, volkovlabs-echarts-panel 7.2.5) alimentado por /data/summary via Infinity.",
    )


def sparse_hist(title: str, column: str, description: str = "", bucket_size: float | None = None) -> Json:
    """Histograma nativo dos valores por imagem de uma coluna do log (`/data/variables`)."""
    target = variables([col(column, title)], root=f'$map($[status_code=200 and {column} != null], function($r){{ {{"{column}": $r.{column}}} }})')
    return histogram(title, [target], description or f"Distribuição por imagem de {column} no intervalo. Fonte: /data/variables via Infinity.", bucket_size=bucket_size)


def numeric_series(title: str, columns: dict[str, str], description: str = "") -> Json:
    """Série temporal por imagem (`ts` x colunas), da fonte /data/variables."""
    select = {"Time": "$r.ts", **{label: f"$r.{name}" for label, name in columns.items()}}
    target = variables(
        [col("Time", "Time", "timestamp"), *[col(label, label) for label in columns]],
        root=map_rows("$[status_code=200]", select),
    )
    return timeseries(title, [target], description or "Um ponto por imagem. Fonte: /data/variables via Infinity.", style="line")


def _le(bound: float) -> str:
    """Rotulo `le` como o prometheus_client o escreve (0.25, 1.0, 16.0, +Inf)."""
    return "+Inf" if bound == float("inf") else repr(float(bound))


def deaccumulated(metric: str, buckets: Sequence[float], title: str, description: str, matchers: str = "") -> Json:
    """Histograma de um histograma Prometheus: cada barra e a diferenca entre buckets `le` consecutivos (janela `$__range`)."""

    def cumulative(bound: float) -> str:
        selector = ",".join(filter(None, [matchers, f'le="{_le(bound)}"']))
        return f"round(sum(increase({metric}_bucket{{{selector}}}[$__range])))"

    targets: list[Json] = []
    bounds = [*buckets, float("inf")]
    for index, bound in enumerate(bounds):
        if index == 0:
            expr, label = cumulative(bound), f"≤ {bound:g}"
        else:
            expr = f"{cumulative(bound)} - {cumulative(bounds[index - 1])}"
            label = f"> {bounds[index - 1]:g}" if bound == float("inf") else f"{bounds[index - 1]:g} a {bound:g}"
        targets.append(prom(expr, ref=f"Q{index}", legend=label, instant=True))
    reduce: Json = {"id": "reduce", "options": {"mode": "seriesToRows", "reducers": ["lastNotNull"]}}
    organize: Json = {"id": "organize", "options": {"renameByName": {"Field": "faixa", "Last *": "observações"}}}
    return barchart(title, targets, "faixa", description, transformations=[reduce, organize])


# --- dashboards ----------------------------------------------------------------------------------
def build_operacao() -> Board:
    b = Board("leiturista-operacao", "Leiturista: Operação", "Tráfego, latência e erros do serviço de inferência (Prometheus; log via Infinity).")
    b.row("Visão geral")
    b.add(stat("Total de requisições (/predict)", [prom(f"sum({REQ}{{{PRED}}})", instant=True)], "Contador acumulado desde que o processo de inferência subiu (zera ao reiniciar).", decimals=0), 4, 5)
    b.add(
        stat(
            "Taxa de 5xx agora",
            [prom(f'sum(rate({REQ}{{{PRED},http_response_code=~"5.."}}[5m]))', instant=True)],
            "Respostas 5xx por segundo na média dos últimos 5 minutos.",
            unit="reqps",
            decimals=4,
            thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 0.0001}],
        ),
        4,
        5,
    )
    b.add(stat("Requisições em andamento", [prom(f"sum(bentoml_service_request_in_progress{{{PRED}}})", instant=True)], "Com um único worker, mais de 1 significa fila.", decimals=0), 4, 5)
    b.add(
        stat(
            "Taxa de erro 4xx (5 min)",
            [prom(f'(sum(rate({REQ}{{{PRED},http_response_code=~"4.."}}[5m])) or vector(0)) / sum(rate({REQ}{{{PRED}}}[5m]))', instant=True)],
            "Entradas inválidas (cliente): fração das requisições dos últimos 5 minutos.",
            unit="percentunit",
            decimals=1,
            thresholds=[{"color": "green", "value": None}, {"color": "yellow", "value": 0.1}],
        ),
        6,
        5,
    )
    b.add(
        stat(
            "Taxa de erro 5xx (5 min)",
            [prom(f'(sum(rate({REQ}{{{PRED},http_response_code=~"5.."}}[5m])) or vector(0)) / sum(rate({REQ}{{{PRED}}}[5m]))', instant=True)],
            "Falhas do servidor (inclui 504 de timeout): fração das requisições dos últimos 5 minutos.",
            unit="percentunit",
            decimals=1,
            thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 0.01}],
        ),
        6,
        5,
    )
    b.row("Tráfego")
    b.add(
        timeseries(
            "Requisições por segundo por código de status",
            [prom(f"sum by (http_response_code) (rate({REQ}{{{PRED}}}[$__rate_interval]))", legend="{{http_response_code}}")],
            "rate() do contador bentoml_service_request_total em /predict, por código HTTP.",
            unit="reqps",
        ),
        12,
        8,
    )
    b.add(
        timeseries(
            "Taxa de erro 4xx e 5xx",
            [
                prom(f'(sum(rate({REQ}{{{PRED},http_response_code=~"4.."}}[5m])) or vector(0)) / sum(rate({REQ}{{{PRED}}}[5m]))', ref="A", legend="4xx (entrada inválida)"),
                prom(f'(sum(rate({REQ}{{{PRED},http_response_code=~"5.."}}[5m])) or vector(0)) / sum(rate({REQ}{{{PRED}}}[5m]))', ref="B", legend="5xx (servidor)"),
            ],
            "Fração das requisições com erro, janela de 5 minutos. 4xx = cliente, 5xx = servidor.",
            unit="percentunit",
            minimum=0,
        ),
        12,
        8,
    )
    b.row("Latência")
    quantiles = [
        prom(f"histogram_quantile({q}, sum by (le) (rate({DUR}{{{PRED}}}[5m])))", ref=chr(ord("A") + i), legend=f"p{int(q * 100)}")
        for i, q in enumerate((0.5, 0.95, 0.99))
    ]
    b.add(timeseries("Latência p50, p95 e p99", quantiles, "histogram_quantile sobre os buckets de bentoml_service_request_duration_seconds (janela de 5 min). Interpola linearmente dentro do bucket.", unit="s", minimum=0), 12, 8)
    heat = prom(f"sum by (le) (increase({DUR}{{{PRED}}}[$__rate_interval]))", legend="{{le}}", fmt="heatmap")
    b.add(
        panel(
            "heatmap",
            "Heatmap de latência",
            [heat],
            "Distribuição da duração por bucket `le` ao longo do tempo (escuro = poucas, claro = muitas observações).",
            options={"calculate": False, "yAxis": {"axisPlacement": "left", "unit": "s"}, "rowsFrame": {"layout": "le"}, "color": {"mode": "scheme", "scheme": "Oranges", "scale": "exponential", "fill": "dark-orange", "steps": 32}, "cellGap": 1, "legend": {"show": True}, "tooltip": {"mode": "single", "yHistogram": False}},
            fieldConfig={"defaults": {"custom": {"scaleDistribution": {"type": "linear"}}}, "overrides": []},
        ),
        12,
        8,
    )
    b.add(
        deaccumulated(
            "bentoml_service_request_duration_seconds",
            DURATION_BUCKETS,
            "Histograma de latência (intervalo do dashboard)",
            "Observações por faixa de duração (segundos) no intervalo selecionado, obtidas subtraindo buckets `le` consecutivos (os buckets são acumulados).",
            matchers=PRED,
        ),
        12,
        8,
    )
    b.add(
        numeric_series("Latência por requisição (log)", {"latency_ms": "latency_ms"}, "Latência em ms de cada requisição respondida com 200, direto do log (inclui fila atrás do worker único e o carregamento preguiçoso do primeiro pedido)."),
        12,
        8,
    )
    b.row("Requisições mais lentas")
    slow_root = (
        '$map(($[status_code=200]^(>latency_ms))[[0..19]], function($r){ { "ts": $r.ts, "request_id": $r.request_id, "latency_ms": $r.latency_ms, '
        '"status": $r.status_code, "funcao": $r.out_funcao, "largura": $r.in_width, "altura": $r.in_height, "bytes": $r.in_file_bytes } })'
    )
    slow = variables(
        [col("ts", "Quando", "timestamp"), col("request_id", "request_id", "string"), col("latency_ms", "latency_ms"), col("status", "status"), col("funcao", "funcao", "string"), col("largura", "largura"), col("altura", "altura"), col("bytes", "bytes")],
        root=slow_root,
    )
    slow_overrides: list[Json] = [{"matcher": {"id": "byName", "options": "latency_ms"}, "properties": [{"id": "unit", "value": "ms"}, {"id": "decimals", "value": 0}]}]
    b.add(table("20 requisições mais lentas (ordenadas por latency_ms)", [slow], "De /data/variables via Infinity, intervalo do dashboard, ordenadas por latency_ms.", overrides=slow_overrides, transformations=[ordered(["Quando", "request_id", "latency_ms", "status", "funcao", "largura", "altura", "bytes"])]), 24, 9)
    return b


def category_counts(name: str) -> Json:
    """Contagem por categoria da janela (`categorical.<name>.window` de /data/summary)."""
    root = f'$each(categorical.{name}.window, function($v,$k){{ {{"{name}": $k, "n": $v}} }})'
    return summary(root, [col(name, name, "string"), col("n", "imagens")])


def build_entrada() -> Board:
    b = Board("leiturista-entrada", "Leiturista: Entrada (imagem)", "Variáveis extraídas da imagem de entrada, janela do dashboard contra a referência (132 fotos).")
    b.row("Histogramas 0-255: janela x referência")
    for channel in CHANNELS:
        b.add(pixel_histogram(channel), 12, 8)
    b.row("Médias por canal ao longo do tempo")
    means = variables(
        [col("Time", "Time", "timestamp"), *[col(c, c) for c in CHANNELS]],
        root=map_rows("$[status_code=200]", {"Time": "$r.ts", **{c: f"$r.in_{name}" for c, name in CHANNELS.items()}}),
    )
    band_root = f'$map({json.dumps(list(CHANNELS))}, function($c){{ ( $k := $lookup({json.dumps(CHANNELS)}, $c); $n := $lookup($$.numeric, $k).reference; {{"canal": $c, "q1": $n.q1, "q3": $n.q3, "media": $n.mean}} ) }})'
    band = summary(band_root, [col("canal", "canal", "string"), col("q1", "q1"), col("q3", "q3"), col("media", "media")], ref="B")
    b.add(
        echarts(
            "Médias dos canais L, R, G e B (com a referência como faixa)",
            [means, band],
            MEANS_BAND_JS,
            "Média 0-255 de cada canal por imagem. A faixa translúcida de cada cor é o intervalo Q1 a Q3 da referência e a linha tracejada é a média da referência. "
            "Painel Business Charts (ECharts) alimentado por /data/variables e /data/summary via Infinity.",
        ),
        24,
        9,
    )
    b.row("Distribuições por imagem (histogramas)")
    b.add(sparse_hist("in_luminance_mean", "in_luminance_mean", "Luminância média (0-255) por imagem."), 8, 8)
    b.add(sparse_hist("luminance_std", "in_luminance_std", "Contraste: desvio-padrão da luminância por imagem."), 8, 8)
    b.add(sparse_hist("sharpness", "in_sharpness", "Nitidez: variância do laplaciano da luminância por imagem."), 8, 8)
    b.row("Box plots (janela x referência)")
    b.add(boxplot("Boxplot das médias por canal", list(CHANNELS.values()), "Médias de L, R, G e B (0-255)."), 12, 9)
    for name in ("luminance_std", "sharpness", "width", "height", "file_bytes"):
        b.add(boxplot(f"Boxplot de {name}", [name]), 6, 9)
    b.row("Categóricas")
    b.add(pie("dominant_channel (canal dominante)", [category_counts("dominant_channel")], "Quantas imagens do intervalo têm R, G ou B como canal de maior média."), 8, 8)
    b.add(barchart("format (formato do arquivo)", [category_counts("format")], "format", "Formato Pillow das imagens recebidas no intervalo."), 8, 8)
    b.add(
        deaccumulated(
            "leiturista_input_luminance_mean",
            LUMINANCE_BUCKETS,
            "Luminância média: métrica customizada do Prometheus",
            "Histograma leiturista_input_luminance_mean (buckets de 16 em 16, 0 a 255) no intervalo do dashboard, subtraindo buckets `le` consecutivos.",
        ),
        8,
        8,
    )
    return b


def rate_transform(column: str, alias: str) -> Json:
    return {"id": "calculateField", "options": {"mode": "windowFunctions", "alias": alias, "window": {"field": column, "reducer": "mean", "windowAlignment": "trailing", "windowSize": 10, "windowSizeMode": "fixed"}}}


def build_saida() -> Board:
    b = Board("leiturista-saida", "Leiturista: Saída (modelo)", "Comportamento do modelo: confiança, função, legibilidade, flags (Infinity e Prometheus).")
    b.row("Confiança")
    b.add(sparse_hist("confianca", "out_confianca", "Distribuição da confiança da resposta (0 a 1) por imagem.", bucket_size=0.05), 8, 8)
    b.add(boxplot("Boxplot de confianca", ["confianca"]), 8, 8)
    b.add(numeric_series("confianca ao longo do tempo", {"confianca": "out_confianca"}, "Confiança de cada requisição."), 8, 8)
    b.row("Função")
    b.add(
        timeseries(
            "Proporção de funcao ao longo do tempo",
            [prom("sum by (funcao) (increase(leiturista_prediction_funcao_total[$__rate_interval]))", legend="{{funcao}}")],
            "Barras empilhadas (100%) do incremento do contador leiturista_prediction_funcao_total por valor de funcao.",
            style="bars",
            stack="percent",
        ),
        12,
        8,
    )
    b.add(pie("funcao no intervalo (pizza)", [category_counts("funcao")], "Contagem por funcao das imagens do intervalo. Fonte: /data/summary via Infinity."), 12, 8)
    b.row("Taxas ao longo do tempo (média móvel de 10 imagens)")
    flags = variables(
        [col("Time", "Time", "timestamp"), col("legivel", "legivel"), col("has_reading", "has_reading"), col("has_serial", "has_serial")],
        root=map_rows("$[status_code=200]", {"Time": "$r.ts", "legivel": "$r.out_legivel ? 1 : 0", "has_reading": "$r.out_has_reading ? 1 : 0", "has_serial": "$r.out_has_serial ? 1 : 0"}),
    )
    transforms = [
        rate_transform("legivel", "legivel (taxa)"),
        rate_transform("has_reading", "has_reading (taxa)"),
        rate_transform("has_serial", "has_serial (taxa)"),
        {"id": "filterFieldsByName", "options": {"include": {"pattern": "^(Time|.* \\(taxa\\))$"}}},
    ]
    b.add(
        timeseries("Taxas de legivel, has_reading e has_serial", [flags], "Média móvel das últimas 10 imagens de cada indicador (1 = sim, 0 = não). Fonte: /data/variables via Infinity.", unit="percentunit", minimum=0, maximum=1, transformations=transforms),
        24,
        8,
    )
    b.row("Contagens")
    b.add(sparse_hist("reading_digits", "out_reading_digits", "Número de dígitos da leitura (nunca o valor).", bucket_size=1), 8, 8)
    b.add(sparse_hist("n_flags", "out_n_flags", "Quantidade de flags por resposta.", bucket_size=1), 8, 8)
    flag_root = '$map($distinct($.out_flags)[$type($)="string"], function($x){ {"flag": $x, "n": $count($$.out_flags[$=$x])} })^(>n)'
    top = variables([col("flag", "flag", "string"), col("n", "ocorrências")], root=flag_root)
    b.add(table("Flags mais frequentes", [top], "Frases fixas de flags (dígitos removidos) contadas no intervalo. Fonte: /data/variables via Infinity.", transformations=[ordered(["flag", "ocorrências"])]), 8, 8)
    b.row("Nitidez do candidato e confiança (Prometheus)")
    b.add(sparse_hist("candidate_sharpness", "out_candidate_sharpness", "Nitidez do recorte candidato a display, por imagem."), 8, 8)
    b.add(boxplot("Boxplot de candidate_sharpness", ["candidate_sharpness"]), 8, 8)
    b.add(
        deaccumulated(
            "leiturista_prediction_confidence",
            CONFIDENCE_BUCKETS,
            "Confiança: métrica customizada do Prometheus",
            "Histograma leiturista_prediction_confidence (buckets de 0,1 em 0,1) no intervalo do dashboard, subtraindo buckets `le` consecutivos.",
        ),
        8,
        8,
    )
    return b


def build_drift() -> Board:
    b = Board("leiturista-drift", "Leiturista: Drift", "KS por canal e PSI por variável da janela deslizante (últimas 100 imagens) contra a referência. Gauges do `monitor`.", time_from="now-3h")
    ks_thresholds: list[Json] = [{"color": "green", "value": None}, {"color": "yellow", "value": 0.05}, {"color": "red", "value": 0.10}]
    b.row("Estado atual")
    for channel in CHANNELS:
        b.add(
            stat(
                f"Drift canal {channel}",
                [prom(f'leiturista_drift_level{{channel="{channel}"}}', instant=True)],
                "Nível do KS (D) do canal: verde (< 0,05), amarelo (>= 0,05), vermelho (>= 0,10), cinza (menos de 8 imagens).",
                mappings=LEVEL_MAPPINGS,
                thresholds=[{"color": "gray", "value": None}, {"color": "green", "value": 0}, {"color": "yellow", "value": 1}, {"color": "red", "value": 2}],
                color_mode="background",
            ),
            4,
            4,
        )
    b.add(stat("Imagens na janela", [prom("leiturista_drift_window_images", instant=True)], "Imagens da janela deslizante usada nos cálculos (mínimo de 8 para um nível conclusivo).", decimals=0), 4, 4)
    b.row("KS por canal")
    b.add(
        timeseries(
            "KS D por canal (L, R, G, B)",
            [prom("leiturista_drift_ks_d", legend="{{channel}}")],
            "D = máximo da diferença entre as CDFs do histograma agregado da janela e o da referência. Linhas: 0,05 (amarelo) e 0,10 (vermelho).",
            minimum=0,
            maximum=1,
            thresholds=ks_thresholds,
        ),
        12,
        8,
    )
    b.add(
        panel(
            "state-timeline",
            "Estado do drift ao longo do tempo",
            [prom("leiturista_drift_level", legend="{{channel}}")],
            "leiturista_drift_level por canal: -1 inconclusivo, 0 verde, 1 amarelo, 2 vermelho.",
            fieldConfig={
                "defaults": {"mappings": LEVEL_MAPPINGS, "color": {"mode": "thresholds"}, "thresholds": {"mode": "absolute", "steps": [{"color": "gray", "value": None}, {"color": "green", "value": 0}, {"color": "yellow", "value": 1}, {"color": "red", "value": 2}]}, "custom": {"fillOpacity": 80, "lineWidth": 0}},
                "overrides": [],
            },
            options={"mergeValues": True, "showValue": "never", "alignValue": "center", "rowHeight": 0.8, "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "single"}},
        ),
        12,
        8,
    )
    b.add(
        timeseries(
            "KS p-valor por canal",
            [prom("leiturista_drift_ks_pvalue", legend="{{channel}}")],
            "p-valor assintótico (kstwobign, n em imagens). Leia sempre junto com D; com referência pequena, p é pouco informativo.",
            minimum=0,
            maximum=1,
        ),
        24,
        7,
    )
    b.row("PSI por variável")
    b.add(
        panel(
            "bargauge",
            "PSI por variável (janela deslizante)",
            [prom("leiturista_drift_psi", legend="{{variable}}", instant=True)],
            "Population Stability Index por variável (10 bins por quantis da referência; categorias para as categóricas). Faixas 0,1 e 0,25 são convenção da indústria, não regra provada. A barra vai até 5; valores maiores aparecem só no número.",
            fieldConfig={"defaults": {"min": 0, "max": 5, "decimals": 3, "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "yellow", "value": 0.1}, {"color": "red", "value": 0.25}]}, "color": {"mode": "thresholds"}}, "overrides": []},
            options={"orientation": "horizontal", "displayMode": "gradient", "showUnfilled": True, "valueMode": "color", "namePlacement": "left", "sizing": "manual", "minVizHeight": 16, "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}},
        ),
        24,
        18,
    )
    b.row("Números do drift (janela deslizante atual)")
    chan = infinity(
        "/data/drift",
        '$each(channels, function($v,$k){ {"canal": $k, "D": $v.d, "p": $v.p_value, "nivel": $v.level_name} })',
        [col("canal", "canal", "string"), col("D", "D"), col("p", "p"), col("nivel", "nivel", "string")],
    )
    psi = infinity(
        "/data/drift",
        '$each(psi, function($v,$k){ {"variavel": $k, "PSI": $v.value, "faixa": $v.band, "bins": $v.bins, "motivo": $v.reason} })',
        [col("variavel", "variavel", "string"), col("PSI", "PSI"), col("faixa", "faixa", "string"), col("bins", "bins"), col("motivo", "motivo", "string")],
    )
    b.add(table("KS por canal (D, p, nível)", [chan], "Fonte: /data/drift via Infinity (a mesma janela dos gauges).", transformations=[ordered(["canal", "D", "p", "nivel"])]), 12, 9)
    b.add(table("PSI de todas as variáveis", [psi], "Fonte: /data/drift via Infinity. PSI nulo traz o motivo (por exemplo referência constante).", transformations=[ordered(["variavel", "PSI", "faixa", "bins", "motivo"]), {"id": "sortBy", "options": {"sort": [{"field": "PSI", "desc": True}]}}]), 12, 9)
    return b


def build_logs() -> Board:
    b = Board("leiturista-logs", "Leiturista: Logs (Loki)", "O log JSONL de requisições e respostas, enviado ao Loki pelo Alloy.")
    b.row("Fluxo")
    stream: Json = panel(
        "logs",
        "Log ao vivo",
        [loki('{job="leiturista"}')],
        "Todas as linhas do log (request e response). Fonte: Loki, enviado pelo Alloy.",
        options={"showTime": True, "wrapLogMessage": False, "prettifyLogMessage": False, "enableLogDetails": True, "sortOrder": "Descending", "dedupStrategy": "none"},
    )
    b.add(stream, 24, 10)
    b.row("Taxas")
    b.add(
        timeseries("Taxa de linhas por event", [loki('sum by (event) (rate({job="leiturista"}[1m]))', legend="{{event}}")], "Linhas por segundo, por rótulo `event` (request ou response).", unit="ops"),
        12,
        8,
    )
    b.add(
        timeseries(
            "Volume de log ao longo do tempo",
            [loki('sum by (event) (count_over_time({job="leiturista"}[$__interval]))', legend="{{event}}")],
            "Linhas por intervalo, empilhadas por event.",
            style="bars",
            stack="normal",
        ),
        12,
        8,
    )
    b.row("Erros")
    errors: Json = panel(
        "logs",
        "Erros (status_code >= 500)",
        [loki('{job="leiturista", event="response"} | json | status_code >= 500')],
        "Respostas 5xx: o status_code é extraído do JSON na consulta (nunca é rótulo no Loki).",
        options={"showTime": True, "wrapLogMessage": False, "prettifyLogMessage": False, "enableLogDetails": True, "sortOrder": "Descending", "dedupStrategy": "none"},
    )
    b.add(errors, 24, 9)
    return b


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for builder in (build_operacao, build_entrada, build_saida, build_drift, build_logs):
        board = builder()
        path = board.write()
        print(f"{path.relative_to(ROOT)}: {sum(1 for p in board.panels if p['type'] != 'row')} paineis")


if __name__ == "__main__":
    main()
