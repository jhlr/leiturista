# SDD: Leiturista MLOps Monitoring (Group Luminus)

Date: 2026-11-01 | Status: ready for an agent with no prior context | Repo: this one (`leiturista`), work on branch `feat/mlops-monitoring`

You are an implementation agent. This document is your only source of truth. Do not ask questions; where a choice is open, **Decisions** already made it. Every acceptance criterion in section 10 must pass before you report done. Read first: `AGENTS.md`, `README.md`, `src/leiturista/service.py`, `src/leiturista/inference.py` (the `Prediction` dataclass and `MeterOCR.predict_image`), `justfile`, `pyproject.toml`.

---

## 1. Goal

Add observability to the existing Leiturista inference service (BentoML, `POST /predict`: photo of an electric meter in, reading/serial/confidence out). Monitoring here means mainly **writing to a file a trace of every request and every response**. A professor must be able, using only the README, to start the service, generate load, and watch the trace grow.

Following the course pattern (https://github.com/lgallindo/bentoML-processamento-imagens), we monitor **features extracted from the input image** (per-channel means and 0-255 histograms of Luminance, Red, Green, Blue, plus extra features) and the **model outputs**, never the pixels.

| # | Deliverable | Points |
|---|---|---|
| D1 | Append-only file log of every request/response, with variables suited to histograms, boxplots, KS and PSI | 50 |
| D2 | `GET /metrics` in Prometheus exposition format + a text (and spreadsheet) explaining how to read it | 10 |
| E1 | Extra: ship the log file to Loki | 5 |
| E2 | Extra: endpoint exposing the variables for histograms, boxplots, KS, PSI | 20 |
| E3 | Extra: Prometheus server scraping `/metrics` | 15 |

### Priority tiers (work in this order, report per tier)

Effort follows the grading, not the amount of text in this document. Do not start a tier until the previous one passes its criteria (section 10); if you run out of room, stop at the last passing tier and report exactly what is left. Never fake a criterion.

| Tier | Scope | Points | Criteria (section 10) |
|---|---|---|---|
| P0 | D1 + D2 + local README path + dependency split + code placement | 60 | 1a, 3 to 8, 12, 13, 14 (local path) |
| P1 | E2, E3, E1, and the Docker environments | 40 | 1b, 2, 9, 10a |
| P2 | Grafana dashboards, drift analysis script, full chart checklist | 0 | 10b, 11 |

Out of scope: building histogram/boxplot/KS/PSI analyses as product features; we only **save the data**. An analysis script is recommended, only to prove the data is sufficient (section 9). Out of scope too: changing models, retraining (**AGENTS.md rule: training is run only by the user; you never run `leiturista train`**), and touching the Streamlit app, `lab-02/` or the existing CLI.

## 2. Architecture: three separate environments (hard requirement)

The professor stressed that **training, inference and monitoring must be different environments**. They share nothing but explicit, read-only interfaces:

```
 TRAINING                    INFERENCE                       MONITORING
 (existing, untouched)       (docker/inference)              (docker/monitoring)
 uv group `train`            image `leiturista-inference`    prometheus, loki, alloy,
 CLI `leiturista train`      bentoml serve :3000             grafana, `monitor` (Bento)
        |                    writes JSONL log                      ^        ^
        |  models/ (files)   exposes /metrics  ----scrape----------+        |
        +--read-only mount-> reads models/                                   |
                             ./serving-logs/ (bind mount) --read-only mount--+
```

| Environment | Contains | Must NOT contain |
|---|---|---|
| Training | existing `leiturista` pipeline, MLflow, datasets | anything from serving or monitoring |
| Inference | `leiturista` library, BentoML, onnxruntime, torch+transformers (TrOCR fallback), Pillow, numpy, logging code, `log_reader` (serves `/data/variables` from its own log only) | MLflow, datasets, accelerate, Streamlit, scipy/pandas/matplotlib, Prometheus/Loki/Grafana servers (the `prometheus_client` library is a BentoML dependency and is fine), any drift or reference analysis |
| Monitoring | Prometheus, Loki, Grafana Alloy, Grafana, `monitor` (a second, small BentoML service + pandas + scipy), analysis scripts | torch, transformers, onnxruntime, model weights |

Interfaces between them, and nothing else:
1. Training -> inference: model files under `models/` and `.model_cache/`, mounted **read-only** into the inference container. The image contains no weights (`just models` downloads a release tarball of about 150 MB; the local `models/` may be larger because it also holds locally trained files, which is harmless read-only). At startup the service checks that the TrOCR weights (`model.safetensors` in `.model_cache/trocr-small-printed`) exist; if not it **exits with a clear message telling the user to run `just models`**. It must not fall back to `artifacts.materialize` from `mlflow.db` (inference has no MLflow). Add a `.dockerignore` excluding `data/`, `models/`, `.model_cache/`, `mlflow.db`, `logs/`, `serving-logs/`, `.venv/` so the build context stays small.
2. Inference -> monitoring, metrics: HTTP scrape of `inference:3000/metrics` over a shared external Docker network `luminus-net`.
3. Inference -> monitoring, logs: the host directory `./serving-logs` (inference mounts it read-write; Alloy and `monitor` mount it **read-only**). Monitoring never writes there. On macOS Docker bind mounts, run the inference container with `user: "${UID}:${GID}"` (exported by the `just` recipes) so the log file is owned by the host user, and read the log from the host side when checking counts.
4. The monitoring environment never calls `/predict` except through the load generator, which is a client script, not part of either runtime.

Deliver this as: `docker/inference/Dockerfile` + `docker/inference/compose.yml`; `docker/monitoring/Dockerfile.monitor` + `docker/monitoring/compose.yml`. **Two separate compose files**, started independently (`just up-inference`, `just up-monitoring`). Both recipes create the shared network when missing (`docker network inspect luminus-net >/dev/null 2>&1 || docker network create luminus-net`), so either can start first, and both compose files declare it `external: true`. Each can be started and stopped without the other; inference keeps logging when monitoring is down. Build inference with CPU-only torch wheels and state the resulting image size and build time in the README.

### 2.1 Dependency split (single `pyproject.toml`, no new package)

Split the current flat `dependencies` into a light base plus extras, so each environment installs only its own set:

- base (`[project].dependencies`): `pillow`, `numpy` (all the feature extractor needs);
- extra `inference`: `bentoml`, `onnxruntime`, `opencv-python-headless`, `torch`, `transformers`, `sentencepiece`, `tokenizers`;
- extra `train`: everything training uses today (`datasets`, `accelerate`, `mlflow`, `pandas`, `pyarrow`, `duckdb`, `torchvision`, ...);
- extra `monitor`: `bentoml`, `pandas`, `scipy`, `matplotlib`, `httpx` (BentoML here is the light serving runtime for `monitor`; no torch or onnxruntime);
- `streamlit` goes to extra `train` (the only app under `app/` is the labeling tool `rotular.py`). Docker builds use `uv sync --extra inference --no-dev` (and `--extra monitor --no-dev`).

Regenerate `uv.lock`. Verify the split by checking that `uv sync --extra inference` does not install `mlflow` and `uv sync --extra monitor` does not install `torch`. The existing `just setup` must keep working for the current workflow (point it at all extras, or document the change in the README). Every other existing `just` recipe keeps working.

## 3. Decisions (already made)

1. **Service under monitoring:** the existing `LeituristaService.predict`. Keep its response contract unchanged.
2. **Monitoring code** lives in the library, `src/leiturista/monitoring/` (AGENTS.md rule 2: reusable code goes in the library), imported by the service. Infra files live in `docker/`, `monitoring/` (configs) and `scripts/`.
3. **Feature extraction has ONE owner:** `src/leiturista/monitoring/features.py`. Drift math and alert thresholds have ONE owner: `src/leiturista/monitoring/drift.py`. Output-variable derivation has ONE owner: `src/leiturista/monitoring/outputs.py`. The service, the reference builder, `monitor`, the load generator and the analysis script only call these.
4. **Log format:** JSON Lines via Python `logging` + `RotatingFileHandler` (mode `a`, 50 MB, 5 backups), file `serving-logs/inference.log`. Append-only: nothing rewrites the file in place; rotation renames and opens a new file. Single BentoML worker, set with `@bentoml.service(workers=1)` and in the build config, because the handler is not multi-process safe; say so in the README. **Where the handler lives:** BentoML's logging docs say `RotatingFileHandler` is not supported inside the Service definition because `bentoml serve` forks. So `logging_setup.get_serving_logger()` creates the handler lazily inside the worker process (called from the service `__init__` and middleware, never at import time), and a test must prove rotation against a **running** service (env `SERVING_LOG_MAX_BYTES` lowers the limit for the test). If rotation does not work that way, fall back to a plain `FileHandler` that renames the file to `.1` (shifting older backups, keeping 5) at service start when it exceeds 50 MB, and report the fallback and its limitation (no rotation during one long run).
5. **Timestamp:** UTC, ISO 8601, microseconds, explicit offset (`2026-11-01T14:03:22.481516+00:00`), from `record.created`.
6. **Privacy (the client distribution company is confidential, `README.md` says its name must not appear):** the log never contains the image, the **meter serial, or the reading value**. It logs only derived numbers and booleans. The `error` field holds only the exception **class name plus a fixed error code** (for example `ValueError/invalid_image`), never the exception message, because messages can carry OCR or input text. `client_ip` is logged only as a truncated SHA-256 hash (first 12 hex chars). The company name does not appear in any file you create.
7. **Reference set and traffic sets** (the group's decision; all public datasets already under `data/`, listed in `docs/origem_dos_dados.md`, all re-downloadable, never committed; the Praekelt dataset declares no license, so images are not redistributed):
   - **Reference:** 132 of the 165 JPEGs in `data/praekelt_meter_readings_img/praekelt_*.jpg` (public dataset `Praekelt/ElectricityMeterReadings1o4`), chosen by a deterministic split: sort file names, shuffle with seed 0, first 132. Ignore `labels.csv`.
   - **Traffic `holdout`:** the other 33 Praekelt images (same distribution, never in the reference). This is the "no drift" control.
   - **Traffic `henrik`:** `data/henrik_energy_meter_img/**/*.png` (168 images, in `train/` and `test/` subfolders, so use a recursive glob) and **traffic `lcd`:** `data/utility_meters_lcd/` (50 images): different cameras and meters, expected to show real drift.
   - Synthetic drift (`mild`, `strong`, section 8) is applied on top of `holdout`.
   `scripts/build_reference.py` runs `features.py`, `outputs.py` and the real `MeterOCR` over the reference images and writes `artifacts/reference.csv` (one row per image, same columns as the log's scalar variables without prefix, no file names) and `artifacts/reference_hist.csv` (per-channel 256-bin histograms, each image normalized to sum 1 and then summed over the set, so each column totals the number of images; `aggregate_histograms` in `drift.py`). It takes `--images-dir`, `--seed` and `--hist-only` (rewrite only `reference_hist.csv`, no MeterOCR). The split is shared code in `src/leiturista/monitoring/` (`split_reference_images`), used by both `build_reference.py` and `load_gen.py`. The reference runs inference once, as a batch job in the **inference** environment (`just build-reference` on the host, or the `build-reference` compose service, section 4). README note: Praekelt photos come from another country and camera setup than the distribution company's field photos, so drift on real field traffic is expected and is itself a finding. With 132 reference images the reference is small: say so next to every KS/PSI number.
8. **Language:** this SDD and any prompt/instruction file are English. Professor-facing deliverables (`README` additions, `docs/metrics_analysis.md`, spreadsheet labels, dashboard titles) are **Brazilian Portuguese**. Code identifiers and log field names stay English.
9. **Docs** go to `docs/`, named `YYYY-MM-DD_name` like the existing ones (AGENTS.md rule 1), committed. Name the new ones `<real date>_monitoramento_*.md`.

## 4. Repository layout (new or changed only)

```
src/leiturista/
  service.py                   # CHANGED: thin wrapper that logs request/response and exports custom metrics
  monitoring/
    __init__.py
    features.py                # ImageFeatures + extract_features (owner)
    outputs.py                 # derive output variables from Prediction (owner)
    drift.py                   # KS (D, p), PSI, alert_level (owner)
    logging_setup.py           # JSON formatter + rotating file handler
    metrics.py                 # prometheus_client custom metrics (input luminance, confidence, funcao counter)
    log_reader.py              # parse current + rotated logs -> rows (owner; used by inference and monitor)
    split.py                   # split_reference_images (owner of the reference/holdout split)
  monitor_service.py           # BentoML service `monitor` (monitoring env): /data/*, drift gauges on /metrics
docker/
  inference/Dockerfile
  inference/compose.yml        # service `inference` (models and .model_cache ro, serving-logs rw) and one-off service `build-reference` (profile `tools`: data ro, models ro, artifacts rw)
  monitoring/Dockerfile.monitor
  monitoring/compose.yml       # prometheus, loki, alloy, grafana, monitor; network luminus-net (external)
monitoring/
  prometheus.yml
  alloy/config.alloy           # Grafana Alloy: tail serving-logs -> Loki
  grafana/provisioning/datasources/datasources.yml   # Prometheus + Loki
  grafana/provisioning/dashboards/                   # five dashboard JSONs, see section 7
scripts/
  build_reference.py
  load_gen.py
  analyze_log.py               # recommended
  make_metrics_xlsx.py
artifacts/                     # reference.csv, reference_hist.csv
serving-logs/                  # gitignored
reports/                       # gitignored
docs/
  2026-11-01_sdd_monitoramento_mlops.md   # this file
  <date>_monitoramento_metrics_analysis.md
  <date>_monitoramento_metrics_analysis.xlsx
tests/monitoring/
```

Add `serving-logs/` and `reports/` to `.gitignore`. Add `just` recipes: `up-inference`, `down-inference`, `up-monitoring`, `down-monitoring`, `build-reference`, `load`, `analyze`. The monitor service runs as `leiturista.monitor_service:MonitorService`.

## 5. D1: file logging (50 pts)

### 5.1 Image features (`features.py`)

Computed on the decoded image after `convert("RGB")`. `L` = Pillow `convert("L")`. Histograms from `Image.histogram()` per channel (256 ints). Means computed **from the histogram** like the course code: `mean = sum(i * counts[i]) / sum(counts)`.

| Feature | Type | Definition |
|---|---|---|
| `width`, `height` | int | pixels |
| `file_bytes` | int | upload size |
| `format` | categorical | Pillow format (JPEG/PNG/...) |
| `luminance_mean`, `r_mean`, `g_mean`, `b_mean` | float | per-channel mean, 0-255 (the four base features of the course pattern) |
| `hist_l`, `hist_r`, `hist_g`, `hist_b` | list[int] (256) | per-channel histograms |
| `luminance_std` | float | contrast: std of L from its histogram |
| `sharpness` | float | variance of the Laplacian of the L channel (numpy 3x3 kernel) |
| `dominant_channel` | categorical R/G/B | channel with the highest mean (ties: R, G, B order) |
| `bright_clip_frac` | float | fraction of pixels with L >= 250 (glare/overexposure), from the L histogram |
| `dark_clip_frac` | float | fraction of pixels with L <= 5 (underexposure), from the L histogram |
| `luminance_entropy` | float | Shannon entropy (bits) of the normalized L histogram |
| `edge_density` | float | fraction of pixels whose gradient magnitude on L (numpy finite differences) exceeds a fixed threshold; define the threshold as a constant in `features.py` |
| `saturation_mean` | float | mean saturation (0-255) of the image converted with Pillow to HSV |
| `orientation` | categorical | `portrait` (height > width), `landscape` (width > height), `square` |
| `dynamic_range` | float | 95th minus 5th percentile of L, from the L histogram |

The features from `luminance_std` down to `dynamic_range` are the implementer's own suggestions, not an assignment requirement; they give the histograms, boxplots, KS and PSI more variables. Adding or removing one means editing only `features.py` and the column list. Every scalar feature automatically gets a histogram, a boxplot (Grafana dashboard 2), and a KS/PSI row (section 9 and the `monitor` gauges).

### 5.2 Output variables (`outputs.py`), from `Prediction`

| Variable | Type | Definition |
|---|---|---|
| `funcao` | categorical | the service's `funcao` (`leitura_normal`, `leitura_incerta`, `sem_leitura_detectada`) |
| `confianca` | float | the service's `confianca` |
| `legivel` | bool | `Prediction.legible` |
| `has_reading` | bool | `Prediction.reading is not None` |
| `has_serial` | bool | `Prediction.serial is not None` |
| `reading_digits` | int or null | number of digits in the reading, never the value |
| `n_boxes` | int | `len(Prediction.boxes)` |
| `n_flags` | int | `len(Prediction.flags)` |
| `flags` | list[str] | the flag strings (fixed phrases; strip any digits so the rotation angle is not a high-cardinality token) |
| `candidate_sharpness` | float or null | `Prediction.sharpness` |

`funcao` and `confianca` must be derived by the **same code** the service uses for its response (refactor that logic into one function, reused by both the response and `outputs.py`; the response contract does not change).

### 5.3 Events

Each `/predict` call writes **two lines** linked by `request_id`.

**Who writes what (decided):** an ASGI middleware generates `request_id` (UUID4), starts `time.perf_counter()`, sets header `X-Request-ID`, and creates a **mutable trace object** (`RequestTrace`: request_id, start time, an outputs dict) stored in `request.state`. Do **not** pass data between the function and the middleware through a `contextvars.ContextVar` set inside the API function: BentoML runs sync APIs in a thread pool on a copied context, so such a `set` may never reach the middleware. The API function receives the trace through `bentoml.Context` (`ctx.request.state.trace`; confirm in the BentoML 1.4 docs that a `ctx: bentoml.Context` parameter is allowed and does not change the input schema or response), writes the **`request` line** as soon as the image decodes, and fills the outputs dict (section 5.2). The middleware, after the response is produced, writes the **`response` line** with latency, status and the outputs from the trace. The request line therefore has the real start-of-processing timestamp and comes before the response line. Add a test that fires concurrent requests and checks every response line carries the outputs of its own request (no cross-talk).

**Fallback (only if the pinned BentoML 1.4 offers no middleware hook; report it):** a decorator on the API function does the same job (creates the trace, times the call, writes both lines), and invalid inputs that never reach the function then get **no** log line; criterion 4 then expects 2 lines instead of 3 and the README says so. Do not hack around the framework to catch them.

**`event: "request"`:** `ts`, `level`, `event`, `request_id`, `endpoint`, `method`, `service_version` (env `SERVICE_VERSION`, set at image build from `git rev-parse --short HEAD` by the `just` recipe; when unset, try `git` locally, else `unknown`; the Docker image has no `.git`), `client_ip` (hashed, Decision 6), and every feature of 5.1 prefixed `in_`.

**`event: "response"`** (always, including errors, so validation failures are covered): `ts`, `level`, `event`, `request_id`, `endpoint`, `method`, `status_code`, `latency_ms` (3 decimals), every variable of 5.2 prefixed `out_` (all null on failure), `error` (class + fixed code, Decision 6; null on success).

Invalid input never reaches the API function, so it has a `response` line only, with null `out_*`. A 5xx logs at ERROR.

`latency_ms` includes time spent queued behind the single worker; the first call after start also includes lazy model loading. Both are expected; document them next to the latency analysis.

Only `/predict` emits these events. `/metrics`, `/livez`, `/readyz` and `/data/*` must not write to the log, and the service writes nothing at startup or rotation, so line counts in tests are exact.

### 5.4 Implementation rules

- Standard library `logging` (https://docs.python.org/pt-br/3/howto/logging.html), logger `leiturista.serving`, `propagate=False`, INFO. Formatter returns `json.dumps(payload)`; no free-text messages.
- Log dir from env `SERVING_LOG_DIR` (default `./serving-logs`); create at startup. Flush every record.
- Never log image bytes, serial, reading value, exception messages or raw client IPs. Privacy test: run `/predict` on a sample with a known serial and reading, force one 4xx and one 5xx (the 5xx through a test-only fault injected in the test, not in production code), and fail if the serial, the reading, any exception message text or the raw client IP appears in any line.
- Logging failures must never fail a request: rely on `logging.handleError`, no try/except around it.
- **Response-contract test** (guard): for each of the 5 images that `just samples` generates in `samples/` (`exemplo_01..04.png` and `sem_medidor.png`; the test generates them if missing), `/predict` returns status 200 and exactly the fields `numero_medidor`, `funcao`, `consumo`, `confianca`, `legivel`, `flags` with the same types as before the change. It does not compare values. Capture the field list and types from the pre-change service before editing it, and keep them in the test.

### 5.5 CRISP-DM link (README section "Entendimento dos dados")

Table plus 5 lines: image features describe photo conditions (brightness, color cast, blur, size); outputs describe model behavior (`confianca`, `legivel`, `funcao` proportions, flags); a lighting or camera change should move inputs first and outputs after; `funcao` and `format`/`dominant_channel` are categorical (proportions); latency and status are operational.

## 6. D2: `/metrics` (10 pts)

- Use BentoML's built-in `GET /metrics`. Before the docs, capture a real snapshot and record the **actual** series names and labels (expected family: `bentoml_service_request_total`, `bentoml_service_request_duration_seconds_{bucket,sum,count}`, `bentoml_service_request_in_progress`; confirm).
- Latency buckets: BentoML 1.4 makes the duration histogram configurable in the service decorator, `@bentoml.service(metrics={"duration": {"buckets": [...]}})` (or `min`/`max`/`factor`; confirm exact keys in the metrics docs). Start from `(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)` seconds, **measure** with `load_gen.py` and adjust until no bucket holds more than half the observations. Record the measured p50/p95, the final buckets, and whether the BentoML duration metric and the log `latency_ms` agree (they may measure different spans).
- Custom metrics with `prometheus_client` `Histogram` and `Counter` (what the BentoML docs recommend; use `bentoml.metrics` wrappers only if the docs confirm them for 1.4), registered so they appear on the same `/metrics`, no labels on the histograms: `leiturista_input_luminance_mean` (0 to 255 step 16) and `leiturista_prediction_confidence` (0.1 to 1.0 step 0.1); and `Counter` `leiturista_prediction_funcao_total{funcao}` (3 values only).
- Valid Prometheus text format; verify with `prometheus_client.parser.text_string_to_metric_families` in a test.

### 6.1 Explanatory text and spreadsheet (Portuguese)

`docs/<date>_monitoramento_metrics_analysis.md` explains, with a worked example from a real `/metrics` snapshot:
1. What each series means (counter, duration histogram `_bucket/_sum/_count`, in-progress gauge, custom metrics). Cumulative counters and `le` buckets.
2. **Latency histogram:** de-accumulate buckets by differencing; plot; mean = `_sum/_count`.
3. **Latency boxplot from buckets:** approximation only. Quantiles by linear interpolation (`histogram_quantile` logic) for Q1, median, Q3, whiskers as min/max non-empty bucket bounds; exact quartiles need `latency_ms` from the log.
4. **Error rate:** `sum(<request counter>{http_response_code=~"5.."}) / sum(<request counter>)` with real names, cumulative and windowed (`increase()`/`rate()`). 4xx tracked separately.
5. **Retention, precisely:** metrics live in the inference process memory, reset on restart; the app persists nothing; Prometheus TSDB keeps 15 days (`--storage.tsdb.retention.time=15d`, set explicitly); the file log lasts until rotation drops the 5th backup (at most 300 MB).

Spreadsheet (`scripts/make_metrics_xlsx.py`, openpyxl, **live formulas**): `raw` (pasted `le`/cumulative), `histogram` (per-bucket by formula), `quantiles` (Q1/median/Q3 interpolation formulas), `errors`. Sample data from a real snapshot.

## 7. Extras

### E1: Loki (5 pts)
In `docker/monitoring/compose.yml`: `loki` and **Grafana Alloy** (its predecessor Promtail reached end of life on 2026-03-02 and is unmaintained; do not use Promtail). Alloy tails `/serving-logs/inference.log*`, active and rotated files (read-only mount), parses each line as JSON, and pushes to Loki with labels only `job` (static `leiturista`) and `event` (never `status_code`, `request_id`, features, histograms or latency as labels). Config in `monitoring/alloy/config.alloy` (components `local.file_match`, `loki.source.file`, `loki.process` with `stage.json`/`stage.labels`, `loki.write`). README explains in two sentences what Alloy does: Loki stores logs but does not read files; a collector reads the file and sends the lines to Loki. README gives a LogQL example (`{job="leiturista", event="response"} | json | status_code >= 500`). **Duplicate check (decided: tail rotated files too):** after forcing a rotation, the number of lines in Loki must equal the number of lines across `inference.log*` on disk (query with `count_over_time`); report both numbers, and if duplicates appear, say so and fix the Alloy config (position tracking) rather than dropping the rotated files.

### E2: Variables endpoint (20 pts), in **both** services (decided)

One owner for parsing: `src/leiturista/monitoring/log_reader.py` (reads the current and rotated files, joins request and response by `request_id`). It uses only the standard library (inference must not need pandas). For the 15 s gauge refresh it must be **incremental**: keep the byte offset per file (detect rotation by inode or size shrink), parse only new bytes, and keep the sliding window in a bounded in-memory deque; never re-parse up to 300 MB of logs on every refresh.

- **Inference service (port 3000):** adds `GET /data/variables` only, reading its own `serving-logs/` through `log_reader`. This is the endpoint a grader finds when only the main server is running.
- **`monitor` service (`leiturista.monitor_service:MonitorService`, host port 3002 to avoid clashing with inference on 3000 and Grafana on 3001):** reads `/serving-logs` and `artifacts/` read-only through the same `log_reader` and offers all `/data/*` routes below.

Both services use `@bentoml.api(route=...)`. These routes should answer GET; if the pinned BentoML cannot serve GET through `@bentoml.api`, mount a minimal ASGI app (`@bentoml.asgi_app`) for the data routes only, and record that in the README. Routes:
- `/data/variables?since=&until=&limit=<default 10000, max 100000>&format=json|csv&include_hist=true|false` (default false), same behavior in both services: one row per `request_id`, joined across the current and rotated files. Scalar `in_*` and `out_*`, `latency_ms`, `status_code`; with `include_hist=true`, JSON adds the 4 arrays and CSV adds wide columns `in_hist_r_000..255` etc. Rows without a request event are included with null inputs.
- (`monitor` only) `/data/reference` and `/data/reference/hist`: the reference files, same column names (no `ts`/`request_id`).
- (`monitor` only) `/data/summary?since=&until=`: what the dashboards need and `/data/variables` does not give: per channel L/R/G/B the aggregated 256-bin histogram of the window and the reference's (equal weight per image, floats summing to the number of images), per numeric variable min, Q1, median, Q3, max and mean (for box plots), and category counts for the categorical variables.
- (`monitor` only) `/data/drift?channel=L|R|G|B|all&since=&until=`: D, p, level per channel from `drift.py`. Without `since`/`until` it uses the same sliding window as the gauges below.
- (`monitor` only) **Drift gauges on its own built-in `/metrics`** (this is how Grafana plots drift over time; Prometheus scrapes it): computed over a sliding window (default last 100 images, env `DRIFT_WINDOW_IMAGES`; recomputed at most every 15 s): `leiturista_drift_ks_d{channel}`, `leiturista_drift_ks_pvalue{channel}`, `leiturista_drift_level{channel}` (-1 inconclusive, 0 green, 1 yellow, 2 red), `leiturista_drift_psi{variable}` for every numeric scalar feature, numeric output **and categorical variable** (`funcao`, `orientation`, `dominant_channel`, `format`, `legivel`), `leiturista_drift_window_images`. Binning rules (all in `drift.py`): numeric scalars use 10 quantile bins taken from the reference; the aggregated pixel histograms use 16 bins (regrouped from 256); categorical variables use their categories.
- The monitor must not write to the log dir.
README (Portuguese) has a 5-line example computing KS and PSI from the data endpoints.

### Grafana (supporting, no points, but the group wants to **see every chart**)
`grafana` on host port 3001, anonymous viewer, provisioned from files only (no manual UI setup). Datasources: Prometheus (inference + monitor metrics), Loki, and the **Infinity** datasource plugin (`yesoreyeram-infinity-datasource`, installed through `GF_INSTALL_PLUGINS`) pointed at the `monitor` data routes for what Prometheus cannot hold (per-image values, 256-bin histograms). Five provisioned dashboards; every chart below is required, titles in Portuguese:

1. **Operação:** requests per second by status code; p50/p95/p99 latency (`histogram_quantile`); latency heatmap from the duration buckets; latency histogram (last window); error rate 4xx and 5xx; requests in progress; stat panels for total requests and current 5xx rate; table of the slowest requests (from `/data/variables` via Infinity, sorted by `latency_ms`).
2. **Entrada (imagem):** (data from `/data/summary` via Infinity) live vs reference 0-255 histograms for L, R, G, B (four panels, bar charts from the aggregated window histogram via Infinity, reference overlaid); time series of the four channel means with the reference mean as a band; histograms of `in_luminance_mean`, `luminance_std` and `sharpness` (native Histogram panel over per-image values from Infinity); **boxplots** of the per-image channel means, `luminance_std`, `sharpness`, `width`, `height`, `file_bytes`; `dominant_channel` pie; `format` bar; input-luminance histogram from the Prometheus custom metric.
3. **Saída (modelo):** `confianca` histogram, boxplot and time series; `funcao` proportions (stacked bars over time and a pie); `legivel`, `has_reading` and `has_serial` rates over time; `reading_digits` histogram; `n_flags` histogram; top flags table; `candidate_sharpness` histogram and boxplot; confidence histogram from the Prometheus custom metric.
4. **Drift:** KS D per channel L/R/G/B over time with threshold lines at 0.05 (yellow) and 0.10 (red); state timeline of `leiturista_drift_level`; stat panels with the current level per channel (green/yellow/red/gray via value mappings); KS p-value per channel; PSI per variable as a bar gauge with thresholds 0.1 and 0.25 (labelled as industry convention); table of all drift numbers; window size stat.
5. **Logs (Loki):** live log stream; event rate by `event`; errors stream (`{job="leiturista", event="response"} | json | status_code >= 500`); log volume over time.

Box plots: Grafana's core panels have no box plot. Before building them, check the installed Grafana version for a supported way (an ECharts or Plotly based panel plugin, installed through `GF_INSTALL_PLUGINS`, fed by Infinity). If none works, build each box plot as a Candlestick panel mapping min/Q1/Q3/max, or as a quartile table plus a bar of the five numbers, and **say so in the README and the panel description**; never silently skip one. Record in the docs which approach worked, with plugin id and version.

### E3: Prometheus (15 pts)
`monitoring/prometheus.yml` scrapes `inference:3000/metrics` and `monitor:3000/metrics` every 5 s over `luminus-net`. Service `prometheus` on 9090 with `--storage.tsdb.retention.time=15d`. README lists 3 example queries (p95 latency via `histogram_quantile`, 5xx rate, requests per second).

## 8. Load generator (`scripts/load_gen.py`, a client, run from the host)

Arguments: `--url` (default `http://localhost:3000`), `--n` (default 200), `--rps` (default 2; the pipeline is slow, about 10 s per photo on the host (MPS) and 10 to 40 s in Docker (CPU); measure before raising), `--traffic {holdout,henrik,lcd}` (default `holdout`, the image sets of Decision 7; the holdout split comes from the shared `split_reference_images`), `--drift {none,mild,strong}` (default `none`, synthetic transform applied with Pillow before upload), `--invalid-pct` (default 5), `--seed`. `httpx` client. `mild` = brightness x0.75 and +15 red cast; `strong` = brightness x0.4, blue cast (R x0.6, B x1.3) and Gaussian noise. Invalid requests: non-image bytes, empty body, missing `image` field, truncated JPEG. Final summary: counts by status code, client-side mean and p95 latency. Optional `--slow-pct` adds header `X-Debug-Sleep-Ms`, honored only when env `ALLOW_DEBUG_SLEEP=1` (off by default, never set in the compose files), to give the latency histogram a tail.

## 9. Drift analysis (recommended, `scripts/analyze_log.py`, monitoring environment)

Reads the log through the `monitor` data routes or directly from `serving-logs/`; produces `reports/`. All math and thresholds in `drift.py`.

**Per-channel KS on aggregated histograms (course method; copy its semantics from `003_ks_drift_alert_streamlit/ks_compare.py` and `alert_rule.py` in the course repo):**
- For each channel L, R, G, B: aggregate the `in_hist_*` arrays over the window with **equal weight per image** (see below), build the CDF, compare with the CDF of `reference_hist.csv` (aggregated the same way), `D = max |F_ref - F_window|`, D in [0, 1].
- **Equal weight per image (changed 2026-11-02; deviation from the course, which sums raw pixel counts):** each image's 256-bin histogram is normalized to sum 1 and then the normalized histograms are summed, so the total of an aggregate equals its number of images. Reason: with raw pixel counts a 4608-pixel-wide photo outweighs a 450-pixel one by two orders of magnitude, so a few large photos decide D. Measured offline on the reference set (19 random images against the other 113, no drift, many draws): D >= 0.10 in 30% of draws with pixel weighting, 9% with equal weighting; median D 0.076 to 0.059. The p-value already used n in images, so the statistic now matches it. One owner: `aggregate_histograms` in `drift.py`, called by `build_reference.py` (`reference_hist.csv`, now floats), `analysis.window_histograms` (window, `/data/summary`, the monitor drift gauges, `analyze_log.py`). `scripts/build_reference.py --hist-only` rewrites only `reference_hist.csv` without running MeterOCR. The same aggregates feed the PSI on pixel histograms. Thresholds are unchanged. Results computed before this change are in `docs/2026-11-02_monitoramento_drift_resultados.md` (pixel weighting); the recomputed table is in `docs/2026-11-02_monitoramento_drift_peso_igual.md`.
- p-value: `scipy.stats.kstwobign` (survival function) applied to `sqrt(n_eff) * D` with `n_eff = n_ref * n_win / (n_ref + n_win)`, where `n` counts **images, not pixels** (the course's choice, since pixels of one photo are not independent). No hand-written p-value series. Report D and p together; D is the monitored quantity.
- **Alert levels on D:** `D >= 0.05` YELLOW, `D >= 0.10` RED, else GREEN; INCONCLUSIVE if either set has fewer than 8 images. Constants `D_YELLOW = 0.05`, `D_RED = 0.10`, `MIN_IMAGES = 8` defined once in `drift.py`. One level per channel, plus the worst of the four.

**Other analyses:**
- Histogram PNG (window vs reference) and boxplot PNG for the scalar features and numeric outputs (`confianca`, `reading_digits`, `n_flags`, `candidate_sharpness`); aggregated 0-255 histograms per channel.
- Scalars: `scipy.stats.ks_2samp` (statistic and p, n = images) and PSI (10 quantile bins from the reference, epsilon 1e-6 as in the course). PSI read-out: <0.1 stable, 0.1 to 0.25 moderate, >0.25 significant, labelled as an industry convention, not a proven rule. PSI on aggregated pixel histograms: regroup 256 bins into 16 first.
- Categorical (`dominant_channel`, `format`, `orientation`, `funcao`, `legivel`, `has_reading`): proportion table and PSI over categories.
- **Discrete or degenerate numerics** (`n_flags`, `reading_digits`, `n_boxes`, any variable whose reference has repeated quantile edges): drop duplicate edges; if the variable has 10 or fewer distinct reference values, use one bin per distinct value; if fewer than 2 bins remain (constant reference), report PSI as null with the reason instead of a number.
- **Deviations from the course (the group's decisions, state them in the docs):** the course (`003_ks_drift_alert_streamlit`, `008_psi_drift_alert_streamlit`) alerts on KS at a single `D >= 0.10` with at least 8 images, and on PSI over per-image mean luminance with 10 reference quantile bins, epsilon 1e-6 and bands 0.10 / 0.50. The group keeps KS red at 0.10, adds yellow at 0.05, keeps epsilon 1e-6, uses the conventional PSI bands 0.10 / 0.25, and adds PSI on aggregated pixel histograms. With small samples (33 holdout images against 132 reference images, about 3 per bin) PSI is inflated by chance, as the course's own control batch showed; report it, do not tune thresholds to make a control pass.
- With many variables tested, also print Benjamini-Hochberg adjusted p-values. Always show the effect size (D or PSI) next to any p-value.
- Exit non-zero if a required column is missing.

## 10. Acceptance criteria

Done only when each is verified by running it. Tier tags in brackets (section 1).

1. [P0 a, P1 b] (a) `uv sync --no-dev --extra inference` does not install `mlflow`, `datasets` or `accelerate` (and importing the inference modules in that environment works); `uv sync --no-dev --extra monitor` does not install `torch` or `onnxruntime`. (b) The inference image has no pandas/scipy/mlflow (`docker run ... pip list`); the monitoring images have no torch/onnxruntime. Report the relevant package list lines.
2. [P1] `just up-inference` serves `/predict` on port 3000 from a container with `models/` and `.model_cache/` mounted read-only (confirm writes there fail; confirm the container exits with the `just models` message when the TrOCR weights are missing) and `serving-logs/` read-write. `just up-monitoring` starts independently, in either order; stopping monitoring leaves inference logging; stopping inference does not crash monitoring.
3. [P0] The response-contract test of 5.4 passes on the 5 sample images, and is seen RED when a field is removed on purpose.
4. [P0] After 1 valid + 1 invalid request, `serving-logs/inference.log` has exactly 3 lines (2 if the decorator fallback of 5.3 had to be used), valid JSON, microsecond ISO timestamps; the request line precedes the response line; request and response of the valid call share `request_id`; the four `in_hist_*` arrays each sum to `width * height`.
5. [P0] Hitting `/metrics`, `/livez` and `/readyz` on the inference service adds zero log lines.
6. [P0] The privacy test of 5.4 passes (serial, reading, exception message text, raw client IP absent).
7. [P0] Append-only test: prefix unchanged and size grows after more traffic; rotation test **against a running service** with `SERVING_LOG_MAX_BYTES` tiny yields `inference.log.1` (or the documented fallback of Decision 4).
8. [P0] `/metrics` parses with the `prometheus_client` parser and contains BentoML's request counter and duration buckets (real names recorded in docs), `leiturista_input_luminance_mean_bucket`, `leiturista_prediction_confidence_bucket`, `leiturista_prediction_funcao_total`.
9. [P1] Hitting `/data/variables` adds zero log lines. `/data/variables?format=csv` on both services returns as many rows as distinct `request_id` values in the log; `include_hist=true` adds histogram columns; on `monitor`, `/data/summary` returns the aggregated histograms and quartiles, `/data/drift` returns a level per channel, and `/metrics` exposes the drift gauges including categorical PSI; the incremental reader parses only new bytes (test: append lines and check the offset advanced without re-reading).
10. [P1 a, P2 b] (a) Prometheus targets `inference` and `monitor` are UP; a LogQL query returns lines via Alloy; the rotation duplicate check of E1 is reported. (b) Grafana loads all five provisioned dashboards with data after a `strong` run; a checklist in the docs lists every required chart of section 7 with a screenshot or a note on how it was built. Box plots may use the documented fallback (candlestick or quartile table) if no plugin works; any chart that could not be built is reported with the reason.
11. [P2, needs the analysis script] Using `load_gen.py --n 100`: `--traffic holdout --drift strong` yields RED on channel L and on R or B; `--traffic holdout --drift none` is the control and is expected GREEN, but with only 33 images chance can push D above 0.05, so report the real D and p whatever they are; also report the levels for `--traffic henrik`, `--traffic lcd` and `--drift mild`. Fewer than 8 images yields INCONCLUSIVE. Report real D and PSI numbers, including those that do not match expectations.
12. [P0] `docs/<date>_monitoramento_metrics_analysis.md` and `.xlsx` exist; the xlsx cells hold formulas and one quantile is recomputed independently in Python and matches.
13. [P0] `pytest tests/monitoring` passes. Guard tests (append-only; no-noise endpoints; privacy; response contract; features computed only in `features.py`; thresholds only in `drift.py`; reference split only in `split.py`) are seen RED before accepted: write them before the implementation or prove with `git stash`, and report how many cases failed.
14. [P0 local, P1 Docker] README (Portuguese) gets a "Monitoramento" section with two paths: local (`just serve` + `just load`, no Docker) and Docker (`just up-inference`, `just up-monitoring`, `just load`), at most 8 commands from clone to seeing the log grow, plus the architecture diagram of section 2 and the three-environment table.

## 11. Facts the agent must measure and write down (do not assume)

1. HTTP status for invalid input (non-image bytes, missing field): send each invalid case and record the real code; 4xx = client error, 5xx = server error.
2. Real `/metrics` series names and labels, from a live snapshot; use only those in docs, PromQL and the dashboard.
3. Latency distribution: run `load_gen.py` and read `latency_ms` from the log, then set the buckets (section 6). Record p50/p95 and the first-call cold-start time separately (models load lazily on the first predict; exclude it from the histogram analysis or call it out).
4. Bytes per request/response pair in the real log.
5. The exact number of files in each image set of Decision 7 (expected: 132 reference, 33 holdout, 168 henrik, 50 lcd) and whether `just models` leaves `.model_cache/trocr-small-printed/model.safetensors` in place.
6. Whether the BentoML 1.4 duration metric and the log `latency_ms` agree (section 6), and whether the pinned BentoML serves GET through `@bentoml.api`, supports the middleware hook of 5.3 and a `bentoml.Context` parameter, and lets `prometheus_client` metrics appear on its `/metrics` (single worker).
7. Whether `RotatingFileHandler` created lazily in the worker really rotates in a served process (Decision 4).

## 12. Rules for the agent

- Create branch `feat/mlops-monitoring` from `main` (it does not exist yet); commit per step of section 13; no attribution lines in commit messages. Do not push.
- Fix causes, not symptoms. No suppressed warnings, no `# noqa`, no broad `except` hiding errors.
- Type hints on all new code; pyright or mypy clean, no gratuitous `Any`.
- Reuse libraries (`prometheus_client`, BentoML's own metrics support, Pillow, numpy, scipy, pandas); do not hand-roll the KS p-value or metrics exposition (D itself is a numpy one-liner over CDFs).
- Never run `leiturista train`. Never `rm`; move to trash. Kill every background process you start before finishing (check `ps aux`). Wrap long jobs in `caffeinate -i`. Run downloads in the background without timeout per AGENTS.md.
- Do not spawn sub-agents.
- Report negative or inconvenient results exactly as measured; never invent or estimate a number in docs.
- Do not modify existing behavior outside the files named in section 4 plus `pyproject.toml`, `uv.lock`, `justfile`, `.gitignore`, `README.md`.

## 13. Order of work

Follow the tiers. P0: 1. Capture the pre-change response contract and write guard tests for logging, privacy and contract (RED). 2. Dependency split in `pyproject.toml`/`uv.lock`. 3. `features.py`, `outputs.py`, `drift.py`, `split.py`, `log_reader.py`. 4. `logging_setup.py` + middleware + service wrapper. 5. `metrics.py` and the bucket check. 6. `build_reference.py` and `load_gen.py`. 7. Docs + xlsx from a real snapshot, README local path. Then check every P0 criterion and report. P1: 8. `/data/variables` on inference and `monitor_service.py`. 9. `docker/inference`, `docker/monitoring` with Prometheus, Loki, Alloy. 10. README Docker path. Check P1 criteria and report. P2: 11. Grafana provisioning and the five dashboards. 12. `analyze_log.py`. 13. Chart checklist. Check P2 criteria and report evidence for everything.
