# Leiturista — comandos de setup/serve/teste do serviço BentoML.
# Ver README.md para o passo a passo comentado.

RELEASE_URL := "https://github.com/jhlr/leiturista/releases/download/modelos-1.0/leiturista-models.tar.gz"

# instala as dependências de todos os ambientes (uv.lock; --all-extras = inference + train + monitor)
setup:
    uv sync --all-extras

# baixa e extrai os pesos dos modelos (det/rec ONNX + TrOCR off-the-shelf), se ainda não existirem
models:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -d models/pp_ocr_v5_mobile_det_onnx ]; then
        echo "models/ já existe — pulando download."
    else
        curl -L {{RELEASE_URL}} -o leiturista-models.tar.gz
        tar -xzf leiturista-models.tar.gz
        echo "modelos extraídos em models/ e .model_cache/"
    fi

# regenera as imagens sintéticas de exemplo em samples/
samples:
    uv run python scripts/gen_sample_images.py

# sobe o serviço BentoML em http://localhost:3000
serve:
    uv run bentoml serve leiturista.service:LeituristaService --port 3000

# chamada de exemplo contra o serviço já no ar (roda `just serve` em outro terminal antes)
predict:
    curl -s -F image=@samples/exemplo_01.png http://localhost:3000/predict | python3 -m json.tool

# setup + download de modelos + geração dos exemplos, um comando só
all: setup models samples

# gera o conjunto de referencia do monitoramento (artifacts/reference*.csv) com o MeterOCR real
build-reference:
    uv run python scripts/build_reference.py

# gera carga contra o serviço no ar (ex.: just load --n 100 --drift strong --traffic holdout)
load *ARGS:
    uv run --extra monitor python scripts/load_gen.py {{ARGS}}

# analise de drift sobre o log (KS por canal, PSI, boxplots) em reports/<label>/ (ex.: just analyze --last 20 --label strong)
analyze *ARGS:
    uv run --extra monitor python scripts/analyze_log.py {{ARGS}}

# --- Docker: ambientes de inferencia e monitoramento separados (docs/2026-11-01_sdd_monitoramento_mlops.md, secao 2) ---

# cria a rede compartilhada se ainda nao existir (os dois compose a declaram `external`)
_luminus-net:
    docker network inspect luminus-net >/dev/null 2>&1 || docker network create luminus-net

# sobe a INFERENCIA em http://localhost:3000 (models/ e .model_cache/ somente leitura; log em ./serving-logs)
up-inference: _luminus-net
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p serving-logs   # criado pelo usuario do host (senao o Docker cria como root)
    export UID GID="$(id -g)" SERVICE_VERSION="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    docker compose -f docker/inference/compose.yml up -d --build --wait --wait-timeout 300

down-inference:
    docker compose -f docker/inference/compose.yml down

# sobe o MONITORAMENTO: Prometheus :9090, Loki :3100, Alloy :12345, Grafana :3001, monitor :3002 (independe da inferencia)
up-monitoring: _luminus-net
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p serving-logs
    export UID GID="$(id -g)" SERVICE_VERSION="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    docker compose -f docker/monitoring/compose.yml up -d --build --wait --wait-timeout 300

down-monitoring:
    docker compose -f docker/monitoring/compose.yml down

# o mesmo job de `build-reference`, dentro da imagem de inferencia (data/ somente leitura, artifacts/ gravavel)
build-reference-docker: _luminus-net
    #!/usr/bin/env bash
    set -euo pipefail
    export UID GID="$(id -g)" SERVICE_VERSION="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    docker compose -f docker/inference/compose.yml --profile tools run --rm --build build-reference
