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
    uv run python scripts/load_gen.py {{ARGS}}
