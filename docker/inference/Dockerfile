# Ambiente de INFERENCIA (SDD secao 2): leiturista + BentoML + onnxruntime + torch/transformers (TrOCR).
# Sem MLflow, datasets, pandas, scipy, pesos ou dados: models/ e .model_cache/ entram por montagem somente leitura.
# Contexto de build = raiz do repositorio:  docker build -f docker/inference/Dockerfile .
FROM python:3.13.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy UV_NO_CACHE=1 PATH=/opt/venv/bin:$PATH

# 1) Dependencias exatas do uv.lock (extra `inference`). O lock resolve torch com CUDA no Linux; aqui o
#    torch/torchvision (mesmas versoes do lock) vem do indice CPU-only da PyTorch e os pacotes nvidia/triton ficam de fora
#    (menos `nvidia-ml-py`, o `pynvml` que o BentoML importa ao medir os recursos da maquina).
COPY pyproject.toml uv.lock ./
RUN uv venv --seed /opt/venv \
 && uv export --frozen --no-dev --extra inference --no-emit-project --no-hashes --no-annotate -o /tmp/requirements.txt \
 && grep -E '^(torch|torchvision)==' /tmp/requirements.txt | sed 's/ ;.*//' > /tmp/torch.txt \
 && grep -v -E '^(torch|torchvision|triton|nvidia-|cuda-)' /tmp/requirements.txt > /tmp/rest.txt \
 && grep -E '^nvidia-ml-py==' /tmp/requirements.txt >> /tmp/rest.txt \
 && uv pip install --python /opt/venv/bin/python --no-deps -r /tmp/rest.txt \
 && uv pip install --python /opt/venv/bin/python --no-deps -r /tmp/torch.txt --index-url https://download.pytorch.org/whl/cpu \
 && rm -f /tmp/requirements.txt /tmp/rest.txt /tmp/torch.txt

# 2) A biblioteca, editavel em /app: `leiturista.paths.ROOT` = /app, onde models/ e .model_cache/ sao montados.
COPY src ./src
COPY scripts ./scripts
RUN uv pip install --python /opt/venv/bin/python --no-deps -e .

# Usuario arbitrario (`user: UID:GID` no compose): precisa de HOME e caches graváveis.
ENV HOME=/tmp BENTOML_HOME=/tmp/bentoml XDG_CACHE_HOME=/tmp/cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    SERVING_LOG_DIR=/serving-logs PYTHONUNBUFFERED=1
ARG SERVICE_VERSION=unknown
ENV SERVICE_VERSION=${SERVICE_VERSION}

COPY --chmod=755 docker/inference/entrypoint.sh /usr/local/bin/leiturista-entrypoint

EXPOSE 3000
ENTRYPOINT ["leiturista-entrypoint"]
CMD ["bentoml", "serve", "leiturista.service:LeituristaService", "--host", "0.0.0.0", "--port", "3000"]
