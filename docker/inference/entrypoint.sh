#!/bin/sh
# Falha rapido, com a mensagem de `just models`, se os pesos do TrOCR nao estao montados.
# `bentoml serve` nao encerra o container quando o worker aborta na partida, entao a checagem roda aqui, antes.
set -e
python -c "from leiturista.service import check_model_weights; check_model_weights()"
exec "$@"
