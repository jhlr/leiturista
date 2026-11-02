"""Serviço BentoML — triagem de fotos de medidor de energia elétrica.

Envelopa `MeterOCR.predict_image` (det+rec ONNX + TrOCR, ver `inference.py`) num
único endpoint HTTP. Não há lógica de OCR nova aqui: o serviço só traduz a
`Prediction` do pipeline já existente para o contrato de resposta pedido pela
triagem (número do medidor, função, consumo, confiança).

Monitoramento (`leiturista.monitoring`): um middleware ASGI grava no log JSON Lines uma linha
`request` e uma `response` por chamada de `/predict` (variáveis derivadas da imagem e da
saída, nunca pixels, serial ou valor da leitura). Um único worker escreve o arquivo.

Rodar:
  bentoml serve leiturista.service:LeituristaService --port 3000
  # ou: just serve
"""

from __future__ import annotations

import os
import time

import bentoml
from PIL import Image as pil

from .inference import TROCR_CACHE, MeterOCR
from .monitoring.features import extract_features, uploaded_size
from .monitoring.logging_setup import get_serving_logger
from .monitoring.metrics import DURATION_BUCKETS, get_metrics
from .monitoring.outputs import CONFIDENCE_OK, derive_outputs
from .monitoring.tracing import RequestTrace, TraceMiddleware

__all__ = ["CONFIDENCE_OK", "LeituristaService"]

DEBUG_SLEEP_HEADER = "X-Debug-Sleep-Ms"


def check_model_weights() -> None:
    """Sai com mensagem clara se os pesos do TrOCR não estão montados (a inferência não tem MLflow)."""
    if not (TROCR_CACHE / "model.safetensors").exists():
        raise SystemExit(f"Pesos do TrOCR ausentes em {TROCR_CACHE}/model.safetensors. Rode `just models` e suba o serviço de novo.")


class _Leiturista:
    def __init__(self) -> None:
        check_model_weights()
        self.ocr = MeterOCR()  # modelos carregam lazy, no primeiro predict
        get_serving_logger()  # cria o handler do log AQUI, dentro do worker (nunca no import)
        self.metrics = get_metrics()  # idem: prometheus_client só depois do PROMETHEUS_MULTIPROC_DIR

    @bentoml.api
    def predict(self, image: pil.Image, ctx: bentoml.Context) -> dict:
        """Recebe a foto do medidor e devolve a extração de leitura."""
        trace: RequestTrace = ctx.request.state.trace
        with trace.stage("invalid_image"):
            features = extract_features(image, uploaded_size(image))  # decodifica a imagem
        trace.log_request(features)

        if os.environ.get("ALLOW_DEBUG_SLEEP") == "1":  # só p/ dar cauda ao histograma de latência
            sleep_ms = ctx.request.headers.get(DEBUG_SLEEP_HEADER, "")
            if sleep_ms.isdigit():
                time.sleep(int(sleep_ms) / 1000)

        with trace.stage("inference_failed"):
            pred = self.ocr.predict_image(image)
        outputs = derive_outputs(pred)
        trace.set_outputs(outputs)
        self.metrics.observe(features, outputs)

        return {
            "numero_medidor": pred.serial,
            "funcao": outputs.funcao,
            "consumo": pred.reading,
            "confianca": outputs.confianca,
            "legivel": pred.legible,
            "flags": pred.flags,
        }


# Forma funcional do decorador (equivale a `@bentoml.service` sobre a classe) para o tipo `Service` ficar visível.
LeituristaService = bentoml.service(name="leiturista", workers=1, metrics={"duration": {"buckets": list(DURATION_BUCKETS)}})(_Leiturista)
LeituristaService.add_asgi_middleware(TraceMiddleware, endpoints=("/predict",))
