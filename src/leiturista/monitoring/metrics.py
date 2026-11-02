"""Metricas Prometheus proprias do servico (SDD secao 6), expostas no `/metrics` do BentoML.

Usam `prometheus_client` direto (o BentoML 1.4 considera `bentoml.metrics` obsoleto). Os objetos
sao criados de forma preguicosa, dentro do worker: o BentoML define `PROMETHEUS_MULTIPROC_DIR`
DEPOIS de importar o modulo do servico, e um `prometheus_client` importado antes ficaria em modo
processo unico, invisivel para o coletor multiprocesso do `/metrics`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prometheus_client import Counter, Histogram

    from .features import ImageFeatures
    from .outputs import OutputVariables

LUMINANCE_BUCKETS: tuple[float, ...] = tuple(float(v) for v in range(16, 256, 16))  # 16..240 (+Inf implicito)
CONFIDENCE_BUCKETS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
# Buckets do histograma de duracao do BentoML: ajustados com latencias reais (docs/*_metrics_analysis.md).
DURATION_BUCKETS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


@dataclass
class ServingMetrics:
    input_luminance: Histogram
    confidence: Histogram
    funcao: Counter

    def observe(self, features: ImageFeatures, outputs: OutputVariables) -> None:
        self.input_luminance.observe(features.luminance_mean)
        self.confidence.observe(outputs.confianca)
        self.funcao.labels(funcao=outputs.funcao).inc()


_metrics: ServingMetrics | None = None


def get_metrics() -> ServingMetrics:
    """Cria (uma vez por processo) os histogramas e o contador do servico."""
    global _metrics
    if _metrics is None:
        from prometheus_client import Counter, Histogram

        from .outputs import FUNCAO_VALUES

        funcao = Counter("leiturista_prediction_funcao", "Predicoes por funcao do servico", ["funcao"])
        for value in FUNCAO_VALUES:  # as 3 series existem desde o inicio
            funcao.labels(funcao=value)
        _metrics = ServingMetrics(
            input_luminance=Histogram(
                "leiturista_input_luminance_mean",
                "Luminancia media (0-255) da imagem de entrada",
                buckets=LUMINANCE_BUCKETS,
            ),
            confidence=Histogram(
                "leiturista_prediction_confidence", "Confianca (confianca da resposta) das predicoes", buckets=CONFIDENCE_BUCKETS
            ),
            funcao=funcao,
        )
    return _metrics
