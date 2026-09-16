"""Serviço BentoML — triagem de fotos de medidor de energia elétrica.

Envelopa `MeterOCR.predict_image` (det+rec ONNX + TrOCR, ver `inference.py`) num
único endpoint HTTP. Não há lógica de OCR nova aqui: o serviço só traduz a
`Prediction` do pipeline já existente para o contrato de resposta pedido pela
triagem (número do medidor, função, consumo, confiança).

Rodar:
  bentoml serve leiturista.service:LeituristaService --port 3000
  # ou: just serve
"""

from __future__ import annotations

import bentoml
from PIL import Image as pil

from .inference import MeterOCR

CONFIDENCE_OK = 0.5  # abaixo disso, a leitura existe mas é tratada como incerta


@bentoml.service(name="leiturista")
class LeituristaService:
    def __init__(self) -> None:
        self.ocr = MeterOCR()  # modelos carregam lazy, no primeiro predict

    @bentoml.api
    def predict(self, image: pil.Image) -> dict:
        """Recebe a foto do medidor e devolve a extração de leitura."""
        pred = self.ocr.predict_image(image)

        leitura_boxes = [b for b in pred.boxes if b.field == "leitura"]
        confianca = max((b.conf for b in leitura_boxes), default=0.0) if pred.reading else 0.0

        if pred.reading is None:
            funcao = "sem_leitura_detectada"
        elif pred.legible and confianca >= CONFIDENCE_OK:
            funcao = "leitura_normal"
        else:
            funcao = "leitura_incerta"

        return {
            "numero_medidor": pred.serial,
            "funcao": funcao,
            "consumo": pred.reading,
            "confianca": round(float(confianca), 3),
            "legivel": pred.legible,
            "flags": pred.flags,
        }
