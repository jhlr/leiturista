"""Gera o recorte sugerido do display das fotos do plano de rotulagem (auxílio ao rotulador).
Usa o pipeline atual (dois detectores + classificador de visor). Uso:
  LEITURISTA_FOTOS_DIR=data/distribuidora_campo LEITURISTA_DET2_ONNX=<det visor-ft> LEITURISTA_VISOR_CLS=models/visor_cls.pt \\
  .venv/bin/python scripts/preparar_rotulagem.py
Saída (gitignored, dado do cliente): data/rotulos_crops/<foto>.png"""
import csv
from pathlib import Path

from PIL import Image

from leiturista.inference import MeterOCR
from leiturista.rotulos import PLANO

FOTOS = Path(__import__("os").environ.get("LEITURISTA_FOTOS_DIR", "data/distribuidora_campo"))
OUT = Path("data/rotulos_crops")
OUT.mkdir(parents=True, exist_ok=True)
ocr = MeterOCR()
for i, r in enumerate(csv.DictReader(open(PLANO, encoding="utf-8")), 1):
    dst = OUT / f"{Path(r['nome_arquivo']).stem}.png"
    if dst.exists():
        continue
    pred = ocr.predict_image(Image.open(FOTOS / r["lote"] / r["nome_arquivo"]).convert("RGB"))
    box = ocr._best_reading(pred.boxes)
    if box is not None:
        box.crop.save(dst)
    if i % 25 == 0:
        print(i, flush=True)
