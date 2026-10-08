"""L4 do Lab 2: onde está a latência do pipeline (p50/p90 por estágio), em CPU.

Embrulha os métodos do `MeterOCR` com cronômetros (rotação, detecção, reconhecimento PP-OCR, TrOCR, recortes,
merge) e roda `predict_image` em `n` fotos estratificadas pelos 4 lotes (seed fixa). Tempos por foto, soma por
estágio (um estágio pode rodar várias vezes por foto). Uso:
  LEITURISTA_FOTOS_DIR=data/distribuidora_campo .venv/bin/python scripts/latencia_por_estagio.py -n 60
"""
import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from leiturista.distribuidora import LOTE_SPLIT
from leiturista.inference import MeterOCR

ap = argparse.ArgumentParser()
ap.add_argument("-n", type=int, default=60)
ap.add_argument("--out", default="docs/figs/l4_latencia.json")
a = ap.parse_args()

FOTOS = Path(os.environ.get("LEITURISTA_FOTOS_DIR", "data/distribuidora_campo"))
rng = random.Random(0)
fotos = [p for lote in sorted(LOTE_SPLIT) for p in rng.sample(sorted((FOTOS / lote).glob("*.jpg")), a.n // 4)]
ocr = MeterOCR()
ocr._load()
ocr._load_trocr()  # carrega fora da medição
acc: dict[str, float] = defaultdict(float)
calls: dict[str, int] = defaultdict(int)
for name, label in (("_detect_rotation", "rotacao"), ("_det_boxes", "deteccao"), ("_rec_recognize", "reconhecimento_ppocr"),
                    ("_trocr_recognize", "reconhecimento_trocr"), ("_crop_rotated", "recortes"), ("_merge_quads", "merge")):
    f = getattr(ocr, name)

    def wrap(*args, _f=f, _l=label, **kw):
        t = time.perf_counter()
        r = _f(*args, **kw)
        acc[_l] += time.perf_counter() - t
        calls[_l] += 1
        return r

    setattr(ocr, name, wrap)
per: dict[str, list[float]] = defaultdict(list)
tot: list[float] = []
for i, p in enumerate(fotos, 1):
    before = dict(acc)
    t = time.perf_counter()
    ocr.predict_image(Image.open(p).convert("RGB"))
    tot.append(time.perf_counter() - t)
    for k in ("rotacao", "deteccao", "reconhecimento_ppocr", "reconhecimento_trocr", "recortes", "merge"):
        per[k].append(acc[k] - before.get(k, 0.0))
    if i % 10 == 0:
        print(i, f"{np.mean(tot):.2f}s", flush=True)
q = lambda v, p: float(np.percentile(v, p))  # noqa: E731
res = {"n": len(tot), "total": {"media": float(np.mean(tot)), "p50": q(tot, 50), "p90": q(tot, 90)},
       "estagios": {k: {"media": float(np.mean(v)), "p50": q(v, 50), "p90": q(v, 90), "fracao_do_total": float(np.sum(v) / np.sum(tot))}
                    for k, v in per.items()}}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps(res, indent=2))
