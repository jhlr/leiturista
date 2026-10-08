"""Constrói o dataset do classificador de visor: todos os candidatos (caixas do det, normal+invertida)
das fotos com quad de visor conhecido, rotulados por IoU com o quad (>=0,4 positivo; <0,1 negativo;
o resto descartado). Guarda também texto/campo do rec (PP-OCRv6) para re-executar a regra antiga
("mais dígitos") offline na avaliação.

Uso: LEITURISTA_FOTOS_DIR=data/distribuidora_campo LEITURISTA_DET2_ONNX=<det visor-ft> \\
     .venv/bin/python scripts/construir_candidatos_visor.py     # retomável
Saída (gitignored): data/visor_cls/{crops/*.png, candidatos.csv}
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from leiturista.distribuidora import FOTOS_DIR
from leiturista.inference import MeterOCR

OUT = Path("data/visor_cls")
FIELDS = ["cand", "split", "lote", "foto", "frame", "label", "iou", "text", "field", "conf", "w", "h"]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    union = cv2.contourArea(a.astype(np.float32)) + cv2.contourArea(b.astype(np.float32)) - inter
    return float(inter / union) if union > 0 else 0.0


def main() -> None:
    boxes = pd.read_csv("data/distribuidora_amr/boxes_visor.csv")
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "candidatos.csv"
    done: set[tuple[str, str]] = set()
    if csv_path.exists():
        done = {(r["lote"], r["foto"]) for r in csv.DictReader(open(csv_path, encoding="utf-8"))}
    ocr = MeterOCR()
    new = not csv_path.exists()
    n = 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for _, r in boxes.iterrows():
            if (r.lote, r.foto) in done:
                continue
            img = cv2.imread(str(FOTOS_DIR / r.lote / r.foto))
            gt = np.array([[r.x1, r.y1], [r.x2, r.y2], [r.x3, r.y3], [r.x4, r.y4]], np.float32)
            for frame, merge in (("normal", True), ("inverted", False)):
                base = img if frame == "normal" else 255 - img
                quads = ocr._det_boxes(base)
                for quad, conf in (ocr._merge_quads(quads) if merge else quads):
                    crop = ocr._crop_rotated(base, quad)
                    if crop.size == 0 or min(crop.shape[:2]) < 8:
                        continue
                    v = iou(quad, gt)
                    if 0.1 <= v < 0.4:
                        continue
                    text = ocr._rec_recognize(crop)
                    cid = f"{r.image[:-4]}_{frame[0]}{n:07d}"
                    cv2.imwrite(str(OUT / "crops" / f"{cid}.png"), crop)
                    w.writerow({"cand": cid, "split": r.split, "lote": r.lote, "foto": r.foto, "frame": frame,
                                "label": int(v >= 0.4), "iou": round(v, 3), "text": text,
                                "field": ocr._classify(text), "conf": round(conf, 3),
                                "w": crop.shape[1], "h": crop.shape[0]})
                    n += 1
            f.flush()
            if n and (n % 500) < 60:
                print(n, "candidatos", flush=True)


if __name__ == "__main__":
    main()
