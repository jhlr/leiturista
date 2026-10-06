"""Passo 0 do plano do detector de visor: desenha TODAS as caixas do det PP-OCRv5
em fotos inteiras de campo e monta folhas de contato para contar, à mão, em quantas
existe uma caixa sobre o visor (recall dos candidatos).

Uso: .venv/bin/python scripts/visor_recall_candidatos.py [--n 50] [--per-sheet 10] --out DIR
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from leiturista.inference import MeterOCR

FOTOS = Path("data/distribuidora_campo")
THUMB = 480


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--per-sheet", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    paths = sorted(FOTOS.rglob("*.jpg"))
    random.Random(a.seed).shuffle(paths)
    ocr = MeterOCR()
    tiles: list[np.ndarray] = []
    index: list[str] = []
    for p in paths[: a.n]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        quads = ocr._det_boxes(img)
        for q, conf in quads:
            cv2.polylines(img, [q.astype(np.int32)], True, (0, 255, 0), max(2, img.shape[1] // 300))
        h, w = img.shape[:2]
        s = THUMB / max(h, w)
        t = cv2.resize(img, (int(w * s), int(h * s)))
        canvas = np.full((THUMB + 24, THUMB, 3), 255, np.uint8)
        canvas[24 : 24 + t.shape[0], : t.shape[1]] = t
        i = len(index) + 1
        cv2.putText(canvas, f"#{i} caixas={len(quads)}", (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        tiles.append(canvas)
        index.append(f"{i}\t{p.name}\t{len(quads)}")

    cols = 5
    for k in range(0, len(tiles), a.per_sheet):
        chunk = tiles[k : k + a.per_sheet]
        while len(chunk) % cols:
            chunk.append(np.full_like(tiles[0], 255))
        rows = [np.hstack(chunk[r : r + cols]) for r in range(0, len(chunk), cols)]
        cv2.imwrite(str(a.out / f"folha_{k // a.per_sheet + 1:02d}.jpg"), np.vstack(rows))
    (a.out / "indice.tsv").write_text("\n".join(index) + "\n")


if __name__ == "__main__":
    main()
