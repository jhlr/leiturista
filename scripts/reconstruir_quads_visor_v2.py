"""Reconstrói os quads do visor p/ data/neoenergia_amr_v2.

Roda _reading_candidate nas fotos aceitas e guarda o quad exato usado para gerar o crop.
Útil pra re-treinar o det visor-ft com dados do novo dataset.

Uso: .venv\\Scripts\\python.exe scripts\\reconstruir_quads_visor_v2.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from leiturista.distribuidora import _reading_candidate
from leiturista.inference import MeterOCR

MAN = Path("data/neoenergia_amr_v2/manifest.csv")
OUT_DIR = Path("data/neoenergia_amr_v2")
FOTOS = Path("data/FotosNeoenergia")
MAX_DIFF = 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default=None, help="train|valid|test ou vazio p/ todos aceitos")
    a = ap.parse_args()

    man = pd.read_csv(MAN, dtype=str, keep_default_na=False)
    man = man[man["status"] == "aceito"].reset_index(drop=True)
    if a.split:
        man = man[man["split"] == a.split].reset_index(drop=True)
    if a.limit:
        man = man.head(a.limit)

    print(f"Reconstruindo quads para {len(man)} aceitos (split={a.split or 'todos'})", flush=True)
    ocr = MeterOCR()
    ok: list[dict] = []
    bad: list[dict] = []

    for i, r in enumerate(man.itertuples()):
        foto = FOTOS / r.lote / r.foto
        img = np.array(Image.open(foto).convert("RGB"))[:, :, ::-1]
        cand = _reading_candidate(ocr, img, r.leitura_original)
        motivo = ""
        if cand is None:
            motivo = "sem_candidato"
        elif cand.source != r.source:
            motivo = f"source {cand.source}!={r.source}"
        else:
            saved = cv2.imread(str(OUT_DIR / r.image))
            if saved is None or saved.shape != cand.crop.shape:
                motivo = "shape_diferente"
            elif float(np.abs(saved.astype(np.int16) - cand.crop).mean()) > MAX_DIFF:
                motivo = "crop_diferente"
        if motivo:
            bad.append({"image": r.image, "foto": r.foto, "lote": r.lote, "split": r.split, "motivo": motivo})
            continue
        h, w = img.shape[:2]
        row = {"image": r.image, "split": r.split, "lote": r.lote, "foto": r.foto,
               "source": cand.source, "img_w": w, "img_h": h}
        for j, (x, y) in enumerate(cand.quad.tolist(), 1):
            row[f"x{j}"], row[f"y{j}"] = float(x), float(y)
        ok.append(row)
        if i % 100 == 0:
            print(f"  {i}/{len(man)} ok={len(ok)} bad={len(bad)}", flush=True)

    pd.DataFrame(ok).to_csv(OUT_DIR / "boxes_visor.csv", index=False)
    pd.DataFrame(bad).to_csv(OUT_DIR / "boxes_visor_divergentes.csv", index=False)
    print(f"RECONSTRUIDOS: {len(ok)} | DIVERGENTES: {len(bad)} de {len(man)}", flush=True)
    if bad:
        print(pd.DataFrame(bad)["motivo"].value_counts().to_string())


if __name__ == "__main__":
    main()
