"""Reconstrói os quads do visor (positivos do detector de visor) a partir do manifest.

`build_dataset` só guardou o recorte, não o quad. O seletor `_reading_candidate` é
determinístico, então reexecutá-lo na foto original com o mesmo rótulo devolve o mesmo
quad. A reconstrução é validada contra o PNG salvo (fonte normal/invertida + diferença
média do recorte); linhas que não batem vão para `boxes_visor_divergentes.csv`.

Uso: .venv/bin/python scripts/reconstruir_quads_visor.py [--limit N]
Saída (data/distribuidora_amr/, gitignored): boxes_visor.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from leiturista.distribuidora import FOTOS_DIR, _reading_candidate
from leiturista.inference import MeterOCR

DATA = Path("data/distribuidora_amr")
MAX_DIFF = 2.0  # diferença média absoluta (0-255) tolerada entre recorte refeito e salvo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    man = pd.read_csv(DATA / "manifest.csv", dtype=str, keep_default_na=False)
    man = man[man["status"] == "aceito"].reset_index(drop=True)
    if a.limit:
        man = man.head(a.limit)

    ocr = MeterOCR()
    ok: list[dict] = []
    bad: list[dict] = []
    for _, r in man.iterrows():
        foto = FOTOS_DIR / r["lote"] / r["foto"]
        img = np.array(Image.open(foto).convert("RGB"))[:, :, ::-1]
        cand = _reading_candidate(ocr, img, r["leitura_original"])
        motivo = ""
        if cand is None:
            motivo = "sem_candidato"
        elif cand.source != r["source"]:
            motivo = f"source {cand.source}!={r['source']}"
        else:
            saved = cv2.imread(str(DATA / r["image"]))
            if saved is None or saved.shape != cand.crop.shape:
                motivo = "shape_diferente"
            elif float(np.abs(saved.astype(np.int16) - cand.crop).mean()) > MAX_DIFF:
                motivo = "crop_diferente"
        if motivo:
            bad.append({"image": r["image"], "foto": r["foto"], "motivo": motivo})
            continue
        h, w = img.shape[:2]
        row = {"image": r["image"], "split": r["split"], "lote": r["lote"], "foto": r["foto"],
               "source": cand.source, "img_w": w, "img_h": h}
        for i, (x, y) in enumerate(cand.quad.tolist(), 1):
            row[f"x{i}"], row[f"y{i}"] = x, y
        ok.append(row)

    pd.DataFrame(ok).to_csv(DATA / "boxes_visor.csv", index=False)
    pd.DataFrame(bad).to_csv(DATA / "boxes_visor_divergentes.csv", index=False)
    print(f"reconstruidos: {len(ok)} | divergentes: {len(bad)} de {len(man)}")
    if bad:
        print(pd.DataFrame(bad)["motivo"].value_counts().to_string())


if __name__ == "__main__":
    main()
