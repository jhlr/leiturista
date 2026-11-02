"""Gera o conjunto de referencia do monitoramento (SDD Decisao 7): `artifacts/reference.csv` e
`artifacts/reference_hist.csv`.

Roda `features.py`, `outputs.py` e o `MeterOCR` real sobre as 132 fotos de referencia (divisao
deterministica de `leiturista.monitoring.split`). E um job em lote do ambiente de INFERENCIA:

  just build-reference
  # ou: uv run python scripts/build_reference.py [--images-dir DIR] [--seed N]

reference.csv: uma linha por imagem, mesmas colunas escalares do log sem prefixo, sem nome de
arquivo. reference_hist.csv: histogramas 0-255 por canal somados sobre o conjunto.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from PIL import Image

from leiturista.inference import MeterOCR
from leiturista.monitoring.features import CATEGORICAL_FEATURES, CHANNELS, HIST_BINS, SCALAR_FEATURES, extract_features, uploaded_size
from leiturista.monitoring.outputs import SCALAR_OUTPUT_COLUMNS, derive_outputs
from leiturista.monitoring.split import DEFAULT_SEED, PRAEKELT_DIR, split_reference_images
from leiturista.paths import ROOT

REFERENCE_COLUMNS = (*SCALAR_FEATURES, *CATEGORICAL_FEATURES, *SCALAR_OUTPUT_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", type=Path, default=PRAEKELT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()

    reference, holdout = split_reference_images(args.images_dir, args.seed)
    print(f"referencia: {len(reference)} imagens (holdout reservado: {len(holdout)}), seed={args.seed}")
    ocr = MeterOCR()
    rows: list[dict[str, object]] = []
    sums = {channel: [0] * HIST_BINS for channel in CHANNELS}
    started = time.perf_counter()
    for index, path in enumerate(reference, start=1):
        with Image.open(path) as image:
            size = uploaded_size(image)
            features = extract_features(image, size)
            outputs = derive_outputs(ocr.predict_image(image))
        rows.append({**features.scalars(), **outputs.scalars()})
        for channel in CHANNELS:
            hist = getattr(features, f"hist_{channel}")
            sums[channel] = [a + b for a, b in zip(sums[channel], hist)]
        if index % 20 == 0 or index == len(reference):
            print(f"  {index}/{len(reference)}  ({time.perf_counter() - started:.0f}s)", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "reference.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REFERENCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with (args.out_dir / "reference_hist.csv").open("w", newline="", encoding="utf-8") as handle:
        hist_writer = csv.writer(handle)
        hist_writer.writerow(["bin", *CHANNELS])
        for bin_index in range(HIST_BINS):
            hist_writer.writerow([bin_index, *(sums[channel][bin_index] for channel in CHANNELS)])
    print(f"gravado: {args.out_dir / 'reference.csv'} ({len(rows)} linhas) e reference_hist.csv")


if __name__ == "__main__":
    main()
