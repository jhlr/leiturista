"""Valida se a nitidez (variância do Laplaciano) prediz acerto/erro do OCR — usando o
UFPR-AMR (gabarito real), não a base de campo (que não tem rótulo de aceite/rejeição).

Motivação: o pipeline `leiturista` expõe `Prediction.sharpness` como sinal de coerência
(ver docs/sbti_artigo/artigo_sbti2026.docx §3.1/4.3), mas nunca foi validado contra um
gabarito de verdade se esse sinal realmente prediz falha de leitura. Aqui usamos o
resultado do baseline PP-OCRv6 (`scripts/ppocr_baseline.py`, coluna "hit" = acerto exato)
como rótulo de referência e comparamos a distribuição de nitidez entre acertos e erros.

Uso:
    .venv/bin/python scripts/ppocr_baseline.py --no-mlflow   # gera data/analise/ppocr_baseline_test.csv
    .venv/bin/python scripts/sharpness_validation_ufpr.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
BASELINE_CSV = BASE / "data" / "analise" / "ppocr_baseline_test.csv"
IMG_DIR = BASE / "data" / "finetune_ufpramr"


def laplacian_var(path: Path) -> float:
    img = np.array(Image.open(path).convert("L"))
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def percentile(vals: list[float], q: float) -> float:
    s = sorted(vals)
    idx = min(len(s) - 1, int(q * len(s)))
    return round(s[idx], 1)


def main() -> None:
    df = pd.read_csv(BASELINE_CSV)
    df["sharpness"] = df["image"].apply(lambda f: laplacian_var(IMG_DIR / f))

    hit = df[df["hit"] == 1]["sharpness"].tolist()
    miss = df[df["hit"] == 0]["sharpness"].tolist()

    report = {
        "n_total": len(df),
        "n_acerto": len(hit),
        "n_erro": len(miss),
        "sharpness_acerto_p10_mediana_p90": [percentile(hit, 0.1), percentile(hit, 0.5), percentile(hit, 0.9)],
        "sharpness_erro_p10_mediana_p90": [percentile(miss, 0.1), percentile(miss, 0.5), percentile(miss, 0.9)],
        "correlacao_pearson_sharpness_x_hit": round(float(df["sharpness"].corr(df["hit"])), 3),
    }

    # AUC simples: sharpness discrimina hit vs miss? (probabilidade de acerto ter sharpness > erro, pares aleatorios)
    rng = np.random.default_rng(42)
    n_pairs = 5000
    wins = 0
    hit_arr, miss_arr = np.array(hit), np.array(miss)
    for _ in range(n_pairs):
        a = hit_arr[rng.integers(0, len(hit_arr))]
        b = miss_arr[rng.integers(0, len(miss_arr))]
        wins += a > b
    report["auc_aproximado_sharpness_discrimina_acerto"] = round(wins / n_pairs, 3)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    out = BASE / "data" / "analise" / "sharpness_validation_ufpr.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSalvo em {out}")


if __name__ == "__main__":
    main()
