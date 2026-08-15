"""Baseline PP-OCRv6_tiny_rec (off-the-shelf) no test set UFPR-AMR.

Roda o modelo ONNX de reconhecimento (1,1M params, sem fine-tune) nas 300
imagens de test e compara com o TrOCR fine-tuned (`leiturista eval`). Sem treino.

Notas:
- As imagens UFPR-AMR são a faixa do display (text-line) — o rec processa direto.
- Preprocess: resize p/ altura 48 mantendo proporção, /255 (a normalização está
  embutida no grafo do ONNX).
- Decode CTC: índice 0 = blank; char = dict[idx-1] (off-by-one do PaddleOCR).
- Registra no MLflow (mlflow.db): métricas + CSV de predições como blob.

Uso:
  .venv/bin/python src/ppocr_baseline.py [--no-mlflow]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

from leiturista import artifacts, paths

BASE = Path(__file__).resolve().parent.parent
MODEL_ONNX = BASE / "models" / "pp_ocr_v6_tiny_rec_onnx" / "inference.onnx"
DICT_JSON = Path("/tmp/ppocr_dict.json")  # extraído do inference.yml (fonte no script)
FINETUNE_DIR = paths.FINETUNE_DIR
OUT_CSV = BASE / "data" / "analise" / "ppocr_baseline_test.csv"
IMG_H = 48
MAX_W = 3200
DICT_FILE = Path(__file__).resolve().parent / "ppocr_v6_dict.json"


def load_dict() -> list[str]:
    if DICT_FILE.exists():
        return json.loads(DICT_FILE.read_text())
    raise SystemExit(
        f"dict não encontrado: {DICT_FILE} (extrair do inference.yml do repo PP-OCRv6_tiny_rec_onnx)"
    )


def prep(path: Path) -> np.ndarray:
    img = np.array(Image.open(path).convert("RGB"))[:, :, ::-1]  # BGR
    h, w = img.shape[:2]
    rw = min(int(math.ceil(IMG_H * w / h)), MAX_W)
    img = np.array(Image.fromarray(img).resize((rw, IMG_H), Image.BILINEAR))
    return (img.transpose(2, 0, 1).astype(np.float32) / 255.0)[None]


def decode(logits: np.ndarray, chars: list[str]) -> str:
    """CTC: argmax, colapsa repetições, pula blank (índice 0), char = dict[idx-1]."""
    idxs = logits.argmax(1)
    prev = None
    out = []
    for i in idxs:
        if i == prev or i == 0 or i >= len(chars) + 1:
            prev = i
            continue
        out.append(chars[i - 1])
        prev = i
    return "".join(out)


def digit_acc(pred: str, label: str) -> float:
    if not label:
        return 0.0
    aligned = zip(pred.rjust(len(label))[-len(label):], label)
    return sum(1 for p, l in aligned if p == l) / len(label)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    chars = load_dict()
    sess = ort.InferenceSession(str(MODEL_ONNX))

    labels = pd.read_csv(FINETUNE_DIR / "labels.csv")
    test = labels[labels["split"] == "test"]
    print(f"test: {len(test)} imagens")

    rows = []
    for _, r in test.iterrows():
        path = FINETUNE_DIR / r["image"]
        logits = sess.run(None, {"x": prep(path)})[0][0]
        pred = decode(logits, chars)
        rows.append({"image": r["image"], "pred": pred, "label": str(r["label"]),
                     "hit": int(pred == str(r["label"]))})

    df = pd.DataFrame(rows)
    exact = float(df["hit"].mean())
    digit = float(df.apply(lambda r: digit_acc(r["pred"], r["label"]), axis=1).mean())
    print(f"PP-OCRv6_tiny_rec exact-match: {exact:.3f} | digit-acc: {digit:.3f} ({len(df)} amostras)")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"CSV: {OUT_CSV}")

    if not args.no_mlflow:
        import mlflow

        mlflow.set_tracking_uri(paths.DEFAULT_TRACKING_URI)
        mlflow.set_experiment("trocr-ufpramr")
        with mlflow.start_run(run_name="baseline-ppocrv6-tiny") as run:
            mlflow.log_params({"model": "PP-OCRv6_tiny_rec", "fine_tuned": False,
                               "test_set": "ufpr_amr_test", "n": len(df)})
            mlflow.log_metrics({"test_exact_match": exact, "test_digit_acc": digit})
            artifacts.save_blob(run.info.run_id, "eval_test.csv",
                                df.to_csv(index=False).encode("utf-8"), "text/csv")
            artifacts.save_text(run.info.run_id, "eval_summary.txt",
                                f"model=PP-OCRv6_tiny_rec\ntest_exact_match={exact:.4f}\ntest_digit_acc={digit:.4f}")
            print(f"MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    main()
