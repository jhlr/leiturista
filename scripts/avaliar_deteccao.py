"""Avaliação isolada do detector (baseline vs det visor-ft vs união)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "neoenergia_amr_v2"
FOTOS = ROOT / "data" / "FotosNeoenergia"
OUT = ROOT / "logs" / "avaliacao_deteccao"
MODELS = ROOT / "models"
CRNN_PT = MODELS / "crnn_dist_v2_ft.pt"

DET_BASE = MODELS / "pp_ocr_v5_mobile_det_onnx" / "inference.onnx"
DET_V2 = MODELS / "pp_ocr_v5_mobile_det_visor_onnx" / "inference.onnx"


def ler_manifest() -> list[dict]:
    with open(DATA / "manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


RUNNER = r"""
import os, sys, json, re, csv, random
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
import importlib
import leiturista.inference as inf
importlib.reload(inf)
from leiturista.inference import MeterOCR
from leiturista.crnn import load_crnn, read_digits

rows = list(csv.DictReader(open(sys.argv[2], newline="", encoding="utf-8")))
fotos_root = Path(sys.argv[3])
crnn_pt = Path(sys.argv[4])
mode = sys.argv[5]
split = sys.argv[6]
n_sem = int(sys.argv[7])

if mode == "aceito":
    man = [r for r in rows if r["status"] == "aceito" and r["split"] == split]
elif mode == "sem_crop":
    sc = [r for r in rows if r["status"] == "sem_crop"]
    random.seed(42)
    random.shuffle(sc)
    man = sc[:n_sem]
else:
    man = []

ocr = MeterOCR()
crnn = load_crnn(crnn_pt)

stats = {"n": len(man), "com_caixa": 0, "com_leitura": 0, "acerto": 0, "oraculo": 0}
for i, r in enumerate(man):
    p = fotos_root / r["lote"] / r["foto"]
    img = np.array(Image.open(p).convert("RGB"))[:, :, ::-1]
    label = re.sub(r"\D", "", r.get("leitura_original", ""))
    cands = []
    for source, merge in (("normal", True), ("inverted", False)):
        base = img if source == "normal" else 255 - img
        quads = ocr._det_boxes(base)
        mquads = ocr._merge_quads(quads) if merge else quads
        for quad, bconf in mquads:
            crop = ocr._crop_rotated(base, quad)
            if crop.size == 0:
                continue
            pil = Image.fromarray(crop[:, :, ::-1])
            text = read_digits(pil, crnn)
            d = re.sub(r"\D", "", text)
            w = float(quad[:, 0].max() - quad[:, 0].min())
            h = float(quad[:, 1].max() - quad[:, 1].min())
            field = ocr._classify(text)
            if field == "leitura" and w < 1.5 * h:
                field = "serial" if (len(text) >= 6 and re.search(r"\d", text)) else "outro"
            cands.append({"digits": d, "field": field})
    if cands:
        stats["com_caixa"] += 1
    leit = [c for c in cands if c["field"] == "leitura" and c["digits"]]
    if leit:
        stats["com_leitura"] += 1
    if any(c["digits"] == label for c in leit):
        stats["acerto"] += 1
    if any(c["digits"] == label for c in cands):
        stats["oraculo"] += 1
    if i % 25 == 0:
        print(i, flush=True)

n = stats["n"] or 1
c = stats["com_caixa"] or 1
l = stats["com_leitura"] or 1
res = {
    "n": stats["n"],
    "com_caixa": stats["com_caixa"],
    "pct_com_caixa": stats["com_caixa"] / n,
    "com_leitura": stats["com_leitura"],
    "pct_com_leitura": stats["com_leitura"] / n,
    "acerto": stats["acerto"],
    "pct_acerto_sobre_todos": stats["acerto"] / n,
    "pct_acerto_sobre_com_leitura": stats["acerto"] / l,
    "oraculo": stats["oraculo"],
    "pct_oraculo": stats["oraculo"] / n,
}
print("RESULT_JSON:" + json.dumps(res))
"""


def run_cfg(cname: str, env: dict, mode: str, split: str, n_sem: int) -> dict:
    tmp = Path(tempfile.mktemp(suffix=".py", prefix="det_eval_"))
    tmp.write_text(RUNNER, encoding="utf-8")
    penv = os.environ.copy()
    penv.update(env)
    penv["PYTHONPATH"] = str(ROOT / "src")
    p = subprocess.run(
        [sys.executable, str(tmp), str(ROOT), str(DATA / "manifest.csv"), str(FOTOS), str(CRNN_PT), mode, split, str(n_sem)],
        cwd=str(ROOT),
        env=penv,
        capture_output=True,
        text=True,
        timeout=2400,
    )
    out = p.stdout + "\n" + p.stderr
    tmp.unlink(missing_ok=True)
    m = re.search(r"RESULT_JSON:(\{.*\})", out)
    if not m:
        print(out[-2500:])
        raise SystemExit(f"falhou {cname}")
    return json.loads(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--sample-sem", type=int, default=200)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    configs = [
        ("baseline", {"LEITURISTA_DET_ONNX": str(DET_BASE), "LEITURISTA_DET2_ONNX": ""}),
        ("det2_solo", {"LEITURISTA_DET_ONNX": str(DET_V2), "LEITURISTA_DET2_ONNX": ""}),
        ("uniao", {"LEITURISTA_DET_ONNX": str(DET_BASE), "LEITURISTA_DET2_ONNX": str(DET_V2), "LEITURISTA_DET2_THRESH": "0.2", "LEITURISTA_DET2_BOX_THRESH": "0.2"}),
    ]

    results = {"test_aceito": {}, "sem_crop_amostra": {}}

    for cname, env in configs:
        print(f"=== {cname} | aceitos {a.split} ===", flush=True)
        res = run_cfg(cname, env, "aceito", a.split, 0)
        results["test_aceito"][cname] = res
        print(json.dumps(res, ensure_ascii=False))

    for cname, env in configs:
        print(f"=== {cname} | sem_crop amostra {a.sample_sem} ===", flush=True)
        res = run_cfg(cname, env, "sem_crop", "", a.sample_sem)
        results["sem_crop_amostra"][cname] = res
        print(json.dumps(res, ensure_ascii=False))

    ts = time.strftime("%Y%m%d_%H%M%S")
    outj = OUT / f"detector_{ts}.json"
    with open(outj, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(OUT / "ultimo.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    evo_dir = ROOT / "data" / "avaliacoes_evolucao" / "detector" / ts
    evo_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy(outj, evo_dir / "detector.json")
    shutil.copy(OUT / "ultimo.json", ROOT / "data" / "avaliacoes_evolucao" / "detector_ultimo.json")

    print("\nSALVO:", outj)
    print("ARTEFATO:", evo_dir / "detector.json")


if __name__ == "__main__":
    main()
