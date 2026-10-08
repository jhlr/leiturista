"""L2 do Lab 2: benchmark em recortes de display do CLIENTE × UFPR-AMR (gap de domínio).

Modelos: PP-OCRv6 tiny rec, TrOCR-small, CRNNDigitos treinado só no UFPR-AMR (zero-shot no cliente) e CRNNDigitos
fine-tunado nos recortes do cliente (partição por lote). Conjuntos: `test` do cliente por lote (lote nunca visto
no treino) e `test` do UFPR-AMR. Métricas: leitura exata e acurácia por dígito (1 - distância de edição / dígitos).
Ressalva: a leitura do cliente nos recortes é a digitada pelo leiturista confirmada pelo OCR (rótulo fraco, viés
para casos fáceis); os rótulos humanos do item C refinam.

Uso: leiturista l2
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import cv2
from PIL import Image

from . import paths
from .crnn import _edit_distance, load_crnn, read_digits


def _items(d: Path, split: str) -> list[tuple[Path, str]]:
    return [(d / r["image"], r["label"]) for r in csv.DictReader(open(d / "labels.csv", encoding="utf-8")) if r["split"] == split]


def run(out_json: Path | str = paths.ROOT / "docs" / "figs" / "l2_gap_dominio.json") -> dict[str, object]:
    from .inference import MeterOCR

    ocr = MeterOCR()
    ocr._load()
    crnn_ufpr = load_crnn(paths.MODELS_DIR / "crnn_bn128_mix500_cur30.pt", cache=False)
    crnn_cli = load_crnn(paths.MODELS_DIR / "crnn_lote_ft.pt", cache=False)
    readers = {
        "PP-OCRv6 tiny (off-the-shelf)": lambda p: re.sub(r"\D", "", ocr._rec_recognize(cv2.imread(str(p)))),
        "TrOCR-small (off-the-shelf)": lambda p: re.sub(r"\D", "", ocr._trocr_recognize(cv2.imread(str(p)))),
        "CRNNDigitos só UFPR-AMR (zero-shot no cliente)": lambda p: read_digits(Image.open(p), crnn_ufpr),
        "CRNNDigitos fine-tunado no cliente (por lote)": lambda p: read_digits(Image.open(p), crnn_cli),
    }
    sets = {"cliente_test_por_lote": _items(paths.DATA_DIR / "distribuidora_amr_lote", "test"),
            "ufpr_amr_test": _items(paths.FINETUNE_DIR, "test")}
    res: dict[str, object] = {"n": {k: len(v) for k, v in sets.items()}, "modelos": {}}
    for name, fn in readers.items():
        r: dict[str, dict[str, float]] = {}
        for sk, items in sets.items():
            preds = [fn(p) for p, _ in items]
            ex = sum(a == y for a, (_, y) in zip(preds, items)) / len(items)
            dig = 1 - sum(_edit_distance(y, a) for a, (_, y) in zip(preds, items)) / sum(len(y) for _, y in items)
            r[sk] = {"leitura_exata": ex, "acuracia_por_digito": dig}
        r["gap_exata"] = {"ufpr_menos_cliente": r["ufpr_amr_test"]["leitura_exata"] - r["cliente_test_por_lote"]["leitura_exata"]}
        res["modelos"][name] = r  # type: ignore[index]
        print(name, json.dumps(r), flush=True)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return res
