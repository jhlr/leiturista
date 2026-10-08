"""L1 do Lab 2: como cada família de reconhecedor erra, em recortes reais.

Famílias: CRNN+CTC off-the-shelf (PP-OCRv6 tiny rec), codificador ViT + decodificador Transformer (TrOCR-small),
e o nosso CRNNDigitos fine-tunado. Cada erro (leitura != verdade) é classificado por alinhamento de edição:
perdeu dígito (só deleções), duplicou dígito (inserção de um dígito igual ao vizinho), trocou dígito
(só substituições), inventou sequência (resto: mistura de operações ou dígitos sem relação, similaridade baixa).

Uso: leiturista errors-l1 --data data/distribuidora_amr_lote --split test -n 150
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import cv2

from . import paths


def classify_error(pred: str, true: str) -> str:
    """Taxonomia do enunciado (L1) por alinhamento de Levenshtein."""
    n, m = len(true), len(pred)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (true[i - 1] != pred[j - 1]))
    i, j, ops = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (true[i - 1] != pred[j - 1]):
            ops.append("sub" if true[i - 1] != pred[j - 1] else "ok")
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append("del")
            i -= 1
        else:
            ops.append("ins")
            j -= 1
    ed = sum(o != "ok" for o in ops)
    if not pred:
        return "perdeu dígito"
    if ed > max(len(true), 2):  # distância maior que o próprio comprimento: sem relação com a verdade
        return "inventou sequência"
    kinds = {o for o in ops if o != "ok"}
    if kinds == {"del"}:
        return "perdeu dígito"
    if kinds == {"sub"}:
        return "trocou dígito"
    if kinds == {"ins"}:
        dup = any(pred[k] == pred[k - 1] for k in range(1, len(pred))) and len(pred) > len(true)
        return "duplicou dígito" if dup else "inventou sequência"
    return "inventou sequência"


def run(data_dir: Path | str, split: str = "test", n: int = 150, ckpt: Path | str = paths.MODELS_DIR / "crnn_lote_ft.pt",
        out_json: Path | str = paths.ROOT / "docs" / "figs" / "l1_erros.json") -> dict[str, object]:
    from .crnn import load_crnn, read_digits
    from .inference import MeterOCR
    from PIL import Image

    d = Path(data_dir)
    items = [(d / r["image"], r["label"]) for r in csv.DictReader(open(d / "labels.csv", encoding="utf-8")) if r["split"] == split][:n]
    ocr, crnn = MeterOCR(), load_crnn(ckpt, cache=False)
    ocr._load()
    fam: dict[str, list[dict[str, str]]] = {"PP-OCRv6 tiny (CRNN+CTC off-the-shelf)": [], "TrOCR-small (ViT+Transformer)": [],
                                            "CRNNDigitos fine-tunado (nosso)": []}
    tot = Counter()
    for p, y in items:
        img = cv2.imread(str(p))
        preds = {
            "PP-OCRv6 tiny (CRNN+CTC off-the-shelf)": re.sub(r"\D", "", ocr._rec_recognize(img)),
            "TrOCR-small (ViT+Transformer)": re.sub(r"\D", "", ocr._trocr_recognize(img)),
            "CRNNDigitos fine-tunado (nosso)": read_digits(Image.open(p), crnn),
        }
        for k, pr in preds.items():
            tot[k] += 1
            if pr != y:
                fam[k].append({"imagem": p.name, "verdade": y, "leitura": pr, "tipo": classify_error(pr, y)})
    res: dict[str, object] = {"n": len(items), "split": split}
    for k, errs in fam.items():
        res[k] = {"acerto_exato": 1 - len(errs) / max(tot[k], 1), "erros": len(errs),
                  "por_tipo": dict(Counter(e["tipo"] for e in errs)), "exemplos": errs[:10]}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return res
