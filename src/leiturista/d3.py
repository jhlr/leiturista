"""D3 do Lab 2: bloco profundo em modo EXTRAÇÃO de características × baseline não profunda, 3 sementes.

Baseline não profunda: HOG (numpy) do recorte 32x128 + um classificador linear (regressão logística por SGD)
por posição, alinhado à direita (5 posições; classe 10 = "sem dígito"). Mesma partição (por lote), mesma
métrica (leitura exata) e mesmo conjunto de validação do bloco profundo.
Profundo (extração): CNN do CRNNDigitos pré-treinada no UFPR-AMR e CONGELADA (`--freeze-cnn`), cabeça GRU+Linear
reiniciada e treinada nos rótulos de treino (`--reinit-head`).

Uso: leiturista d3 --seeds 0 1 2
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import SGDClassifier

from . import paths

POS = 5


def _hog(g: np.ndarray, cell: int = 8, bins: int = 9) -> np.ndarray:
    """HOG simples (gradientes sem sinal, histograma por célula 8x8, normalização L2 por bloco 2x2)."""
    gx = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    b = np.minimum((ang % 180) / (180 / bins), bins - 1).astype(int)
    h, w = g.shape
    H = np.zeros((h // cell, w // cell, bins), np.float32)
    for i in range(h // cell):
        for j in range(w // cell):
            sl = (slice(i * cell, (i + 1) * cell), slice(j * cell, (j + 1) * cell))
            H[i, j] = np.bincount(b[sl].ravel(), weights=mag[sl].ravel(), minlength=bins)
    blocks = [H[i:i + 2, j:j + 2].ravel() for i in range(H.shape[0] - 1) for j in range(H.shape[1] - 1)]
    return np.concatenate([v / (np.linalg.norm(v) + 1e-6) for v in blocks])


def _feat(p: Path) -> np.ndarray:
    return _hog(cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (128, 32)))


def _targets(y: str) -> list[int]:
    y = y[-POS:].rjust(POS, "x")
    return [10 if c == "x" else int(c) for c in y]


def load(dirs: list[Path], split: str) -> tuple[list[Path], list[str]]:
    ps, ys = [], []
    for d in dirs:
        for r in csv.DictReader(open(d / "labels.csv", encoding="utf-8")):
            if r["split"] == split:
                ps.append(d / r["image"])
                ys.append(r["label"])
    return ps, ys


def hog_baseline(train_dirs: list[Path], eval_dir: Path, seed: int) -> dict[str, float]:
    """Treina nos `train` de `train_dirs`; alpha escolhido no VALID de `eval_dir` (nunca no teste)."""
    tr_p, tr_y = load(train_dirs, "train")
    X = np.stack([_feat(p) for p in tr_p])
    T = np.array([_targets(y) for y in tr_y])
    ev = {sp: load([eval_dir], sp) for sp in ("valid", "test")}
    F = {sp: np.stack([_feat(p) for p in ev[sp][0]]) for sp in ev}

    def exact(heads: list[SGDClassifier], sp: str) -> float:
        P = np.stack([h.predict(F[sp]) for h in heads], 1)
        pred = ["".join(str(d) for d in row if d != 10) for row in P]
        return float(np.mean([a == b[-POS:] for a, b in zip(pred, ev[sp][1])]))

    best: tuple[float, float, list[SGDClassifier]] | None = None
    for alpha in (1e-5, 1e-4, 1e-3, 1e-2):
        heads = [SGDClassifier(loss="log_loss", alpha=alpha, max_iter=60, random_state=seed, tol=None).fit(X, T[:, k]) for k in range(POS)]
        v = exact(heads, "valid")
        if best is None or v > best[0]:
            best = (v, alpha, heads)
    assert best is not None
    return {"valid_exact": best[0], "test_exact": exact(best[2], "test"), "alpha": best[1]}


def run(seeds: list[int], ckpt: Path | str, out_json: Path | str = paths.ROOT / "docs" / "figs" / "d3_resultados.json",
        epochs: int = 30) -> dict[str, object]:
    from .crnn import train_crnn

    lote, ufpr = paths.DATA_DIR / "distribuidora_amr_lote", paths.FINETUNE_DIR
    dirs = [lote, ufpr]  # as duas famílias treinam nos MESMOS dados (lote train + UFPR train); avaliam no lote valid/test
    res: dict[str, list[dict[str, float]]] = {"hog_linear": [], "crnn_extracao": []}
    for s in seeds:
        res["hog_linear"].append(hog_baseline(dirs, lote, s))
        r = train_crnn(data_dirs=dirs, out=paths.MODELS_DIR / f"crnn_extracao_s{s}.pt", epochs=epochs, lr=1e-3, seed=s, init=ckpt,
                       invert_prob=0.4, crop_jitter=0.07, freeze_cnn=True, reinit_head=True, warmup=20, experiment="crnn-d3",
                       tracking_uri=f"sqlite:///{paths.DATA_DIR / 'mlflow_crnn.db'}")
        res["crnn_extracao"].append({"valid_exact": r["distribuidora_amr_lote_valid_exact"], "test_exact": r["distribuidora_amr_lote_test_exact"]})
    summ = {k: {m: {"media": float(np.mean([x[m] for x in v])), "desvio": float(np.std([x[m] for x in v]))} for m in v[0]}
            for k, v in res.items()}
    out = {"seeds": seeds, "por_semente": res, "resumo": summ}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
