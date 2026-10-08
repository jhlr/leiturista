"""Fine-tune do localizador PP-OCRv5_mobile_det para achar o VISOR (leitura do medidor).

O ONNX do det é convertido para PyTorch (`onnx2torch`, saída idêntica ao ORT, ~1e-7), treinado
só no mapa de probabilidade DB (BCE com hard negative mining + dice) contra o quad do visor
(shrink 0,4 como no DB) e exportado de volta para ONNX no mesmo contrato (`x` -> mapa 1xHxW),
então o pós-processamento (`MeterOCR.score_to_quads`) e o pipeline não mudam.

Positivos: `data/distribuidora_amr/boxes_visor.csv` (ver scripts/reconstruir_quads_visor.py);
viés conhecido: só há quad nas fotos em que o det original já achava algo. Avaliação: recall do
visor (IoU >= 0,5 com alguma caixa, em foto original OU invertida, como o pipeline) e nº médio de
caixas por foto, no valid/test, contra o det original.

Uso: leiturista train-det
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

from . import paths
from .distribuidora import FOTOS_DIR
from .inference import DET_MEAN, DET_ONNX, DET_STD, MeterOCR

BOXES_CSV = paths.DATA_DIR / "distribuidora_amr" / "boxes_visor.csv"
OUT_ONNX = paths.MODELS_DIR / "pp_ocr_v5_mobile_det_visor_onnx" / "inference.onnx"
SHRINK = 0.4  # razão de shrink do DB


def load_det_torch(onnx_path: Path | str = DET_ONNX) -> nn.Module:
    import onnx
    from onnx2torch import convert

    return convert(onnx.load(str(onnx_path)))


def _quad(r: pd.Series) -> np.ndarray:
    return np.array([[r.x1, r.y1], [r.x2, r.y2], [r.x3, r.y3], [r.x4, r.y4]], np.float32)


def _shrunk(quad: np.ndarray) -> np.ndarray:
    """Quad encolhido (DB: d = A(1-r²)/L) em torno do centro, via retângulo mínimo."""
    (cx, cy), (w, h), ang = cv2.minAreaRect(quad)
    d = w * h * (1 - SHRINK**2) / max(2 * (w + h), 1e-6)
    return cv2.boxPoints(((cx, cy), (max(w - 2 * d, 1.0), max(h - 2 * d, 1.0)), ang))


def _read(r: pd.Series) -> np.ndarray:
    return cv2.imread(str(FOTOS_DIR / r.lote / r.foto))


def _tensor(img_bgr: np.ndarray) -> tuple[torch.Tensor, float, float]:
    x, sx, sy = MeterOCR._det_prep(img_bgr)
    return torch.tensor(x), sx, sy


def _sample(r: pd.Series, rng: np.random.Generator, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """(imagem 3xHxW normalizada, alvo 1xHxW) com o alvo no frame da imagem já com padding."""
    img, quad = _read(r), _quad(r)
    if train:
        if rng.random() < 0.4:  # o pipeline também roda o det na foto invertida
            img = 255 - img
        s = rng.uniform(0.7, 1.3)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        quad = quad * s
        img = np.clip(img.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30), 0, 255).astype(np.uint8)
    x, _, _ = _tensor(img)
    hp, wp = x.shape[2], x.shape[3]
    tgt = np.zeros((hp, wp), np.uint8)  # fotos <= 960 px: _det_prep só faz padding, coordenadas preservadas
    cv2.fillPoly(tgt, [np.round(_shrunk(quad)).astype(np.int32)], 1)
    return x[0], torch.tensor(tgt[None], dtype=torch.float32)


def _collate(items: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    hp, wp = max(i[0].shape[1] for i in items), max(i[0].shape[2] for i in items)
    hp, wp = math.ceil(hp / 32) * 32, math.ceil(wp / 32) * 32
    x = torch.zeros(len(items), 3, hp, wp)
    y = torch.zeros(len(items), 1, hp, wp)
    m = torch.zeros(len(items), 1, hp, wp)
    for k, (a, b) in enumerate(items):
        x[k, :, :a.shape[1], :a.shape[2]] = a
        y[k, :, :b.shape[1], :b.shape[2]] = b
        m[k, :, :a.shape[1], :a.shape[2]] = 1
    return x, torch.cat([y, m], 1)


def db_loss(prob: torch.Tensor, ym: torch.Tensor, neg_ratio: float = 3.0) -> torch.Tensor:
    """BCE com hard negative mining (3:1) + dice, só no mapa de probabilidade."""
    y, m = ym[:, :1], ym[:, 1:]
    p = prob.clamp(1e-6, 1 - 1e-6)
    bce = -(y * p.log() + (1 - y) * (1 - p).log())
    pos = (y * m).bool()
    neg = ((1 - y) * m).bool()
    n_pos = int(pos.sum())
    n_neg = min(int(neg.sum()), max(int(n_pos * neg_ratio), 1000))
    neg_loss = bce[neg].topk(n_neg).values if n_neg else bce.new_zeros(0)
    bce_l = (bce[pos].sum() + neg_loss.sum()) / max(n_pos + n_neg, 1)
    inter = (p * y * m).sum()
    dice = 1 - 2 * inter / ((p * m).sum() + (y * m).sum() + 1e-6)
    return bce_l + dice


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    union = cv2.contourArea(a.astype(np.float32)) + cv2.contourArea(b.astype(np.float32)) - inter
    return float(inter / union) if union > 0 else 0.0


@torch.no_grad()
def evaluate_det(model: nn.Module, df: pd.DataFrame, iou: float = 0.5) -> tuple[float, float]:
    """(recall do visor em foto original OU invertida, nº médio de caixas por foto [original+inv])."""
    model.eval()
    hit, boxes = 0, 0
    for _, r in df.iterrows():
        img, gt = _read(r), _quad(r)
        found = False
        for frame in (img, 255 - img):
            x, sx, sy = _tensor(frame)
            quads = MeterOCR.score_to_quads(model(x)[0, 0].numpy(), sx, sy)
            boxes += len(quads)
            found = found or any(_iou(q, gt) >= iou for q, _ in quads)
        hit += int(found)
    return hit / len(df), boxes / len(df)


def export_onnx(model: nn.Module, path: Path | str = OUT_ONNX) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(model, torch.zeros(1, 3, 480, 384), str(path), input_names=["x"], output_names=["out"],
                      dynamic_axes={"x": {0: "b", 2: "h", 3: "w"}, "out": {0: "b", 2: "h", 3: "w"}},
                      opset_version=17, dynamo=False)
    return path


def train_det(
    epochs: int = 15,
    batch: int = 8,
    lr: float = 3e-4,
    seed: int = 0,
    out: Path | str = OUT_ONNX,
    tracking_uri: str = paths.DEFAULT_TRACKING_URI,
    experiment: str = "det-visor",
) -> dict[str, float]:
    import mlflow

    torch.manual_seed(seed)
    torch.set_num_threads(4)
    rng = np.random.default_rng(seed)
    df = pd.read_csv(BOXES_CSV)
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "valid", "test"))
    model = load_det_torch()
    base = load_det_torch().eval()
    res: dict[str, float] = {}
    for nome, d in (("valid", va), ("test", te)):
        res[f"base_{nome}_recall"], res[f"base_{nome}_boxes"] = evaluate_det(base, d)
        print(f"det original {nome}: recall visor {res[f'base_{nome}_recall']:.3f}, "
              f"{res[f'base_{nome}_boxes']:.1f} caixas/foto", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    best, best_state = -1.0, None
    with mlflow.start_run(run_name=f"det-{epochs}ep"):
        mlflow.log_params({"epochs": epochs, "batch": batch, "lr": lr, "n_train": len(tr), "shrink": SHRINK})
        for ep in range(epochs):
            t0, soma, n = time.time(), 0.0, 0
            model.train()
            order = rng.permutation(len(tr))
            for i in range(0, len(order), batch):
                x, ym = _collate([_sample(tr.iloc[j], rng, True) for j in order[i:i + batch]])
                loss = db_loss(model(x), ym)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                soma, n = soma + loss.item(), n + 1
            rec, bx = evaluate_det(model, va)
            mlflow.log_metrics({"train_loss": soma / n, "valid_recall": rec, "valid_boxes": bx}, step=ep)
            mark = ""
            if rec > best:
                best, mark = rec, " *"
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"época {ep:>2}: perda {soma / n:.4f}  valid recall {rec:.3f}  {bx:.1f} caixas/foto  "
                  f"{time.time() - t0:.0f}s{mark}", flush=True)
        assert best_state is not None
        model.load_state_dict(best_state)
        for nome, d in (("valid", va), ("test", te)):
            res[f"ft_{nome}_recall"], res[f"ft_{nome}_boxes"] = evaluate_det(model, d)
            print(f"det fine-tunado {nome}: recall visor {res[f'ft_{nome}_recall']:.3f}, "
                  f"{res[f'ft_{nome}_boxes']:.1f} caixas/foto", flush=True)
        mlflow.log_metrics(res)
        print("ONNX:", export_onnx(model, out), flush=True)
    return res
