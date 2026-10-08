"""Classificador "este candidato é o visor de leitura?" (passo A de docs/2026-10-01_gargalo_amarelo_e_plano_visor.md).

MobileNetV3-Small (mesmo `scene.build_model`) sobre o recorte de cada caixa do det (visor ×
serial/placa/outro). Substitui a regra "mais dígitos" do `_best_reading`, que escolhe serial/placa.
Dataset: `scripts/construir_candidatos_visor.py` (data/visor_cls). Avaliação por foto: top-1 = o
candidato de maior P(visor) é o visor, contra a regra antiga re-executada offline.

Uso: leiturista train-visor        # salva models/visor_cls.pt
"""

from __future__ import annotations

import csv
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from . import paths
from .scene import _MEAN, _STD, build_model

DATA = paths.DATA_DIR / "visor_cls"
WEIGHTS = paths.MODELS_DIR / "visor_cls.pt"
SIZE = (64, 192)  # (altura, largura): recorte de visor ~3:1

_eval_tf = transforms.Compose([transforms.Resize(SIZE), transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])
_train_tf = transforms.Compose([
    transforms.Resize(SIZE),
    transforms.RandomAffine(5, translate=(0.04, 0.08), scale=(0.9, 1.1)),
    transforms.ColorJitter(0.4, 0.4, 0.3),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


def _read(split: str) -> list[dict[str, str]]:
    with open(DATA / "candidatos.csv", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def _load(rows: list[dict[str, str]], tf: transforms.Compose) -> torch.Tensor:
    return torch.stack([tf(Image.open(DATA / "crops" / f"{r['cand']}.png").convert("RGB")) for r in rows])


def old_rule_pick(cands: list[dict[str, str]]) -> dict[str, str] | None:
    """Regra antiga (`MeterOCR._best_reading`) re-executada nos candidatos gravados."""
    readings = [c for c in cands if c["field"] == "leitura"]
    if not readings:
        return None

    def pure(c: dict[str, str]) -> bool:
        t = re.sub(r"[\s.,]", "", c["text"])
        return bool(t) and bool(re.fullmatch(r"\d+", t))

    pool = [c for c in readings if pure(c)] or readings
    return max(pool, key=lambda c: (len(re.sub(r"\D", "", c["text"])), float(c["conf"])))


@torch.no_grad()
def scores(model: nn.Module, rows: list[dict[str, str]], batch: int = 128) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(_load(rows[i:i + batch], _eval_tf)).squeeze(1)) for i in range(0, len(rows), batch)]
    return torch.cat(out).numpy()


_CACHE: dict[str, nn.Module] = {}


def load_visor(path: Path | str = WEIGHTS) -> nn.Module:
    key = str(path)
    if key not in _CACHE:
        m = build_model(pretrained=False)
        m.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
        _CACHE[key] = m.eval()
    return _CACHE[key]


@torch.no_grad()
def score_crops(crops: list[Image.Image], path: Path | str = WEIGHTS) -> np.ndarray:
    """P(visor) para cada recorte (PIL RGB)."""
    m = load_visor(path)
    x = torch.stack([_eval_tf(c.convert("RGB")) for c in crops])
    return torch.sigmoid(m(x).squeeze(1)).numpy()


def top1(model: nn.Module, rows: list[dict[str, str]]) -> dict[str, float]:
    """Top-1 por foto (só fotos com algum positivo entre os candidatos): novo vs. regra antiga."""
    s = scores(model, rows)
    by: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by[(r["lote"], r["foto"])].append(i)
    n = new = old = 0
    for idx in by.values():
        if not any(rows[i]["label"] == "1" for i in idx):
            continue
        n += 1
        new += rows[max(idx, key=lambda i: s[i])]["label"] == "1"
        pick = old_rule_pick([rows[i] for i in idx])
        old += pick is not None and pick["label"] == "1"
    return {"fotos": n, "novo": new / n, "regra_antiga": old / n}


def train_visor(epochs: int = 8, batch: int = 64, lr: float = 1e-3, seed: int = 0, out: Path | str = WEIGHTS,
                tracking_uri: str = paths.DEFAULT_TRACKING_URI, experiment: str = "visor-cls") -> dict[str, float]:
    import mlflow

    torch.manual_seed(seed)
    torch.set_num_threads(4)
    rng = np.random.default_rng(seed)
    tr, va, te = _read("train"), _read("valid"), _read("test")
    y = np.array([int(r["label"]) for r in tr])
    print(f"train {len(tr)} candidatos ({y.sum()} positivos), valid {len(va)}, test {len(te)}", flush=True)
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    per_epoch = min(len(tr), 8 * len(pos))  # 1:3 positivo:negativo por época, negativos reamostrados
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    best, out = -1.0, Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_name=f"visor-{epochs}ep"):
        mlflow.log_params({"epochs": epochs, "batch": batch, "lr": lr, "n_train": len(tr)})
        for ep in range(epochs):
            t0, soma, n = time.time(), 0.0, 0
            idx = rng.permutation(np.concatenate([rng.choice(pos, per_epoch // 4), rng.choice(neg, per_epoch - per_epoch // 4)]))
            model.train()
            for i in range(0, len(idx), batch):
                rows = [tr[j] for j in idx[i:i + batch]]
                x = _load(rows, _train_tf)
                yy = torch.tensor([float(r["label"]) for r in rows])
                loss = loss_fn(model(x).squeeze(1), yy)
                opt.zero_grad()
                loss.backward()
                opt.step()
                soma, n = soma + loss.item(), n + 1
            m = top1(model, va)
            mlflow.log_metrics({"train_loss": soma / n, "valid_top1": m["novo"], "valid_top1_regra_antiga": m["regra_antiga"]}, step=ep)
            mark = ""
            if m["novo"] > best:
                best, mark = m["novo"], " *"
                torch.save({"state_dict": model.state_dict(), "size": SIZE}, out)
            print(f"época {ep}: perda {soma / n:.4f}  valid top-1 {m['novo']:.3f} (regra antiga {m['regra_antiga']:.3f}, "
                  f"{m['fotos']:.0f} fotos)  {time.time() - t0:.0f}s{mark}", flush=True)
        model.load_state_dict(torch.load(out, map_location="cpu")["state_dict"])
        res: dict[str, float] = {}
        for nome, rows in (("valid", va), ("test", te)):
            m = top1(model, rows)
            res.update({f"{nome}_top1": m["novo"], f"{nome}_top1_regra_antiga": m["regra_antiga"]})
            print(f"{nome}: top-1 {m['novo']:.3f} (regra antiga {m['regra_antiga']:.3f}, {m['fotos']:.0f} fotos)", flush=True)
        mlflow.log_metrics(res)
    return res
