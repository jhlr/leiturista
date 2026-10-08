"""Classificador de cena "tem medidor na foto?" (estágio 1 da triagem).

MobileNetV3-Small (ImageNet) com 1 logit: P(tem medidor). Roda sobre a foto inteira, antes do
OCR; "sem medidor" vira vermelho em vez de amarelo (ver docs/2026-10-01_gargalo_amarelo_e_plano_visor.md).

Rótulos: `llm_meter_visible` do CSV da distribuidora (rótulo FRACO, gerado por LLM, ~8% sem
medidor). O CSV e as fotos ficam só em disco local (data/, gitignored). Split determinístico,
estratificado pelo rótulo, 80/10/10. O limiar é escolhido no `valid` (F1 da classe "sem medidor")
e gravado no checkpoint; o `test` só reporta.

Uso:
    leiturista train-scene            # salva models/scene_medidor.pt (MLflow em mlflow.db)
    from leiturista.scene import load_scene, predict_scene
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from . import paths

LABELS_CSV = paths.SCENE_LABELS_CSV
FOTOS_DIR = Path(os.environ.get("LEITURISTA_FOTOS_DIR", str(paths.CAMPO_DIR)))
WEIGHTS = paths.SCENE_WEIGHTS
READABLE_WEIGHTS = paths.MODELS_DIR / "scene_legivel.pt"
# alvo -> (coluna do rótulo fraco do LLM, nome da classe negativa)
TARGETS = {"meter": ("llm_meter_visible", "sem medidor"), "readable": ("llm_display_readable", "ilegível")}
SIZE = (320, 240)  # (altura, largura): fotos 360x480 em pé
_MEAN, _STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

_eval_tf = transforms.Compose([transforms.Resize(SIZE), transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])
_train_tf = transforms.Compose([
    transforms.RandomChoice([transforms.RandomRotation((a, a), expand=True) for a in (0, 90, 180, 270)]),
    transforms.Resize(SIZE),
    transforms.ColorJitter(0.4, 0.4, 0.3, 0.05),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2.0))], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


def build_model(pretrained: bool = True) -> nn.Module:
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    return m


def read_labels(csv_path: Path = LABELS_CSV, fotos_dir: Path = FOTOS_DIR, target: str = "meter") -> list[tuple[Path, int, str]]:
    """(foto, positivo 0/1, lote) para as linhas com rótulo do LLM e foto no disco.
    `target`: "meter" = tem medidor; "readable" = o display é legível (ilegível inclui foto sem medidor)."""
    col = TARGETS[target][0]
    rows: list[tuple[Path, int, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if r[col] not in ("True", "False"):
                continue
            p = fotos_dir / r["file"]
            if p.is_file():
                rows.append((p, int(r[col] == "True"), r["lote"]))
    return rows


def split_labels(rows: list[tuple[Path, int, str]], seed: int = 42) -> dict[str, list[tuple[Path, int, str]]]:
    """80/10/10 estratificado pelo rótulo (poucos negativos: mantém a proporção em cada split)."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[tuple[Path, int, str]]] = {"train": [], "valid": [], "test": []}
    for y in (0, 1):
        grp = [r for r in rows if r[1] == y]
        idx = rng.permutation(len(grp))
        n_tr, n_va = int(len(grp) * 0.8), int(len(grp) * 0.1)
        for k, i in enumerate(idx):
            out["train" if k < n_tr else "valid" if k < n_tr + n_va else "test"].append(grp[i])
    return out


class _DS(torch.utils.data.Dataset):
    def __init__(self, items: list[tuple[Path, int, str]], tf: transforms.Compose) -> None:
        self.items, self.tf = items, tf

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        p, y, _ = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), torch.tensor(float(y))


def _scores(model: nn.Module, items: list[tuple[Path, int, str]], batch: int) -> np.ndarray:
    """P(tem medidor) para cada item, sem augmentation."""
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in torch.utils.data.DataLoader(_DS(items, _eval_tf), batch_size=batch):
            out.append(torch.sigmoid(model(x)).squeeze(1).numpy())
    return np.concatenate(out)


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) (Mann-Whitney), empates valem 0,5."""
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def _f1_neg(p: np.ndarray, y: np.ndarray, thr: float) -> tuple[float, float, float]:
    """(precisão, recall, F1) da classe 'sem medidor' (y=0) prevista quando p < thr."""
    pred_neg = p < thr
    tp = float((pred_neg & (y == 0)).sum())
    prec = tp / max(pred_neg.sum(), 1)
    rec = tp / max((y == 0).sum(), 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)


def best_threshold(p: np.ndarray, y: np.ndarray) -> float:
    cands = np.unique(np.round(p, 3))
    return float(max(cands, key=lambda t: _f1_neg(p, y, t)[2]))


def train_scene(
    out: Path | str = WEIGHTS,
    labels_csv: Path | str = LABELS_CSV,
    fotos_dir: Path | str = FOTOS_DIR,
    epochs: int = 10,
    freeze_epochs: int = 2,
    batch: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
    pretrained: bool = True,
    tracking_uri: str = paths.DEFAULT_TRACKING_URI,
    experiment: str | None = None,
    target: str = "meter",
) -> dict[str, float]:
    """Treina com BCE ponderada pela classe rara; backbone congelado nas primeiras
    `freeze_epochs`. Salva a época de melhor AUROC no `valid` (o `test` não seleciona nada)."""
    import mlflow

    torch.manual_seed(seed)
    neg_name = TARGETS[target][1]
    experiment = experiment or ("scene-medidor" if target == "meter" else "scene-legivel")
    split = split_labels(read_labels(Path(labels_csv), Path(fotos_dir), target))
    ys = {k: np.array([r[1] for r in v]) for k, v in split.items()}
    n_neg, n_pos = int((ys["train"] == 0).sum()), int((ys["train"] == 1).sum())
    # logit = P(tem medidor): pos_weight < 1 desconta a classe majoritária
    pos_weight = torch.tensor(n_neg / max(n_pos, 1))
    model = build_model(pretrained)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = torch.utils.data.DataLoader(_DS(split["train"], _train_tf), batch_size=batch, shuffle=True, num_workers=2)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    best_auc, best_state, best_thr = -1.0, None, 0.5
    with mlflow.start_run(run_name=f"scene-{epochs}ep"):
        mlflow.log_params({"epochs": epochs, "freeze_epochs": freeze_epochs, "batch": batch, "lr": lr, "seed": seed,
                           "pretrained": pretrained, "n_train": len(split["train"]), "n_train_neg": n_neg,
                           "size": f"{SIZE[0]}x{SIZE[1]}", "target": target})
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        for ep in range(epochs):
            for p in model.features.parameters():
                p.requires_grad = ep >= freeze_epochs
            model.train()
            soma, n = 0.0, 0
            for x, y in loader:
                opt.zero_grad()
                loss = loss_fn(model(x).squeeze(1), y)
                loss.backward()
                opt.step()
                soma += loss.item()
                n += 1
            pv = _scores(model, split["valid"], batch)
            auc = _auroc(pv[ys["valid"] == 1], pv[ys["valid"] == 0])
            mlflow.log_metrics({"train_loss": soma / n, "valid_auroc": auc}, step=ep)
            print(f"época {ep:>2}: perda = {soma / n:.4f}  valid AUROC = {auc:.3f}", flush=True)
            if auc > best_auc:
                best_auc, best_thr = auc, best_threshold(pv, ys["valid"])
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        assert best_state is not None
        model.load_state_dict(best_state)
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "threshold": best_thr}, out)

        res: dict[str, float] = {"valid_auroc": best_auc, "threshold": best_thr}
        for nome in ("valid", "test"):
            p = _scores(model, split[nome], batch)
            prec, rec, f1 = _f1_neg(p, ys[nome], best_thr)
            res.update({f"{nome}_neg_precision": prec, f"{nome}_neg_recall": rec, f"{nome}_neg_f1": f1})
            if nome == "test":
                res["test_auroc"] = _auroc(p[ys[nome] == 1], p[ys[nome] == 0])
            print(f"{nome:>5}: {neg_name}  precisão={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}  "
                  f"(n={len(ys[nome])}, negativos={(ys[nome] == 0).sum()})", flush=True)
        mlflow.log_metrics(res)
    return res


_CACHE: dict[str, tuple[nn.Module, float]] = {}


def load_scene(path: Path | str = WEIGHTS) -> tuple[nn.Module, float]:
    key = str(path)
    if key not in _CACHE:
        ck = torch.load(path, map_location="cpu")
        m = build_model(pretrained=False)
        m.load_state_dict(ck["state_dict"])
        _CACHE[key] = (m.eval(), float(ck["threshold"]))
    return _CACHE[key]


def predict_scene(img: Image.Image, path: Path | str = WEIGHTS) -> tuple[float, bool]:
    """(P(tem medidor), sem_medidor) com o limiar gravado no checkpoint."""
    m, thr = load_scene(path)
    with torch.no_grad():
        p = float(torch.sigmoid(m(_eval_tf(img.convert("RGB"))[None])))
    return p, p < thr
