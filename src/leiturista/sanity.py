"""Verificações de sanidade do Lab 2 (itens B1, B2, D1, D2) sobre o `CRNNDigitos`, no dado REAL."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .crnn import CRNNDigitos, _batch, decodificar_ctc


def shape_table(model: nn.Module, x: torch.Tensor) -> list[tuple[str, tuple[int, ...]]]:
    """Forma de saída de cada submódulo de primeiro nível (B1) e dentro da CNN."""
    rows: list[tuple[str, tuple[int, ...]]] = []
    hooks = []
    for name, mod in model.named_children():
        subs = list(mod.named_children()) if name == "cnn" else [("", mod)]
        for sub_name, sub in subs:
            label = f"{name}.{sub_name}" if sub_name else name
            hooks.append(sub.register_forward_hook(
                lambda _m, _i, o, lab=label: rows.append((lab, tuple((o[0] if isinstance(o, tuple) else o).shape)))))
    with torch.no_grad():
        model.eval()(x)
    for h in hooks:
        h.remove()
    return rows


def count_params(model: nn.Module, freeze_cnn: bool = False) -> tuple[int, int]:
    """(total, treinável); `freeze_cnn` = configuração de extração de características (B2)."""
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for n, p in model.named_parameters() if not (freeze_cnn and n.startswith("cnn")))
    return total, train


def initial_loss(items: list[tuple[Path, str]], seed: int = 0) -> tuple[float, float]:
    """(perda CTC inicial com pesos aleatórios, referência T*ln(11)/L da saída uniforme) em um lote REAL (D1)."""
    torch.manual_seed(seed)
    model = CRNNDigitos().eval()
    x, alvos, comp = _batch(items)
    with torch.no_grad():
        lp = model(x).log_softmax(-1).permute(1, 0, 2)
        loss = nn.CTCLoss(blank=0, zero_infinity=True)(lp, alvos, torch.full((x.size(0),), lp.size(0), dtype=torch.long), comp)
    t, comprimento = lp.size(0), float(comp.float().mean())
    # a CTC (reduction="mean") divide pela comprimento do alvo: sob saída uniforme a perda ~ T*ln(11)/L
    return float(loss), t * math.log(11) / comprimento


def overfit_one_batch(items: list[tuple[Path, str]], steps: int = 300, seed: int = 0) -> dict[str, object]:
    """Sobreajusta um lote de 10-20 recortes REAIS sem regularização; a perda tem que ir a ~0 (D2)."""
    torch.manual_seed(seed)
    model = CRNNDigitos()
    x, alvos, comp = _batch(items)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    curve: list[float] = []
    for _ in range(steps):
        model.train()
        lp = model(x).log_softmax(-1).permute(1, 0, 2)
        loss = ctc(lp, alvos, torch.full((x.size(0),), lp.size(0), dtype=torch.long), comp)
        opt.zero_grad()
        loss.backward()
        opt.step()
        curve.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        pred = decodificar_ctc(model(x))
    return {"perda_inicial": curve[0], "perda_final": curve[-1], "acerto_exato": float(np.mean([p == y for p, (_, y) in zip(pred, items)])),
            "n": len(items), "lidos": pred, "alvos": [y for _, y in items]}
