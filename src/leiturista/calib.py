"""Confiança, calibração e recusa do CRNNDigitos (Lab 2, itens E1 e E2).

Confiança de uma leitura = média geométrica das probabilidades máximas por passo da CTC, com temperatura T
nos logits (`softmax(logits / T)`); "correta" = leitura exata. `fit_temperature` escolhe T no VALID (nunca no
teste) minimizando a entropia cruzada binária de (confiança × acerto). ECE em 10 bins; curva cobertura × risco
= aceitar só leituras com confiança >= limiar; risco = taxa de erro entre as aceitas.

Uso: leiturista calibrate-crnn --ckpt models/crnn_lote_ft.pt --data data/distribuidora_amr_lote
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from . import paths
from .crnn import _batch, decodificar_ctc, load_crnn


@torch.no_grad()
def collect_logits(model, items: list[tuple[Path, str]], batch: int = 64) -> tuple[list[torch.Tensor], list[str]]:
    """Logits (T, C) por exemplo e rótulos."""
    model.eval()
    out: list[torch.Tensor] = []
    for i in range(0, len(items), batch):
        x, _, _ = _batch(items[i:i + batch], width=model.width)
        out += list(model(x))
    return out, [y for _, y in items]


def confidences(logits: list[torch.Tensor], T: float = 1.0) -> np.ndarray:
    return np.array([float(torch.softmax(l / T, -1).max(-1).values.clamp_min(1e-9).log().mean().exp()) for l in logits])


def correctness(logits: list[torch.Tensor], labels: list[str]) -> np.ndarray:
    return np.array([p == y for p, y in zip(decodificar_ctc(torch.stack(logits)), labels)], dtype=float)


def fit_temperature(logits: list[torch.Tensor], labels: list[str]) -> float:
    """T que minimiza a entropia cruzada binária de (confiança, acerto) no conjunto dado (use o VALID)."""
    y = correctness(logits, labels)
    best, best_t = float("inf"), 1.0
    for T in np.exp(np.linspace(np.log(0.3), np.log(5.0), 60)):
        c = np.clip(confidences(logits, float(T)), 1e-6, 1 - 1e-6)
        nll = float(-(y * np.log(c) + (1 - y) * np.log(1 - c)).mean())
        if nll < best:
            best, best_t = nll, float(T)
    return best_t


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(tot)


def coverage_risk(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(limiar, cobertura, risco) para cada limiar de confiança (aceita conf >= limiar)."""
    order = np.argsort(-conf)
    c, ok = conf[order], correct[order]
    n = np.arange(1, len(c) + 1)
    return c, n / len(c), 1 - np.cumsum(ok) / n


def max_coverage_at_risk(conf: np.ndarray, correct: np.ndarray, risk: float) -> tuple[float, float]:
    """Maior cobertura com risco <= `risk` (e o limiar que a dá)."""
    thr, cov, rk = coverage_risk(conf, correct)
    ok = np.where(rk <= risk)[0]
    return (float(cov[ok.max()]), float(thr[ok.max()])) if len(ok) else (0.0, float("nan"))


def run(ckpt: Path | str, data_dir: Path | str, out_dir: Path | str = paths.ROOT / "docs" / "figs",
        target_cov: float = 0.6, target_risk: float = 0.02) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = Path(data_dir)
    items: dict[str, list[tuple[Path, str]]] = {"valid": [], "test": []}
    for r in csv.DictReader(open(d / "labels.csv", encoding="utf-8")):
        if r["split"] in items:
            items[r["split"]].append((d / r["image"], r["label"]))
    model = load_crnn(ckpt, cache=False)
    lv, yv = collect_logits(model, items["valid"])
    lt, yt = collect_logits(model, items["test"])
    T = fit_temperature(lv, yv)  # só no VALID
    okt = correctness(lt, yt)
    c0, c1 = confidences(lt, 1.0), confidences(lt, T)
    res: dict[str, object] = {"T": T, "n_valid": len(yv), "n_test": len(yt), "acerto_test": float(okt.mean()),
                              "ece_test_antes": ece(c0, okt), "ece_test_depois": ece(c1, okt),
                              "meta": {"cobertura_min": target_cov, "risco_max": target_risk}}
    cov, thr = max_coverage_at_risk(c1, okt, target_risk)
    res["cobertura_no_risco_alvo"] = cov
    res["limiar_no_risco_alvo"] = thr
    res["cobertura_por_risco"] = {f"risco<={r:.2f}": max_coverage_at_risk(c1, okt, r)[0] for r in (0.02, 0.05, 0.10, 0.20)}
    res["risco_na_cobertura_alvo"] = float(1 - okt[np.argsort(-c1)][:max(1, int(round(target_cov * len(okt))))].mean())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    edges = np.linspace(0, 1, 11)
    for conf, lab in ((c0, f"antes (T=1), ECE {res['ece_test_antes']:.3f}"), (c1, f"depois (T={T:.2f}), ECE {res['ece_test_depois']:.3f}")):
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 3:
                xs.append(conf[m].mean())
                ys.append(okt[m].mean())
        ax[0].plot(xs, ys, "o-", label=lab)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(xlabel="confiança", ylabel="acerto observado", title="Confiabilidade (teste)")
    ax[0].legend(fontsize=7)
    thr_, cov_, rk_ = coverage_risk(c1, okt)
    ax[1].plot(cov_, rk_)
    ax[1].axvline(target_cov, color="r", ls=":", lw=0.8)
    ax[1].axhline(target_risk, color="r", ls=":", lw=0.8)
    ax[1].set(xlabel="cobertura", ylabel="risco (erro entre aceitas)", title="Cobertura × risco (teste)")
    fig.tight_layout()
    fig.savefig(out / "e1_e2_calibracao_cobertura_risco.png", dpi=130)
    (out / "e1_e2_metricas.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res
