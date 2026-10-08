"""Sequências sintéticas de dígitos p/ pré-treino do `CRNNDigitos` (estágio 1 do plano,
docs/dl-lab2/plano-treino-crnn.md): concatena dígitos isolados do LCD digits (Kaggle, CC0,
`data/lcd_digits/data/0-9/`) numa tira, com comprimento sorteado da distribuição real do UFPR-AMR.

Glare, cutout e ruído entram ANTES de concatenar, por dígito (degradação real de campo atinge 1-2
dígitos, não a tira); o jitter de aspecto entra DEPOIS, no resize final.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import paths

LCD_DIR = paths.DATA_DIR / "lcd_digits" / "data"
CROP_DIGITO = (20, 30, 120, 225)  # (x0,y0,x1,y1) em 150x300: tira a vizinhança cortada e as faixas do display
ALTURA_DIGITO = 64  # altura comum antes de concatenar; o resize final leva a tira a 32x128


def length_distribution(labels_csv: Path | str = paths.FINETUNE_DIR / "labels.csv") -> tuple[np.ndarray, np.ndarray]:
    """Distribuição empírica dos comprimentos de rótulo no split train (comprimentos, probabilidades)."""
    with open(labels_csv) as f:
        c = Counter(len(r["label"]) for r in csv.DictReader(f) if r["split"] == "train")
    comp = np.array(sorted(c))
    p = np.array([c[k] for k in comp], dtype=np.float64)
    return comp, p / p.sum()


class DigitBank:
    """Índice dos dígitos isolados; carrega (cinza, altura fixa) sob demanda e guarda em cache."""

    def __init__(self, root: Path | str = LCD_DIR) -> None:
        self.files = {d: sorted((Path(root) / str(d)).glob("*.jpg")) for d in range(10)}
        if not all(self.files.values()):
            raise FileNotFoundError(f"LCD digits não encontrado em {root}")
        self._cache: dict[Path, np.ndarray] = {}

    def sample(self, d: int, rng: np.random.Generator) -> np.ndarray:
        p = self.files[d][int(rng.integers(len(self.files[d])))]
        if p not in self._cache:
            im = Image.open(p).convert("L").crop(CROP_DIGITO)
            w = max(1, round(im.width * ALTURA_DIGITO / im.height))
            self._cache[p] = np.asarray(im.resize((w, ALTURA_DIGITO)), dtype=np.float32) / 255.0
        return self._cache[p]


def _glare(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Elipse clara semi-transparente (opacidade 0,3-0,6) sobre o dígito."""
    h, w = a.shape
    m = Image.new("L", (w, h), 0)
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    rx, ry = rng.uniform(0.2, 0.6) * w, rng.uniform(0.2, 0.6) * h
    ImageDraw.Draw(m).ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
    alpha = np.asarray(m, dtype=np.float32) / 255.0 * rng.uniform(0.3, 0.6)
    return a * (1 - alpha) + alpha


def _cutout(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Retângulo cobrindo até ~30% da ÁREA do dígito (nunca o dígito inteiro)."""
    h, w = a.shape
    frac = rng.uniform(0.1, 0.3)
    rw = int(w * rng.uniform(0.4, 1.0))
    rh = max(1, int(frac * h * w / max(rw, 1)))
    rh = min(rh, h)
    x, y = int(rng.integers(0, w - rw + 1)), int(rng.integers(0, h - rh + 1))
    out = a.copy()
    out[y:y + rh, x:x + rw] = rng.uniform(0.0, 1.0)
    return out


def _noise(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        return a + rng.normal(0, rng.uniform(0.02, 0.08), a.shape).astype(np.float32)
    out = a.copy()
    mask = rng.random(a.shape)
    p = rng.uniform(0.03, 0.05)
    out[mask < p / 2] = 0.0
    out[mask > 1 - p / 2] = 1.0
    return out


def degrade_digit(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Degradação independente por dígito: glare ~17%, cutout ~12%, ruído ~20%."""
    if rng.random() < 0.17:
        a = _glare(a, rng)
    if rng.random() < 0.12:
        a = _cutout(a, rng)
    if rng.random() < 0.20:
        a = _noise(a, rng)
    return np.clip(a, 0.0, 1.0)


def make_sequence(
    bank: DigitBank,
    lengths: tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
    size: tuple[int, int] = (128, 32),
) -> tuple[Image.Image, str]:
    """Uma tira sintética (imagem cinza `size`=(L,A) + rótulo). Jitter de aspecto (largura 85-100%)
    com borda replicada até o tamanho fixo."""
    n = int(rng.choice(lengths[0], p=lengths[1]))
    label = "".join(str(int(rng.integers(10))) for _ in range(n))
    strip = np.concatenate([degrade_digit(bank.sample(int(c), rng), rng) for c in label], axis=1)
    im = Image.fromarray((strip * 255).astype(np.uint8))
    w = max(1, round(size[0] * rng.uniform(0.85, 1.0)))  # só estreita: cortar apagaria dígitos do rótulo
    arr = np.asarray(im.resize((w, size[1])))
    pad = size[0] - w
    left = int(rng.integers(0, pad + 1))
    arr = np.pad(arr, ((0, 0), (left, pad - left)), mode="edge")
    return Image.fromarray(arr), label
