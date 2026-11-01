"""Variaveis extraidas da imagem de entrada (dono unico, SDD secao 5.1).

Tudo e calculado sobre a imagem decodificada e convertida para RGB. As medias saem dos
histogramas 0-255 (`Image.histogram()`), como no codigo do curso. Adicionar ou remover uma
variavel significa editar so este arquivo e as listas de colunas abaixo.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

EDGE_GRADIENT_THRESHOLD = 30.0  # magnitude do gradiente de L (0-255) acima da qual o pixel e "borda"
BRIGHT_CLIP_LEVEL = 250  # L >= 250: estouro de brilho (reflexo/superexposicao)
DARK_CLIP_LEVEL = 5  # L <= 5: subexposicao
PERCENTILE_LOW = 0.05
PERCENTILE_HIGH = 0.95
HIST_BINS = 256

SCALAR_FEATURES: tuple[str, ...] = (
    "width",
    "height",
    "file_bytes",
    "luminance_mean",
    "r_mean",
    "g_mean",
    "b_mean",
    "luminance_std",
    "sharpness",
    "bright_clip_frac",
    "dark_clip_frac",
    "luminance_entropy",
    "edge_density",
    "saturation_mean",
    "dynamic_range",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("format", "dominant_channel", "orientation")
HIST_FEATURES: tuple[str, ...] = ("hist_l", "hist_r", "hist_g", "hist_b")
# Ordem de colunas do log (sem prefixo): escalares, categoricas e depois os histogramas.
FEATURE_COLUMNS: tuple[str, ...] = SCALAR_FEATURES + CATEGORICAL_FEATURES + HIST_FEATURES
CHANNELS: tuple[str, ...] = ("l", "r", "g", "b")


@dataclass(frozen=True)
class ImageFeatures:
    width: int
    height: int
    file_bytes: int | None
    format: str | None
    luminance_mean: float
    r_mean: float
    g_mean: float
    b_mean: float
    hist_l: tuple[int, ...]
    hist_r: tuple[int, ...]
    hist_g: tuple[int, ...]
    hist_b: tuple[int, ...]
    luminance_std: float
    sharpness: float
    dominant_channel: str
    bright_clip_frac: float
    dark_clip_frac: float
    luminance_entropy: float
    edge_density: float
    saturation_mean: float
    orientation: str
    dynamic_range: float

    def as_dict(self) -> dict[str, object]:
        """Todas as variaveis, na ordem de `FEATURE_COLUMNS` (histogramas como listas)."""
        out: dict[str, object] = {name: getattr(self, name) for name in FEATURE_COLUMNS}
        for name in HIST_FEATURES:
            out[name] = list(getattr(self, name))
        return out

    def scalars(self) -> dict[str, object]:
        """So as variaveis escalares e categoricas (colunas do reference.csv)."""
        return {name: getattr(self, name) for name in SCALAR_FEATURES + CATEGORICAL_FEATURES}


def mean_from_histogram(counts: Sequence[int]) -> float:
    """mean = sum(i * counts[i]) / sum(counts), como no codigo do curso."""
    total = sum(counts)
    return sum(i * c for i, c in enumerate(counts)) / total if total else 0.0


def std_from_histogram(counts: Sequence[int]) -> float:
    total = sum(counts)
    if not total:
        return 0.0
    mean = mean_from_histogram(counts)
    return math.sqrt(sum(c * (i - mean) ** 2 for i, c in enumerate(counts)) / total)


def entropy_from_histogram(counts: Sequence[int]) -> float:
    """Entropia de Shannon (bits) do histograma normalizado."""
    total = sum(counts)
    if not total:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c)


def percentile_from_histogram(counts: Sequence[int], q: float) -> float:
    """Menor nivel i cuja frequencia acumulada atinge a fracao q."""
    total = sum(counts)
    if not total:
        return 0.0
    cumulative = 0
    for level, count in enumerate(counts):
        cumulative += count
        if cumulative / total >= q:
            return float(level)
    return float(len(counts) - 1)


def _laplacian_variance(luma: np.ndarray) -> float:
    """Variancia do Laplaciano (kernel 3x3 4-vizinhos) do canal L."""
    if luma.shape[0] < 3 or luma.shape[1] < 3:
        return 0.0
    gray = luma.astype(np.int32)
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    return float(lap.var(dtype=np.float64))


def _edge_density(luma: np.ndarray) -> float:
    """Fracao de pixels com gradiente (diferencas finitas) acima do limiar."""
    if luma.shape[0] < 2 or luma.shape[1] < 2:
        return 0.0
    gray = luma.astype(np.int32)
    gx = gray[:-1, 1:] - gray[:-1, :-1]
    gy = gray[1:, :-1] - gray[:-1, :-1]
    return float(np.mean(gx * gx + gy * gy > EDGE_GRADIENT_THRESHOLD**2))


def _orientation(width: int, height: int) -> str:
    if height > width:
        return "portrait"
    if width > height:
        return "landscape"
    return "square"


def uploaded_size(image: Image.Image) -> int | None:
    """Tamanho em bytes do arquivo enviado, lido do arquivo aberto pelo Pillow.

    Chamar ANTES de decodificar a imagem (o Pillow fecha o arquivo depois do `load`).
    """
    fp = getattr(image, "fp", None)
    if fp is None or not hasattr(fp, "seek") or not hasattr(fp, "tell"):
        return None
    position = fp.tell()
    fp.seek(0, 2)
    size = int(fp.tell())
    fp.seek(position)
    return size


def extract_features(image: Image.Image, file_bytes: int | None = None, image_format: str | None = None) -> ImageFeatures:
    """Calcula as variaveis da secao 5.1 do SDD.

    `file_bytes` e o tamanho do upload (use `uploaded_size` antes de decodificar) e
    `image_format` o formato do arquivo; quando omitido usa `image.format`.
    """
    fmt = image_format if image_format is not None else image.format
    rgb = image.convert("RGB")
    luma_image = rgb.convert("L")
    width, height = rgb.size

    raw = rgb.histogram()
    hist_r, hist_g, hist_b = raw[:HIST_BINS], raw[HIST_BINS : 2 * HIST_BINS], raw[2 * HIST_BINS :]
    hist_l = luma_image.histogram()
    means = {"R": mean_from_histogram(hist_r), "G": mean_from_histogram(hist_g), "B": mean_from_histogram(hist_b)}
    dominant = max(means, key=lambda channel: means[channel])  # empate: ordem R, G, B

    luma = np.asarray(luma_image)
    total = sum(hist_l)
    saturation = rgb.convert("HSV").getchannel("S").histogram()

    return ImageFeatures(
        width=width,
        height=height,
        file_bytes=file_bytes,
        format=fmt,
        luminance_mean=mean_from_histogram(hist_l),
        r_mean=means["R"],
        g_mean=means["G"],
        b_mean=means["B"],
        hist_l=tuple(hist_l),
        hist_r=tuple(hist_r),
        hist_g=tuple(hist_g),
        hist_b=tuple(hist_b),
        luminance_std=std_from_histogram(hist_l),
        sharpness=_laplacian_variance(luma),
        dominant_channel=dominant,
        bright_clip_frac=sum(hist_l[BRIGHT_CLIP_LEVEL:]) / total,
        dark_clip_frac=sum(hist_l[: DARK_CLIP_LEVEL + 1]) / total,
        luminance_entropy=entropy_from_histogram(hist_l),
        edge_density=_edge_density(luma),
        saturation_mean=mean_from_histogram(saturation),
        orientation=_orientation(width, height),
        dynamic_range=percentile_from_histogram(hist_l, PERCENTILE_HIGH) - percentile_from_histogram(hist_l, PERCENTILE_LOW),
    )
