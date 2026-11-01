"""Divisao referencia x holdout e listas de imagens de trafego (dono unico, SDD Decisao 7).

Referencia = 132 das 165 fotos Praekelt, escolhidas por uma divisao deterministica: nomes
ordenados, embaralhados com seed 0, primeiras 132. O restante (33) e o holdout, o controle
"sem drift". Compartilhado por `scripts/build_reference.py` e `scripts/load_gen.py`.
"""

from __future__ import annotations

import random
from pathlib import Path

from ..paths import DATA_DIR

PRAEKELT_DIR = DATA_DIR / "praekelt_meter_readings_img"
HENRIK_DIR = DATA_DIR / "henrik_energy_meter_img"
LCD_DIR = DATA_DIR / "utility_meters_lcd"
N_REFERENCE = 132
DEFAULT_SEED = 0
TRAFFIC_SETS: tuple[str, ...] = ("holdout", "henrik", "lcd")


def split_reference_images(
    images_dir: Path = PRAEKELT_DIR, seed: int = DEFAULT_SEED, n_reference: int = N_REFERENCE
) -> tuple[list[Path], list[Path]]:
    """Devolve (referencia, holdout) a partir dos `praekelt_*.jpg` de `images_dir`."""
    names = sorted(images_dir.glob("praekelt_*.jpg"), key=lambda p: p.name)
    if len(names) < n_reference:
        raise ValueError(f"{images_dir} tem {len(names)} imagens, menos que a referencia ({n_reference})")
    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:n_reference], shuffled[n_reference:]


def traffic_images(traffic: str, seed: int = DEFAULT_SEED, data_dir: Path | None = None) -> list[Path]:
    """Imagens de um conjunto de trafego: `holdout`, `henrik` (recursivo) ou `lcd`."""
    base = data_dir if data_dir is not None else DATA_DIR
    if traffic == "holdout":
        return split_reference_images(base / PRAEKELT_DIR.name, seed)[1]
    if traffic == "henrik":
        return sorted((base / HENRIK_DIR.name).rglob("*.png"))
    if traffic == "lcd":
        return sorted((base / LCD_DIR.name).glob("*.jpg"))
    raise ValueError(f"trafego desconhecido: {traffic!r} (use um de {TRAFFIC_SETS})")
