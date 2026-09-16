"""Gera imagens sintéticas de medidor para README/demo (sem foto real do cliente).

Uso:
  .venv/bin/python scripts/gen_sample_images.py

Regrava `samples/exemplo_0{1..4}.png` — determinístico (seed fixa), para o
`curl` de exemplo do README não depender de uma foto real de ninguém.
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parents[1] / "samples"
SEED = 42

SAMPLES = [
    ("exemplo_01.png", "017355", 0),
    ("exemplo_02.png", "204921", 3),
    ("exemplo_03.png", "0006234", -2),
    ("exemplo_04.png", "883012", 0),
]


def _draw_meter(reading: str, angle: int, rng: random.Random) -> Image.Image:
    w, h = 480, 360
    body = tuple(rng.randint(150, 175) for _ in range(3))
    img = Image.new("RGB", (w, h), body)
    draw = ImageDraw.Draw(img)

    # carcaça oval simplificada
    draw.ellipse((30, 30, w - 30, h - 30), outline=(90, 90, 90), width=6)

    # display LCD (preto, dígitos brancos)
    disp_w, disp_h = 260, 60
    x0, y0 = (w - disp_w) // 2, (h - disp_h) // 2
    draw.rectangle((x0, y0, x0 + disp_w, y0 + disp_h), fill=(15, 15, 15))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 40)
    except OSError:
        font = ImageFont.load_default()
    draw.text((x0 + 14, y0 + 8), reading, fill=(230, 230, 230), font=font)

    # placa/serial abaixo (texto pequeno, ruído leve)
    draw.text((x0, y0 + disp_h + 20), f"SN {rng.randint(10_000_000, 99_999_999)}",
               fill=(40, 40, 40), font=ImageFont.load_default())

    if angle:
        img = img.rotate(angle, expand=True, fillcolor=body)
    return img


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    for name, reading, angle in SAMPLES:
        img = _draw_meter(reading, angle, rng)
        img.save(OUT_DIR / name)
        print(f"{OUT_DIR / name} ({reading})")


if __name__ == "__main__":
    main()
