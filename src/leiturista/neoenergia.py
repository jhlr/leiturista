"""Dados reais da distribuidora (fotos distribuidora) -> dataset de fine-tune.

Formato de origem (4 lotes `PSP_EXTRATLEITIMPL_*/BaseExtracao_*.csv`, fotos .jpg
360x480):

    Numero do medidor;Posicao do medidor lida;Nota de Leitura Atual;Foto do medidor

A `Nota de Leitura Atual` = 'NA' indica leitura normal (display lido pelo leiturista);
a `Posicao do medidor lida` é o rótulo (leitura registrada). As fotos são do medidor
inteiro (display eletrônico) — diferente do UFPR-AMR, que já são crops do display.

Este módulo:
1. parseia os CSVs (`load_rows`),
2. localiza e recorta o display com o det (PP-OCRv5) normal+invertido e classifica o
   candidato de leitura com o rec (PP-OCRv6_tiny),
3. filtra por nitidez (variância do Laplaciano do crop),
4. grava `labels.csv` (split,image,label) no formato consumido por `build_dataset`
   (pipeline `leiturista train/eval`) + `manifest.csv` de auditoria + mosaicos QA.
"""

from __future__ import annotations

import glob
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from . import paths
from .inference import MeterOCR

distribuidora_DIR = paths.ROOT / "Fotosdistribuidora"
LAPLACIAN_LEGIBLE = 25.0  # referencia do pipeline de inferência
MIN_SHARPNESS_ACCEPT = 10.0  # gate de nitidez frouxo: o match vs rótulo é o filtro real
SPLITS = (("train", 0.8), ("valid", 0.1), ("test", 0.1))


@dataclass
class ReadingCandidate:
    quad: np.ndarray
    crop: np.ndarray  # BGR
    text: str
    conf: float
    sharpness: float
    source: str  # 'normal' | 'inverted'
    match: float  # similaridade dígito-a-dígito (alinhado à direita) vs rótulo


def _digit_match(pred: str, label: str) -> float:
    """Fração de dígitos iguais, alinhando da direita (como `_digit_acc` do eval)."""
    if not pred or not label:
        return 0.0
    n = min(len(pred), len(label))
    aligned = zip(pred[-n:], label[-n:])
    return sum(1 for a, b in aligned if a == b) / len(label)


def load_rows(data_root: Path | str = distribuidora_DIR) -> pd.DataFrame:
    """Lê todos os BaseExtracao_*.csv e devolve linhas com foto existente no disco."""
    data_root = Path(data_root)
    rows: list[pd.DataFrame] = []
    for c in sorted(glob.glob(str(data_root / "*" / "BaseExtracao_*.csv"))):
        d = pd.read_csv(c, sep=";", keep_default_na=False, dtype=str)
        d["lote"] = c.split("\\")[-2] if "\\" in c else c.split("/")[-2]
        d["csv_dir"] = str(Path(c).parent)
        rows.append(d)
    D = pd.concat(rows, ignore_index=True)
    D = D[~D["Foto do medidor"].astype(str).str.upper().isin(["NA", "", "NAN"])].copy()
    D["foto_path"] = D.apply(
        lambda r: str(Path(r["csv_dir"]) / r["Foto do medidor"]), axis=1
    )
    D = D[D["foto_path"].apply(os.path.isfile)]
    D["leitura"] = D["Posicao do medidor lida"].str.strip().str.replace(r"\D", "", regex=True)
    D = D[D["leitura"] != ""]
    return D.reset_index(drop=True)


def _reading_candidate(ocr: MeterOCR, img_bgr: np.ndarray, label: str) -> ReadingCandidate | None:
    """Localiza o crop de leitura do display (det normal+invertido, rec p/ classificar).

    Não roda TrOCR (caro) — a classificação usa o PP-OCRv6_tiny do pipeline.
    Seleção: o display eletrônico (leitura) tem até 8 dígitos e seus dígitos batem
    com a leitura registrada — isso descarta placas de identidade do medidor (~10
    dígitos) e datas (ex.: 03072026).
    """
    label_digits = re.sub(r"\D", "", label)
    if not label_digits:
        return None
    best: ReadingCandidate | None = None
    for source, merge in (("normal", True), ("inverted", False)):
        base = img_bgr if source == "normal" else 255 - img_bgr
        quads = ocr._det_boxes(base)
        merged = ocr._merge_quads(quads) if merge else quads
        for quad, bconf in merged:
            crop = ocr._crop_rotated(base, quad)
            if crop.size == 0:
                continue
            text = ocr._rec_recognize(crop)
            field = ocr._classify(text)
            w = float(quad[:, 0].max() - quad[:, 0].min())
            h = float(quad[:, 1].max() - quad[:, 1].min())
            if field == "leitura" and w < 1.5 * h:
                field = "serial" if (len(text) >= 6 and re.search(r"\d", text)) else "outro"
            if field != "leitura":
                continue
            digits = "".join(re.findall(r"\d", text))
            if not (3 <= len(digits) <= 8):  # display eletrônico: 3..8 dígitos
                continue
            if len(digits) < len(label_digits):
                # display costuma mostrar zeros à esquerda (>= dígitos da leitura);
                # menos dígitos que a leitura = crop parcial/ruído (ex.: "120V").
                continue
            if len(digits) > len(label_digits) and set(digits[:-len(label_digits)]) != {"0"}:
                # dígitos extras à esquerda NÃO podem ser dígitos de valor — só zeros à
                # esquerda. Isso elimina voltagem colada à leitura ("12038034"), crop de
                # placa/ID ("83015067") e rótulos incompatíveis ("300085" vs leitura "85").
                continue
            if re.search(r"\d\s*(?:[vV]{1,2}\b|[aA]\b|hz\b|kwh\b|kw\b)", text, re.I):
                # rótulo de tensão/corrente/unidade (120V, 240V, 60Hz...) — não é display.
                continue
            match = _digit_match(digits, label_digits)
            if best is None or match > best.match:
                best = ReadingCandidate(
                    quad=np.array(quad, dtype=np.int32),
                    crop=crop,
                    text=text,
                    conf=bconf,
                    sharpness=cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var(),
                    source=source,
                    match=match,
                )
    return best if best is not None and best.match >= 0.5 else None


def _make_montage(items: list[tuple[np.ndarray, str]], out: Path, cols: int = 4) -> None:
    """Mosaico de crops com legenda (rótulo/nitidez)."""
    if not items:
        return
    h = max(im.shape[0] for im, _ in items)
    w = max(im.shape[1] for im, _ in items)
    if w > 360 or h > 200:
        scale = min(360 / max(w, 1), 200 / max(h, 1))
        w, h = max(int(w * scale), 1), max(int(h * scale), 1)
    rows_n = int(math.ceil(len(items) / cols))
    pad, cap = 8, 22
    W = cols * (w + pad) + pad
    H = rows_n * (h + pad + cap) + pad
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (im, label) in enumerate(items):
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA) if im.shape[:2] != (h, w) else im
        pil = Image.fromarray(im[:, :, ::-1])
        c, r = i % cols, i // cols
        x, y = pad + c * (w + pad), pad + r * (h + pad + cap)
        canvas.paste(pil, (x, y))
        draw.text((x, y + h + 2), label[:50], fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"QA montage: {out.resolve()} ({len(items)} itens)")


def build_dataset(
    out_dir: Path | str,
    data_root: Path | str = distribuidora_DIR,
    notes: tuple[str, ...] = ("NA",),
    min_sharpness: float = MIN_SHARPNESS_ACCEPT,
    max_samples: int | None = None,
    seed: int = 42,
    qa_n: int = 40,
) -> tuple[Path, pd.DataFrame]:
    """Constrói data/<out>/labels.csv com crops do display (aceitos por nitidez).

    Retorna (caminho de labels.csv, manifest). O split é determinístico (seed).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(data_root)
    rows = rows[rows["Nota de Leitura Atual"].isin(notes)]
    if max_samples:
        rows = rows.sample(min(max_samples, len(rows)), random_state=seed)
    rows = rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    ocr = MeterOCR()
    labels: list[str] = ["split,image,label"]
    manifest: list[dict] = []
    accepted_: list[tuple[str, str]] = []  # (nome do arquivo, leitura)
    accepted_qa: list[tuple[np.ndarray, str]] = []
    rejected_: list[tuple[np.ndarray, str]] = []

    for _i, r in rows.iterrows():
        img = np.array(Image.open(r["foto_path"]).convert("RGB"))[:, :, ::-1]
        cand = _reading_candidate(ocr, img, r["leitura"])
        item = {
            "lote": r["lote"],
            "foto": r["Foto do medidor"],
            "medidor": r["Numero do medidor"],
            "leitura_original": r["leitura"],
            "image": "",
            "split": "",
            "source": cand.source if cand else "",
            "rec": cand.text if cand else "",
            "match": round(cand.match, 2) if cand else "",
            "sharpness": round(cand.sharpness, 1) if cand else "",
        }
        if cand is None:
            item["status"] = "sem_crop"
            manifest.append(item)
            continue
        if cand.sharpness < min_sharpness:
            item["status"] = "borrado"
            manifest.append(item)
            if len(rejected_) < qa_n:
                rejected_.append((cand.crop, f"leitura={r['leitura']} blur={cand.sharpness:.0f}"))
            continue

        name = f"neo_{r['leitura']}_{len(accepted_):05d}.png"
        Image.fromarray(cand.crop[:, :, ::-1]).save(out_dir / name)
        item.update({"status": "aceito", "image": name})
        manifest.append(item)
        accepted_.append((name, r["leitura"]))
        if len(accepted_qa) < qa_n:
            accepted_qa.append((cand.crop, f"leitura={r['leitura']} match={cand.match:.2f} blur={cand.sharpness:.0f}"))

    # split determinístico sobre o pool de aceitos (80/10/10)
    _rng = random.Random(seed)
    _rng.shuffle(accepted_)
    fracs = [int(len(accepted_) * f) for _n, f in SPLITS]
    for k, (name, leitura) in enumerate(accepted_):
        split = "train"
        if k >= sum(fracs[:1]):
            split = "valid"
        if k >= sum(fracs[:2]):
            split = "test"
        labels.append(f"{split},{name},{leitura}")
        for rec in manifest:
            if rec["image"] == name:
                rec["split"] = split
                break

    df = pd.DataFrame(manifest)
    csv = out_dir / "labels.csv"
    csv.write_text("\n".join(labels) + "\n", encoding="utf-8")
    df.to_csv(out_dir / "manifest.csv", index=False)

    qa = out_dir / "_qa"
    _make_montage(accepted_qa, qa / "aceitos.png")
    _make_montage(rejected_, qa / "rejeitados.png")

    total = df["image"].notna().sum()
    print(f"fotos processadas: {len(df)} | aceitas: {total} | borradas: {(df['status']=='borrado').sum()} | sem_crop: {(df['status']=='sem_crop').sum()}")
    print(df["status"].value_counts().to_string())
    print(f"labels.csv -> {csv.resolve()}")
    return csv, df