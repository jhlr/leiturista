"""Preparação de dados do subprojeto CV: extração UFPR-AMR e datasets de fine-tune.

Fluxo padrão:
1. `extract_ufpr_amr()`  — extrai imagens dos parquets UFPR-AMR + `labels.csv`.
2. `load_labels()`       — lê o CSV.
3. `build_dataset()`     — Dataset HuggingFace (pixel_values + labels) pronto p/ treino.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
from datasets import Dataset, disable_caching

from . import paths

disable_caching()


def _ext(b: bytes) -> str:
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if b[:2] == b"\xff\xd8":
        return "jpg"
    return "bin"


def extract_ufpr_amr(out: Path | str | None = None, per_split: int | None = None) -> Path:
    """Extrai todas as imagens UFPR-AMR (test/train/valid) para `out`.

    Retorna o caminho de `labels.csv` (colunas: split, image, label).
    """
    out = Path(out) if out else paths.FINETUNE_DIR
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    rows_total = 0
    labels: list[str] = ["split,image,label"]
    for split in ("test", "train", "valid"):
        query = (
            f"SELECT image, ground_truth FROM read_parquet("
            f"'{paths.UFPR_AMR_DIR}/{split}-00000-of-00001-*.parquet')"
        )
        if per_split:
            query += f" LIMIT {per_split}"
        rows = con.execute(query).fetchall()
        for i, (img, gt_raw) in enumerate(rows):
            gt = json.loads(gt_raw).get("gt_parse", "x")
            name = f"ufpr_{split}_{i:02d}_{gt}.{_ext(img['bytes'])}"
            (out / name).write_bytes(img["bytes"])
            labels.append(f"{split},{name},{gt}")
        rows_total += len(rows)
        print(f"{split}: {len(rows)}")
    csv = out / "labels.csv"
    csv.write_text("\n".join(labels) + "\n")
    print(f"total: {rows_total} -> {out.resolve()} (+ labels.csv)")
    return csv


def load_labels(data_dir: Path | str, split: str) -> pd.DataFrame:
    """Lê `labels.csv` filtrando por split; devolve colunas (image, label)."""
    csv = Path(data_dir) / "labels.csv"
    df = pd.read_csv(csv)
    df = df[df["split"] == split][["image", "label"]].reset_index(drop=True)
    df["image"] = df["image"].map(lambda n: str((Path(data_dir) / n).resolve()))
    return df


def build_dataset(
    data_dir: Path | str,
    split: str,
    processor,
    max_len: int = 16,
    max_samples: int | None = None,
) -> Dataset:
    """Monta um Dataset HF com `pixel_values` (imagem) e `labels` (token da leitura)."""
    df = load_labels(data_dir, split)
    if max_samples:
        df = df.head(max_samples)
    ds = Dataset.from_pandas(df)

    def prep(ex):
        enc = processor(
            images=ex["image"],
            text=str(ex["label"]),
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return {"pixel_values": enc["pixel_values"][0], "labels": enc["labels"][0]}

    return ds.map(prep, remove_columns=["image", "label"])
