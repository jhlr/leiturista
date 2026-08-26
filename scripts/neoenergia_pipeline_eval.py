"""Testa, sobre fotos reais da Neoenergia PE, se os sinais que o pipeline MeterOCR já emite
(leitura não detectada, legibilidade via Laplaciano) se correlacionam com o que a semântica da
ocorrência registrada pelo leiturista sugere — sem depender do rótulo de aceite/rejeição que
ainda não temos.

Hipóteses testadas (grupos amostrados do CSV real, coluna "Nota de Leitura Atual"):
  H1: T111 ("Caixa/Tampa Embaçada/Danificada/Ausente - c/ Leitura") deve ter taxa MAIOR de
      "leitura não detectada"/ilegibilidade que o grupo NA (leitura normal, sem obstrução
      relatada) — a nota já documenta obstrução visual da caixa de medição.
  H2: L101 ("Leitura Informada Pelo Cliente") deve ter taxa MAIOR de "leitura não detectada"
      que o grupo NA — se o leiturista precisou que o cliente informasse a leitura, é porque
      não conseguiu obtê-la da foto.
  Baseline: NA (nota normal, leitura esperada e presumivelmente obtida sem problema).

Uso:
    .venv/bin/python scripts/neoenergia_pipeline_eval.py --n-per-group 60 --json out.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "neoenergia_pe"
GROUPS = ["NA", "T111", "L101"]


def collect_rows() -> dict[str, list[tuple[Path, str]]]:
    """Retorna, por nota, lista de (caminho_da_imagem, numero_do_medidor)."""
    by_group: dict[str, list[tuple[Path, str]]] = {g: [] for g in GROUPS}
    for batch_dir in sorted(DATA_DIR.iterdir()):
        if not batch_dir.is_dir():
            continue
        csvs = list(batch_dir.glob("BaseExtracao_*.csv"))
        if not csvs:
            continue
        rows = list(csv.DictReader(open(csvs[0], encoding="latin-1"), delimiter=";"))
        for r in rows:
            nota = r["Nota de Leitura Atual"].strip()
            foto = r["Foto do medidor"].strip()
            if nota in GROUPS and foto != "NA":
                img_path = batch_dir / foto
                if img_path.exists():
                    by_group[nota].append((img_path, r["Numero do medidor"]))
    return by_group


def run_eval(n_per_group: int, seed: int) -> dict:
    from leiturista.inference import MeterOCR  # import tardio: evita custo se só --help

    random.seed(seed)
    by_group = collect_rows()
    ocr = MeterOCR()

    results: dict[str, list[dict]] = {g: [] for g in GROUPS}
    for group, items in by_group.items():
        sample = random.sample(items, min(n_per_group, len(items)))
        for img_path, medidor in sample:
            img = Image.open(img_path).convert("RGB")
            pred = ocr.predict_image(img)
            results[group].append({
                "arquivo": img_path.name,
                "medidor": medidor,
                "leitura": pred.reading,
                "legivel": pred.legible,
                "flags": pred.flags,
            })

    summary = {}
    for group, preds in results.items():
        total = len(preds)
        sem_leitura = sum(1 for p in preds if p["leitura"] is None)
        ilegivel = sum(1 for p in preds if not p["legivel"])
        summary[group] = {
            "amostra": total,
            "disponivel_no_lote": len(by_group[group]),
            "pct_sem_leitura_detectada": round(100 * sem_leitura / total, 1) if total else None,
            "pct_ilegivel_laplaciano": round(100 * ilegivel / total, 1) if total else None,
            "flags_mais_comuns": Counter(f for p in preds for f in p["flags"]).most_common(5),
        }

    return {"resumo": summary, "detalhe": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-group", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    report = run_eval(args.n_per_group, args.seed)
    print(json.dumps(report["resumo"], ensure_ascii=False, indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nDetalhe completo em {args.json}")


if __name__ == "__main__":
    main()
