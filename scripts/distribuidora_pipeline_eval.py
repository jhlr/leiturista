"""Testa, sobre fotos reais de uma distribuidora de energia parceira, se os sinais que o pipeline MeterOCR já emite
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

Além de H1/H2, o padrão agora cobre as 9 notas mais frequentes da base (DEFAULT_GROUPS)
para uma correlação sinal-x-semântica mais ampla, e mede latência por imagem (segundos)
para benchmark formal de custo operacional.

Uso:
    .venv/bin/python scripts/distribuidora_pipeline_eval.py --n-per-group 40 \
        --json data/distribuidora_campo/pipeline_eval.json
    # ou restringindo os grupos: --groups NA T111 L101

ATENÇÃO: o --json inclui, em "detalhe", número de medidor e nome de arquivo reais do
cliente. Sempre gravar dentro de data/ (gitignored) — nunca em docs/ ou outro caminho
rastreado pelo git.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

from PIL import Image

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "distribuidora_campo"
# Grupos padrão: baseline (NA) + as 9 notas mais frequentes na base (ver
# scripts/distribuidora_stats.py --top_10_notas), cobrindo obstrução (T111),
# substituição (P111), leitura via cliente (L101), duplicidade (B111),
# campanha (L131), função inexistente no medidor/sistema (T161/T181),
# registrador parado (M141) e alocação incorreta (R111).
DEFAULT_GROUPS = ["NA", "T181", "P111", "L101", "T111", "B111", "L131", "T161", "M141", "R111"]


def collect_rows(groups: list[str]) -> dict[str, list[tuple[Path, str]]]:
    """Retorna, por nota, lista de (caminho_da_imagem, numero_do_medidor)."""
    by_group: dict[str, list[tuple[Path, str]]] = {g: [] for g in groups}
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
            if nota in groups and foto != "NA":
                img_path = batch_dir / foto
                if img_path.exists():
                    by_group[nota].append((img_path, r["Numero do medidor"]))
    return by_group


def run_eval(groups: list[str], n_per_group: int, seed: int) -> dict:
    from leiturista.inference import MeterOCR  # import tardio: evita custo se só --help

    random.seed(seed)
    by_group = collect_rows(groups)
    ocr = MeterOCR()

    results: dict[str, list[dict]] = {g: [] for g in groups}
    latencies: list[float] = []
    for group, items in by_group.items():
        sample = random.sample(items, min(n_per_group, len(items)))
        for img_path, medidor in sample:
            img = Image.open(img_path).convert("RGB")
            t0 = time.perf_counter()
            pred = ocr.predict_image(img)
            dt = time.perf_counter() - t0
            latencies.append(dt)
            results[group].append({
                "arquivo": img_path.name,
                "medidor": medidor,
                "leitura": pred.reading,
                "legivel": pred.legible,
                "sharpness": pred.sharpness,
                "flags": pred.flags,
                "latencia_s": round(dt, 2),
            })

    def _pct(vals: list[float], q: float) -> float | None:
        if not vals:
            return None
        s = sorted(vals)
        idx = min(len(s) - 1, int(q * len(s)))
        return round(s[idx], 1)

    summary = {}
    for group, preds in results.items():
        total = len(preds)
        sem_leitura = sum(1 for p in preds if p["leitura"] is None)
        ilegivel = sum(1 for p in preds if not p["legivel"])
        sharp_vals = [p["sharpness"] for p in preds if p["sharpness"] is not None]
        sem_candidato = sum(1 for p in preds if p["sharpness"] is None)

        summary[group] = {
            "amostra": total,
            "disponivel_no_lote": len(by_group[group]),
            "pct_sem_leitura_detectada": round(100 * sem_leitura / total, 1) if total else None,
            "pct_ilegivel_laplaciano": round(100 * ilegivel / total, 1) if total else None,
            "sem_candidato_nitidez": sem_candidato,
            "sharpness_p10_mediana_p90": [_pct(sharp_vals, 0.1), _pct(sharp_vals, 0.5), _pct(sharp_vals, 0.9)],
            "flags_mais_comuns": Counter(f for p in preds for f in p["flags"]).most_common(5),
        }

    latency_summary = {
        "n": len(latencies),
        "media_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "p10_mediana_p90_s": [_pct(latencies, 0.1), _pct(latencies, 0.5), _pct(latencies, 0.9)],
        "min_s": round(min(latencies), 2) if latencies else None,
        "max_s": round(max(latencies), 2) if latencies else None,
    }

    return {"resumo": summary, "latencia": latency_summary, "detalhe": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    ap.add_argument("--n-per-group", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    report = run_eval(args.groups, args.n_per_group, args.seed)
    print(json.dumps({"resumo": report["resumo"], "latencia": report["latencia"]}, ensure_ascii=False, indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nDetalhe completo em {args.json}")


if __name__ == "__main__":
    main()
