"""Estatísticas agregadas da base real Neoenergia PE (data/neoenergia_pe/).

Junta os 4 lotes diários (`BaseExtracao_<data>_Dia.csv`) com o catálogo de ocorrências
(`DESCRIÇÃO NOTAS LEITURISTAS X SOLICITAÇÃO DE FOTO.xlsx`) e reporta, para o conjunto
completo (não apenas 1 lote): volume, distribuição de notas, taxa de NA-com-foto, taxa de
medidor duplo (`A/B`), auditoria de imagens órfãs/ausentes e cruzamento SIM(exige foto) x
Foto=NA.

Uso:
    .venv/bin/python scripts/neoenergia_stats.py [--json out.json]
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "neoenergia_pe"
CATALOG_XLSX = DATA_DIR / "DESCRIÇÃO NOTAS LEITURISTAS X SOLICITAÇÃO DE FOTO.xlsx"


def load_catalog(path: Path) -> dict[str, dict[str, str]]:
    z = zipfile.ZipFile(path)
    shared = re.findall(r"<t[^>]*>(.*?)</t>", z.read("xl/sharedStrings.xml").decode("utf-8"), re.S)
    strings = [html.unescape(re.sub(r"<[^>]+>", "", s)) for s in shared]
    sheet = z.read("xl/worksheets/sheet1.xml").decode("utf-8")
    rows = re.findall(r"<row[^>]*>(.*?)</row>", sheet, re.S)
    catalog: dict[str, dict[str, str]] = {}
    for row in rows[1:]:
        cells = re.findall(r'<c[^>]*t="s"[^>]*><v>(\d+)</v></c>', row)
        if len(cells) == 3:
            nota, desc, exige = (strings[int(i)] for i in cells)
            catalog[nota] = {"descricao": desc, "exige_foto": exige}
    return catalog


def load_batch(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, encoding="latin-1") as f:
        return list(csv.DictReader(f, delimiter=";"))


def analyze() -> dict:
    catalog = load_catalog(CATALOG_XLSX)
    batches = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())

    all_rows: list[dict[str, str]] = []
    per_batch: dict[str, dict] = {}

    for batch_dir in batches:
        csvs = list(batch_dir.glob("BaseExtracao_*.csv"))
        if not csvs:
            continue
        rows = load_batch(csvs[0])
        images_on_disk = {p.name for p in batch_dir.glob("*.jpg")}
        referenced = {r["Foto do medidor"].strip() for r in rows if r["Foto do medidor"].strip() != "NA"}

        for r in rows:
            r["_batch"] = batch_dir.name
        all_rows.extend(rows)

        per_batch[batch_dir.name] = {
            "linhas_csv": len(rows),
            "imagens_disco": len(images_on_disk),
            "referenciadas_no_csv": len(referenced),
            "orfas_disco_sem_csv": len(images_on_disk - referenced),
            "referenciadas_ausentes_disco": len(referenced - images_on_disk),
        }

    total = len(all_rows)
    na_nota = [r for r in all_rows if r["Nota de Leitura Atual"].strip() == "NA"]
    na_nota_com_foto = [r for r in na_nota if r["Foto do medidor"].strip() != "NA"]
    slash = [r for r in all_rows if "/" in r["Numero do medidor"]]
    slash_notas = Counter(r["Nota de Leitura Atual"].strip() for r in slash)

    exige_sim_notas = {n for n, v in catalog.items() if v["exige_foto"] == "SIM"}
    exige_sim_rows = [r for r in all_rows if r["Nota de Leitura Atual"].strip() in exige_sim_notas]
    exige_sim_sem_foto = [r for r in exige_sim_rows if r["Foto do medidor"].strip() == "NA"]

    nota_counts = Counter(r["Nota de Leitura Atual"].strip() for r in all_rows)

    report = {
        "lotes": list(per_batch.keys()),
        "total_linhas_csv": total,
        "total_imagens_disco": sum(v["imagens_disco"] for v in per_batch.values()),
        "por_lote": per_batch,
        "catalogo": {
            "total_notas": len(catalog),
            "exige_foto_sim": sum(1 for v in catalog.values() if v["exige_foto"] == "SIM"),
            "exige_foto_nao": sum(1 for v in catalog.values() if v["exige_foto"] == "NÃO"),
        },
        "nota_na": {
            "total": len(na_nota),
            "pct_do_total": round(100 * len(na_nota) / total, 1) if total else 0,
            "com_foto": len(na_nota_com_foto),
            "pct_com_foto": round(100 * len(na_nota_com_foto) / len(na_nota), 1) if na_nota else 0,
        },
        "medidor_duplo_barra": {
            "total": len(slash),
            "pct_do_total": round(100 * len(slash) / total, 1) if total else 0,
            "top_notas": slash_notas.most_common(5),
        },
        "exige_foto_sim_mas_sem_foto": {
            "total_exige_sim": len(exige_sim_rows),
            "sem_foto": len(exige_sim_sem_foto),
            "pct": round(100 * len(exige_sim_sem_foto) / len(exige_sim_rows), 1) if exige_sim_rows else 0,
            "top_notas": Counter(r["Nota de Leitura Atual"].strip() for r in exige_sim_sem_foto).most_common(5),
        },
        "top_10_notas": nota_counts.most_common(10),
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None, help="grava o relatório em JSON")
    args = ap.parse_args()

    report = analyze()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
