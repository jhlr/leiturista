"""Rotulagem do Lab 2 (item C): plano de amostragem, atribuição e kappa de Cohen.

Esquema (lab-02/rotulos/esquema.md): por foto, `classe` ∈ {legivel, ilegivel, sem_medidor} e, quando legível,
`leitura` = os dígitos que o rotulador lê no display. Amostra: `n` fotos estratificadas pelos 4 lotes,
`n_dupla` delas rotuladas por DOIS rotuladores, às cegas (cada um salva a própria planilha).

Uso:
    leiturista rotulos-plano --raters ana,bruno,carla,diego,eva   # rotulos/plano.csv (+ recortes sugeridos)
    streamlit run app/rotular.py                                  # cada integrante rotula o seu bloco
    leiturista rotulos-kappa                                      # une as planilhas + kappa por classe
As fotos NÃO vão para o repositório (dado do cliente): o app lê de LEITURISTA_FOTOS_DIR.
"""

from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from . import paths
from .distribuidora import LOTE_SPLIT

ROTULOS_DIR = paths.ROOT / "lab-02" / "rotulos"
PLANO = ROTULOS_DIR / "plano.csv"
CLASSES = ("legivel", "ilegivel", "sem_medidor")


def make_plan(raters: list[str], fotos_root: Path, n: int = 300, n_dupla: int = 50, seed: int = 42) -> list[dict[str, str]]:
    """Amostra estratificada por lote (n/4 por lote) e atribuição: cada foto a 1 rotulador (round-robin),
    `n_dupla` fotos (distribuídas igualmente entre os lotes) a 2 rotuladores distintos."""
    rng = random.Random(seed)
    lotes = sorted(LOTE_SPLIT)
    per = n // len(lotes)
    chosen: list[tuple[str, str]] = []
    for lote in lotes:
        fotos = sorted(p.name for p in (fotos_root / lote).glob("*.jpg"))
        chosen += [(lote, f) for f in rng.sample(fotos, per)]
    rng.shuffle(chosen)
    dup_per = {lote: n_dupla // len(lotes) + (i < n_dupla % len(lotes)) for i, lote in enumerate(lotes)}
    seen: Counter[str] = Counter()
    rows: list[dict[str, str]] = []
    k = 0
    for lote, foto in chosen:
        dupla = seen[lote] < dup_per[lote]
        seen[lote] += dupla
        r1 = raters[k % len(raters)]
        k += 1
        r2 = raters[(k + len(raters) // 2) % len(raters)] if dupla else ""
        if dupla and r2 == r1:
            r2 = raters[(raters.index(r1) + 1) % len(raters)]
        rows.append({"nome_arquivo": foto, "lote": lote, "split": LOTE_SPLIT[lote], "rotulador_1": r1,
                     "rotulador_2": r2, "dupla": str(int(dupla))})
    return rows


def write_plan(rows: list[dict[str, str]], path: Path = PLANO) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Kappa de Cohen entre dois rotuladores (rótulos categóricos)."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(a) | set(b)) / n**2
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def merge_and_kappa(dir_: Path = ROTULOS_DIR) -> dict[str, object]:
    """Une `rotulos_<nome>.csv`, grava `rotulos.csv` (um rótulo por foto: o do rotulador_1; a dupla fica em
    `rotulos_dupla.csv`) e devolve kappa geral e por classe (um-contra-todos) nas fotos em dupla."""
    plan = {r["nome_arquivo"]: r for r in csv.DictReader(open(dir_ / "plano.csv", encoding="utf-8"))}
    by: dict[tuple[str, str], dict[str, str]] = {}
    for f in sorted(dir_.glob("rotulos_*.csv")):
        if f.name in ("rotulos_dupla.csv",):
            continue
        for r in csv.DictReader(open(f, encoding="utf-8")):
            by[(r["nome_arquivo"], r["rotulador"])] = r
    final: list[dict[str, str]] = []
    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    for nome, p in plan.items():
        r1 = by.get((nome, p["rotulador_1"]))
        if r1 is None:
            continue
        final.append({"nome_arquivo": nome, "lote": p["lote"], "split": p["split"], "rotulador": p["rotulador_1"],
                      "classe": r1["classe"], "leitura": r1["leitura"]})
        if p["dupla"] == "1":
            r2 = by.get((nome, p["rotulador_2"]))
            if r2 is not None:
                pairs.append((r1, r2))
    with open(dir_ / "rotulos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["nome_arquivo", "lote", "split", "rotulador", "classe", "leitura"])
        w.writeheader()
        w.writerows(final)
    a, b = [x["classe"] for x, _ in pairs], [y["classe"] for _, y in pairs]
    rep: dict[str, object] = {"rotuladas": len(final), "pares": len(pairs), "kappa_classe": cohen_kappa(a, b),
                              "concordancia": sum(x == y for x, y in zip(a, b)) / max(len(a), 1)}
    rep["kappa_por_classe"] = {c: cohen_kappa([str(x == c) for x in a], [str(y == c) for y in b]) for c in CLASSES}
    leg = [(x["leitura"], y["leitura"]) for x, y in pairs if x["classe"] == y["classe"] == "legivel"]
    rep["leitura_igual_na_dupla"] = (sum(p == q for p, q in leg) / len(leg)) if leg else float("nan")
    return rep
