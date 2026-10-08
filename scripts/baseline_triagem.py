"""Baseline da triagem verde/amarelo/vermelho (Abordagem A, `docs/triagem-vermelho-amarelo-verde.md`).

Responde: que % das fotos da distribuidora cai no AMARELO com as regras heurísticas mais
simples possíveis, usando só o que já existe (catálogo de notas + pipeline `MeterOCR`).

Regras (ordem de precedência, a primeira que casa decide):
  R1  nota NÃO exige foto                                   -> verde    (sem olhar a imagem)
  R2  nota exige foto e a foto está ausente (NA / sem arquivo) -> vermelho (sem olhar a imagem)
  R2b nota fora do catálogo (na prática "NA") e sem foto     -> amarelo  (catálogo não diz se exige)
  R3  nota exige, foto presente, pipeline não detecta NENHUMA caixa de texto -> vermelho
  R4  leitura detectada == leitura digitada pelo leiturista E legível (Laplaciano) -> verde
  O leitor de dígitos é intercambiável (`--reader`): `ppocr` (pipeline atual: PP-OCRv6/TrOCR) ou
  `crnn` (CRNNDigitos do professor, relendo o recorte que o det achou). Detecção e nitidez são
  as mesmas nos dois, então a diferença de cor é só do leitor.
  R5  todo o resto                                          -> amarelo
      motivos: leitura_nao_detectada | leitura_diverge_da_digitada | ilegivel_laplaciano

R1/R2 valem para a população inteira (sem OCR). R3-R5 precisam rodar o pipeline (~4 s/foto em
CPU), então rodam numa amostra aleatória (seed fixa) das fotos que "precisam de imagem" e o
resultado é extrapolado para a população com IC de Wilson 95%.

Uso:
    uv run python scripts/baseline_triagem.py census                 # só R1/R2, população toda
    uv run python scripts/baseline_triagem.py sample --n 500         # roda OCR, retomável
    uv run python scripts/baseline_triagem.py crnn                   # relê os recortes com o CRNN
    uv run python scripts/baseline_triagem.py report --reader ppocr  # ou crnn; grava JSON agregado

ATENÇÃO: o CSV por foto (`data/analise/baseline_triagem_amostra.csv`) tem medidor e arquivo
reais; fica em data/ (gitignored). O JSON agregado (sem identificadores) vai para docs/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

from PIL import Image

from distribuidora_stats import CATALOG_XLSX, DATA_DIR, load_batch, load_catalog

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = ROOT / "data" / "analise" / "baseline_triagem_amostra.csv"
BASE_SAMPLE = SAMPLE_CSV  # amostra da baseline (mesmas 500 fotos), usada pela cena
SAMPLE_CRNN_CSV = ROOT / "data" / "analise" / "baseline_triagem_amostra_crnn.csv"
CROPS_DIR = ROOT / "data" / "analise" / "crops_baseline"  # recorte da leitura (dado do cliente, gitignored)
REPORT_JSON = ROOT / "docs" / "2026-10-01_baseline_triagem.json"  # sufixo _<leitor> adicionado ao gravar
SCENE_CSV = ROOT / "data" / "analise" / "baseline_triagem_cena.csv"  # P(tem medidor) das fotos da amostra
LAPLACIAN_LEGIBLE = 25.0  # mesmo limiar do pipeline (src/leiturista/inference.py)
FIELDS = ["lote", "foto", "nota", "leitura_digitada", "leitura_ocr", "n_caixas", "nitidez",
          "legivel", "cor", "regra", "motivo", "seg"]


def digits(s: str) -> str:
    return "".join(c for c in (s or "") if c.isdigit()).lstrip("0")


def load_population() -> list[dict]:
    """Uma linha por registro dos 4 lotes, já com a decisão de R1/R2 quando não precisa de imagem."""
    catalog = load_catalog(CATALOG_XLSX)
    out: list[dict] = []
    for batch_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        csvs = list(batch_dir.glob("BaseExtracao_*.csv"))
        if not csvs:
            continue
        for r in load_batch(csvs[0]):
            nota = r["Nota de Leitura Atual"].strip()
            foto = r["Foto do medidor"].strip()
            path = batch_dir / foto
            tem_foto = foto.upper() not in {"NA", ""} and path.is_file()
            exige = catalog.get(nota, {}).get("exige_foto", "?")
            row = {"lote": batch_dir.name, "foto": foto, "nota": nota, "exige": exige,
                   "tem_foto": tem_foto, "path": str(path),
                   "leitura_digitada": r["Posicao do medidor lida"].strip()}
            if exige == "NÃO":
                row.update(cor="verde", regra="R1")
            elif exige == "SIM" and not tem_foto:
                row.update(cor="vermelho", regra="R2")
            elif exige == "SIM":
                row.update(cor=None, regra=None)  # precisa de imagem
            elif tem_foto:  # nota fora do catálogo (na prática "NA", leitura normal): precisa de imagem
                row.update(cor=None, regra=None)
            else:  # fora do catálogo e sem foto: não há regra de "exige foto", então não rejeita sozinho
                row.update(cor="amarelo", regra="R2b")
            out.append(row)
    return out


def classify(n_caixas: int, leitura: str, alvo: str, legivel: bool) -> tuple[str, str, str]:
    if n_caixas == 0:
        return "vermelho", "R3", "nenhuma_caixa_de_texto"
    if leitura and leitura == alvo and legivel:
        return "verde", "R4", "leitura_bate_e_legivel"
    if not leitura:
        return "amarelo", "R5", "leitura_nao_detectada"
    if leitura != alvo:
        return "amarelo", "R5", "leitura_diverge_da_digitada"
    return "amarelo", "R5", "ilegivel_laplaciano"


def crop_path(lote: str, foto: str) -> Path:
    return CROPS_DIR / f"{lote}__{foto}.png"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def cmd_census(_: argparse.Namespace) -> None:
    pop = load_population()
    n = len(pop)
    cnt = Counter((r["regra"] or "precisa_imagem") for r in pop)
    print(f"registros: {n}")
    for k, v in cnt.most_common():
        print(f"  {k:16s} {v:6d}  {100 * v / n:5.1f}%")
    sem_catalogo = sum(1 for r in pop if r["exige"] == "?")
    print(f"notas fora do catálogo: {sem_catalogo}")


def cmd_sample(args: argparse.Namespace) -> None:
    from leiturista.inference import MeterOCR  # import tardio: carrega modelos pesados

    pop = load_population()
    pool = [r for r in pop if r["regra"] is None]
    random.Random(args.seed).shuffle(pool)
    pool = pool[: args.n]

    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if SAMPLE_CSV.exists():
        with open(SAMPLE_CSV, encoding="utf-8") as f:
            done = {(r["lote"], r["foto"]) for r in csv.DictReader(f)}
    new_file = not SAMPLE_CSV.exists()
    ocr = MeterOCR()
    t0 = time.time()
    with open(SAMPLE_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for i, r in enumerate(pool, 1):
            if (r["lote"], r["foto"]) in done:
                continue
            t1 = time.time()
            pred = ocr.predict_image(Image.open(r["path"]).convert("RGB"))
            leitura = digits(pred.reading or "")
            alvo = digits(r["leitura_digitada"])
            legivel = bool(pred.legible)
            cor, regra, motivo = classify(len(pred.boxes), leitura, alvo, legivel)
            box = ocr._best_reading(pred.boxes)
            if box is not None:
                CROPS_DIR.mkdir(parents=True, exist_ok=True)
                box.crop.save(crop_path(r["lote"], r["foto"]))
            w.writerow({"lote": r["lote"], "foto": r["foto"], "nota": r["nota"],
                        "leitura_digitada": r["leitura_digitada"], "leitura_ocr": pred.reading or "",
                        "n_caixas": len(pred.boxes),
                        "nitidez": "" if pred.sharpness is None else round(float(pred.sharpness), 2),
                        "legivel": legivel, "cor": cor, "regra": regra, "motivo": motivo,
                        "seg": round(time.time() - t1, 2)})
            f.flush()
            if i % 20 == 0:
                print(f"{i}/{len(pool)}  {time.time() - t0:.0f}s", flush=True)


def cmd_crnn(args: argparse.Namespace) -> None:
    """Relê os recortes salvos com o CRNNDigitos e regrava a amostra como baseline do leitor `crnn`."""
    from leiturista.crnn import load_crnn, read_digits

    model = load_crnn(args.ckpt) if args.ckpt else load_crnn()
    with open(SAMPLE_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(SAMPLE_CRNN_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            cp = crop_path(r["lote"], r["foto"])
            leitura = digits(read_digits(Image.open(cp), model)) if cp.is_file() else ""
            alvo = digits(r["leitura_digitada"])
            cor, regra, motivo = classify(int(r["n_caixas"]), leitura, alvo, r["legivel"] == "True")
            w.writerow({**r, "leitura_ocr": leitura, "cor": cor, "regra": regra, "motivo": motivo})
    print(f"{len(rows)} fotos relidas -> {SAMPLE_CRNN_CSV}")


def cmd_report(args: argparse.Namespace) -> None:
    pop = load_population()
    n = len(pop)
    fixos = Counter(r["cor"] for r in pop if r["regra"] is not None)
    n_img = sum(1 for r in pop if r["regra"] is None)
    with open(SAMPLE_CRNN_CSV if args.reader == "crnn" else SAMPLE_CSV, encoding="utf-8") as f:
        amostra = list(csv.DictReader(f))
    m = len(amostra)
    ca = Counter(r["cor"] for r in amostra)
    motivos = Counter(r["motivo"] for r in amostra if r["cor"] == "amarelo")
    seg = sorted(float(r["seg"]) for r in amostra)

    rep: dict = {
        "leitor": args.reader,
        "registros": n,
        "decididos_sem_imagem": {"verde_R1": fixos["verde"], "vermelho_R2": fixos["vermelho"], "amarelo_R2b": fixos["amarelo"]},
        "precisam_de_imagem": n_img,
        "amostra_n": m,
        "amostra_cores": dict(ca),
        "amostra_motivos_amarelo": dict(motivos),
        "latencia_s": {"media": round(sum(seg) / m, 2), "p50": seg[m // 2], "p90": seg[int(m * 0.9)]},
    }
    pop_cor = {}
    for cor in ("verde", "amarelo", "vermelho"):
        base = fixos[cor]
        k = ca[cor]
        lo, hi = wilson(k, m)
        est = base + (k / m) * n_img
        pop_cor[cor] = {
            "pct_estimado": round(100 * est / n, 1),
            "ic95_pct": [round(100 * (base + lo * n_img) / n, 1), round(100 * (base + hi * n_img) / n, 1)],
        }
    rep["populacao"] = pop_cor
    # sensibilidade: amarelo se a coerência com a digitada não fosse exigida
    sem_diverge = sum(1 for r in amostra if r["cor"] == "amarelo" and r["motivo"] != "leitura_diverge_da_digitada")
    rep["sensibilidade_amarelo_pct_sem_regra_diverge"] = round(100 * (fixos["amarelo"] + (sem_diverge / m) * n_img) / n, 1)
    if SCENE_CSV.exists():  # "sem medidor" na foto vira vermelho em vez de amarelo
        sem = {(r["lote"], r["foto"]) for r in csv.DictReader(open(SCENE_CSV, encoding="utf-8")) if r["sem_medidor"] == "True"}
        k_am = sum(1 for r in amostra if r["cor"] == "amarelo" and (r["lote"], r["foto"]) not in sem)
        rep["com_cena"] = {"fotos_sem_medidor": len(sem),
                           "amarelo_pct": round(100 * (fixos["amarelo"] + (k_am / m) * n_img) / n, 1),
                           "amarelo_amostra": k_am}
    out = REPORT_JSON.with_name(REPORT_JSON.stem + f"_{args.reader}.json") if args.tag else REPORT_JSON.with_name(f"2026-10-01_baseline_triagem_{args.reader}.json")
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


def apply_tag(tag: str) -> None:
    """`--tag X` isola as saídas de uma configuração (det/leitor diferentes) sem sobrescrever a baseline."""
    global SAMPLE_CSV, SAMPLE_CRNN_CSV, CROPS_DIR, REPORT_JSON
    if not tag:
        return
    SAMPLE_CSV = SAMPLE_CSV.with_name(f"baseline_triagem_amostra_{tag}.csv")
    SAMPLE_CRNN_CSV = SAMPLE_CRNN_CSV.with_name(f"baseline_triagem_amostra_crnn_{tag}.csv")
    CROPS_DIR = CROPS_DIR.with_name(f"crops_{tag}")
    REPORT_JSON = REPORT_JSON.with_name(f"2026-10-08_triagem_{tag}.json")


def cmd_scene(_: argparse.Namespace) -> None:
    """P(tem medidor) (classificador de cena) para as fotos da amostra baseline; reaproveitado por todas as tags."""
    from leiturista.scene import predict_scene

    pop = {(r["lote"], r["foto"]): r["path"] for r in load_population()}
    with open(BASE_SAMPLE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(SCENE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lote", "foto", "p_medidor", "sem_medidor"])
        for r in rows:
            p, sem = predict_scene(Image.open(pop[(r["lote"], r["foto"])]))
            w.writerow([r["lote"], r["foto"], round(p, 4), sem])
    print(f"{len(rows)} fotos -> {SCENE_CSV}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="sufixo das saídas (uma config por tag)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("census").set_defaults(fn=cmd_census)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=500)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_sample)
    c = sub.add_parser("crnn", help="relê os recortes salvos com o CRNNDigitos")
    c.add_argument("--ckpt", default=None, help="checkpoint do CRNN (default: models/crnn_digitos.pt)")
    c.set_defaults(fn=cmd_crnn)
    sub.add_parser("scene", help="P(tem medidor) da amostra (classificador de cena)").set_defaults(fn=cmd_scene)
    r = sub.add_parser("report")
    r.add_argument("--reader", choices=["ppocr", "crnn"], default="ppocr")
    r.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    apply_tag(args.tag)
    args.fn(args)


if __name__ == "__main__":
    main()
