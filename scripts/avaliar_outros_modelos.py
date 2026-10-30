"""Avaliação dos modelos NÃO-leitores (classificador de cena e de visor) na base v2.

Contexto: os datasets originais desses dois modelos (`data/distribuidora_campo_rotulado.csv`
com rótulos fracos do LLM, e `data/visor_cls/candidatos.csv`) NÃO estão no disco, então não dá
para reproduzir o test split original. Este script mede duas coisas que dão para medir com os
rótulos que existem hoje:

1. `--part scene` — taxa de "vermelho falso" em fotos onde o medidor/visor comprovadamente
   aparece e foi lido (`status=aceito` no manifest da v2). Se o pipeline achou e leu o display,
   existe medidor visível ⇒ qualquer "sem medidor"/"ilegível" ali é erro. Também mede quantos
   `sem_crop` (falha de detecção) o classificador de cena pega — que é o uso operacional dele.

2. `--part visor` — avaliação ponta a ponta do seletor de candidato no split de test: mesmo
   conjunto de caixas + mesmo leitor (CRNN v2), variando só a REGRA de escolha:
     A) `visor_cls` (argmax P(visor) >= 0.5)
     B) regra antiga ("mais dígitos")
     C) A com fallback para B (comportamento real do `_best_reading`)
     D) oráculo (existe alguma caixa cujo CRNN bate a leitura?)
   A diferença A vs B isola a qualidade do classificador de visor.

Uso:
    .venv\\Scripts\\python.exe scripts\\avaliar_outros_modelos.py --part all
    .venv\\Scripts\\python.exe scripts\\avaliar_outros_modelos.py --part visor --limit 60
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import time
from pathlib import Path

# ATENÇÃO: as envs de inferência são lidas no import de `leiturista.inference`.
os.environ.setdefault("LEITURISTA_DET2_ONNX", "models/pp_ocr_v5_mobile_det_visor_onnx/inference.onnx")
os.environ.setdefault("LEITURISTA_DET2_THRESH", "0.2")
os.environ.setdefault("LEITURISTA_DET2_BOX_THRESH", "0.2")

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "neoenergia_amr_v2"
FOTOS = ROOT / "data" / "FotosNeoenergia"
OUT = ROOT / "logs" / "avaliacao_outros_modelos"
MODELS = ROOT / "models"

VISOR_PT = MODELS / "visor_cls.pt"
SCENE_MEDIDOR = MODELS / "scene_medidor.pt"
SCENE_LEGIVEL = MODELS / "scene_legivel.pt"
CRNN_PT = MODELS / "crnn_dist_v2_ft.pt"


def ler_manifest() -> list[dict[str, str]]:
    with open(DATA / "manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def foto_path(r: dict[str, str]) -> Path:
    return FOTOS / r["lote"] / r["foto"]


# ---------------------------------------------------------------- cena
def _load_cls(path: Path) -> tuple[torch.nn.Module, float]:
    from leiturista.scene import build_model

    ck = torch.load(path, map_location="cpu")
    m = build_model(pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), float(ck["threshold"])


@torch.no_grad()
def _scores(path: Path, fotos: list[Path], batch: int = 32) -> np.ndarray:
    from leiturista.scene import _eval_tf

    model, _thr = _load_cls(path)
    out: list[np.ndarray] = []
    for i in range(0, len(fotos), batch):
        ch = fotos[i:i + batch]
        x = torch.stack([_eval_tf(Image.open(p).convert("RGB")) for p in ch])
        out.append(torch.sigmoid(model(x).squeeze(1)).numpy())
    return np.concatenate(out)


def _montage(items: list[tuple[Path, str]], out: Path, cols: int = 6, tw: int = 200) -> None:
    if not items:
        return
    th = int(tw * 4 / 3)
    rows_n = (len(items) + cols - 1) // cols
    pad, cap = 6, 18
    canvas = Image.new("RGB", (cols * (tw + pad) + pad, rows_n * (th + pad + cap) + pad), "white")
    d = ImageDraw.Draw(canvas)
    for i, (p, label) in enumerate(items):
        im = Image.open(p).convert("RGB").resize((tw, th))
        c, r = i % cols, i // cols
        x, y = pad + c * (tw + pad), pad + r * (th + pad + cap)
        canvas.paste(im, (x, y))
        d.text((x + 2, y + th + 2), label[:34], fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"mosaico: {out}  ({len(items)} fotos)")


def parte_scene(limit_sem: int, seed: int) -> dict:
    man = ler_manifest()
    aceitos = [r for r in man if r["status"] == "aceito"]
    borrados = [r for r in man if r["status"] == "borrado"]
    sem_crop = [r for r in man if r["status"] == "sem_crop"]
    rng = np.random.default_rng(seed)
    sem_amostra = [sem_crop[i] for i in rng.choice(len(sem_crop), min(limit_sem, len(sem_crop)), replace=False)]

    res: dict = {}

    # medidor: positivo garantido = aceito; negativo fraco = sem_crop (é falha de detecção,
    # não "sem medidor" — reportamos como cobertura operacional, não como recall de classe)
    p_med_ace = _scores(SCENE_MEDIDOR, [foto_path(r) for r in aceitos])
    thr_med = _load_cls(SCENE_MEDIDOR)[1]
    p_med_bor = _scores(SCENE_MEDIDOR, [foto_path(r) for r in borrados])
    p_med_sem = _scores(SCENE_MEDIDOR, [foto_path(r) for r in sem_amostra])

    # legivel: positivo garantido = aceito com match >= 0.8
    match = np.array([float(r["match"] or 0) for r in aceitos])
    p_leg_ace = _scores(SCENE_LEGIVEL, [foto_path(r) for r in aceitos])
    thr_leg = _load_cls(SCENE_LEGIVEL)[1]
    p_leg_bor = _scores(SCENE_LEGIVEL, [foto_path(r) for r in borrados])
    p_leg_sem = _scores(SCENE_LEGIVEL, [foto_path(r) for r in sem_amostra])

    res["scene_medidor"] = {
        "threshold": thr_med,
        "aceito_n": len(aceitos),
        "aceito_falso_vermelho": float((p_med_ace < thr_med).mean()),
        "aceito_p_media": float(p_med_ace.mean()),
        "aceito_p_p05": float(np.percentile(p_med_ace, 5)),
        "borrado_n": len(borrados),
        "borrado_vermelho": float((p_med_bor < thr_med).mean()),
        "sem_crop_n": len(sem_amostra),
        "sem_crop_vermelho": float((p_med_sem < thr_med).mean()),
        "sem_crop_p_media": float(p_med_sem.mean()),
    }
    res["scene_legivel"] = {
        "threshold": thr_leg,
        "aceito_n": len(aceitos),
        "aceito_falso_vermelho": float((p_leg_ace < thr_leg).mean()),
        "aceito_match08_n": int((match >= 0.8).sum()),
        "aceito_match08_falso_vermelho": float((p_leg_ace[match >= 0.8] < thr_leg).mean()),
        "aceito_match_baixo_falso_vermelho": float((p_leg_ace[match < 0.8] < thr_leg).mean()),
        "borrado_n": len(borrados),
        "borrado_vermelho": float((p_leg_bor < thr_leg).mean()),
        "sem_crop_n": len(sem_amostra),
        "sem_crop_vermelho": float((p_leg_sem < thr_leg).mean()),
    }

    # mosaicos de auditoria: as fotos que o classificador chama de "sem medidor" entre os aceitos
    idx_red = np.where(p_med_ace < thr_med)[0][:36]
    _montage([(foto_path(aceitos[i]), f"p={p_med_ace[i]:.2f}") for i in idx_red],
             OUT / "scene_medidor_falsos_vermelhos.png")
    idx_sem = np.argsort(p_med_sem)[:36]
    _montage([(foto_path(sem_amostra[i]), f"p={p_med_sem[i]:.2f}") for i in idx_sem],
             OUT / "sem_crop_que_o_scene_reprova.png")
    idx_sem_ok = np.argsort(-p_med_sem)[:36]
    _montage([(foto_path(sem_amostra[i]), f"p={p_med_sem[i]:.2f}") for i in idx_sem_ok],
             OUT / "sem_crop_que_o_scene_aprova.png")

    return res


# ---------------------------------------------------------------- visor
def _digits(t: str) -> str:
    return re.sub(r"\D", "", t)


def parte_visor(split: str, limit: int | None, seed: int) -> dict:
    from leiturista.crnn import load_crnn, read_digits
    from leiturista.inference import MeterOCR
    from leiturista.visor import score_crops

    man = [r for r in ler_manifest() if r["status"] == "aceito" and r["split"] == split]
    man.sort(key=lambda r: r["image"])
    if limit:
        rng = np.random.default_rng(seed)
        man = [man[i] for i in rng.choice(len(man), min(limit, len(man)), replace=False)]

    ocr = MeterOCR()
    crnn = load_crnn(CRNN_PT)

    stats = collections.Counter()
    per_photo: list[dict] = []
    t0 = time.time()

    for k, r in enumerate(man):
        img = np.array(Image.open(foto_path(r)).convert("RGB"))[:, :, ::-1]
        label = _digits(r["leitura_original"])

        # mesmo conjunto de candidatos de `distribuidora._reading_candidate`
        # (normal mesclado + invertida sem mesclar), com leitura pelo CRNN v2
        cands: list[dict] = []
        for source, merge in (("normal", True), ("inverted", False)):
            base = img if source == "normal" else 255 - img
            quads = ocr._det_boxes(base)
            for quad, bconf in (ocr._merge_quads(quads) if merge else quads):
                crop = ocr._crop_rotated(base, quad)
                if crop.size == 0:
                    continue
                pil = Image.fromarray(crop[:, :, ::-1])
                text = read_digits(pil, crnn)
                d = re.sub(r"\D", "", text)
                w = float(quad[:, 0].max() - quad[:, 0].min())
                h = float(quad[:, 1].max() - quad[:, 1].min())
                field = MeterOCR._classify(text)
                if field == "leitura" and w < 1.5 * h:
                    field = "serial" if (len(text) >= 6 and re.search(r"\d", text)) else "outro"
                cands.append({"text": text, "digits": d, "field": field, "conf": bconf, "crop": pil})

        rec = {"image": r["image"], "label": label, "n_boxes": len(cands)}
        if not cands:
            stats["sem_caixa"] += 1
            per_photo.append(rec)
            continue
        stats["com_caixa"] += 1

        p = score_crops([c["crop"] for c in cands], VISOR_PT)
        i_visor = int(p.argmax())
        rec["p_visor"] = float(p[i_visor])
        rec["pick_visor"] = cands[i_visor]["digits"]
        rec["ok_visor"] = cands[i_visor]["digits"] == label
        rec["ok_visor_thr"] = bool(p[i_visor] >= 0.5 and rec["ok_visor"])

        readings = [c for c in cands if c["field"] == "leitura"]
        pick_old = None
        if readings:
            pure = [c for c in readings if c["digits"] and re.fullmatch(r"\d+", c["text"])]
            pool = pure or readings
            pick_old = max(pool, key=lambda c: (len(c["digits"]), c["conf"]))
        rec["pick_antiga"] = pick_old["digits"] if pick_old else ""
        rec["ok_antiga"] = rec["pick_antiga"] == label

        rec["ok_fallback"] = bool(
            (p[i_visor] >= 0.5 and rec["ok_visor"]) or (p[i_visor] < 0.5 and rec["ok_antiga"])
        )
        rec["ok_oraculo"] = any(c["digits"] == label for c in cands)
        per_photo.append(rec)

        for key in ("ok_visor", "ok_visor_thr", "ok_antiga", "ok_fallback", "ok_oraculo"):
            stats[key] += int(rec[key])
        if k % 25 == 0:
            print(f"  {k}/{len(man)}  {time.time() - t0:.0f}s", flush=True)

    n = len(man)
    com = stats["com_caixa"]
    orac = stats["ok_oraculo"]
    res = {
        "split": split,
        "n": n,
        "sem_caixa": stats["sem_caixa"],
        "com_caixa": com,
        "taxa_sem_caixa": stats["sem_caixa"] / max(n, 1),
        "acuracia_visor": stats["ok_visor"] / max(com, 1),
        "acuracia_visor_thr": stats["ok_visor_thr"] / max(com, 1),
        "acuracia_regra_antiga": stats["ok_antiga"] / max(com, 1),
        "acuracia_fallback": stats["ok_fallback"] / max(com, 1),
        "oraculo": orac / max(com, 1),
        "n_oraculo": orac,
        "acuracia_visor_dado_oraculo": stats["ok_visor"] / max(orac, 1),
        "acuracia_antiga_dado_oraculo": stats["ok_antiga"] / max(orac, 1),
        "acuracia_fallback_dado_oraculo": stats["ok_fallback"] / max(orac, 1),
    }
    return {"resumo": res, "por_foto": per_photo}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("scene", "visor", "all"), default="all")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit-visor", type=int, default=None)
    ap.add_argument("--limit-sem-crop", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    if a.part in ("scene", "all"):
        print("== classificadores de cena ==", flush=True)
        out["scene"] = parte_scene(a.limit_sem_crop, a.seed)
        print(json.dumps(out["scene"], indent=2, ensure_ascii=False))
    if a.part in ("visor", "all"):
        print("== classificador de visor (ponta a ponta) ==", flush=True)
        out["visor"] = parte_visor(a.split, a.limit_visor, a.seed)
        print(json.dumps(out["visor"]["resumo"], indent=2, ensure_ascii=False))
        with open(OUT / f"visor_{a.split}.json", "w", encoding="utf-8") as f:
            json.dump(out["visor"], f, ensure_ascii=False, indent=1)
    with open(OUT / "resumo.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nJSON: {OUT / 'resumo.json'}")


if __name__ == "__main__":
    main()
