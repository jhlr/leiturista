"""`CRNNDigitos` — reconhecedor de dígitos do esqueleto do professor (Lab 2, bloco Luminus).

CNN reduz a altura a 1 e mantém a largura como eixo de tempo; GRU bidirecional lê da esquerda
para a direita; CTC dispensa saber onde cada dígito está no recorte. Mesma arquitetura e mesma
receita de `notebooks/lab02-dl.py` (fonte única do modelo agora é este módulo; o notebook
mantém a cópia didática do esqueleto).

Uso:
    leiturista train-crnn                 # treina em data/finetune_ufpramr, salva models/crnn_digitos.pt
    from leiturista.crnn import load_crnn, read_digits
"""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch import nn

from . import paths

N_SIMBOLOS = 10  # dígitos 0-9; a CTC soma 1 símbolo "branco" (índice 0)
ALTURA, LARGURA = 32, 128  # LARGURA = padrão; o modelo guarda a largura com que foi treinado (`model.width`)
WEIGHTS = paths.MODELS_DIR / "crnn_digitos.pt"


class CRNNDigitos(nn.Module):
    def __init__(self, n_simbolos: int = N_SIMBOLOS, oculto: int = 128, norm: str = "batch", width: int = LARGURA) -> None:
        super().__init__()
        self.norm, self.width = norm, width

        def bloco(cin: int, cout: int, pool: tuple[int, int]) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout) if norm == "batch" else nn.GroupNorm(8, cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            )

        self.cnn = nn.Sequential(
            bloco(1, 32, (2, 2)),     # 32x128 -> 16x64
            bloco(32, 64, (2, 2)),    # -> 8x32
            bloco(64, 128, (2, 1)),   # -> 4x32 (pool só na altura: preserva passos de tempo)
            bloco(128, 128, (4, 1)),  # -> 1x32
        )
        self.rnn = nn.GRU(128, oculto, bidirectional=True, batch_first=True)
        self.saida = nn.Linear(2 * oculto, n_simbolos + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x).squeeze(2).permute(0, 2, 1)  # (B, T, C)
        h, _ = self.rnn(f)
        return self.saida(h)  # (B, T, n_simbolos+1)


def decodificar_ctc(logits: torch.Tensor) -> list[str]:
    """Guloso: argmax por passo, colapsa repetições, remove brancos. Dígito d -> índice d+1."""
    seqs: list[str] = []
    for linha in logits.argmax(-1).tolist():
        s: list[str] = []
        anterior = 0
        for k in linha:
            if k != anterior and k != 0:
                s.append(str(k - 1))
            anterior = k
        seqs.append("".join(s))
    return seqs


def preprocess(img: Image.Image, width: int = LARGURA) -> np.ndarray:
    """Cinza, squash para 32 x `width`, [0,1]. Mesmo pré-processamento do treino."""
    return np.asarray(img.convert("L").resize((width, ALTURA)), dtype=np.float32) / 255.0


def augment(
    a: np.ndarray, rng: np.random.Generator, invert_prob: float, crop_jitter: float = 0.0, width: int = LARGURA,
    rot_deg: float = 10.0, blur_p: float = 0.3,
) -> np.ndarray:
    """Augmentation do plano (docs/dl-lab2/plano-treino-crnn.md). Núcleo: brilho/contraste, giro
    ±10°, desfoque 5x5 (p=0,3). Inversão de polaridade (41,7% dos recortes reais da distribuidora
    vêm da fase invertida) e jitter de recorte (`crop_jitter`=fração máx. por borda; o det não
    entrega caixa pixel-perfeita) são do estágio 2. Sem espelhamento nem rotação grande."""
    if rng.random() < invert_prob:
        a = 1.0 - a
    a = a * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30) / 255.0
    im = Image.fromarray((np.clip(a, 0.0, 1.0) * 255).astype(np.uint8))
    if rot_deg > 0:
        im = im.rotate(rng.uniform(-rot_deg, rot_deg), resample=Image.BILINEAR, fillcolor=int(np.median(np.asarray(im))))
    if crop_jitter > 0:  # borda replicada + janela aleatória: sorteia zoom-in e zoom-out
        pw, ph = round(crop_jitter * width), round(crop_jitter * ALTURA)
        big = Image.fromarray(np.pad(np.asarray(im), ((ph, ph), (pw, pw)), mode="edge"))
        l, r = rng.integers(0, 2 * pw + 1, 2) if pw else (0, 0)
        t, b = rng.integers(0, 2 * ph + 1, 2) if ph else (0, 0)
        im = big.resize((width, ALTURA), box=(int(l), int(t), width + 2 * pw - int(r), ALTURA + 2 * ph - int(b)))
    if rng.random() < blur_p:
        im = im.filter(ImageFilter.GaussianBlur(1.0))
    return np.asarray(im, dtype=np.float32) / 255.0


def _batch(
    items: list[tuple[Path | Image.Image, str]],
    rng: np.random.Generator | None = None,
    invert_prob: float = 0.0,
    crop_jitter: float = 0.0,
    width: int = LARGURA,
    rot_deg: float = 10.0,
    blur_p: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    arrs = [preprocess(Image.open(p) if isinstance(p, Path) else p, width) for p, _ in items]
    if rng is not None:
        arrs = [augment(a, rng, invert_prob, crop_jitter, width, rot_deg, blur_p) for a in arrs]
    x = torch.tensor(np.stack(arrs))[:, None]
    alvos = torch.tensor([int(c) + 1 for _, r in items for c in r])
    comp = torch.tensor([len(r) for _, r in items])
    return x, alvos, comp


def _edit_distance(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b) + 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return dp[len(b)]


def evaluate(model: CRNNDigitos, items: list[tuple[Path, str]], batch: int = 32) -> tuple[float, float]:
    """(leitura exata, acurácia por dígito = 1 - distância de edição / dígitos do alvo)."""
    model.eval()
    exatos = erros = total = 0
    with torch.no_grad():
        for i in range(0, len(items), batch):
            lote = items[i:i + batch]
            x, _, _ = _batch(lote, width=model.width)
            for pred, (_, alvo) in zip(decodificar_ctc(model(x)), lote):
                exatos += int(pred == alvo)
                erros += _edit_distance(alvo, pred)
                total += len(alvo)
    return exatos / len(items), 1 - erros / total


def predict(model: CRNNDigitos, items: list[tuple[Path, str]], batch: int = 32) -> list[str]:
    model.eval()
    out: list[str] = []
    with torch.no_grad():
        for i in range(0, len(items), batch):
            x, _, _ = _batch(items[i:i + batch], width=model.width)
            out += decodificar_ctc(model(x))
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """p-valor bicaudal do McNemar exato (binomial, p=0,5) sobre os pares discordantes b e c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare_crnn(
    checkpoints: Sequence[Path | str], data_dir: Path | str = paths.FINETUNE_DIR, split: str = "test"
) -> None:
    """Leitura exata por checkpoint + McNemar exato pareado em todos os pares (mesmo split)."""
    d = Path(data_dir)
    with open(d / "labels.csv") as f:
        items = [(d / r["image"], r["label"]) for r in csv.DictReader(f) if r["split"] == split]
    hits = {}
    for ck in checkpoints:
        pred = predict(load_crnn(ck, cache=False), items)
        hits[str(ck)] = [p == y for p, (_, y) in zip(pred, items)]
        print(f"{ck}: leitura exata {sum(hits[str(ck)]) / len(items):.3f} ({split}, n={len(items)})")
    names = list(hits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            only_a = sum(x and not y for x, y in zip(hits[a], hits[b]))
            only_b = sum(y and not x for x, y in zip(hits[a], hits[b]))
            print(f"McNemar {Path(a).name} x {Path(b).name}: só A acerta {only_a}, só B acerta {only_b}, "
                  f"p = {mcnemar_exact(only_a, only_b):.3f}")


def train_crnn(
    data_dirs: Sequence[Path | str] = (paths.FINETUNE_DIR,),
    out: Path | str = WEIGHTS,
    epochs: int = 15,
    batch: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
    init: Path | str | None = None,
    invert_prob: float = 0.0,
    crop_jitter: float = 0.0,
    norm: str = "batch",
    width: int = LARGURA,
    synth_per_epoch: int = 0,
    synth_only: bool = False,
    warmup: int = 0,
    cosine: bool = False,
    augment_on: bool = True,
    aug_start: int = 0,
    freeze_cnn: bool = False,
    reinit_head: bool = False,
    rot_deg: float = 10.0,
    blur_p: float = 0.3,
    clip_calib_iters: int = 0,
    tracking_uri: str = paths.DEFAULT_TRACKING_URI,
    experiment: str = "crnn-digitos",
) -> dict[str, float]:
    """CTC + Adam, seed fixa. Seleciona o checkpoint pela leitura exata no `valid` REAL (nunca
    pelo teste); `test` é reportado uma vez, no fim, com o checkpoint escolhido.

    `synth_per_epoch`>0: soma ao treino real N sequências sintéticas (synth.py), regeneradas a
    cada época; `synth_only` descarta o treino real (estágio 1 puro). valid/test são sempre reais.
    `warmup`: passos de aquecimento linear do LR; `cosine`: depois decai em cosseno até 0.
    `freeze_cnn`: modo EXTRAÇÃO de características (CNN congelada, BN em eval; só GRU+Linear treinam);
    `reinit_head`: reinicia GRU+Linear com a semente (extração com cabeça nova, seeds de verdade).
    `augment_on=False` desliga TODA augmentation (diagnóstico); `aug_start`=N: currículo, as N
    primeiras épocas sem augmentation (ela alonga o platô inicial da CTC) e depois tudo ligado.
    `clip_calib_iters`>0: clipping de gradiente com limiar = percentil 90 da norma observada
    nesses primeiros passos (sem clipping); 0 = sem clipping.

    `data_dirs`: um ou mais datasets (`labels.csv` com split,image,label); o treino concatena os
    `train`, e valid/test são reportados POR dataset (ex.: UFPR-AMR e distribuidora separados).
    `init`: pesos de partida (fine-tune, sem congelar nada). `invert_prob`/brilho: augmentation
    só no treino."""
    import mlflow

    per_ds: dict[str, dict[str, list[tuple[Path, str]]]] = {}
    for d in map(Path, data_dirs):
        sp: dict[str, list[tuple[Path, str]]] = {"train": [], "valid": [], "test": []}
        with open(d / "labels.csv") as f:
            for r in csv.DictReader(f):
                sp[r["split"]].append((d / r["image"], r["label"]))
        per_ds[d.name] = sp
    split = {k: [it for sp in per_ds.values() for it in sp[k]] for k in ("train", "valid", "test")}

    torch.manual_seed(42 + seed)  # seed=0 -> 42 (comportamento anterior)
    torch.set_num_threads(2)
    rng = np.random.default_rng(seed)
    if init is not None:  # fine-tune herda norm/width do checkpoint (a largura não está no state_dict)
        _, norm, width = _read_ckpt(init)
    model = CRNNDigitos(norm=norm, width=width)
    if synth_per_epoch:
        from .synth import DigitBank, length_distribution, make_sequence

        bank, lens = DigitBank(), length_distribution(Path(next(iter(data_dirs))) / "labels.csv")
    if init is not None:
        model.load_state_dict(_read_ckpt(init)[0])
    if reinit_head:
        model.rnn.reset_parameters()
        model.saida.reset_parameters()
    if freeze_cnn:
        for prm in model.cnn.parameters():
            prm.requires_grad = False
    opt = torch.optim.Adam([prm for prm in model.parameters() if prm.requires_grad], lr=lr)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"crnn-{epochs}ep"):
        mlflow.log_params({"epochs": epochs, "batch": batch, "lr": lr, "seed": seed,
                           "n_train": len(split["train"]), "altura": ALTURA, "largura": width, "norm": norm,
                           "datasets": ",".join(per_ds), "init": str(init), "invert_prob": invert_prob, "synth_per_epoch": synth_per_epoch, "synth_only": synth_only, "cosine": cosine, "augment": augment_on, "aug_start": aug_start, "freeze_cnn": freeze_cnn, "reinit_head": reinit_head, "rot_deg": rot_deg, "blur_p": blur_p})
        train = split["train"]
        if synth_only:
            train = []
        steps_ep = -(-(synth_per_epoch + len(train)) // batch)
        total = epochs * steps_ep

        def lr_at(step: int) -> float:
            if step < warmup:
                return lr * (step + 1) / warmup
            if not cosine:
                return lr
            return lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

        step, norms, clip = 0, [], float("inf")
        best, out = (-1.0, -1.0), Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        for ep in range(epochs):
            t0 = time.time()
            model.train()
            if freeze_cnn:
                model.cnn.eval()
            items: list[tuple[Path | Image.Image, str]] = list(train)
            if synth_per_epoch:
                items += [make_sequence(bank, lens, rng, (width, ALTURA)) for _ in range(synth_per_epoch)]
            items = [items[j] for j in rng.permutation(len(items))]
            soma, n = 0.0, 0
            for i in range(0, len(items), batch):
                x, alvos, comp = _batch(items[i:i + batch], rng if augment_on and ep >= aug_start else None, invert_prob, crop_jitter, width, rot_deg, blur_p)
                lp = model(x).log_softmax(-1).permute(1, 0, 2)
                comp_ent = torch.full((x.size(0),), lp.size(0), dtype=torch.long)
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                opt.zero_grad()
                loss = ctc(lp, alvos, comp_ent, comp)
                loss.backward()
                if clip_calib_iters:
                    gn = float(nn.utils.clip_grad_norm_(model.parameters(), clip))
                    if step < clip_calib_iters:
                        norms.append(gn)
                        if step == clip_calib_iters - 1:
                            clip = float(np.percentile(norms, 90))
                            mlflow.log_param("clip_norm", clip)
                            print(f"clip calibrado (p90 da norma, {clip_calib_iters} passos): {clip:.3f}", flush=True)
                opt.step()
                soma += loss.item()
                n += 1
                step += 1
            exato, dig = evaluate(model, split["valid"], batch)
            mlflow.log_metrics({"train_loss": soma / n, "valid_exact": exato, "valid_digit_acc": dig}, step=ep)
            marca = ""
            if (exato, dig) > best:  # desempate por acurácia de dígito (nas primeiras épocas a exata é 0)
                best, marca = (exato, dig), " *"
                _save(model, out)
            print(f"época {ep:>2}: perda {soma / n:.4f}  valid exata {exato:.3f} dígito {dig:.3f}  "
                  f"{time.time() - t0:.0f}s{marca}", flush=True)

        model = load_crnn(out, cache=False)
        res: dict[str, float] = {}
        for ds, sp in per_ds.items():
            for nome in ("valid", "test"):
                if not sp[nome]:
                    continue
                exato, dig = evaluate(model, sp[nome], batch)
                res[f"{ds}_{nome}_exact"], res[f"{ds}_{nome}_digit_acc"] = exato, dig
                print(f"{ds:>20} {nome:>5}: leitura exata = {exato:.3f}  por dígito = {dig:.3f}", flush=True)
        mlflow.log_metrics(res)
    return res


def _save(model: CRNNDigitos, path: Path) -> None:
    torch.save({"state": model.state_dict(), "norm": model.norm, "width": model.width}, path)


def _read_ckpt(path: Path | str) -> tuple[dict[str, torch.Tensor], str, int]:
    """(state_dict, norm, width). Checkpoint antigo (state_dict puro) = BatchNorm, largura 128."""
    obj = torch.load(path, map_location="cpu")
    if "state" in obj:
        return obj["state"], obj.get("norm", "batch"), obj.get("width", LARGURA)
    return obj, "batch", LARGURA


_CACHE: dict[str, CRNNDigitos] = {}


def load_crnn(path: Path | str = WEIGHTS, cache: bool = True) -> CRNNDigitos:
    """Carrega pesos; checkpoints antigos (state_dict puro) = BatchNorm, largura 128."""
    key = str(path)
    if not cache or key not in _CACHE:
        state, norm, width = _read_ckpt(path)
        m = CRNNDigitos(norm=norm, width=width)
        m.load_state_dict(state)
        if not cache:
            return m.eval()
        _CACHE[key] = m.eval()
    return _CACHE[key]


def read_digits(img: Image.Image, model: CRNNDigitos | None = None) -> str:
    """Lê o recorte de um display com o CRNN. Devolve só dígitos ('' se nada)."""
    m = model or load_crnn()
    x = torch.tensor(preprocess(img, m.width))[None, None]
    with torch.no_grad():
        return decodificar_ctc(m(x))[0]
