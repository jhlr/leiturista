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
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from . import paths

N_SIMBOLOS = 10  # dígitos 0-9; a CTC soma 1 símbolo "branco" (índice 0)
ALTURA, LARGURA = 32, 128
WEIGHTS = paths.MODELS_DIR / "crnn_digitos.pt"


class CRNNDigitos(nn.Module):
    def __init__(self, n_simbolos: int = N_SIMBOLOS, oculto: int = 128) -> None:
        super().__init__()

        def bloco(cin: int, cout: int, pool: tuple[int, int]) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
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


def preprocess(img: Image.Image) -> np.ndarray:
    """Cinza, squash para 32x128, [0,1]. Mesmo pré-processamento do treino."""
    return np.asarray(img.convert("L").resize((LARGURA, ALTURA)), dtype=np.float32) / 255.0


def _batch(items: list[tuple[Path, str]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor(np.stack([preprocess(Image.open(p)) for p, _ in items]))[:, None]
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
            x, _, _ = _batch(lote)
            for pred, (_, alvo) in zip(decodificar_ctc(model(x)), lote):
                exatos += int(pred == alvo)
                erros += _edit_distance(alvo, pred)
                total += len(alvo)
    return exatos / len(items), 1 - erros / total


def train_crnn(
    data_dir: Path | str = paths.FINETUNE_DIR,
    out: Path | str = WEIGHTS,
    epochs: int = 15,
    batch: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
    tracking_uri: str = paths.DEFAULT_TRACKING_URI,
    experiment: str = "crnn-digitos",
) -> dict[str, float]:
    """Receita do notebook (Adam, CTC, lotes de 32, 15 épocas, seed fixa). Seleciona nada pelo
    teste: salva o estado da ÚLTIMA época e só então reporta valid/test."""
    import mlflow

    data_dir = Path(data_dir)
    split: dict[str, list[tuple[Path, str]]] = {"train": [], "valid": [], "test": []}
    with open(data_dir / "labels.csv") as f:
        for r in csv.DictReader(f):
            split[r["split"]].append((data_dir / r["image"], r["label"]))

    torch.manual_seed(42)
    torch.set_num_threads(2)
    rng = np.random.default_rng(seed)
    model = CRNNDigitos()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"crnn-{epochs}ep"):
        mlflow.log_params({"epochs": epochs, "batch": batch, "lr": lr, "seed": seed,
                           "n_train": len(split["train"]), "altura": ALTURA, "largura": LARGURA})
        train = split["train"]
        for ep in range(epochs):
            model.train()
            ordem = rng.permutation(len(train))
            soma, n = 0.0, 0
            for i in range(0, len(ordem), batch):
                x, alvos, comp = _batch([train[j] for j in ordem[i:i + batch]])
                lp = model(x).log_softmax(-1).permute(1, 0, 2)
                comp_ent = torch.full((x.size(0),), lp.size(0), dtype=torch.long)
                opt.zero_grad()
                loss = ctc(lp, alvos, comp_ent, comp)
                loss.backward()
                opt.step()
                soma += loss.item()
                n += 1
            mlflow.log_metric("train_loss", soma / n, step=ep)
            print(f"época {ep:>2}: perda treino = {soma / n:.4f}", flush=True)

        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out)
        res: dict[str, float] = {}
        for nome in ("valid", "test"):
            exato, dig = evaluate(model, split[nome], batch)
            res[f"{nome}_exact"], res[f"{nome}_digit_acc"] = exato, dig
            print(f"{nome:>5}: leitura exata = {exato:.3f}  acurácia por dígito = {dig:.3f}", flush=True)
        mlflow.log_metrics(res)
    return res


_CACHE: dict[str, CRNNDigitos] = {}


def load_crnn(path: Path | str = WEIGHTS) -> CRNNDigitos:
    key = str(path)
    if key not in _CACHE:
        m = CRNNDigitos()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        _CACHE[key] = m.eval()
    return _CACHE[key]


def read_digits(img: Image.Image, model: CRNNDigitos | None = None) -> str:
    """Lê o recorte de um display com o CRNN. Devolve só dígitos ('' se nada)."""
    m = model or load_crnn()
    x = torch.tensor(preprocess(img))[None, None]
    with torch.no_grad():
        return decodificar_ctc(m(x))[0]
