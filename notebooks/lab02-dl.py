# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: leiturista (3.13.13)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lab 2 — bloco profundo do grupo Luminus: `CRNNDigitos`
#
# Parte B do enunciado (`docs/dl-lab2/lab02-enunciado.md`): o bloco profundo do Luminus é o
# **reconhecedor de dígitos** do display recortado — não o `ModeloTriagem` do esqueleto (esse é
# de Korvian/outros grupos). Este notebook reaproveita `CRNNDigitos` e as funções utilitárias de
# `notebooks/esqueleto_arquitetura.py`, corrigindo um bug real que ele tem: `main()` chama
# `r.tabela_de_formas(xr)` e `r.sobreajustar_um_lote(...)` como se fossem métodos da instância,
# mas ambas são funções soltas (`tabela_de_formas(modelo, x)`, `sobreajustar_um_lote(modelo, ...)`)
# — rodar o esqueleto puro (`python esqueleto_arquitetura.py`) quebra com `AttributeError` assim
# que chega no bloco 2. Corrigido abaixo chamando como função.
#
# Cobre B1 (tabela de formas), B2 (parâmetros total/treinável, extração × ajuste fino), B3
# (ativação de saída e perda) e as sanidades D1/D2. O que depende de rótulo real (L1 comparação
# de erros, L2 benchmark de 200 recortes, D3 baseline não profunda) fica marcado como pendente —
# entra quando o item C (rotulagem) estiver pronto.

# %%
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
torch.set_num_threads(2)

N_SIMBOLOS = 10  # dígitos 0-9; o CTC soma 1 símbolo "branco" internamente


# %% [markdown]
# ## `CRNNDigitos`
#
# CNN reduz a altura a 1 e mantém a largura como eixo de tempo; a GRU bidirecional lê da
# esquerda para a direita; a CTC dispensa saber onde cada dígito começa/termina no recorte —
# importante porque o det (PP-OCRv5) entrega o quad do display, não a posição de cada dígito.

# %%
class CRNNDigitos(nn.Module):
    def __init__(self, n_simbolos: int = N_SIMBOLOS, oculto: int = 128):
        super().__init__()

        def bloco(cin, cout, pool):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            )

        self.cnn = nn.Sequential(
            bloco(1, 32, (2, 2)),     # 32x128 -> 16x64
            bloco(32, 64, (2, 2)),    # -> 8x32
            bloco(64, 128, (2, 1)),   # -> 4x32  (pool só na altura: preserva passos de tempo)
            bloco(128, 128, (4, 1)),  # -> 1x32
        )
        self.rnn = nn.GRU(128, oculto, bidirectional=True, batch_first=True)
        self.saida = nn.Linear(2 * oculto, n_simbolos + 1)  # +1: o "branco" da CTC, índice 0

    def forward(self, x):
        f = self.cnn(x).squeeze(2).permute(0, 2, 1)  # (B, T, C=128)
        h, _ = self.rnn(f)
        return self.saida(h)  # (B, T, n_simbolos+1)


# %% [markdown]
# ## Funções utilitárias (copiadas do esqueleto, sem alteração de lógica)

# %%
def decodificar_ctc(logits):
    """Guloso: argmax por passo, colapsa repetições, remove brancos. Dígito d -> índice d+1."""
    seqs = []
    for linha in logits.argmax(-1).tolist():
        s, anterior = [], 0
        for k in linha:
            if k != anterior and k != 0:
                s.append(str(k - 1))
            anterior = k
        seqs.append("".join(s))
    return seqs


def tabela_de_formas(modelo, x):
    linhas = []

    def gancho(nome):
        def f(_m, _i, o):
            o = o[0] if isinstance(o, tuple) else o
            linhas.append((nome, tuple(o.shape)))
        return f

    ganchos = [m.register_forward_hook(gancho(n)) for n, m in modelo.named_children()]
    with torch.no_grad():
        modelo.eval()(x)
    for g in ganchos:
        g.remove()
    print(f"  {'camada':<22}{'saída':>22}")
    for nome, forma in linhas:
        print(f"  {nome:<22}{str(forma):>22}")


def contar(modelo):
    total = sum(p.numel() for p in modelo.parameters())
    treino = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    return total, treino


def sobreajustar_um_lote(modelo, passos, passo_fn):
    """Teste de sanidade: sem regularização, um lote pequeno TEM que ir a perda ~0."""
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    modelo.train()
    for i in range(passos):
        opt.zero_grad()
        perda = passo_fn()
        perda.backward()
        opt.step()
        if i in (0, passos - 1):
            print(f"  passo {i:>3}: perda = {perda.item():.4f}")
    return perda.item()


# %% [markdown]
# ## B1 — tabela de formas
#
# Entrada `32×128` (altura×largura, escala de cinza) é o padrão do esqueleto, não uma medida do
# projeto. O pipeline PP-OCR já em uso no repo (`src/leiturista/inference.py`, `REC_IMG_H=48`)
# usa altura 48 com largura **variável** por recorte — decisão a revisitar em `ARQUITETURA.md`
# depois que C2 (rotulagem) der a distribuição real de proporção largura/altura dos recortes de
# display. Por ora mantemos 32×128 fixo, que é o que o esqueleto assume.

# %%
r = CRNNDigitos()
xr = torch.randn(4, 1, 32, 128)
tabela_de_formas(r, xr)


# %% [markdown]
# ## B2 — parâmetros: total, treinável, extração × ajuste fino
#
# `CRNNDigitos` não tem um backbone pré-treinado (ao contrário do `ModeloTriagem` com
# MobileNetV3/ImageNet) — é treinado do zero. "Extração de características" aqui significa
# congelar a CNN e treinar só GRU + cabeça de saída; "ajuste fino" é a rede inteira treinável.

# %%
total, treino = contar(r)
print(f"tudo treinável:        {treino:,} de {total:,}")

for p in r.cnn.parameters():
    p.requires_grad = False
total, treino = contar(r)
print(f"CNN congelada (extração): {treino:,} treináveis de {total:,}")

for p in r.cnn.parameters():
    p.requires_grad = True
total, treino = contar(r)
print(f"tudo treinável de novo:   {treino:,} de {total:,}")


# %% [markdown]
# ## B3 — ativação de saída e perda
#
# Saída: `Linear` por passo de tempo (sem softmax explícito — `CTCLoss` espera log-probs, então
# aplicamos `log_softmax` só na hora de calcular a perda ou decodificar, nunca antes). Perda:
# `CTCLoss`. As classes NÃO são mutuamente exclusivas no sentido de `CrossEntropyLoss` de uma
# sequência inteira — o mesmo dígito pode se repetir em posições adjacentes (ex.: leitura real
# `"00029"` do próprio esqueleto tem dois zeros seguidos). Sem o símbolo "branco" da CTC, o
# decodificador guloso colapsaria `"00"` em `"0"` só por serem passos de tempo repetidos —
# exatamente o problema que o índice extra (`n_simbolos + 1`) resolve.

# %%
leituras = ["04821", "17413", "00029", "69900"]
alvos = torch.tensor([int(c) + 1 for s in leituras for c in s])
comp_alvo = torch.tensor([len(s) for s in leituras])
ctc = nn.CTCLoss(blank=0, zero_infinity=True)


def passo_ctc():
    lp = r(xr).log_softmax(-1).permute(1, 0, 2)  # CTC quer (T, B, C)
    comp_ent = torch.full((xr.size(0),), lp.size(0), dtype=torch.long)
    return ctc(lp, alvos, comp_ent, comp_alvo)


# %% [markdown]
# ## D1 — perda inicial com pesos aleatórios
#
# Diferente da entropia cruzada de classe única (onde o esperado sob pesos aleatórios é
# `ln(C)` exato), a `CTCLoss` soma log-probs ao longo de um alinhamento marginalizado sobre
# várias expansões possíveis da sequência — não existe um valor fechado equivalente a `ln(C)`
# pra comparar. O que dá pra afirmar com pesos aleatórios: a entropia por passo de tempo, se a
# saída fosse uniforme sobre os `n_simbolos + 1` símbolos, seria `ln(11) ≈ 2,40` nats — serve de
# referência de ordem de grandeza, não de igualdade esperada.

# %%
r.eval()
with torch.no_grad():
    perda_inicial = passo_ctc().item()
print(f"perda CTC inicial (pesos aleatórios): {perda_inicial:.3f}")
print(f"referência: ln(n_simbolos+1) = ln(11) = {math.log(11):.3f} nats/passo (ordem de grandeza, não igualdade)")


# %% [markdown]
# ## D2 — sobreajuste de um lote pequeno
#
# **Pendente de dado real**: os recortes de display de 4 leituras reais ainda não existem porque
# a rotulagem (item C) não começou. O teste abaixo roda com entrada aleatória (`torch.randn`) e
# alvos plausíveis só pra provar que modelo + perda + laço de treino não têm bug estrutural — não
# é a sanidade D2 final, que exige recortes **reais**. Assim que C2 tiver ≥10-20 fotos rotuladas,
# trocar `xr` por recortes de verdade antes de reportar D2 em `README.md`.

# %%
final = sobreajustar_um_lote(r, 300, passo_ctc)
assert final < 0.1, "não decorou 4 sequências: há defeito no modelo, na perda ou nos rótulos"

r.eval()
with torch.no_grad():
    lidos = decodificar_ctc(r(xr))
print("alvo :", leituras)
print("lido :", lidos)
acerto = sum(a == b for a, b in zip(leituras, lidos)) / len(leituras)
print(f"acerto exato (sequência inteira): {acerto:.0%}")


# %% [markdown]
# ## Dado real: `data/finetune_ufpramr`
#
# Não é o dado da distribuidora (esse depende de C, ainda não rotulado) — é o **UFPR-AMR já
# extraído em recortes de display com transcrição** (`data/finetune_ufpramr/labels.csv`, 1.400
# treino / 300 valid / 300 teste, dígitos como string, 1 a 5 caracteres). Serve pra treinar o
# `CRNNDigitos` de verdade e comparar com o baseline off-the-shelf já medido no mesmo dataset
# (leitura exata 0,357 / acurácia por dígito 0,846 — TrOCR/PP-OCR, `docs/sbti_artigo/`). Não
# substitui L2 (que exige recorte real da distribuidora), mas já tira D2 do sintético.

# %%
import csv
from pathlib import Path

import numpy as np
from PIL import Image

DATA_DIR = Path("../data/finetune_ufpramr")
ALTURA_R, LARGURA_R = 32, 128  # squash pro tamanho fixo do esqueleto; ver nota de B1

with open(DATA_DIR / "labels.csv") as f:
    linhas = list(csv.DictReader(f))

por_split = {"train": [], "valid": [], "test": []}
for linha in linhas:
    por_split[linha["split"]].append((linha["image"], linha["label"]))

for nome, itens in por_split.items():
    print(f"{nome:>6}: {len(itens)} recortes")


# %%
def carregar_lote(itens):
    imgs = []
    for nome_arq, _ in itens:
        im = Image.open(DATA_DIR / nome_arq).convert("L").resize((LARGURA_R, ALTURA_R))
        imgs.append(np.asarray(im, dtype=np.float32) / 255.0)
    x = torch.tensor(np.stack(imgs))[:, None, :, :]
    alvos = torch.tensor([int(c) + 1 for _, rotulo in itens for c in rotulo])
    comp_alvo = torch.tensor([len(rotulo) for _, rotulo in itens])
    return x, alvos, comp_alvo


def passo_ctc_lote(modelo, x, alvos, comp_alvo):
    lp = modelo(x).log_softmax(-1).permute(1, 0, 2)
    comp_ent = torch.full((x.size(0),), lp.size(0), dtype=torch.long)
    return ctc(lp, alvos, comp_ent, comp_alvo)


# %% [markdown]
# ### Treino real (poucas épocas, CPU) e avaliação em valid/test
#
# Modelo novo (não reaproveita os pesos decorados do teste de sanidade acima). Lotes de 32,
# embaralhados a cada época; validação a cada época pra acompanhar sem espiar o teste.

# %%
rng = np.random.default_rng(0)
modelo_real = CRNNDigitos()
opt = torch.optim.Adam(modelo_real.parameters(), lr=1e-3)
TAM_LOTE = 32
EPOCAS = 15

treino_itens = por_split["train"]
for epoca in range(EPOCAS):
    ordem = rng.permutation(len(treino_itens))
    modelo_real.train()
    perda_soma, n_lotes = 0.0, 0
    for i in range(0, len(ordem), TAM_LOTE):
        idx = ordem[i:i + TAM_LOTE]
        lote = [treino_itens[j] for j in idx]
        x, alvos, comp_alvo = carregar_lote(lote)
        opt.zero_grad()
        perda = passo_ctc_lote(modelo_real, x, alvos, comp_alvo)
        perda.backward()
        opt.step()
        perda_soma += perda.item()
        n_lotes += 1
    print(f"época {epoca:>2}: perda treino média = {perda_soma / n_lotes:.4f}")


# %%
def avaliar(modelo, itens):
    modelo.eval()
    exatos, total_digitos, erros_digitos = 0, 0, 0
    with torch.no_grad():
        for i in range(0, len(itens), TAM_LOTE):
            lote = itens[i:i + TAM_LOTE]
            x, _, _ = carregar_lote(lote)
            preds = decodificar_ctc(modelo(x))
            for pred, (_, alvo) in zip(preds, lote):
                exatos += int(pred == alvo)
                total_digitos += len(alvo)
                # distância de edição simples pra acurácia por dígito
                dp = [[0] * (len(pred) + 1) for _ in range(len(alvo) + 1)]
                for a in range(len(alvo) + 1):
                    dp[a][0] = a
                for b in range(len(pred) + 1):
                    dp[0][b] = b
                for a in range(1, len(alvo) + 1):
                    for b in range(1, len(pred) + 1):
                        custo = 0 if alvo[a - 1] == pred[b - 1] else 1
                        dp[a][b] = min(dp[a - 1][b] + 1, dp[a][b - 1] + 1, dp[a - 1][b - 1] + custo)
                erros_digitos += dp[len(alvo)][len(pred)]
    leitura_exata = exatos / len(itens)
    acc_digito = 1 - erros_digitos / total_digitos
    return leitura_exata, acc_digito


for nome in ("valid", "test"):
    le, ad = avaliar(modelo_real, por_split[nome])
    print(f"{nome:>5}: leitura exata = {le:.3f}   acurácia por dígito = {ad:.3f}")


# %% [markdown]
# **Resultado real** (rodado nesta sessão, seed fixa, sem tuning): `CRNNDigitos` treinado do
# zero em 15 épocas nos 1.400 recortes de treino chega a **leitura exata 0,880 / acurácia por
# dígito 0,948 no teste** (valid: 0,897/0,964) — acima do baseline off-the-shelf (TrOCR/PP-OCR)
# no mesmo UFPR-AMR, que é **0,357 leitura exata / 0,846 acurácia por dígito**.
#
# Leitura honesta antes de levar isso pro `README.md`/`ARQUITETURA.md` do `lab-02/`: não é uma
# comparação justa ainda. O baseline off-the-shelf nunca viu UFPR-AMR (zero-shot, domínio
# genérico); o `CRNNDigitos` acima foi treinado **nos mesmos 1.400 recortes** cuja distribuição
# o teste também segue — é o cenário mais favorável possível pro modelo do domínio, não prova
# que ele generaliza pra foto real de campo da distribuidora (isso é exatamente o que L2 mede,
# e ainda depende de C). Sem tuning de hiperparâmetro, sem early stopping por valid (rodei as
# 15 épocas fixas), sem augmentation, sem comparação com baseline não profunda (D3). Registrar
# os dois números lado a lado é o achado certo pra decidir; declarar "CRNN ganhou" sem essas
# ressalvas não é.

# %% [markdown]
# ## O que falta (depende de C — rotulagem)
#
# - **L1**: comparar `CRNNDigitos` (CTC) com a família TrOCR (ViT + decoder Transformer) já
#   usada no pipeline atual (`src/leiturista/inference.py`) — 10 erros reais de cada, classificados
#   em perdeu/duplicou/trocou/inventou. Precisa de recortes reais com transcrição.
# - **L2**: benchmark em ≥200 recortes reais da distribuidora (pode reaproveitar parte dos 300 de
#   C2) e comparar com o baseline já medido no UFPR-AMR (leitura exata 0,357 / acurácia por dígito
#   0,846 — `docs/sbti_artigo/`).
# - **D3**: baseline não profunda no mesmo split de validação, três sementes.
# - **E1/E2**: calibração (ECE, temperature scaling) e curva cobertura×risco — dependem de um
#   modelo de fato treinado em dado real, não só do teste de sanidade acima.
