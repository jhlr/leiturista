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
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: leiturista (3.13.13)
#     language: python
#     name: python3
# ---

# %%
"""Esqueleto de referência do Lab 2: dois blocos profundos que aparecem, com nomes
diferentes, nas arquiteturas dos quatro grupos do Projeto 4 (distribuidora).

1. ModeloTriagem  - backbone CNN pré-treinável + duas cabeças (cena e qualidade).
2. CRNNDigitos    - leitor de dígitos do display recortado, treinado com CTC.

Roda em CPU, sem baixar nada: `python esqueleto_arquitetura.py`.
Com `--pretreinado`, baixa os pesos ImageNet da MobileNetV3 (precisa de rede).
"""
import argparse, math, torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
torch.manual_seed(42)
torch.set_num_threads(2)

CLASSES_CENA = ["digital", "ciclometrico", "cena_ocorrencia", "nao_medidor"]
DEFEITOS = ["reflexo", "fora_de_foco", "tampa_suja", "enquadramento", "display_apagado"]
ALTURA, LARGURA = 480, 360  # resolução nativa das fotos do PDA (retrato, 3:4)



# %%
class ModeloTriagem(nn.Module):
    """Cena é exclusiva (uma foto é de um tipo só): softmax + entropia cruzada.
    Defeitos coexistem (reflexo E fora de foco): uma sigmoide por defeito + BCE."""

    def __init__(self, pretreinado: bool = False, p_dropout: float = 0.2):
        super().__init__()
        pesos = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretreinado else None
        base = mobilenet_v3_small(weights=pesos)
        self.extrator = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        dim = base.classifier[0].in_features  # 576
        self.cabeca_cena = nn.Sequential(nn.Dropout(p_dropout), nn.Linear(dim, len(CLASSES_CENA)))
        self.cabeca_qualidade = nn.Sequential(nn.Dropout(p_dropout), nn.Linear(dim, len(DEFEITOS)))

    def forward(self, x):
        z = self.pool(self.extrator(x)).flatten(1)
        return self.cabeca_cena(z), self.cabeca_qualidade(z)



# %%
class CRNNDigitos(nn.Module):
    """CNN reduz a altura a 1 e mantém a largura como eixo de tempo; a GRU lê da esquerda
    para a direita; a CTC dispensa saber onde cada dígito está no recorte."""

    def __init__(self, n_simbolos: int = 10, oculto: int = 128):
        super().__init__()

        def bloco(cin, cout, pool):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(pool))

        self.cnn = nn.Sequential(
            bloco(1, 32, (2, 2)),    # 32x128 -> 16x64
            bloco(32, 64, (2, 2)),   # -> 8x32
            bloco(64, 128, (2, 1)),  # -> 4x32  (pool só na altura: preserva passos de tempo)
            bloco(128, 128, (4, 1)), # -> 1x32
        )
        self.rnn = nn.GRU(128, oculto, bidirectional=True, batch_first=True)
        self.saida = nn.Linear(2 * oculto, n_simbolos + 1)  # +1: o "branco" da CTC, índice 0

    def forward(self, x):
        f = self.cnn(x).squeeze(2).permute(0, 2, 1)  # (B, T=32, C=128)
        h, _ = self.rnn(f)
        return self.saida(h)                         # (B, T, n_simbolos+1)



# %%
def perda_triagem(logits_cena, logits_qual, y_cena, y_qual, peso_pos=None, lam=1.0):
    return F.cross_entropy(logits_cena, y_cena) + lam * F.binary_cross_entropy_with_logits(
        logits_qual, y_qual, pos_weight=peso_pos
    )

def decidir(logits_cena, logits_qual, lim_cena=0.90, lim_defeito=0.50, lim_limpo=0.10):
    """Três saídas. DÚVIDA não é falha do modelo: é a faixa em que ele se recusa a decidir.
    Os limiares são placeholders; devem sair da validação, nunca do teste."""
    p_cena = logits_cena.softmax(-1)
    p_def = logits_qual.sigmoid()
    conf_cena, cena = p_cena.max(-1)
    decisoes = []
    for c, pc, pd in zip(cena.tolist(), conf_cena.tolist(), p_def):
        if CLASSES_CENA[c] == "nao_medidor" and pc >= lim_cena:
            decisoes.append("REJEITADA")
        elif pd.max().item() >= lim_defeito and pc >= lim_cena:
            decisoes.append("REJEITADA")
        elif pd.max().item() <= lim_limpo and pc >= lim_cena:
            decisoes.append("ACEITA")
        else:
            decisoes.append("DUVIDA")
    return decisoes

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

    alvos = [(n, m) for n, m in modelo.named_children()]
    if hasattr(modelo, "extrator"):
        alvos = [(f"extrator.{n}", m) for n, m in modelo.extrator.named_children()] + alvos[1:]
    ganchos = [m.register_forward_hook(gancho(n)) for n, m in alvos]
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



# %%
def main(pretreinado: bool):
    print("=" * 64, "\n1) ModeloTriagem (MobileNetV3-Small, duas cabeças)\n" + "=" * 64)
    m = ModeloTriagem(pretreinado=pretreinado)
    x = torch.randn(8, 3, ALTURA, LARGURA)
    tabela_de_formas(m, x)
    total, treino = contar(m)
    print(f"  parâmetros: {total:,} total, {treino:,} treináveis")

    for p in m.extrator.parameters():
        p.requires_grad = False
    total, treino = contar(m)
    print(f"  extrator congelado: {treino:,} treináveis de {total:,}")
    for p in m.extrator.parameters():
        p.requires_grad = True

    y_cena = torch.randint(0, len(CLASSES_CENA), (8,))
    y_qual = torch.randint(0, 2, (8, len(DEFEITOS))).float()
    m.eval()
    with torch.no_grad():
        lc, lq = m(x)
        ce = F.cross_entropy(lc, y_cena).item()
        bce = F.binary_cross_entropy_with_logits(lq, y_qual).item()
    print(f"  perda inicial CE  = {ce:.3f}  (esperado ~ ln({len(CLASSES_CENA)}) = {math.log(len(CLASSES_CENA)):.3f})")
    print(f"  perda inicial BCE = {bce:.3f}  (esperado ~ ln(2) = {math.log(2):.3f})")
    print("  decisões com pesos aleatórios:", decidir(lc, lq))

    m.cabeca_cena[0].p = m.cabeca_qualidade[0].p = 0.0
    x_peq, yc_peq, yq_peq = x[:4], y_cena[:4], y_qual[:4]
    print("  sobreajuste de um lote (4 imagens):")
    final = sobreajustar_um_lote(m, 60, lambda: perda_triagem(*m(x_peq), yc_peq, yq_peq))
    assert final < 0.1, "não decorou 4 imagens: há defeito no modelo, na perda ou nos rótulos"

    print("\n" + "=" * 64, "\n2) CRNNDigitos (CNN + GRU bidirecional + CTC)\n" + "=" * 64)
    r = CRNNDigitos()
    xr = torch.randn(4, 1, 32, 128)
    r.tabela_de_formas(xr)
    total, _ = contar(r)
    print(f"  parâmetros: {total:,}")

    leituras = ["04821", "17413", "00029", "69900"]
    alvos = torch.tensor([int(c) + 1 for s in leituras for c in s])
    comp_alvo = torch.tensor([len(s) for s in leituras])
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    def passo_ctc():
        lp = r(xr).log_softmax(-1).permute(1, 0, 2)  # CTC quer (T, B, C)
        comp_ent = torch.full((xr.size(0),), lp.size(0), dtype=torch.long)
        return ctc(lp, alvos, comp_ent, comp_alvo)

    print("  sobreajuste de um lote (4 recortes):")
    r.sobreajustar_um_lote(300, passo_ctc)
    r.eval()
    with torch.no_grad():
        lidos = decodificar_ctc(r(xr))
    print("  alvo :", leituras)
    print("  lido :", lidos)
    acerto = sum(a == b for a, b in zip(leituras, lidos)) / len(leituras)
    print(f"  acerto exato (sequência inteira): {acerto:.0%}")



# %%
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretreinado", action="store_true")
    main(ap.parse_args().pretreinado)


