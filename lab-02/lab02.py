# %% [markdown]
# # Lab 2 — Luminus: arquitetura profunda do reconhecedor de dígitos (`CRNNDigitos`)
#
# Notebook executado do zero (`Reiniciar e executar tudo`). Caminho das fotos configurável abaixo; as fotos
# **não** vão para o repositório. Treinos longos (D3, L1, L2, L4) rodam por CLI e gravam JSON em `docs/figs/`;
# o notebook os lê e os mostra, e recalcula o que é rápido (B, C4, D1, D2, E).

# %%
import json
import os
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "lab-02" else Path.cwd()
os.chdir(ROOT)
FOTOS_DIR = Path(os.environ.get("LEITURISTA_FOTOS_DIR", ROOT / "data" / "distribuidora_campo"))  # configurável
import csv

import numpy as np
import torch
from PIL import Image

from leiturista import calib, paths
from leiturista.crnn import CRNNDigitos, load_crnn
from leiturista.distribuidora import split_by_lote
from leiturista.sanity import count_params, initial_loss, overfit_one_batch, shape_table

torch.set_num_threads(2)
FIGS = ROOT / "docs" / "figs"
LOTE = paths.DATA_DIR / "distribuidora_amr_lote"
def itens(split):
    return [(LOTE / r["image"], r["label"]) for r in csv.DictReader(open(LOTE / "labels.csv")) if r["split"] == split]

# %% [markdown]
# ## B1 — tabela de formas (entrada real: recorte 1 × 32 × 128)

# %%
x = torch.zeros(1, 1, 32, 128)
print(f"{'camada':<12}{'saída':>22}")
for nome, forma in shape_table(CRNNDigitos(), x):
    print(f"{nome:<12}{str(forma):>22}")
print("\n(B, T, C): T = 32 passos de tempo (largura), C = 11 = 10 dígitos + branco da CTC")

# %% [markdown]
# ## B2 — parâmetros: extração × ajuste fino

# %%
m = CRNNDigitos()
for nome, congela in (("ajuste fino (tudo treinável)", False), ("extração (CNN congelada)", True)):
    tot, tr = count_params(m, congela)
    print(f"{nome:<32} total {tot:>9,}   treináveis {tr:>9,}   ({tr / tot:.0%})")

# %% [markdown]
# ## B3 — ativação de saída e perda
#
# **Cabeça única: `Linear(256 → 11)` por passo de tempo, sem softmax no modelo; perda `CTCLoss(blank=0)`.**
# As classes **não** são exclusivas no nível da sequência (vários dígitos por recorte), mas são exclusivas
# **por passo de tempo** (cada passo emite um símbolo: um dígito ou o branco): isso é um `softmax` sobre 11 símbolos,
# aplicado como `log_softmax` dentro da perda (a CTC marginaliza todos os alinhamentos). Um `sigmoid` por dígito
# (multirrótulo) não serve: não diria *onde* nem *quantas vezes* cada dígito aparece. Exemplo real da base que prova
# a necessidade do branco: leituras com dígito repetido, onde sem o branco `55` colapsaria em `5`.

# %%
todos = [y for p, y in itens("train")] + [y for r in csv.DictReader(open(paths.FINETUNE_DIR / "labels.csv")) for y in [r["label"]]]
rep = [y for y in todos if any(a == b for a, b in zip(y, y[1:]))]
print(f"{len(rep)} de {len(todos)} leituras reais têm dígito repetido em sequência; exemplos:", rep[:8])

# %% [markdown]
# ## C4 — partição POR LOTE e prova de ausência de vazamento

# %%
print(json.dumps(split_by_lote(paths.DATA_DIR / "distribuidora_amr", LOTE), indent=1))

# %% [markdown]
# ## C3 — kappa de Cohen (50 fotos em dupla, às cegas)

# %%
rotulos = ROOT / "lab-02" / "rotulos" / "rotulos.csv"
if rotulos.exists():
    from leiturista.rotulos import merge_and_kappa
    print(json.dumps(merge_and_kappa(), indent=1, ensure_ascii=False))
else:
    print("PENDENTE: rotulagem humana (C2/C3) ainda não concluída; `leiturista rotulos-kappa` calcula quando as planilhas chegarem.")

# %% [markdown]
# ## D1 — perda inicial com pesos aleatórios (lote REAL de 16 recortes)

# %%
treino = itens("train")[:16]
perda, ref = initial_loss(treino)
print(f"perda CTC inicial = {perda:.3f}; referência (saída uniforme sobre 11 símbolos) T·ln(11)/L = 32·ln(11)/L̄ = {ref:.3f}.")
print("A CTC soma sobre alinhamentos, então não há igualdade exata; estar na ordem de grandeza da referência é o esperado (≫ seria inicialização ruim).")

# %% [markdown]
# ## D2 — sobreajuste de 16 recortes REAIS

# %%
d2 = overfit_one_batch(treino, steps=300)
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d2.items() if k not in ("lidos", "alvos")})
print("alvos:", d2["alvos"][:8], "\nlidos:", d2["lidos"][:8])
assert d2["perda_final"] < 0.1

# %% [markdown]
# ## D3 — extração de características × baseline não profunda (3 sementes; `leiturista d3`)

# %%
d3 = json.load(open(FIGS / "d3_resultados.json"))
for k, v in d3["resumo"].items():
    print(k, {m: f"{s['media']:.3f} ± {s['desvio']:.3f}" for m, s in v.items() if m != "alpha"})

# %% [markdown]
# ## E1/E2 — calibração (temperature scaling no VALID) e cobertura × risco (`leiturista calibrate-crnn`)

# %%
res = calib.run(paths.MODELS_DIR / "crnn_lote_ft.pt", LOTE)
print(json.dumps(res, indent=1))
from IPython.display import Image as _I, display
display(_I(filename=str(FIGS / "e1_e2_calibracao_cobertura_risco.png")))

# %% [markdown]
# ## L1 — como cada família erra (`leiturista errors-l1`)

# %%
l1 = json.load(open(FIGS / "l1_erros.json"))
for fam, v in l1.items():
    if isinstance(v, dict):
        print(f"{fam}: acerto exato {v['acerto_exato']:.3f}, erros {v['erros']}, por tipo {v['por_tipo']}")
        for e in v["exemplos"]:
            print("   ", e)

# %% [markdown]
# ## L2 — benchmark do cliente × UFPR-AMR (`leiturista l2`)

# %%
l2 = json.load(open(FIGS / "l2_gap_dominio.json"))
print("n:", l2["n"])
for k, v in l2["modelos"].items():
    print(f"{k:<50} cliente {v['cliente_test_por_lote']['leitura_exata']:.3f}/{v['cliente_test_por_lote']['acuracia_por_digito']:.3f}   "
          f"UFPR {v['ufpr_amr_test']['leitura_exata']:.3f}/{v['ufpr_amr_test']['acuracia_por_digito']:.3f}")

# %% [markdown]
# ## L4 — latência por estágio (`scripts/latencia_por_estagio.py`)

# %%
l4 = json.load(open(FIGS / "l4_latencia.json"))
print("total:", {k: round(v, 2) for k, v in l4["total"].items()})
for k, v in l4["estagios"].items():
    print(f"{k:<24} média {v['media']:.2f}s  p50 {v['p50']:.2f}s  p90 {v['p90']:.2f}s  ({v['fracao_do_total']:.0%} do total)")
