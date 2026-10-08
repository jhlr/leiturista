"""App de rotulagem do Lab 2 (item C). Cada integrante rotula SÓ o seu bloco (às cegas: não vê o rótulo do outro).

    LEITURISTA_FOTOS_DIR=data/distribuidora_campo streamlit run app/rotular.py

Grava `lab-02/rotulos/rotulos_<nome>.csv` (nome_arquivo, lote, rotulador, classe, leitura, observacao) a cada foto.
O recorte sugerido do display (data/rotulos_crops/, gerado por scripts/preparar_rotulagem.py) é só um auxílio:
o rótulo vale para a FOTO inteira.
"""
import csv
import os
from pathlib import Path

import streamlit as st
from PIL import Image

from leiturista.rotulos import CLASSES, PLANO, ROTULOS_DIR

FOTOS = Path(os.environ.get("LEITURISTA_FOTOS_DIR", "data/distribuidora_campo"))
CROPS = Path("data/rotulos_crops")
FIELDS = ["nome_arquivo", "lote", "rotulador", "classe", "leitura", "observacao"]

st.set_page_config(page_title="Rotulagem", layout="wide")
plano = list(csv.DictReader(open(PLANO, encoding="utf-8")))
nomes = sorted({r["rotulador_1"] for r in plano} | {r["rotulador_2"] for r in plano if r["rotulador_2"]})
nome = st.sidebar.selectbox("Quem é você?", [""] + nomes)
if not nome:
    st.info("Escolha seu nome na barra lateral.")
    st.stop()

meu = [r for r in plano if nome in (r["rotulador_1"], r["rotulador_2"])]
arq = ROTULOS_DIR / f"rotulos_{nome}.csv"
feitos: dict[str, dict[str, str]] = {}
if arq.exists():
    feitos = {r["nome_arquivo"]: r for r in csv.DictReader(open(arq, encoding="utf-8"))}
pend = [r for r in meu if r["nome_arquivo"] not in feitos]
st.sidebar.progress(len(feitos) / max(len(meu), 1), text=f"{len(feitos)}/{len(meu)} rotuladas")
with st.sidebar.expander("Esquema (resumo)"):
    st.markdown("- **legivel**: dá para ler os dígitos do display (digite-os)\n- **ilegivel**: há medidor, mas o display não "
                "dá para ler (borrado, reflexo, escuro, longe, coberto)\n- **sem_medidor**: não há medidor na foto\n\n"
                "Detalhes e exemplos-limite em `lab-02/rotulos/esquema.md`.")
if not pend:
    st.success("Terminou o seu bloco. Envie `lab-02/rotulos/rotulos_%s.csv` ao grupo." % nome)
    st.stop()

r = pend[0]
st.subheader(f"{len(feitos) + 1}/{len(meu)} — {r['nome_arquivo']}")
c1, c2 = st.columns([3, 2])
c1.image(Image.open(FOTOS / r["lote"] / r["nome_arquivo"]), use_container_width=True)
crop = CROPS / f"{Path(r['nome_arquivo']).stem}.png"
if crop.exists():
    c2.caption("Recorte sugerido do display (auxílio automático, pode estar errado)")
    c2.image(Image.open(crop), use_container_width=True)
with c2.form(key=r["nome_arquivo"], clear_on_submit=True):
    classe = st.radio("Classe", CLASSES, horizontal=True)
    leitura = st.text_input("Leitura (só dígitos; só se legivel)")
    obs = st.text_input("Observação (opcional)")
    if st.form_submit_button("Salvar e próxima"):
        if classe == "legivel" and not leitura.strip().isdigit():
            st.error("Classe legivel exige a leitura em dígitos.")
        else:
            novo = not arq.exists()
            with open(arq, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if novo:
                    w.writeheader()
                w.writerow({"nome_arquivo": r["nome_arquivo"], "lote": r["lote"], "rotulador": nome, "classe": classe,
                            "leitura": leitura.strip() if classe == "legivel" else "", "observacao": obs})
            st.rerun()
