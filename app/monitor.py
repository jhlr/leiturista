"""Dashboard de monitoramento do pipeline Leiturista.

Roda inferência em batch sobre um diretório de fotos e exibe:
- Distribuição semáforo (verde / amarelo / vermelho)
- Latência por estágio (det / TrOCR / PP-OCR)
- Galeria interativa por bucket com a imagem anotada
- Debug de caixas detectadas (texto bruto, campo, confiança) por foto
- Tabela de resultados exportável

Uso:
    .venv/bin/streamlit run app/monitor.py
    # Com fotos reais (não versionadas):
    LEITURISTA_FOTOS_DIR=data/distribuidora_campo .venv/bin/streamlit run app/monitor.py

Variáveis de ambiente opcionais (herdadas do inference.py):
    LEITURISTA_DET_ONNX      caminho do modelo det (padrão: models/pp_ocr_v5_mobile_det_onnx)
    LEITURISTA_VISOR_CLS     caminho do classificador de visor (desativado por padrão)
    LEITURISTA_DET2_ONNX     segundo detector (desativado por padrão)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Leiturista — Monitor",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constantes / regras de triagem
# ---------------------------------------------------------------------------
# Verde: leitura detectada + legível + confiança alta  → aprovação com 1 clique
# Vermelho: sem leitura ou explicitamente ilegível     → rejeição com 1 clique
# Amarelo: qualquer caso intermediário                 → revisão humana completa

CONF_VERDE = 0.55       # limiar mínimo de confiança para o bucket verde
CONF_VERMELHO = 0.0     # confiança = 0 e sem leitura → vermelho
FOTOS_DIR_DEFAULT = Path(os.environ.get("LEITURISTA_FOTOS_DIR", "samples"))

IMAGENS_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

from leiturista.ocorrencias import (
    CATALOGO_NOTAS,
    NOTAS_IMPEDIMENTO_AUSENCIA,
    NOTAS_LEITURA_REGULAR,
    avaliar_coerencia,
    normalizar_nota,
)

# ---------------------------------------------------------------------------
# Triagem semáforo contextual (Tarefa 2: Coerência Foto <-> Ocorrência)
# ---------------------------------------------------------------------------

def triagem(pred, nota: str = "NA", leitura_sistema: str = "") -> tuple[str, str, str]:
    """Avalia coerência da evidência com a nota registrada.
    - Impedimento (I100, B100): sem leitura na foto = SUCESSO (Verde: comprova o impedimento).
    - Leitura regular (NA, L101): leitura nítida detectada = SUCESSO (Verde: comprova o consumo).
    - Falhas ou discordâncias = Vermelho ou Amarelo."""
    res = avaliar_coerencia(pred, nota, leitura_sistema)
    return res.bucket, res.justificativa, res.categoria_nota


BUCKET_LABEL = {
    "verde":    "🟢 Verde — Evidência Consistente (Aprovar)",
    "amarelo":  "🟡 Amarelo — Revisão Humana",
    "vermelho": "🔴 Vermelho — Inconsistência de Campo",
}
BUCKET_COLOR = {"verde": "#1a7a4a", "amarelo": "#b58c00", "vermelho": "#a31515"}


@st.cache_data
def carregar_metadados_csv(root_str: str) -> dict[str, dict]:
    """Indexa os metadados dos CSVs BaseExtracao_*.csv por nome da foto."""
    root = Path(root_str)
    meta = {}
    if not root.exists():
        return meta
    for c in root.rglob("BaseExtracao_*.csv"):
        try:
            df = pd.read_csv(c, sep=";", encoding="latin1", dtype=str, on_bad_lines="skip")
            col_foto = "Foto do medidor"
            col_nota = "Nota de Leitura Atual"
            col_pos = "Posicao do medidor lida"
            col_med = "Numero do medidor"
            if col_foto in df.columns:
                for _, row in df.iterrows():
                    f_name = str(row.get(col_foto, "")).strip()
                    if f_name and f_name.lower() not in ("nan", "none", ""):
                        raw_nota = row.get(col_nota)
                        nota_val = str(raw_nota).strip() if pd.notna(raw_nota) and str(raw_nota).strip() else "NA"
                        meta[f_name] = {
                            "nota": normalizar_nota(nota_val),
                            "leitura_sistema": str(row.get(col_pos, "")).strip() if pd.notna(row.get(col_pos)) else "",
                            "medidor_sistema": str(row.get(col_med, "")).strip() if pd.notna(row.get(col_med)) else "",
                        }
        except Exception:
            pass
    return meta

# ---------------------------------------------------------------------------
# Cache do modelo (singleton por sessão)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Carregando modelos OCR…")
def get_ocr():
    from leiturista.inference import MeterOCR
    return MeterOCR()


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def listar_fotos(root: Path, lote: str | None) -> list[Path]:
    """Lista recursivamente todas as imagens dentro de root (ou root/lote)."""
    base = root / lote if lote else root
    return sorted(
        p for p in base.rglob("*")
        if p.suffix.lower() in IMAGENS_EXT and p.is_file()
    )


def _instrumentar_ocr(ocr):
    """Monkey-patch nos métodos de MeterOCR para medir latência por estágio.
    Devolve um dict compartilhado que acumula tempos (zerado a cada foto)."""
    tempos: dict[str, float] = {}

    def _wrap(nome, metodo):
        def wrapped(*args, **kwargs):
            t = time.perf_counter()
            resultado = metodo(*args, **kwargs)
            tempos[nome] = tempos.get(nome, 0.0) + (time.perf_counter() - t)
            return resultado
        return wrapped

    ocr._det_boxes_orig = ocr._det_boxes
    ocr._trocr_recognize_orig = ocr._trocr_recognize
    ocr._rec_recognize_orig = ocr._rec_recognize
    ocr._detect_rotation_orig = ocr._detect_rotation

    ocr._det_boxes = _wrap("det", ocr._det_boxes)
    ocr._trocr_recognize = _wrap("trocr", ocr._trocr_recognize)
    ocr._rec_recognize = _wrap("ppocr", ocr._rec_recognize)
    ocr._detect_rotation = _wrap("rotacao", ocr._detect_rotation)

    return tempos


def processar(paths_: list[Path], ocr, barra, status_txt, limite: int | None, metadados: dict[str, dict] | None = None) -> list[dict]:
    """Roda inferência e devolve lista de dicts com resultado + bucket + latência por estágio."""
    total = len(paths_) if limite is None else min(len(paths_), limite)
    resultados = []
    meta_dict = metadados or {}

    # Instrumenta uma única vez (idempotente se já foi feito)
    if not hasattr(ocr, "_det_boxes_orig"):
        tempos_ref = _instrumentar_ocr(ocr)
    else:
        # Recupera o dict compartilhado já criado — precisamos acessar o closure
        # Estratégia: recria o wrapper (idempotente, não aninha)
        tempos_ref = _instrumentar_ocr(ocr)

    for i, p in enumerate(paths_[:total]):
        status_txt.text(f"Processando {i + 1}/{total}: {p.name}")
        barra.progress((i + 1) / total)

        # Zera acumulador desta foto
        tempos_ref.clear()
        t0 = time.perf_counter()

        # Metadados de campo da foto (se houver BaseExtracao_*.csv)
        meta_foto = meta_dict.get(p.name, {})
        nota = meta_foto.get("nota", "NA")
        leitura_sistema = meta_foto.get("leitura_sistema", "")
        medidor_sistema = meta_foto.get("medidor_sistema", "")

        try:
            img = Image.open(p).convert("RGB")
            pred = ocr.predict_image(img)
            lat_total = round((time.perf_counter() - t0) * 1000)
            
            # Triagem contextual foto <-> nota de ocorrência
            bucket, justificativa, cat_nota = triagem(pred, nota, leitura_sistema)
            desc_nota = CATALOGO_NOTAS.get(nota, {}).get("desc", "Ocorrência")

            leitura_boxes = [b for b in pred.boxes if b.field == "leitura"]
            confianca = max((b.conf for b in leitura_boxes), default=0.0) if pred.reading else 0.0
            # Caixas cruas para debug de serial
            caixas = [
                {"campo": b.field, "texto": b.text, "conf": round(b.conf, 3), "source": b.source}
                for b in pred.boxes
            ]
            resultados.append({
                "arquivo": p.name,
                "lote": p.parent.name,
                "caminho": str(p),
                "bucket": bucket,
                "nota": nota,
                "desc_nota": desc_nota,
                "categoria_nota": cat_nota,
                "justificativa": justificativa,
                "leitura_sistema": leitura_sistema or "—",
                "medidor_sistema": medidor_sistema or "—",
                "leitura": pred.reading or "—",
                "serial": pred.serial or "—",
                "confianca": round(float(confianca), 3),
                "legivel": pred.legible,
                "nitidez": round(pred.sharpness, 1) if pred.sharpness is not None else None,
                "flags": ", ".join(pred.flags) if pred.flags else "",
                "latencia_ms": lat_total,
                "lat_det_ms": round(tempos_ref.get("det", 0) * 1000),
                "lat_trocr_ms": round(tempos_ref.get("trocr", 0) * 1000),
                "lat_ppocr_ms": round(tempos_ref.get("ppocr", 0) * 1000),
                "lat_rot_ms": round(tempos_ref.get("rotacao", 0) * 1000),
                "annotated": pred.annotated,
                "caixas": caixas,
            })
        except Exception as e:
            resultados.append({
                "arquivo": p.name,
                "lote": p.parent.name,
                "caminho": str(p),
                "bucket": "vermelho",
                "nota": nota,
                "desc_nota": CATALOGO_NOTAS.get(nota, {}).get("desc", "Ocorrência"),
                "categoria_nota": "erro",
                "justificativa": f"Erro de processamento da imagem: {e}",
                "leitura_sistema": leitura_sistema or "—",
                "medidor_sistema": medidor_sistema or "—",
                "leitura": "—",
                "serial": "—",
                "confianca": 0.0,
                "legivel": False,
                "nitidez": None,
                "flags": f"ERRO: {e}",
                "latencia_ms": round((time.perf_counter() - t0) * 1000),
                "lat_det_ms": 0, "lat_trocr_ms": 0, "lat_ppocr_ms": 0, "lat_rot_ms": 0,
                "annotated": None,
                "caixas": [],
            })
    return resultados


# ---------------------------------------------------------------------------
# UI — barra lateral (configuração)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuração")

    # Status de Aceleração por Hardware
    import onnxruntime as ort
    tem_dml = "DmlExecutionProvider" in ort.get_available_providers()
    if tem_dml:
        st.success("⚡ **GPU Ativa:** DirectML (AMD Radeon)")
    else:
        st.info("🖥️ **Processamento:** CPU")

    fotos_dir_str = st.text_input(
        "Diretório de fotos",
        value=str(FOTOS_DIR_DEFAULT),
        help="Pode ser o caminho absoluto para data/distribuidora_campo ou qualquer pasta com imagens.",
    )
    fotos_dir = Path(fotos_dir_str)

    # Carrega metadados dos CSVs de ocorrências
    metadados = carregar_metadados_csv(str(fotos_dir))
    if metadados:
        st.success(f"📋 **{len(metadados):,}** ocorrências mapeadas de CSVs de campo")
    else:
        st.caption("ℹ Nenhum BaseExtracao_*.csv encontrado nesta pasta (fotos tratadas como NA/Leitura Normal)")

    # Detecta lotes (subpastas)
    lotes_disponiveis: list[str] = []
    if fotos_dir.exists():
        lotes_disponiveis = sorted(
            d.name for d in fotos_dir.iterdir()
            if d.is_dir() and any(f.suffix.lower() in IMAGENS_EXT for f in d.rglob("*"))
        )

    lote_sel = st.selectbox(
        "Lote (subpasta)",
        ["— todos —"] + lotes_disponiveis,
        help="Filtra um lote específico ou processa todos.",
    )
    lote: str | None = None if lote_sel == "— todos —" else lote_sel

    limite = st.slider(
        "Limite de fotos",
        min_value=1, max_value=500, value=50, step=10,
        help="Processa no máximo N fotos (útil para inspeção rápida).",
    )

    st.divider()
    st.markdown("**Limiares de triagem**")
    conf_verde = st.slider("Confiança mínima para 🟢 Verde", 0.0, 1.0, CONF_VERDE, 0.05)
    st.caption("Abaixo desse limiar → 🟡 Amarelo. Sem leitura → 🔴 Vermelho.")

    st.divider()
    rodar = st.button("▶ Rodar pipeline", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# UI — cabeçalho
# ---------------------------------------------------------------------------

st.title("🔍 Leiturista — Monitor de Pipeline")
st.caption(
    "Triagem Semáforo Contextual: 🟢 Evidência Consistente (comprova leitura ou ausência/impedimento) · 🟡 Revisão Humana · 🔴 Inconsistência de Evidência"
)

# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

if rodar:
    if not fotos_dir.exists():
        st.error(f"Diretório não encontrado: `{fotos_dir}`")
        st.stop()

    fotos = listar_fotos(fotos_dir, lote)
    if not fotos:
        st.warning("Nenhuma imagem encontrada no diretório selecionado.")
        st.stop()

    total_fotos = len(fotos)
    st.info(f"Encontradas **{total_fotos}** imagens. Processando até **{limite}**…")

    ocr = get_ocr()

    barra = st.progress(0)
    status_txt = st.empty()

    resultados = processar(fotos, ocr, barra, status_txt, limite, metadados=metadados)

    barra.empty()
    status_txt.empty()

    # Salva na sessão para persistir entre reruns do Streamlit
    st.session_state["resultados"] = resultados
    st.session_state["conf_verde_usado"] = conf_verde


# ---------------------------------------------------------------------------
# Exibição (persiste mesmo sem re-rodar)
# ---------------------------------------------------------------------------

if "resultados" not in st.session_state:
    st.info("Configure os parâmetros e clique **▶ Rodar pipeline** para começar.")
    st.stop()

resultados: list[dict] = st.session_state["resultados"]

# Re-classifica com o limiar atual considerando a nota de ocorrência
for r in resultados:
    nota = r.get("nota", "NA")
    if nota in NOTAS_IMPEDIMENTO_AUSENCIA:
        if r["leitura"] == "—":
            r["bucket_atual"] = "verde"
        else:
            r["bucket_atual"] = "amarelo"
    elif nota in NOTAS_LEITURA_REGULAR or nota == "NA":
        if r["leitura"] != "—" and r["legivel"] and r["nitidez"] is not None and r["nitidez"] >= 25.0 and r["confianca"] >= conf_verde:
            r["bucket_atual"] = "verde"
        elif r["leitura"] == "—":
            r["bucket_atual"] = "vermelho"
        else:
            r["bucket_atual"] = "amarelo"
    else:
        if r["leitura"] != "—" and r["legivel"]:
            r["bucket_atual"] = "verde"
        else:
            r["bucket_atual"] = "amarelo"

contagem = {"verde": 0, "amarelo": 0, "vermelho": 0}
for r in resultados:
    contagem[r["bucket_atual"]] += 1

total = len(resultados)

# ---------------------------------------------------------------------------
# Resumo — métricas principais
# ---------------------------------------------------------------------------

st.subheader("📊 Resumo")

col_v, col_a, col_r, col_lat = st.columns(4)
col_v.metric(
    "🟢 Verde",
    f"{contagem['verde']}",
    f"{contagem['verde'] / total:.0%} do total",
    delta_color="off",
)
col_a.metric(
    "🟡 Amarelo",
    f"{contagem['amarelo']}",
    f"{contagem['amarelo'] / total:.0%} do total",
    delta_color="off",
)
col_r.metric(
    "🔴 Vermelho",
    f"{contagem['vermelho']}",
    f"{contagem['vermelho'] / total:.0%} do total",
    delta_color="off",
)
lat_media = sum(r["latencia_ms"] for r in resultados) / total
col_lat.metric("⏱ Latência média", f"{lat_media:.0f} ms/foto")

# --- Latência por estágio -------------------------------------------------
with st.expander("⏱ Latência por estágio (det / TrOCR / PP-OCR / rotação)", expanded=False):
    lat_det   = sum(r["lat_det_ms"]   for r in resultados) / total
    lat_trocr = sum(r["lat_trocr_ms"] for r in resultados) / total
    lat_ppocr = sum(r["lat_ppocr_ms"] for r in resultados) / total
    lat_rot   = sum(r["lat_rot_ms"]   for r in resultados) / total
    lat_outro = lat_media - lat_det - lat_trocr - lat_ppocr - lat_rot

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🔲 Det (ONNX)", f"{lat_det:.0f} ms", f"{lat_det/lat_media:.0%}")
    c2.metric("🤖 TrOCR", f"{lat_trocr:.0f} ms", f"{lat_trocr/lat_media:.0%}")
    c3.metric("⚡ PP-OCRv6", f"{lat_ppocr:.0f} ms", f"{lat_ppocr/lat_media:.0%}")
    c4.metric("🔄 Rotação", f"{lat_rot:.0f} ms", f"{lat_rot/lat_media:.0%}")
    c5.metric("🔧 Resto (crop/merge)", f"{max(lat_outro,0):.0f} ms", f"{max(lat_outro,0)/lat_media:.0%}")

    st.caption(
        "**TrOCR** (62M params, PyTorch) é o gargalo em CPU. "
        "Para reduzir: desative-o via `LEITURISTA_DET_ONNX` ou use `PP-OCRv6` como único leitor."
    )

    # Mini gráfico de barras horizontal
    estagios = {"Det": lat_det, "TrOCR": lat_trocr, "PP-OCR": lat_ppocr, "Rotação": lat_rot, "Resto": max(lat_outro, 0)}
    df_lat = pd.DataFrame({"Estágio": list(estagios.keys()), "ms (média)": list(estagios.values())})
    st.bar_chart(df_lat.set_index("Estágio"), height=180)

    # Distribuição por foto
    with st.expander("Ver distribuição por foto"):
        df_lat_foto = pd.DataFrame([
            {"arquivo": r["arquivo"], "total": r["latencia_ms"], "det": r["lat_det_ms"],
             "trocr": r["lat_trocr_ms"], "ppocr": r["lat_ppocr_ms"], "rotação": r["lat_rot_ms"]}
            for r in resultados
        ])
        st.dataframe(df_lat_foto, use_container_width=True, height=200)

# Barra de proporção visual
verde_pct = contagem["verde"] / total
amarelo_pct = contagem["amarelo"] / total
vermelho_pct = contagem["vermelho"] / total

st.markdown(
    f"""
<div style="display:flex;height:20px;border-radius:4px;overflow:hidden;margin-bottom:1em">
  <div style="width:{verde_pct:.1%};background:#1a7a4a" title="Verde {verde_pct:.1%}"></div>
  <div style="width:{amarelo_pct:.1%};background:#d4a800" title="Amarelo {amarelo_pct:.1%}"></div>
  <div style="width:{vermelho_pct:.1%};background:#a31515" title="Vermelho {vermelho_pct:.1%}"></div>
</div>
""",
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Tabela completa + download
# ---------------------------------------------------------------------------

df = pd.DataFrame([
    {
        "bucket": r["bucket_atual"],
        "nota": r.get("nota", "NA"),
        "ocorrência": r.get("desc_nota", "Normal"),
        "arquivo": r["arquivo"],
        "lote": r["lote"],
        "leitura OCR": r["leitura"],
        "leitura sistema": r.get("leitura_sistema", "—"),
        "serial OCR": r["serial"],
        "medidor sistema": r.get("medidor_sistema", "—"),
        "confiança": r["confianca"],
        "coerência": r.get("justificativa", ""),
        "latência (ms)": r["latencia_ms"],
    }
    for r in resultados
])

BUCKET_ICON = {"verde": "🟢", "amarelo": "🟡", "vermelho": "🔴"}
df["bucket"] = df["bucket"].map(BUCKET_ICON)

col_tbl, col_dl = st.columns([4, 1])
col_tbl.subheader("📋 Resultados")
col_dl.download_button(
    "⬇ Exportar CSV",
    data=df.to_csv(index=False, encoding="utf-8-sig"),
    file_name="monitor_resultados.csv",
    mime="text/csv",
    use_container_width=True,
)

filtro_bucket = st.multiselect(
    "Filtrar por bucket",
    options=["🟢", "🟡", "🔴"],
    default=["🟢", "🟡", "🔴"],
)
df_filtrado = df[df["bucket"].isin(filtro_bucket)] if filtro_bucket else df
st.dataframe(df_filtrado, use_container_width=True, height=300)

st.divider()

# ---------------------------------------------------------------------------
# Galeria por bucket
# ---------------------------------------------------------------------------

st.subheader("🖼 Galeria por bucket")

bucket_map = {"🟢 Verde": "verde", "🟡 Amarelo": "amarelo", "🔴 Vermelho": "vermelho"}
aba_selecionada = st.radio(
    "Bucket",
    list(bucket_map.keys()),
    horizontal=True,
    label_visibility="collapsed",
)
bucket_filtro = bucket_map[aba_selecionada]

galeria = [r for r in resultados if r["bucket_atual"] == bucket_filtro]

if not galeria:
    st.info(f"Nenhuma foto neste bucket.")
else:
    max_galeria = st.slider("Fotos na galeria", 4, min(50, len(galeria)), min(12, len(galeria)), 4)
    galeria_exibida = galeria[:max_galeria]

    colunas = st.columns(4)
    for i, r in enumerate(galeria_exibida):
        col = colunas[i % 4]
        img = r["annotated"] if r["annotated"] is not None else Image.open(r["caminho"])
        nota_cap = r.get("nota", "NA")
        col.image(img, caption=f"{r['arquivo']}\nNota: {nota_cap} · Leitura: {r['leitura']}", use_container_width=True)

    if len(galeria) > max_galeria:
        st.caption(f"Exibindo {max_galeria} de {len(galeria)} fotos neste bucket.")

st.divider()

# ---------------------------------------------------------------------------
# Inspeção individual
# ---------------------------------------------------------------------------

st.subheader("🔬 Inspeção individual")

arquivos = [r["arquivo"] for r in resultados]
sel = st.selectbox("Escolha uma foto", arquivos)
r_sel = next(r for r in resultados if r["arquivo"] == sel)

c1, c2 = st.columns([3, 2])
img_exibir = r_sel["annotated"] if r_sel["annotated"] is not None else Image.open(r_sel["caminho"])
c1.image(img_exibir, caption=r_sel["arquivo"], use_container_width=True)

with c2:
    buc = r_sel["bucket_atual"]
    st.markdown(
        f"<h3 style='color:{BUCKET_COLOR[buc]}'>{BUCKET_LABEL[buc]}</h3>",
        unsafe_allow_html=True,
    )
    nota = r_sel.get("nota", "NA")
    desc_nota = r_sel.get("desc_nota", "Normal")
    cat_nota = r_sel.get("categoria_nota", "leitura")
    just = r_sel.get("justificativa", "")

    st.markdown(f"**Ocorrência:** `{nota}` — {desc_nota}")
    if cat_nota == "impedimento":
        st.info("🛡️ **Evidência de Impedimento:** Foto tirada para justificar ausência/bloqueio de medidor.")

    st.markdown(f"**Avaliação:** _{just}_")
    st.divider()

    col_l1, col_l2 = st.columns(2)
    col_l1.markdown(f"**Leitura OCR:** `{r_sel['leitura']}`")
    col_l2.markdown(f"**Leitura Sistema:** `{r_sel.get('leitura_sistema', '—')}`")

    col_s1, col_s2 = st.columns(2)
    serial_val = r_sel["serial"]
    if serial_val == "—":
        col_s1.markdown("**Serial OCR:** :red[não detectado]")
    else:
        col_s1.markdown(f"**Serial OCR:** `{serial_val}`")
    col_s2.markdown(f"**Medidor Sistema:** `{r_sel.get('medidor_sistema', '—')}`")

    st.markdown(f"**Confiança:** `{r_sel['confianca']:.3f}` · **Legível:** `{r_sel['legivel']}` · **Nitidez:** `{r_sel['nitidez']}`")
    st.markdown(
        f"**Latência:** `{r_sel['latencia_ms']} ms` "
        f"_(det {r_sel['lat_det_ms']}ms · TrOCR {r_sel['lat_trocr_ms']}ms · PP-OCR {r_sel['lat_ppocr_ms']}ms)_"
    )
    if r_sel["flags"]:
        st.warning(f"⚠ Flags: {r_sel['flags']}")
    else:
        st.success("Sem flags de erro")

# Debug de caixas detectadas (texto bruto que o OCR leu)
with st.expander("🔍 Debug: todas as caixas detectadas (texto bruto)", expanded=False):
    caixas = r_sel.get("caixas", [])
    if not caixas:
        st.info("Nenhuma caixa detectada nesta foto.")
    else:
        df_caixas = pd.DataFrame(caixas)
        CAMPO_ICON = {"leitura": "🟩 leitura", "serial": "🟦 serial", "outro": "⬜ outro"}
        df_caixas["campo"] = df_caixas["campo"].map(lambda x: CAMPO_ICON.get(x, x))
        st.dataframe(df_caixas, use_container_width=True)

        n_serial = sum(1 for c in caixas if "serial" in c["campo"])
        n_leitura = sum(1 for c in caixas if "leitura" in c["campo"])
        n_outro = len(caixas) - n_serial - n_leitura
        st.caption(
            f"Total: **{len(caixas)}** caixas — "
            f"🟩 leitura: {n_leitura} · 🟦 serial: {n_serial} · ⬜ outro: {n_outro}  \n"
            "Serial é heurístico: texto com ≥6 chars, mistura letra+dígito, sem símbolos de spec. "
            "Se o OCR leu só dígitos ou o texto tem `= ° () %`, cai em 'outro'."
        )
