"""Demo Leiturista — leitura de medidor a partir de foto.

Rodar:
  .venv/bin/streamlit run app/app.py
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from leiturista import paths
from leiturista.inference import MeterOCR

st.set_page_config(page_title="Leiturista — Leitura de medidor", page_icon="🔢",
                   layout="wide")

st.title("🔢 Leiturista — leitura automática de medidor")
st.caption(
    "Projeto 4 · Neoenergia PE · Demo: OCR do display (PP-OCRv5 det + TrOCR-small rec, "
    "com retry por inversão p/ odômetro). Extrai a **leitura** e o **serial** do medidor "
    "e avalia a **legibilidade**."
)

OCR = MeterOCR()


def _annotate(annotated: Image.Image) -> str:
    import io
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data
def _predict(png_bytes: bytes) -> dict:
    pred = OCR.predict_image(Image.open(io.BytesIO(png_bytes)))
    rows = [
        {"campo": b.field, "texto": b.text, "conf": round(b.conf, 3)}
        for b in pred.boxes
    ]
    return {
        "reading": pred.reading,
        "serial": pred.serial,
        "signature": pred.signature,
        "signature_tokens": pred.signature_tokens,
        "legible": pred.legible,
        "flags": pred.flags,
        "boxes": rows,
        "annotated_png": _annotate(pred.annotated),
    }


col_a, col_b = st.columns([1, 2])

with col_a:
    st.subheader("Foto do medidor")
    upload = st.file_uploader("Envie uma foto (jpg/png)", type=["jpg", "jpeg", "png"])
    st.divider()
    st.markdown("**Testar com uma amostra** do test set (UFPR-AMR):")
    test_dir = Path(paths.FINETUNE_DIR)
    samples = sorted(test_dir.glob("ufpr_test_*.png"))[:6] if test_dir.is_dir() else []
    sample_names = [p.name for p in samples]
    chosen = st.selectbox("Amostra", [""] + sample_names) if sample_names else None
    if chosen:
        image = Image.open(next(p for p in samples if p.name == chosen))
    elif upload is not None:
        image = Image.open(upload)
    else:
        st.info("Envie uma foto ou escolha uma amostra.")
        st.stop()

    st.image(image, caption="Entrada", use_container_width=True)

with col_b:
    png = io.BytesIO()
    image.save(png, format="PNG")
    result = _predict(png.getvalue())

    r1, r2, r3 = st.columns(3)
    r1.metric("Leitura (kWh)", result["reading"] or "—")
    r2.metric("Serial", result["serial"] or "—")
    r3.metric("Legibilidade", "OK" if result["legible"] else "Duvidosa")

    st.caption(f"Identidade do medidor: `{result['signature'] or '—'}` "
               f"({len(result['signature_tokens'])} tokens de placa/serial)")

    st.image(result["annotated_png"], caption="Caixas detectadas (verde=leitura, laranja=serial)")

    if result["flags"]:
        st.warning("\n".join(f"- {f}" for f in result["flags"]))

    if result["boxes"]:
        st.dataframe(pd.DataFrame(result["boxes"]), use_container_width=True)
    else:
        st.info("Nenhuma caixa de texto detectada na foto.")

st.divider()
st.caption(
    "Modelos: PP-OCRv5_mobile_det (ONNX, Apache-2.0) + TrOCR-small-printed (MIT) via "
    "mlflow.db · Código: `leiturista.inference` · docs: `docs/app_demo_streamlit.md`"
)
