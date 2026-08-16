"""Leiturista - visão computacional para leitura de medidores (Projeto 4 / distribuidoras de energia).

Biblioteca reutilizável do pipeline CV/OCR:
- `leiturista.data`  — extração de imagens (parquets UFPR-AMR) e preparação de datasets
- `leiturista.models`— carregamento/ajuste de modelos TrOCR (processor + vision-encoder-decoder)
- `leiturista.train` — fine-tune com tracking em MLflow
- `leiturista.eval`  — avaliação no split test (exact-match) com registro em MLflow

Uso via CLI (ver `leiturista.cli`):
    leiturista extract
    leiturista train   --epochs 8 --batch 4 --grad-accum 8
    leiturista eval
"""

from __future__ import annotations

from .paths import (
    DEFAULT_EXPERIMENT,
    DEFAULT_TRACKING_URI,
    DATA_DIR,
    FINETUNE_DIR,
    MLFLOW_DB,
    MODELS_DIR,
    ROOT,
    UFPR_AMR_DIR,
)

__all__ = [
    "ROOT",
    "DATA_DIR",
    "UFPR_AMR_DIR",
    "FINETUNE_DIR",
    "MODELS_DIR",
    "MLFLOW_DB",
    "DEFAULT_TRACKING_URI",
    "DEFAULT_EXPERIMENT",
]

__version__ = "0.1.0"
