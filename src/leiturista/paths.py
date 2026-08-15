"""Caminhos canônicos e padrões do subprojeto CV (fonte única de verdade)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
UFPR_AMR_DIR = DATA_DIR / "ufpr_amr"
FINETUNE_DIR = DATA_DIR / "finetune_ufpramr"
MODELS_DIR = ROOT / "models"

# MLflow: tracking local em SQLite (o file store caiu em maintenance mode no
# mlflow 3.15 — sqlite é o backend recomendado). Ver `docs/finetune_trocr_ufpramr.md`.
MLFLOW_DB = ROOT / "mlflow.db"
DEFAULT_TRACKING_URI = f"sqlite:///{MLFLOW_DB}"
DEFAULT_EXPERIMENT = "trocr-ufpramr"

# UFPR-AMR: contadores com 5 dígitos (leitura = inteiro até 99999).
NUM_READING_DIGITS = 5
