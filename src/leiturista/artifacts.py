"""Artefatos como blobs no mesmo sqlite do tracking do MLflow.

O MLflow só tem backends de artefato em filesystem/nuvem — usar
`log_artifact`/`log_text` criaria o diretório `mlruns/` espalhado.
Para concentrar TUDO no `mlflow.db`, guardamos os artefatos pequenos
(sumários, CSV de predição) como blobs numa tabela própria, chaveada por
run_id, no mesmo arquivo do tracking.

Artefatos grandes (checkpoint do modelo) ficam em `models/` e entram no
MLflow como param `checkpoint_dir`; sob demanda, também podem ser salvos como
blob (`save_zip`, ex.: `model.zip` do TrOCR-small-printed) no mesmo `.db`.
"""

from __future__ import annotations

import io
import json
import sqlite3
import zipfile
from pathlib import Path

from .paths import MLFLOW_DB

_TABLE = "mapen_artifact_blobs"

_CREATE = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    run_id      TEXT NOT NULL,
    name        TEXT NOT NULL,
    content_type TEXT,
    data        BLOB,
    PRIMARY KEY (run_id, name)
)
"""


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(MLFLOW_DB, timeout=30)
    conn.execute(_CREATE)
    return conn


def save_blob(run_id: str, name: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    conn = _conn()
    try:
        conn.execute(
            f"INSERT OR REPLACE INTO {_TABLE} (run_id, name, content_type, data) VALUES (?, ?, ?, ?)",
            (run_id, name, content_type, data),
        )
        conn.commit()
    finally:
        conn.close()


def save_text(run_id: str, name: str, text: str) -> None:
    save_blob(run_id, name, text.encode("utf-8"), "text/plain")


def save_json(run_id: str, name: str, obj: object) -> None:
    save_blob(run_id, name, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"), "application/json")


def save_zip(run_id: str, name: str, src_dir: Path | str) -> None:
    """Empacota um diretório como blob zip (ex.: checkpoint do modelo)."""
    src_dir = Path(src_dir)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(src_dir))
    save_blob(run_id, name, buf.getvalue(), "application/zip")


def save_model_files(run_id: str, prefix: str, src_dir: Path | str, names: list[str]) -> None:
    """Salva um modelo pronto-para-usar como blobs individuais (`{prefix}/{arquivo}`)
    no mlflow.db — fonte de verdade que o app materializa direto."""
    src_dir = Path(src_dir)
    for name in names:
        save_blob(run_id, f"{prefix}/{name}", (src_dir / name).read_bytes(), "application/octet-stream")


def materialize(run_id: str, prefix: str, out_dir: Path | str, names: list[str] | None = None) -> Path:
    """Grava os blobs `{prefix}/*` de um run num diretório (pronto p/ transformers)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = _conn()
    try:
        rows = conn.execute(
            f"SELECT name, data FROM {_TABLE} WHERE run_id = ? AND name LIKE ?",
            (run_id, f"{prefix}/%"),
        ).fetchall()
    finally:
        conn.close()
    for name, data in rows:
        rel = name[len(prefix) + 1:]
        if names is not None and rel not in names:
            continue
        (out_dir / rel).write_bytes(bytes(data))
    return out_dir


def delete_blob(run_id: str, name: str) -> None:
    conn = _conn()
    try:
        conn.execute(f"DELETE FROM {_TABLE} WHERE run_id = ? AND name = ?", (run_id, name))
        conn.commit()
    finally:
        conn.close()


def read_blob(run_id: str, name: str) -> bytes | None:
    conn = _conn()
    try:
        row = conn.execute(
            f"SELECT data FROM {_TABLE} WHERE run_id = ? AND name = ?",
            (run_id, name),
        ).fetchone()
    finally:
        conn.close()
    return bytes(row[0]) if row else None


def list_names(run_id: str) -> list[tuple[str, int]]:
    conn = _conn()
    try:
        rows = conn.execute(
            f"SELECT name, length(data) FROM {_TABLE} WHERE run_id = ? ORDER BY name",
            (run_id,),
        ).fetchall()
    finally:
        conn.close()
    return [(name, size) for name, size in rows]
