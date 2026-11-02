"""Privacidade (SDD 5.4), com um serial e uma leitura CONHECIDOS e uma falha 5xx injetada no teste.

Roda o servico em processo (TestClient): a falha vem de um monkeypatch no `MeterOCR` feito aqui,
nunca de codigo de producao. O log nao pode conter serial, leitura, texto de excecao nem IP cru.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from PIL import Image
from starlette.testclient import TestClient

from leiturista.inference import MeterOCR, Prediction
from leiturista.monitoring.logging_setup import LOGGER_NAME
from leiturista.service import LeituristaService

SERIAL = "SN-998877665"
READING = "0123456"
ERROR_TEXT = "texto-secreto-da-excecao-OCR-4242"


def _reset_logger() -> None:
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("SERVING_LOG_DIR", str(tmp_path))
    _reset_logger()
    with TestClient(LeituristaService.to_asgi(), raise_server_exceptions=False) as test_client:
        yield test_client
    _reset_logger()


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), (120, 60, 30)).save(buffer, "PNG")
    return buffer.getvalue()


def _fake_prediction() -> Prediction:
    box = SimpleNamespace(field="leitura", conf=0.93)
    fake: Any = SimpleNamespace(reading=READING, serial=SERIAL, boxes=[box], legible=True, sharpness=40.0, flags=["flag fixa 90"])
    return cast(Prediction, fake)


def test_log_has_no_serial_reading_exception_text_or_ip_even_on_4xx_and_5xx(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {"image": ("a.png", _png(), "image/png")}
    monkeypatch.setattr(MeterOCR, "predict_image", lambda self, image: _fake_prediction())
    ok = client.post("/predict", files=files)
    assert ok.status_code == 200 and ok.json()["numero_medidor"] == SERIAL and ok.json()["consumo"] == READING

    bad = client.post("/predict", data={"outro": "x"})
    assert bad.status_code == 400

    def boom(self: MeterOCR, image: Image.Image) -> Prediction:
        raise RuntimeError(ERROR_TEXT)

    monkeypatch.setattr(MeterOCR, "predict_image", boom)
    broken = client.post("/predict", files=files)
    assert broken.status_code == 500

    text = (tmp_path / "inference.log").read_text(encoding="utf-8")
    events = [json.loads(line) for line in text.splitlines()]
    assert [(e["event"], e.get("status_code")) for e in events] == [
        ("request", None), ("response", 200), ("response", 400), ("request", None), ("response", 500)
    ]
    for secret in (SERIAL, READING, ERROR_TEXT, "testclient", "127.0.0.1"):
        assert secret not in text
    assert events[-1]["error"] == "RuntimeError/inference_failed" and events[-1]["level"] == "ERROR"
    assert events[1]["out_reading_digits"] == len(READING) and events[1]["out_has_serial"] is True
    assert events[1]["out_flags"] == ["flag fixa"]  # digitos removidos
