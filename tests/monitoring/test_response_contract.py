"""Guarda do contrato de resposta: o monitoramento nao pode mudar o `/predict`.

Campos e tipos capturados do servico ANTES da alteracao (2026-11-01, `just serve` sem
monitoramento) para as 5 imagens de `just samples`. Nao compara valores.
"""

from __future__ import annotations

import pytest

from .conftest import SAMPLE_NAMES, LiveService

NoneType = type(None)
CONTRACT: dict[str, tuple[type, ...]] = {
    "numero_medidor": (str, NoneType),
    "funcao": (str,),
    "consumo": (str, NoneType),
    "confianca": (float,),
    "legivel": (bool,),
    "flags": (list,),
}


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_predict_response_contract(service: LiveService, name: str) -> None:
    response = service.predict(name)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == set(CONTRACT)
    for field, types in CONTRACT.items():
        assert isinstance(body[field], types), f"{field}: {type(body[field]).__name__}"
    assert all(isinstance(flag, str) for flag in body["flags"])
