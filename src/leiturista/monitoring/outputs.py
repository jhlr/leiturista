"""Variaveis de saida do modelo (dono unico, SDD secao 5.2) e a regra `funcao`/`confianca`.

`funcao_and_confianca` e a MESMA funcao usada pelo servico para montar a resposta e por este
modulo para o log: nao existe segunda copia da regra. Nunca expomos o valor da leitura nem o
serial, so contagens e booleanos.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..inference import Prediction

CONFIDENCE_OK = 0.5  # abaixo disso, a leitura existe mas e tratada como incerta

NUMERIC_OUTPUTS: tuple[str, ...] = ("confianca", "reading_digits", "n_boxes", "n_flags", "candidate_sharpness")
CATEGORICAL_OUTPUTS: tuple[str, ...] = ("funcao", "legivel", "has_reading", "has_serial")
LIST_OUTPUTS: tuple[str, ...] = ("flags",)
OUTPUT_COLUMNS: tuple[str, ...] = (
    "funcao",
    "confianca",
    "legivel",
    "has_reading",
    "has_serial",
    "reading_digits",
    "n_boxes",
    "n_flags",
    "flags",
    "candidate_sharpness",
)
SCALAR_OUTPUT_COLUMNS: tuple[str, ...] = tuple(c for c in OUTPUT_COLUMNS if c not in LIST_OUTPUTS)
FUNCAO_VALUES: tuple[str, ...] = ("leitura_normal", "leitura_incerta", "sem_leitura_detectada")

_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")


def funcao_and_confianca(pred: Prediction) -> tuple[str, float]:
    """Regra do servico: `funcao` e `confianca` (arredondada em 3 casas) de uma `Prediction`."""
    leitura_boxes = [b for b in pred.boxes if b.field == "leitura"]
    confianca = max((b.conf for b in leitura_boxes), default=0.0) if pred.reading else 0.0

    if pred.reading is None:
        funcao = "sem_leitura_detectada"
    elif pred.legible and confianca >= CONFIDENCE_OK:
        funcao = "leitura_normal"
    else:
        funcao = "leitura_incerta"
    return funcao, round(float(confianca), 3)


def normalize_flag(flag: str) -> str:
    """Remove os digitos (o angulo de rotacao viraria um token de alta cardinalidade)."""
    return _SPACES.sub(" ", _DIGITS.sub("", flag)).strip()


@dataclass(frozen=True)
class OutputVariables:
    funcao: str
    confianca: float
    legivel: bool
    has_reading: bool
    has_serial: bool
    reading_digits: int | None
    n_boxes: int
    n_flags: int
    flags: tuple[str, ...]
    candidate_sharpness: float | None

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {name: getattr(self, name) for name in OUTPUT_COLUMNS}
        out["flags"] = list(self.flags)
        return out

    def scalars(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in SCALAR_OUTPUT_COLUMNS}


def derive_outputs(pred: Prediction) -> OutputVariables:
    funcao, confianca = funcao_and_confianca(pred)
    return OutputVariables(
        funcao=funcao,
        confianca=confianca,
        legivel=bool(pred.legible),
        has_reading=pred.reading is not None,
        has_serial=pred.serial is not None,
        reading_digits=sum(ch.isdigit() for ch in pred.reading) if pred.reading is not None else None,
        n_boxes=len(pred.boxes),
        n_flags=len(pred.flags),
        flags=tuple(normalize_flag(flag) for flag in pred.flags),
        candidate_sharpness=None if pred.sharpness is None else float(pred.sharpness),
    )
