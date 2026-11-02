from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from leiturista.inference import Prediction
from leiturista.monitoring import outputs as o


def _pred(reading: str | None, legible: bool = True, confs: tuple[float, ...] = (0.9,), serial: str | None = None,
          flags: tuple[str, ...] = (), sharpness: float | None = 30.0) -> Prediction:
    boxes = [SimpleNamespace(field="leitura", conf=c) for c in confs] + [SimpleNamespace(field="serial", conf=0.7)]
    fake: Any = SimpleNamespace(reading=reading, legible=legible, boxes=boxes, serial=serial, flags=list(flags), sharpness=sharpness)
    return cast(Prediction, fake)


def test_funcao_rule() -> None:
    assert o.funcao_and_confianca(_pred("00123")) == ("leitura_normal", 0.9)
    assert o.funcao_and_confianca(_pred("00123", confs=(0.4,))) == ("leitura_incerta", 0.4)
    assert o.funcao_and_confianca(_pred("00123", legible=False)) == ("leitura_incerta", 0.9)
    assert o.funcao_and_confianca(_pred(None)) == ("sem_leitura_detectada", 0.0)


def test_outputs_never_carry_reading_or_serial_and_strip_digits_from_flags() -> None:
    out = o.derive_outputs(
        _pred("00123", serial="SN 998877", flags=("medidor rotacionado na foto - corrigida (90°)", "x"), sharpness=None)
    )
    d = out.as_dict()
    assert tuple(d) == o.OUTPUT_COLUMNS
    assert out.reading_digits == 5 and out.has_serial and out.n_boxes == 2 and out.n_flags == 2
    assert out.candidate_sharpness is None
    assert not any(ch.isdigit() for flag in out.flags for ch in flag)
    assert "00123" not in str(d) and "998877" not in str(d)
    assert o.derive_outputs(_pred(None)).reading_digits is None
