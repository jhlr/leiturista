"""Catálogo de ocorrências (notas de leitura) e regras de coerência da evidência fotográfica.

No fluxo de campo da distribuidora (Tarefa 2), as fotos servem como evidência de dois objetivos distintos:
1. LEITURA REGULAR: a foto deve comprovar o consumo registrado no visor (ex.: NA, L101).
2. IMPEDIMENTO / AUSÊNCIA: a foto é tirada propositalmente para demonstrar a ausência do
   medidor ou a impossibilidade de leitura (ex.: portão trancado I100, medidor retirado B100,
   imóvel demolido D100, cão no local I160). Nesses casos, a ausência de leitura na foto NÃO É
   ERRO — é a confirmação do impedimento relatado pelo leiturista!
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

# ---------------------------------------------------------------------------
# Catálogo oficial de notas (extraído de DESCRIÇÃO NOTAS LEITURISTAS X SOLICITAÇÃO DE FOTO.xlsx)
# ---------------------------------------------------------------------------

CATALOGO_NOTAS: dict[str, dict[str, Any]] = {
    "NA":   {"desc": "Leitura Normal", "exige_foto": True},
    "B100": {"desc": "Local Cons Desligado - E.M. Retirado", "exige_foto": True},
    "B101": {"desc": "Local Cons Desligado - E.M. Desconectado c/Leitura", "exige_foto": False},
    "B111": {"desc": "Local Cons. Implantado em Duplicidade", "exige_foto": True},
    "B121": {"desc": "Local Cons. Fornecendo/Fornecido a/por Terceiros", "exige_foto": False},
    "B131": {"desc": "Suspensão com Selo", "exige_foto": True},
    "C111": {"desc": "Bairro Incorreto", "exige_foto": False},
    "C121": {"desc": "Rua Incorreta", "exige_foto": False},
    "C131": {"desc": "Número da Porta/Portal Incorreto", "exige_foto": False},
    "C151": {"desc": "Atividade Incorreta", "exige_foto": False},
    "C161": {"desc": "Quant. Incorreta de Dígitos do E.M.", "exige_foto": True},
    "C181": {"desc": "Número do Poste Incorreto", "exige_foto": False},
    "CE00": {"desc": "Calamidade Pública ou Emergência", "exige_foto": False},
    "D100": {"desc": "Local Cons. Demolido - sem E.M. com Ramal", "exige_foto": True},
    "D101": {"desc": "Local Cons. Demolido - com E.M. e Ramal", "exige_foto": True},
    "D110": {"desc": "Local Cons. Demolido - sem E.M. e Ramal", "exige_foto": True},
    "D111": {"desc": "Local Cons. Demolido - com E.M. sem Ramal", "exige_foto": True},
    "E101": {"desc": "Poste Inclinado ou Quebrado", "exige_foto": True},
    "E111": {"desc": "Objeto, Árvore, Imóvel Encostado/Próximo à Rede", "exige_foto": True},
    "G100": {"desc": "Display Apagado por Falta de Energia/E.M. Cortado", "exige_foto": True},
    "G200": {"desc": "Estimativa Automática", "exige_foto": False},
    "I100": {"desc": "Loc. Cons. Fechado Ocupado E.M./Display Interno", "exige_foto": True},
    "I110": {"desc": "Loc. Cons. Fechado Desocupado E.M./Display Interno", "exige_foto": True},
    "I120": {"desc": "Loc. Cons. Fechado Veraneio E.M./Display Interno", "exige_foto": True},
    "I130": {"desc": "Proibida a Entrada do Leiturista na UC", "exige_foto": True},
    "I140": {"desc": "Quadro/Caixa de Medição Trancado/Pintado/Obstruído", "exige_foto": True},
    "I160": {"desc": "Cão no Local de Consumo", "exige_foto": True},
    "I170": {"desc": "Animais na Caixa de Medição", "exige_foto": True},
    "I180": {"desc": "Local Perigoso", "exige_foto": False},
    "I240": {"desc": "Local Interditado/Alagado/Intransitável", "exige_foto": True},
    "L101": {"desc": "Leitura Informada Pelo Cliente", "exige_foto": True},
    "L121": {"desc": "Campanha 1", "exige_foto": True},
    "L131": {"desc": "Campanha 2", "exige_foto": False},
    "M100": {"desc": "E.M./Display Requer Verificação - s/ Leitura", "exige_foto": True},
    "M101": {"desc": "E.M./Display Requer Verificação - c/ Leitura", "exige_foto": True},
    "M121": {"desc": "E.M. c/ Registrador Parado - Loc. Cons. Habitado", "exige_foto": False},
    "M131": {"desc": "NÃO UTILIZ - CHAM 8000058533 (med volt leit)", "exige_foto": False},
    "M141": {"desc": "E.M. c/ Registrador Parado - Loc. Cons. Desabitado", "exige_foto": False},
    "M200": {"desc": "E.M./Display Requer Verif. - Desab. s/ Leitura", "exige_foto": True},
    "N160": {"desc": "Local de Consumo Não Localizado", "exige_foto": False},
    "P101": {"desc": "E.M. Solto/Inclinado/Embaçado/Pintado - c/ Leitura", "exige_foto": False},
    "P111": {"desc": "E.M. Substituído ou Número Incorreto", "exige_foto": True},
    "P131": {"desc": "E.M. Vizinho Não Cadastrado", "exige_foto": False},
    "P140": {"desc": "Loc. Cons. c/ Ligação Direta -  s/ E.M. ou Apagado", "exige_foto": True},
    "P141": {"desc": "Loc. Cons. c/ Ligação Direta - c/ E.M. c/ Leitura", "exige_foto": False},
    "P151": {"desc": "Suspeita de Irregularidade", "exige_foto": False},
    "P191": {"desc": "U.C. Vizinha com Ligação Clandestina", "exige_foto": True},
    "P200": {"desc": "E.M. Solto/Inclinado/Embaçado/Pintado - s/ Leitura", "exige_foto": True},
    "P210": {"desc": "Caixa/Tampa Embaçada/Danificada  - s/ Leitura", "exige_foto": False},
    "P211": {"desc": "Gambiarra de Iluminação na Rede", "exige_foto": False},
    "P220": {"desc": "E.M. em Local/Altura Irreg/Lente c/Prob -s/Leitura", "exige_foto": True},
    "P231": {"desc": "Iluminação Pública Acesa Durante o Dia", "exige_foto": False},
    "R111": {"desc": "Medidor Alocado Incorretamente", "exige_foto": False},
    "R161": {"desc": "Ponto de Referência Incorreto", "exige_foto": False},
    "T101": {"desc": "Ramal/Caixa/Parede Vazando Corrente/em Risco", "exige_foto": False},
    "T111": {"desc": "Caixa/Tampa Embaçando/Danif./Ausente - c/ Leitura", "exige_foto": False},
    "T121": {"desc": "Verificação de Leitura", "exige_foto": True},
    "T131": {"desc": "E.M. em Local/Altura Irreg/Lente c/Prob -c/Leitura", "exige_foto": True},
    "T151": {"desc": "Ligação Nova com Medição Interna", "exige_foto": False},
    "T161": {"desc": "Função Não Existe no E.M.", "exige_foto": False},
    "T171": {"desc": "Data/Hora do E.M. Incorreta", "exige_foto": False},
    "T181": {"desc": "Função Não Existe no Sistema", "exige_foto": True},
}

# Notas onde a foto é EVIDÊNCIA DE IMPEDIMENTO / AUSÊNCIA (foto de portão, muro, cão, ausência):
NOTAS_IMPEDIMENTO_AUSENCIA: set[str] = {
    "B100",  # E.M. Retirado
    "D100", "D110",  # Demolido sem E.M.
    "I100", "I110", "I120",  # Casa Fechada (display interno)
    "I130",  # Entrada Proibida
    "I140",  # Caixa Trancada/Obstruída
    "I160",  # Cão no Local
    "I170",  # Animais na Caixa
    "I240",  # Local Alagado / Intransitável
    "G100",  # Display Apagado por Falta de Energia
    "M100", "M200",  # Display Requer Verificação sem Leitura
    "P140",  # Ligação Direta sem E.M.
    "P200", "P220",  # Embaçado/Local Irregular sem Leitura
    "E101", "E111",  # Poste Inclinado / Rede Obstruída
}

# Notas onde a foto DEVE CONTER LEITURA REGULAR:
NOTAS_LEITURA_REGULAR: set[str] = {
    "NA", "LNA", "L101", "T121", "M101", "P111", "C161", "T131",
}


@dataclasses.dataclass
class CoerenciaResult:
    bucket: str               # 'verde' | 'amarelo' | 'vermelho'
    tipo: str                 # 'impedimento_confirmado' | 'leitura_confirmada' | 'inconsistencia' | 'revisao'
    categoria_nota: str       # 'impedimento' | 'leitura' | 'outra'
    descricao_nota: str
    justificativa: str
    nota_normalizada: str


def normalizar_nota(nota: str | None) -> str:
    """Normaliza o código da nota de leitura."""
    if not nota or str(nota).strip().upper() in ("NAN", "NONE", "", "NA"):
        return "NA"
    return str(nota).strip().upper()


def avaliar_coerencia(pred: Any, nota_raw: str | None, leitura_registrada: str | None = None) -> CoerenciaResult:
    """Avalia a coerência entre a evidência da foto e a nota de ocorrência registrada."""
    nota = normalizar_nota(nota_raw)
    info_nota = CATALOGO_NOTAS.get(nota, {"desc": "Ocorrência Não Catalogada", "exige_foto": True})
    desc_nota = info_nota["desc"]

    # 1. Caso: Nota de Impedimento / Ausência de Medidor
    if nota in NOTAS_IMPEDIMENTO_AUSENCIA:
        if pred.reading is None:
            # SUCESSO DE EVIDÊNCIA: A foto cumpriu exatamente o papel de provar o impedimento
            return CoerenciaResult(
                bucket="verde",
                tipo="impedimento_confirmado",
                categoria_nota="impedimento",
                descricao_nota=desc_nota,
                justificativa=f"Evidência consistente: Nota {nota} ({desc_nota}) comprovada pela ausência de leitura/medidor na foto.",
                nota_normalizada=nota,
            )
        else:
            # INCONSISTÊNCIA: A nota diz que a casa estava fechada ou o medidor ausente, mas achou leitura!
            return CoerenciaResult(
                bucket="amarelo",
                tipo="revisao",
                categoria_nota="impedimento",
                descricao_nota=desc_nota,
                justificativa=f"Revisão necessária: Nota {nota} ({desc_nota}) indica ausência/impedimento, porém o OCR detectou leitura {pred.reading}.",
                nota_normalizada=nota,
            )

    # 2. Caso: Nota de Leitura Regular esperada (NA, L101, T121, etc.)
    elif nota in NOTAS_LEITURA_REGULAR or nota == "NA":
        if pred.reading is not None and getattr(pred, "legible", True):
            # Se tivermos a leitura registrada no sistema, podemos comparar
            leitura_esperada = re.sub(r"\D", "", str(leitura_registrada or ""))
            leitura_obtida = getattr(pred, "reading", "")

            # Match exato ou alinhado com a leitura registrada
            bate_com_registro = bool(leitura_esperada and (
                leitura_obtida == leitura_esperada or
                leitura_obtida.endswith(leitura_esperada) or
                leitura_esperada.endswith(leitura_obtida)
            ))

            if bate_com_registro:
                just = f"Leitura confirmada: {leitura_obtida} confere com registro do sistema ({leitura_esperada})."
            else:
                just = f"Leitura detectada: {leitura_obtida} com nitidez adequada para nota {nota}."

            return CoerenciaResult(
                bucket="verde",
                tipo="leitura_confirmada",
                categoria_nota="leitura",
                descricao_nota=desc_nota,
                justificativa=just,
                nota_normalizada=nota,
            )
        elif pred.reading is None:
            # FALHA DE EVIDÊNCIA: A nota exigia leitura, mas nada foi detectado
            return CoerenciaResult(
                bucket="vermelho",
                tipo="inconsistencia",
                categoria_nota="leitura",
                descricao_nota=desc_nota,
                justificativa=f"Inconsistência de evidência: Nota {nota} ({desc_nota}) exige foto do display, mas nenhuma leitura foi detectada.",
                nota_normalizada=nota,
            )
        else:
            return CoerenciaResult(
                bucket="amarelo",
                tipo="revisao",
                categoria_nota="leitura",
                descricao_nota=desc_nota,
                justificativa=f"Revisão manual: Leitura incerta ou display borrado para nota {nota}.",
                nota_normalizada=nota,
            )

    # 3. Caso Geral / Outras Notas (ex.: T181, P111, B111)
    else:
        if pred.reading is not None and getattr(pred, "legible", True):
            return CoerenciaResult(
                bucket="verde",
                tipo="leitura_confirmada",
                categoria_nota="outra",
                descricao_nota=desc_nota,
                justificativa=f"Leitura {pred.reading} identificada para ocorrência {nota} ({desc_nota}).",
                nota_normalizada=nota,
            )
        else:
            return CoerenciaResult(
                bucket="amarelo",
                tipo="revisao",
                categoria_nota="outra",
                descricao_nota=desc_nota,
                justificativa=f"Ocorrência {nota} ({desc_nota}): requer conferência manual de evidência.",
                nota_normalizada=nota,
            )
