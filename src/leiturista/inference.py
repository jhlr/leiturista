"""Inferência de leitura de medidor: det (PP-OCRv5_mobile) + rec (PP-OCRv6_tiny).

Fluxo:
  1. det: caixas de texto na foto do medidor (display, placa/serial).
  2. rec: OCR de cada crop → texto + confiança.
  3. classifica cada box como `leitura`, `serial` ou `outro`.
  4. flags de coerência: legibilidade (Laplacian), múltiplas leituras, sem texto.

Uso:
  from leiturista.inference import MeterOCR
  ocr = MeterOCR()
  pred = ocr.predict_image(pil_image)   # -> Prediction
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from . import artifacts, paths
from .models import configure_model, load_processor, load_trocr, resolve_device

DET_ONNX = paths.MODELS_DIR / "pp_ocr_v5_mobile_det_onnx" / "inference.onnx"
REC_ONNX = paths.MODELS_DIR / "pp_ocr_v6_tiny_rec_onnx" / "inference.onnx"
TROCR_DIR = paths.MODELS_DIR / "trocr-small-printed"
TROCR_MODEL_RUN = "9c14db6248834e7e80f2e4356959d5aa"
TROCR_CACHE = paths.ROOT / ".model_cache" / "trocr-small-printed"
TROCR_FILES = ["config.json", "generation_config.json", "preprocessor_config.json",
               "sentencepiece.bpe.model", "special_tokens_map.json", "tokenizer_config.json",
               "model.safetensors"]
DICT_FILE = paths.ROOT / "scripts" / "ppocr_v6_dict.json"

DET_LONG_SIDE = 960
DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DET_THRESH = 0.3
DET_BOX_THRESH = 0.6
DET_UNCLIP = 1.5
DET_MIN_SIZE = 3

REC_IMG_H = 48
REC_MAX_W = 3200

SERIAL_RE = re.compile(r"[A-Za-z]")
SPEC_CHARS = re.compile(r"[=°()%/\u00b7\"_~]|\.\.")
LAPLACIAN_LEGIBLE = 25.0


@dataclasses.dataclass
class Box:
    text: str
    conf: float  # score do box (det), 0..1
    field: str  # 'leitura' | 'serial' | 'outro'
    quad: np.ndarray  # 4x2 (x, y)
    crop: Image.Image
    source: str = "normal"  # 'normal' (preto-no-branco) | 'inverted' (branco-no-preto)


@dataclasses.dataclass
class Prediction:
    reading: str | None
    serial: str | None
    signature: str
    signature_tokens: list[str]
    boxes: list[Box]
    legible: bool
    flags: list[str]
    annotated: Image.Image


class MeterOCR:
    """Pipeline det + rec de medidores (modelos ONNX locais)."""

    def __init__(self) -> None:
        self._det: ort.InferenceSession | None = None
        self._rec: ort.InferenceSession | None = None
        self._chars: list[str] = []
        self._trocr: tuple | None = None

    # -- carregamento lazy ------------------------------------------------
    def _load(self) -> None:
        if self._det is not None:
            return
        self._det = ort.InferenceSession(str(DET_ONNX))
        self._rec = ort.InferenceSession(str(REC_ONNX))
        self._chars = json.loads(DICT_FILE.read_text())

    def _load_trocr(self) -> tuple:
        """TrOCR-small-printed materializado do mlflow.db (cache em disco).
        Fonte de verdade = blobs `trocr-small-printed/*` do run TROCR_MODEL_RUN."""
        if self._trocr is not None:
            return self._trocr
        if not (TROCR_CACHE / "model.safetensors").exists():
            try:
                artifacts.materialize(TROCR_MODEL_RUN, "trocr-small-printed", TROCR_CACHE, TROCR_FILES)
            except Exception:
                artifacts.materialize(TROCR_MODEL_RUN, "trocr-small-printed", TROCR_CACHE)
            if not (TROCR_CACHE / "model.safetensors").exists():
                for name in TROCR_FILES:
                    (TROCR_CACHE / name).write_bytes((TROCR_DIR / name).read_bytes())
        device = "cpu"
        try:
            device = resolve_device()
        except Exception:
            pass
        processor = load_processor(TROCR_CACHE)
        model = load_trocr(TROCR_CACHE)
        configure_model(model, processor.tokenizer)
        model.to(device)
        model.eval()
        self._trocr = (processor, model, device)
        return self._trocr

    # -- det ---------------------------------------------------------------
    @staticmethod
    def _det_prep(img_bgr: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Pré-processa p/ det e devolve (tensor, sx, sy) com sx/sy p/ mapear caixas
        de volta às coordenadas da imagem original (o resize de proporção se anula)."""
        h0, w0 = img_bgr.shape[:2]
        long_side = max(h0, w0)
        scale = DET_LONG_SIDE / long_side if long_side > DET_LONG_SIDE else 1.0
        img = cv2.resize(img_bgr, (int(round(w0 * scale)), int(round(h0 * scale))),
                         interpolation=cv2.INTER_LINEAR) if scale != 1.0 else img_bgr
        h1, w1 = img.shape[:2]
        # ONNX Paddle2ONNX exige dims múltiplas de 32 (Resize interno quebra com impar)
        hp = int(math.ceil(h1 / 32) * 32)
        wp = int(math.ceil(w1 / 32) * 32)
        if (hp, wp) != (h1, w1):
            padded = np.full((hp, wp, 3), 127, np.uint8)
            padded[:h1, :w1] = img
            img = padded
        norm = img.astype(np.float32) / 255.0
        norm = (norm - DET_MEAN) / DET_STD
        return norm.transpose(2, 0, 1)[None], w0 / wp, h0 / hp

    def _det_boxes(self, img_bgr: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Retorna (quad 4x2, score do box) acima do box_thresh, com unclip."""
        self._load()
        x, sx, sy = self._det_prep(img_bgr)
        out = self._det.run(None, {"x": x})[0]
        score = out[0, 0, :, :].astype(np.float32)
        if score.max() > 1.0:
            score = 1.0 / (1.0 + np.exp(-score))

        bitmap = (score > DET_THRESH).astype(np.uint8)
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        quads: list[tuple[np.ndarray, float]] = []
        for c in contours:
            if len(c) < 4:
                continue
            mask = np.zeros(bitmap.shape, np.uint8)
            cv2.drawContours(mask, [c], -1, 1, -1)
            conf = float(score[mask == 1].mean())
            if conf < DET_BOX_THRESH:
                continue
            rect = cv2.minAreaRect(c)
            (cx, cy), (w, h), ang = rect
            if min(w, h) < DET_MIN_SIZE:
                continue
            # unclip: expanda o retângulo em torno do centro
            w *= DET_UNCLIP
            h *= DET_UNCLIP
            quad = cv2.boxPoints(((cx, cy), (w, h), ang))
            quad[:, 0] *= sx
            quad[:, 1] *= sy
            quads.append((quad, conf))
        return quads

    # -- rec ---------------------------------------------------------------
    @staticmethod
    def _rec_prep(crop_bgr: np.ndarray) -> np.ndarray:
        h, w = crop_bgr.shape[:2]
        rw = min(int(math.ceil(REC_IMG_H * w / h)), REC_MAX_W)
        img = cv2.resize(crop_bgr, (max(rw, 1), REC_IMG_H),
                         interpolation=cv2.INTER_LINEAR)
        return (img.transpose(2, 0, 1).astype(np.float32) / 255.0)[None]

    def _rec_recognize(self, crop_bgr: np.ndarray) -> str:
        """Reconhece o texto do crop. O grafo já aplica softmax (saída em 0..1)."""
        self._load()
        logits = self._rec.run(None, {"x": self._rec_prep(crop_bgr)})[0][0]
        idxs = logits.argmax(1)
        prev = None
        out: list[str] = []
        for i in idxs:
            if i == prev or i == 0 or i >= len(self._chars) + 1:
                prev = int(i)
                continue
            out.append(self._chars[i - 1])
            prev = int(i)
        return "".join(out)

    def _trocr_recognize(self, crop_bgr: np.ndarray) -> str:
        """Leitura por linha com TrOCR-small-printed (leitor principal)."""
        processor, model, device = self._load_trocr()
        img = crop_bgr
        h, w = img.shape[:2]
        if w < 160 or h < 48:
            scale = max(2.0, 160.0 / max(w, 1), 48.0 / max(h, 1))
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        pil = Image.fromarray(img[:, :, ::-1])
        inputs = processor(images=pil, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(
                inputs["pixel_values"].to(device),
                max_new_tokens=24, num_beams=4, do_sample=False,
            )
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        return re.sub(r"\s+", " ", text).strip()

    # -- crop rotacionado --------------------------------------------------
    @staticmethod
    def _crop_rotated(img_bgr: np.ndarray, quad: np.ndarray, pad: int = 8) -> np.ndarray:
        (cx, cy), (w, h), ang = cv2.minAreaRect(quad.astype(np.float32))
        if w < h:
            ang += 90
            w, h = h, w
        w = int(w + 2 * pad)
        h = int(h + 2 * pad)
        M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        img_rot = cv2.warpAffine(img_bgr, M, (img_bgr.shape[1], img_bgr.shape[0]),
                                 flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
        x0 = max(0, int(cx - w / 2))
        y0 = max(0, int(cy - h / 2))
        return img_rot[y0:y0 + h, x0:x0 + w]

    # -- classificação leitura/serial ---------------------------------------
    @staticmethod
    def _classify(text: str) -> str:
        """leitura = majoritariamente dígitos (tolerante a ruído do OCR: letra/símbolo
        solto); serial = alfanumérico longo sem símbolos de especificação."""
        t = text.strip()
        if not t:
            return "outro"
        digits = re.findall(r"\d", t)
        n_digits = len(digits)
        ratio = n_digits / len(t)
        if n_digits >= 2 and ratio >= 0.6 and len(t) <= 10:
            return "leitura"
        if (len(t) >= 6 and len(t) <= 24 and SERIAL_RE.search(t)
                and n_digits >= 1 and ratio < 0.9
                and SPEC_CHARS.search(t) is None):
            return "serial"
        return "outro"

    # -- merge de caixas (det fragmenta dígitos do display) ------------------
    @staticmethod
    def _merge_quads(quads: list[tuple[np.ndarray, float]]) -> list[tuple[np.ndarray, float]]:
        """Agrupa caixas com sobreposição vertical em linhas; une no retângulo
        envolvente. O det costuma separar cada dígito do display numa caixa."""
        rects = [cv2.boundingRect(q.astype(np.int32)) for q, _ in quads]
        groups: list[dict] = []
        for i, r in enumerate(rects):
            placed = False
            for g in groups:
                x0, y0, w0, h0 = g["rect"]
                x1, y1, w1, h1 = r
                y_over = min(y0 + h0, y1 + h1) - max(y0, y1)
                if y_over > 0.4 * min(h0, h1):
                    g["idxs"].append(i)
                    g["rect"] = (min(x0, x1), min(y0, y1),
                                 max(x0 + w0, x1 + w1) - min(x0, x1),
                                 max(y0 + h0, y1 + h1) - min(y0, y1))
                    placed = True
                    break
            if not placed:
                groups.append({"idxs": [i], "rect": r})
        merged: list[tuple[np.ndarray, float]] = []
        for g in groups:
            x, y, w, h = g["rect"]
            quad = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
            conf = float(np.mean([quads[i][1] for i in g["idxs"]]))
            merged.append((quad, conf))
        return merged

    # -- pipeline -----------------------------------------------------------
    def _recognize_rows(self, img_src: np.ndarray, quads: list[tuple[np.ndarray, float]],
                        merge: bool, source: str = "normal") -> list[Box]:
        rows = self._merge_quads(quads) if merge else quads
        out: list[Box] = []
        for quad, box_conf in rows:
            crop = self._crop_rotated(img_src, quad)
            if crop.size == 0:
                continue
            text = self._trocr_recognize(crop) or self._rec_recognize(crop)
            field = self._classify(text)
            w = float(quad[:, 0].max() - quad[:, 0].min())
            h = float(quad[:, 1].max() - quad[:, 1].min())
            if field == "leitura" and w < 1.5 * h:
                # serial estampado na vertical (w << h) lê como só-dígitos e vira
                # "leitura" — display real de medidor é horizontal.
                field = "serial" if (len(text) >= 6 and re.search(r"\d", text)) else "outro"
            out.append(Box(
                text=text, conf=box_conf, field=field, source=source,
                quad=np.array(quad, dtype=np.int32),
                crop=Image.fromarray(crop[:, :, ::-1]),
            ))
        return out

    @staticmethod
    def _digits(text: str) -> str:
        return re.sub(r"[^\d]", "", text)

    @staticmethod
    def _best_reading(boxes: list[Box]) -> Box | None:
        readings = [b for b in boxes if b.field == "leitura"]
        if not readings:
            return None

        def pure_digits(b: Box) -> bool:
            t = re.sub(r"[\s.,]", "", b.text)
            return bool(t) and re.fullmatch(r"\d+", t)

        pool = [b for b in readings if pure_digits(b)] or readings
        # preferência: texto só-dígitos (exclui placa tipo ': 83415801' que o
        # TrOCR injeta); depois mais dígitos. NÃO priorizar fonte invertida —
        # display preto-no-branco (contador mecânico) perderia pra ruído.
        return max(pool, key=lambda b: (len(MeterOCR._digits(b.text)), b.conf))

    @staticmethod
    def _signature(boxes: list[Box]) -> tuple[str, list[str]]:
        """Assinatura/identidade do medidor: tudo que NÃO é leitura (serial + placa)
        vira tokens normalizados (maiúsculo, alfanumérico), ordenados por posição
        (y, x). Hash sha1 determinístico + tokens p/ comparação fuzzy."""
        tokens: list[str] = []
        for b in sorted((b for b in boxes if b.field != "leitura"),
                        key=lambda b: (int(b.quad[:, 1].mean()), int(b.quad[:, 0].mean()))):
            for tok in re.findall(r"[A-Z0-9]+", b.text.upper()):
                if len(tok) >= 2:
                    tokens.append(tok)
        digest = hashlib.sha1("|".join(tokens).encode("utf-8")).hexdigest()[:12] if tokens else ""
        return digest, tokens

    def predict_image(self, image: Image.Image) -> Prediction:
        """Orquestra: detecta a rotação da foto (medidor deitado), corrige para a
        orientação em pé e roda o pipeline normal + invertida."""
        img_bgr = np.array(image.convert("RGB"))[:, :, ::-1].copy()
        deg = self._detect_rotation(img_bgr)
        if deg != 0:
            code = {90: cv2.ROTATE_90_CLOCKWISE,
                    180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE}[deg]
            img_bgr = cv2.rotate(img_bgr, code)
        pred = self._predict_bgr(img_bgr)
        if deg != 0:
            pred.flags.append(f"medidor rotacionado na foto — foto corrigida ({deg}°)")
        return pred

    def _detect_rotation(self, img_bgr: np.ndarray) -> int:
        """0/90/180/270 — orientação em que o texto fica mais horizontal. Usa o
        próprio det em imagem reduzida: na orientação correta as linhas de texto
        são largas (w >> h) e dominam a área; deitado, viram colunas."""
        h0, w0 = img_bgr.shape[:2]
        scale = 640 / max(h0, w0)
        small = cv2.resize(img_bgr, (int(w0 * scale), int(h0 * scale))) if scale < 1 else img_bgr
        codes = [(0, None), (90, cv2.ROTATE_90_CLOCKWISE),
                 (180, cv2.ROTATE_180), (270, cv2.ROTATE_90_COUNTERCLOCKWISE)]
        best, best_score = 0, -1.0
        for deg, code in codes:
            img = cv2.rotate(small, code) if code is not None else small
            horiz_area = total_area = 0.0
            for quad, _c in self._merge_quads(self._det_boxes(img)):
                xs, ys = quad[:, 0], quad[:, 1]
                w, h = xs.max() - xs.min(), ys.max() - ys.min()
                if min(w, h) < 4:
                    continue
                total_area += w * h
                if w > 1.2 * h:
                    horiz_area += w * h
            if total_area == 0:
                continue
            score = horiz_area / total_area
            if score > best_score:
                best, best_score = deg, score
        return best

    def _predict_bgr(self, img_bgr: np.ndarray) -> Prediction:
        self._load()
        img_rgb = img_bgr[:, :, ::-1]
        annotated = img_rgb.copy()

        flags: list[str] = []
        boxes: list[Box] = []

        # fase 1: imagem normal (texto preto-no-branco, ex.: serial estampado)
        quads_orig = self._det_boxes(img_bgr)
        for quad, _conf in quads_orig:
            cv2.polylines(annotated, [quad.astype(np.int32)], True, (180, 180, 180), 1)
        boxes += self._recognize_rows(img_bgr, quads_orig, merge=True, source="normal")

        # fase 2: imagem invertida (display branco-no-preto, ex.: odômetro)
        inv_bgr = 255 - img_bgr
        quads_inv = self._det_boxes(inv_bgr)
        for quad, _conf in quads_inv:
            cv2.polylines(annotated, [quad.astype(np.int32)], True, (200, 200, 200), 1)
        boxes += self._recognize_rows(inv_bgr, quads_inv, merge=False, source="inverted")

        def _laplacian(b: Box) -> float:
            return cv2.Laplacian(np.array(b.crop.convert("L")), cv2.CV_64F).var()

        reading_box = self._best_reading(boxes)
        reading = self._digits(reading_box.text) if reading_box else None
        if reading is None:
            full = self._trocr_recognize(img_bgr)
            digits_full = self._digits(full)
            if digits_full:
                reading = digits_full
                flags.append("leitura via OCR da imagem inteira (caixa de leitura não segmentada)")

        legible = True
        if reading_box is not None:
            if _laplacian(reading_box) < LAPLACIAN_LEGIBLE:
                legible = False
                flags.append("display borrado/desfocado — conferir leitura")
            if reading_box.source == "inverted":
                flags.append("leitura via imagem invertida (display claro-em-escuro)")

        for b in boxes:
            color = {"leitura": (0, 200, 0), "serial": (0, 150, 255), "outro": (220, 220, 220)}[b.field]
            cv2.polylines(annotated, [b.quad], True, color, 2)
            label = f"{b.field}:{b.text}"
            y = int(b.quad[:, 1].min()) - 6
            cv2.putText(annotated, label, (int(b.quad[:, 0].min()), max(y, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        serial_box = max((b for b in boxes if b.field == "serial"), key=lambda b: b.conf) if any(b.field == "serial" for b in boxes) else None
        serial = serial_box.text.strip() if serial_box else None
        if reading is None:
            flags.append("leitura não detectada (medidor ausente, display ilegível ou foto fora de foco)")
        if serial is None:
            flags.append("serial não detectado (foto pode ser só do display)")

        signature, signature_tokens = self._signature(boxes)

        return Prediction(
            reading=reading,
            serial=serial,
            signature=signature,
            signature_tokens=signature_tokens,
            boxes=boxes,
            legible=legible,
            flags=flags,
            annotated=Image.fromarray(annotated),
        )


def meter_similarity(a: Prediction, b: Prediction) -> float:
    """Jaccard dos tokens da assinatura — quão provável é ser o MESMO medidor."""
    ta, tb = set(a.signature_tokens), set(b.signature_tokens)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
