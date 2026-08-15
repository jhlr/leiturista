"""Carregamento e configuração de modelos TrOCR (processor + vision-encoder-decoder).

Nota: o auto-load do `TrOCRProcessor.from_pretrained` quebra no transformers >= 5.14
quando o repo não tem `tokenizer.json` (só `sentencepiece.bpe.model`): ele tenta
converter pra tokenizer fast sem `sentencepiece` instalado. Por isso o processor é
montado manualmente — mas a classe do tokenizer é a declarada no `tokenizer_config.json`
(nessas repos, `XLMRobertaTokenizer`), não `RobertaTokenizer` (vocab desalinhado).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import transformers
from transformers import (
    XLMRobertaTokenizer,
    TrOCRProcessor,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
)


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_tokenizer(model_path: str) -> XLMRobertaTokenizer:
    """Carrega o tokenizer slow declarado no tokenizer_config.json do repo."""
    config_path = Path(model_path) / "tokenizer_config.json"
    tokenizer_class: str = "XLMRobertaTokenizer"
    if config_path.is_file():
        raw = json.loads(config_path.read_text())
        tokenizer_class = raw.get("tokenizer_class") or tokenizer_class
    tokenizer_cls = getattr(transformers, tokenizer_class, XLMRobertaTokenizer)
    return tokenizer_cls.from_pretrained(model_path, use_fast=False)


def load_processor(model_path: Path | str) -> TrOCRProcessor:
    model_path = str(model_path)
    image_processor = ViTImageProcessor.from_pretrained(model_path)
    tokenizer = _load_tokenizer(model_path)
    return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)


def load_trocr(model_path: Path | str) -> VisionEncoderDecoderModel:
    return VisionEncoderDecoderModel.from_pretrained(str(model_path))


def configure_model(model: VisionEncoderDecoderModel, tokenizer: XLMRobertaTokenizer) -> None:
    """Ajusta IDs especiais do decoder para geração (TrOCR)."""
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id


def save_model(model: VisionEncoderDecoderModel, processor: TrOCRProcessor, out_dir: Path | str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))
