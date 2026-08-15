"""Fine-tune de TrOCR (stage1) em leituras UFPR-AMR, com tracking em MLflow.

Cada execução gera um run no experimento `trocr-ufpramr` com:
- params: hiperparâmetros (modelo base, epochs, batch, grad-accum, lr, max_len, data)
- metrics: eval loss + exact-match por passo (callback) e best metric
- blobs (tabela `mapen_artifact_blobs` no mesmo mlflow.db): `checkpoint.zip`
  (modelo treinado) e `run_summary.txt` (resumo do run)
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import mlflow
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    VisionEncoderDecoderModel,
    XLMRobertaTokenizer,
)

from . import artifacts, models, paths
from .data import build_dataset

_METRIC_KEYS = ("eval_loss", "eval_exact_match")


class _MlflowCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if not logs:
            return
        step = logs.pop("step", None)
        mlflow.log_metrics({k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}, step=step)


def _exact_match(processor, pred_ids, label_ids) -> float:
    preds = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels = processor.tokenizer.batch_decode(
        [[t for t in row if t != -100] for row in label_ids], skip_special_tokens=True
    )
    return sum(1 for p, l in zip(preds, labels) if p == l) / max(len(preds), 1)


def _collate(batch: list[dict], tokenizer: XLMRobertaTokenizer) -> dict:
    """Collate p/ vision-encoder-decoder: stacka pixel_values e mascara pad→-100.

    (DataCollatorForSeq2Seq serve a texto — exige `input_ids`, que não existe aqui.)
    """
    pixel_values = torch.stack([torch.as_tensor(ex["pixel_values"]) for ex in batch])
    labels = torch.stack([torch.as_tensor(ex["labels"]) for ex in batch])
    pad_token_id = tokenizer.pad_token_id
    assert isinstance(pad_token_id, int)
    labels = labels.masked_fill(labels == pad_token_id, -100)
    return {"pixel_values": pixel_values, "labels": labels}


def train(
    *,
    model_path: Path | str,
    data_dir: Path | str,
    out_dir: Path | str,
    epochs: int = 8,
    batch: int = 4,
    grad_accum: int = 8,
    lr: float = 2e-5,
    max_len: int = 16,
    max_samples: int | None = None,
    tracking_uri: str = paths.DEFAULT_TRACKING_URI,
    experiment: str = paths.DEFAULT_EXPERIMENT,
) -> dict:
    params = {
        "base_model": str(model_path),
        "data_dir": str(data_dir),
        "epochs": epochs,
        "batch": batch,
        "grad_accum": grad_accum,
        "lr": lr,
        "max_len": max_len,
        "max_samples": max_samples or "all",
    }
    out_dir = Path(out_dir)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=f"train-{params['epochs']}ep") as run:
        mlflow.log_params(params)

        processor = models.load_processor(model_path)
        train_ds = build_dataset(data_dir, "train", processor, max_len, max_samples)
        valid_ds = build_dataset(data_dir, "valid", processor, max_len, max_samples)

        model = models.load_trocr(model_path)
        models.configure_model(model, processor.tokenizer)

        def compute_metrics(ep):
            pred_ids = ep.predictions if isinstance(ep.predictions, torch.Tensor) else torch.tensor(ep.predictions)
            return {"exact_match": _exact_match(processor, pred_ids, ep.label_ids)}

        collator = lambda batch: _collate(batch, processor.tokenizer)
        targs = Seq2SeqTrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=batch,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            weight_decay=0.01,
            warmup_steps=30,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,
            predict_with_generate=True,
            generation_num_beams=4,
            load_best_model_at_end=True,
            metric_for_best_model="exact_match",
            greater_is_better=True,
            report_to=[],
            fp16=False,
            bf16=False,
            dataloader_num_workers=2,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            data_collator=collator,
            compute_metrics=compute_metrics,
            processing_class=processor,
            callbacks=[_MlflowCallback()],
        )
        trainer.train()
        models.save_model(cast(VisionEncoderDecoderModel, trainer.model), processor, out_dir)

        best = trainer.state.best_metric
        if best is None:
            raise RuntimeError("treino terminou sem avaliação — metric_for_best_model nunca avaliada")
        best_metric_name = getattr(trainer.state, "best_metric_name", "eval_exact_match")
        mlflow.log_metrics({f"best_{best_metric_name}": float(best)})
        mlflow.log_params({"checkpoint_dir": str(out_dir.resolve())})

        artifacts.save_text(
            run.info.run_id,
            "run_summary.txt",
            f"base_model={params['base_model']}\ncheckpoint={out_dir.resolve()}\n"
            f"{best_metric_name}={best}",
        )
        artifacts.save_zip(run.info.run_id, "checkpoint.zip", out_dir)

        summary = {"run_id": run.info.run_id, **params, "best_metric": best}
        print(f"MLflow run: {mlflow.get_tracking_uri()} (run_id={run.info.run_id})")
        print(f"modelo salvo em {out_dir.resolve()}")
        return summary
