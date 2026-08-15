"""Avaliação do checkpoint fine-tuned no split test, com registro em MLflow.

Gera um run de eval com:
- params: checkpoint, dados, max_len
- metrics: test_exact_match (e test_digit_acc)
- blobs (tabela `mapen_artifact_blobs` no mesmo mlflow.db): `eval_test.csv`
  (pred, label por amostra) e `eval_summary.txt`
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
import torch

from . import artifacts, models, paths
from .data import build_dataset


def _digit_acc(pred: str, label: str) -> float:
    """Acurácia por dígito (alinhando da direita para a esquerda, leituras com 5 dígitos)."""
    if not label:
        return 0.0
    aligned = zip(pred.rjust(len(label))[-len(label):], label)
    return sum(1 for p, l in aligned if p == l) / len(label)


def evaluate(
    *,
    model_dir: Path | str,
    data_dir: Path | str,
    max_len: int = 16,
    max_samples: int | None = None,
    tracking_uri: str = paths.DEFAULT_TRACKING_URI,
    experiment: str = paths.DEFAULT_EXPERIMENT,
) -> dict:
    model_dir = Path(model_dir)
    device = models.resolve_device()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name="eval-test") as run:
        mlflow.log_params(
            {"model_dir": str(model_dir), "data_dir": str(data_dir), "max_len": max_len,
             "max_samples": max_samples or "all"}
        )

        processor = models.load_processor(model_dir)
        ds = build_dataset(data_dir, "test", processor, max_len, max_samples)
        labels = [
            processor.tokenizer.decode([t for t in row if t != -100], skip_special_tokens=True)
            for row in ds["labels"]
        ]

        model = models.load_trocr(model_dir)
        models.configure_model(model, processor.tokenizer)
        model.to(device).eval()

        rows: list[dict] = []
        with torch.no_grad():
            for i, pixel_values in enumerate(ds["pixel_values"]):
                pv = torch.tensor(pixel_values).unsqueeze(0).to(device)
                gen = model.generate(pv, max_new_tokens=max_len, num_beams=4)
                pred = processor.tokenizer.decode(gen[0], skip_special_tokens=True)
                rows.append({"pred": pred, "label": labels[i], "hit": int(pred == labels[i])})

        df = pd.DataFrame(rows)
        exact = float(df["hit"].mean())
        digit = float(df.apply(lambda r: _digit_acc(r["pred"], r["label"]), axis=1).mean())
        mlflow.log_metrics({"test_exact_match": exact, "test_digit_acc": digit})

        report = Path(model_dir) / "eval_test.csv"
        df.to_csv(report, index=False)
        artifacts.save_blob(
            run.info.run_id,
            "eval_test.csv",
            df.to_csv(index=False).encode("utf-8"),
            "text/csv",
        )
        artifacts.save_text(
            run.info.run_id,
            "eval_summary.txt",
            f"checkpoint={model_dir.resolve()}\ntest_exact_match={exact:.4f}\ntest_digit_acc={digit:.4f}",
        )

        print(f"test exact-match: {exact:.3f} | digit-acc: {digit:.3f} ({len(df)} amostras)")
        for _, r in df.head(15).iterrows():
            print(f"  pred={r['pred']!r:>10}  label={r['label']!r:>10}  {'OK' if r['hit'] else 'X'}")

        return {"run_id": run.info.run_id, "test_exact_match": exact, "test_digit_acc": digit}
