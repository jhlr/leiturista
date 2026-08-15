"""CLI da biblioteca leiturista.

Subcomandos:
  extract    — extrai imagens UFPR-AMR + labels.csv para data/finetune_ufpramr
  train      — fine-tune TrOCR-small-stage1 com tracking em MLflow
  eval       — avalia checkpoint no split test e registra no MLflow
  artifacts  — lista/dump blobs de artefato de um run (mlflow.db)
  restore    — extrai o checkpoint.zip de um run de volta para disco

Exemplos:
  leiturista extract
  leiturista train --epochs 8 --batch 4 --grad-accum 8
  leiturista eval
  leiturista artifacts <run_id>
  leiturista restore <run_id>
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from . import paths


def _add_mlflow_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--tracking-uri", default=paths.DEFAULT_TRACKING_URI)
    ap.add_argument("--experiment", default=paths.DEFAULT_EXPERIMENT)


def _cmd_extract(args: argparse.Namespace) -> None:
    from .data import extract_ufpr_amr

    extract_ufpr_amr(out=args.out, per_split=args.per_split)


def _cmd_train(args: argparse.Namespace) -> None:
    from .train import train

    train(
        model_path=args.model,
        data_dir=args.data,
        out_dir=args.out,
        epochs=args.epochs,
        batch=args.batch,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_len=args.max_len,
        max_samples=args.max_samples,
        tracking_uri=args.tracking_uri,
        experiment=args.experiment,
    )


def _cmd_eval(args: argparse.Namespace) -> None:
    from .eval import evaluate

    evaluate(
        model_dir=args.out,
        data_dir=args.data,
        max_len=args.max_len,
        max_samples=args.max_samples,
        tracking_uri=args.tracking_uri,
        experiment=args.experiment,
    )


def _cmd_artifacts(args: argparse.Namespace) -> None:
    from . import artifacts

    if not args.name:
        for name, size in artifacts.list_names(args.run_id):
            print(f"{name}  ({size} bytes)")
        return
    data = artifacts.read_blob(args.run_id, args.name)
    if data is None:
        raise SystemExit(f"artefato não encontrado: run_id={args.run_id} name={args.name}")
    if args.output:
        Path(args.output).write_bytes(data)
    else:
        print(data.decode("utf-8", errors="replace"), end="")


def _cmd_restore(args: argparse.Namespace) -> None:
    import zipfile

    from . import artifacts

    data = artifacts.read_blob(args.run_id, args.name)
    if data is None:
        raise SystemExit(f"artefato não encontrado: run_id={args.run_id} name={args.name}")
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(target)
    print(f"restaurado em {target.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="leiturista", description="Leiturista - visão computacional de medidores (TrOCR/OCR)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract", help="extrai imagens UFPR-AMR + labels.csv")
    p.add_argument("-o", "--out", default=str(paths.FINETUNE_DIR))
    p.add_argument("-n", "--per-split", type=int, default=None)
    p.set_defaults(func=_cmd_extract)

    p = sub.add_parser("train", help="fine-tune TrOCR-small-stage1 em UFPR-AMR (MLflow)")
    p.add_argument("--model", default=str(paths.MODELS_DIR / "trocr-small-stage1"))
    p.add_argument("--data", default=str(paths.FINETUNE_DIR))
    p.add_argument("--out", default=str(paths.MODELS_DIR / "trocr-small-finetuned-ufpramr"))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=16)
    p.add_argument("--max-samples", type=int, default=None, help="debug: subset por split")
    _add_mlflow_args(p)
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("eval", help="avalia checkpoint no split test (MLflow)")
    p.add_argument("--data", default=str(paths.FINETUNE_DIR))
    p.add_argument("--out", default=str(paths.MODELS_DIR / "trocr-small-finetuned-ufpramr"))
    p.add_argument("--max-len", type=int, default=16)
    p.add_argument("--max-samples", type=int, default=None, help="debug: subset do test")
    _add_mlflow_args(p)
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("artifacts", help="lista/dump blobs de artefato de um run no mlflow.db")
    p.add_argument("run_id")
    p.add_argument("name", nargs="?", help="nome do blob (sem nome = lista)")
    p.add_argument("-o", "--output", help="grava o blob num arquivo (senão imprime como texto)")
    p.set_defaults(func=_cmd_artifacts)

    p = sub.add_parser("restore", help="extrai um zip-blob (ex.: checkpoint.zip) para disco")
    p.add_argument("run_id")
    p.add_argument("name", nargs="?", default="checkpoint.zip")
    p.add_argument("-o", "--out", default=str(paths.MODELS_DIR / "trocr-small-finetuned-ufpramr"))
    p.set_defaults(func=_cmd_restore)

    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
