"""CLI da biblioteca leiturista.

Subcomandos:
  extract    — extrai imagens UFPR-AMR + labels.csv para data/finetune_ufpramr
  train      — fine-tune TrOCR-small-stage1 com tracking em MLflow
  train-crnn — treina o CRNNDigitos (esqueleto do professor) em UFPR-AMR
  train-scene — classificador 'tem medidor na foto?' (MobileNetV3-Small, MLflow)
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
import os
from pathlib import Path

from . import paths


def _add_mlflow_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--tracking-uri", default=paths.DEFAULT_TRACKING_URI)
    ap.add_argument("--experiment", default=paths.DEFAULT_EXPERIMENT)


def _cmd_extract(args: argparse.Namespace) -> None:
    from .data import extract_ufpr_amr

    extract_ufpr_amr(out=args.out, per_split=args.per_split)


def _cmd_import_distribuidora(args: argparse.Namespace) -> None:
    from .distribuidora import MIN_SHARPNESS_ACCEPT, build_dataset

    build_dataset(
        out_dir=args.out,
        data_root=args.data_root,
        notes=tuple(args.notes.split(",")),
        min_sharpness=args.min_sharpness if args.min_sharpness is not None else MIN_SHARPNESS_ACCEPT,
        max_samples=args.max_samples,
        seed=args.seed,
        qa_n=args.qa_n,
    )


def _cmd_split_lote(args: argparse.Namespace) -> None:
    import json

    from .distribuidora import split_by_lote

    print(json.dumps(split_by_lote(args.src, args.out), ensure_ascii=False, indent=2))


def _cmd_rotulos_plano(args: argparse.Namespace) -> None:
    from pathlib import Path

    from .rotulos import PLANO, make_plan, write_plan

    rows = make_plan([r.strip() for r in args.raters.split(",")], Path(args.fotos_dir), n=args.n, n_dupla=args.n_dupla,
                     seed=args.seed)
    write_plan(rows)
    print(f"{len(rows)} fotos, {sum(r['dupla'] == '1' for r in rows)} em dupla -> {PLANO}")


def _cmd_rotulos_kappa(_: argparse.Namespace) -> None:
    import json

    from .rotulos import merge_and_kappa

    print(json.dumps(merge_and_kappa(), ensure_ascii=False, indent=2))


def _cmd_train_crnn(args: argparse.Namespace) -> None:
    from .crnn import train_crnn

    train_crnn(data_dirs=args.data or [paths.FINETUNE_DIR], out=args.out, epochs=args.epochs, batch=args.batch,
               lr=args.lr, seed=args.seed, init=args.init, invert_prob=args.invert_prob,
               crop_jitter=args.crop_jitter, norm=args.norm, width=args.width, synth_per_epoch=args.synth_per_epoch, synth_only=args.synth_only, warmup=args.warmup, cosine=args.cosine, augment_on=not args.no_augment, aug_start=args.aug_start, rot_deg=args.rot_deg, blur_p=args.blur_p,
               clip_calib_iters=args.clip_calib_iters,
               tracking_uri=args.tracking_uri, experiment=args.experiment)


def _cmd_compare_crnn(args: argparse.Namespace) -> None:
    from .crnn import compare_crnn

    compare_crnn(args.checkpoints, data_dir=args.data, split=args.split)


def _cmd_train_det(args: argparse.Namespace) -> None:
    from .det import train_det

    train_det(epochs=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed, out=args.out,
              tracking_uri=args.tracking_uri, experiment=args.experiment)


def _cmd_train_visor(args: argparse.Namespace) -> None:
    from .visor import train_visor

    train_visor(epochs=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed, out=args.out,
                tracking_uri=args.tracking_uri, experiment=args.experiment)


def _cmd_train_scene(args: argparse.Namespace) -> None:
    from .scene import train_scene

    train_scene(out=args.out, labels_csv=args.labels, fotos_dir=args.fotos_dir, epochs=args.epochs,
                freeze_epochs=args.freeze_epochs, batch=args.batch, lr=args.lr, seed=args.seed,
                pretrained=not args.no_pretrained, tracking_uri=args.tracking_uri, experiment=args.experiment,
                target=args.target)


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

    p = sub.add_parser("import-distribuidora", help="importa fotos da distribuidora -> crops do display + labels.csv")
    p.add_argument("--data-root", default=str(paths.ROOT / "FotosDistribuidora"))
    p.add_argument("-o", "--out", default=str(paths.DATA_DIR / "distribuidora_amr"))
    p.add_argument("--notes", default="NA", help="notas de leitura aceitas, separadas por vírgula")
    p.add_argument("--min-sharpness", type=float, default=None, help="limiar de nitidez do crop (default lib: 10.0)")
    p.add_argument("-n", "--max-samples", type=int, default=None, help="debug: subconjunto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--qa-n", type=int, default=40)
    p.set_defaults(func=_cmd_import_distribuidora)

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

    p = sub.add_parser("split-lote", help="reparticiona o dataset da distribuidora POR LOTE e mede vazamento")
    p.add_argument("--src", default=str(paths.DATA_DIR / "distribuidora_amr"))
    p.add_argument("--out", default=str(paths.DATA_DIR / "distribuidora_amr_lote"))
    p.set_defaults(func=_cmd_split_lote)

    p = sub.add_parser("rotulos-plano", help="plano de rotulagem do Lab 2: 300 fotos por lote, 50 em dupla às cegas")
    p.add_argument("--raters", default="R1,R2,R3,R4,R5", help="nomes dos rotuladores, separados por vírgula")
    p.add_argument("--fotos-dir", default=os.environ.get("LEITURISTA_FOTOS_DIR", str(paths.CAMPO_DIR)))
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--n-dupla", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=_cmd_rotulos_plano)

    p = sub.add_parser("rotulos-kappa", help="une as planilhas de rotulagem e calcula kappa de Cohen por classe")
    p.set_defaults(func=_cmd_rotulos_kappa)

    p = sub.add_parser("train-crnn", help="treina o CRNNDigitos (esqueleto do professor, Lab 2) em UFPR-AMR")
    p.add_argument("--data", action="append", help="dataset com labels.csv; repetir p/ vários (default: UFPR-AMR)")
    p.add_argument("--out", default=str(paths.MODELS_DIR / "crnn_digitos.pt"))
    p.add_argument("--init", default=None, help="pesos de partida (fine-tune)")
    p.add_argument("--invert-prob", type=float, default=0.0, help="prob. de inverter polaridade no treino")
    p.add_argument("--crop-jitter", type=float, default=0.0, help="jitter de recorte por borda (estágio 2: 0.07)")
    p.add_argument("--synth-per-epoch", type=int, default=0, help="nº de sequências sintéticas somadas ao treino por época")
    p.add_argument("--synth-only", action="store_true", help="descarta o treino real (estágio 1 puro)")
    p.add_argument("--norm", choices=["batch", "group"], default="batch", help="normalização da CNN")
    p.add_argument("--width", type=int, default=128, help="largura do recorte de entrada (altura fixa 32)")
    p.add_argument("--cosine", action="store_true", help="decaimento cosseno do LR após o warmup")
    p.add_argument("--aug-start", type=int, default=0, help="currículo: épocas iniciais sem augmentation")
    p.add_argument("--rot-deg", type=float, default=10.0, help="giro máximo da augmentation (graus)")
    p.add_argument("--blur-p", type=float, default=0.3, help="prob. de desfoque na augmentation")
    p.add_argument("--no-augment", action="store_true", help="desliga toda augmentation (diagnóstico)")
    p.add_argument("--warmup", type=int, default=0, help="passos de warmup do LR (depois decaimento cosseno)")
    p.add_argument("--clip-calib-iters", type=int, default=0, help="clipping com limiar = p90 da norma nesses passos")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tracking-uri", default=paths.DEFAULT_TRACKING_URI)
    p.add_argument("--experiment", default="crnn-digitos")
    p.set_defaults(func=_cmd_train_crnn)

    p = sub.add_parser("compare-crnn", help="leitura exata + McNemar pareado entre checkpoints do CRNN")
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--data", default=str(paths.FINETUNE_DIR))
    p.add_argument("--split", default="test")
    p.set_defaults(func=_cmd_compare_crnn)

    p = sub.add_parser("train-det", help="fine-tune do PP-OCRv5_mobile_det para localizar o visor (ONNX -> torch -> ONNX)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(paths.MODELS_DIR / "pp_ocr_v5_mobile_det_visor_onnx" / "inference.onnx"))
    p.add_argument("--tracking-uri", default=paths.DEFAULT_TRACKING_URI)
    p.add_argument("--experiment", default="det-visor")
    p.set_defaults(func=_cmd_train_det)

    p = sub.add_parser("train-visor", help="classificador 'este candidato é o visor?' (MobileNetV3-Small)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(paths.MODELS_DIR / "visor_cls.pt"))
    p.add_argument("--tracking-uri", default=paths.DEFAULT_TRACKING_URI)
    p.add_argument("--experiment", default="visor-cls")
    p.set_defaults(func=_cmd_train_visor)

    p = sub.add_parser("train-scene", help="treina o classificador 'tem medidor na foto?' (rótulos fracos do LLM)")
    p.add_argument("--labels", default=str(paths.SCENE_LABELS_CSV))
    p.add_argument("--fotos-dir", default=os.environ.get("LEITURISTA_FOTOS_DIR", str(paths.CAMPO_DIR)))
    p.add_argument("--out", default=str(paths.SCENE_WEIGHTS))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--freeze-epochs", type=int, default=2, help="épocas iniciais só com a cabeça")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true", help="não baixar pesos ImageNet")
    p.add_argument("--tracking-uri", default=paths.DEFAULT_TRACKING_URI)
    p.add_argument("--experiment", default=None, help="default: scene-medidor / scene-legivel")
    p.add_argument("--target", choices=["meter", "readable"], default="meter",
                   help="meter = tem medidor; readable = display legível (ilegível -> vermelho)")
    p.set_defaults(func=_cmd_train_scene)

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
