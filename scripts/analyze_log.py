"""Analise de drift sobre o log de servico (SDD secao 9, ambiente de MONITORAMENTO).

  just analyze                                   # as 100 imagens mais recentes de serving-logs/
  uv run --extra monitor python scripts/analyze_log.py --last 20 --label strong
  uv run --extra monitor python scripts/analyze_log.py --since 2026-11-02T00:10:00+00:00 --until ...

Le o log direto de `--log-dir` (padrao `serving-logs/`, somente leitura) pelo mesmo `LogReader` do
`monitor`, compara com `artifacts/reference*.csv` e escreve em `reports/<label>/`: `drift.md`,
`drift.csv` e PNGs (histogramas 0-255 por canal, histogramas e boxplots das variaveis numericas).
Toda a matematica e os limiares (D 0.05/0.10, minimo de 8 imagens, PSI, Benjamini-Hochberg) estao
em `leiturista.monitoring.drift`; aqui so se orquestra e se desenha. Sai com codigo 2 se faltar
uma coluna obrigatoria no log ou na referencia, e com 1 se nao houver imagens na janela.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from leiturista.monitoring.analysis import (
    CATEGORICAL_VARIABLES,
    CHANNEL_LABELS,
    DEFAULT_WINDOW_IMAGES,
    NUMERIC_VARIABLES,
    DriftReport,
    Reference,
    ReferenceMissing,
    compute_drift,
    load_reference,
    window_histograms,
    window_rows,
)
from leiturista.monitoring.drift import benjamini_hochberg, psi_histogram, scalar_ks
from leiturista.monitoring.log_reader import INPUT_HIST_COLUMNS, LogReader
from leiturista.monitoring.logging_setup import log_dir
from leiturista.paths import ROOT

CHANNEL_COLORS = {"L": "#444444", "R": "#d62728", "G": "#2ca02c", "B": "#1f77b4"}
PSI_NOTE = "faixas de PSI (<0,10 estavel, 0,10 a 0,25 moderado, >0,25 significativo) sao convencao da industria, nao regra provada"
# numericas desenhadas em histograma/boxplot (SDD secao 9): escalares de imagem e saidas numericas
PLOTTED = tuple(NUMERIC_VARIABLES)


class MissingColumns(Exception):
    """Coluna obrigatoria ausente no log ou na referencia."""


def check_columns(reference: Reference, rows: Sequence[Mapping[str, object]]) -> None:
    missing_ref = [name for name in (*NUMERIC_VARIABLES, *CATEGORICAL_VARIABLES) if name not in reference.numeric and name not in reference.categorical]
    missing_log = [
        column
        for column in (*NUMERIC_VARIABLES.values(), *CATEGORICAL_VARIABLES.values(), *INPUT_HIST_COLUMNS)
        if rows and column not in rows[0]
    ]
    if missing_ref or missing_log:
        raise MissingColumns(f"colunas ausentes: referencia={missing_ref} log={missing_log}")


def numeric_column(rows: Sequence[Mapping[str, object]], column: str) -> list[float]:
    return [
        float(value)
        for row in rows
        if isinstance(value := row.get(column), (int, float)) and not isinstance(value, bool)
    ]


def category_proportions(values: Sequence[str]) -> dict[str, float]:
    total = len(values)
    return {} if total == 0 else {k: values.count(k) / total for k in sorted(set(values))}


def variable_table(reference: Reference, rows: Sequence[Mapping[str, object]], report: DriftReport) -> list[dict[str, object]]:
    """Uma linha por variavel numerica: KS de 2 amostras (D, p, p ajustado BH) e PSI."""
    entries: list[dict[str, object]] = []
    for name, column in NUMERIC_VARIABLES.items():
        ref_values = [v for v in reference.numeric[name] if v is not None]
        win_values = numeric_column(rows, column)
        psi = report.psi[name]
        entry: dict[str, object] = {
            "variavel": name,
            "n_ref": len(ref_values),
            "n_janela": len(win_values),
            "ks_d": None,
            "ks_p": None,
            "ks_p_bh": None,
            "psi": psi.value,
            "psi_faixa": psi.band,
            "psi_bins": psi.n_bins,
            "psi_motivo": psi.reason,
        }
        if ref_values and win_values:
            entry["ks_d"], entry["ks_p"] = scalar_ks(ref_values, win_values)
        entries.append(entry)
    testable = [e for e in entries if isinstance(e["ks_p"], float)]
    for entry, adjusted in zip(testable, benjamini_hochberg([float(str(e["ks_p"])) for e in testable])):
        entry["ks_p_bh"] = adjusted
    return entries


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return "-" if math.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def channel_markdown(report: DriftReport) -> list[str]:
    lines = ["| canal | D | p | nivel | n ref | n janela |", "|---|---|---|---|---|---|"]
    for label, ch in report.channels.items():
        lines.append(f"| {label} | {fmt(ch.d)} | {fmt(ch.p, 4)} | {ch.level.name} | {ch.n_ref} | {ch.n_win} |")
    return lines


def histogram_psi_markdown(reference: Reference, rows: Sequence[Mapping[str, object]]) -> list[str]:
    hists = window_histograms(rows)
    lines = ["| canal | PSI (16 bins) | faixa |", "|---|---|---|"]
    for label, channel in CHANNEL_LABELS.items():
        result = psi_histogram(reference.hist[channel], hists[channel])
        lines.append(f"| {label} | {fmt(result.value)} | {result.band or '-'} |")
    return lines


def write_report(
    out_dir: Path,
    label: str,
    reference: Reference,
    rows: Sequence[Mapping[str, object]],
    report: DriftReport,
    table: list[dict[str, object]],
) -> None:
    lines = [
        f"# Drift: {label}",
        "",
        f"Janela: {report.n_window} imagens. Referencia: {report.n_reference} imagens (pequena: leia cada KS e PSI com essa ressalva).",
        f"Pior nivel entre os canais: **{report.worst.name}**.",
        "",
        "## KS por canal (histogramas 0-255 agregados com peso igual por imagem, n em imagens)",
        "",
        *channel_markdown(report),
        "",
        "## PSI dos histogramas de pixels agregados, peso igual por imagem (256 regrupados em 16 bins)",
        "",
        *histogram_psi_markdown(reference, rows),
        "",
        "## Variaveis numericas (KS de 2 amostras e PSI com 10 bins por quantis da referencia)",
        "",
        "| variavel | n ref | n janela | KS D | KS p | KS p (BH) | PSI | faixa | bins |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in table:
        reason = f" ({e['psi_motivo']})" if e["psi_motivo"] else ""
        lines.append(
            f"| {e['variavel']} | {e['n_ref']} | {e['n_janela']} | {fmt(e['ks_d'])} | {fmt(e['ks_p'], 4)} | {fmt(e['ks_p_bh'], 4)} "
            f"| {fmt(e['psi'])}{reason} | {e['psi_faixa'] or '-'} | {e['psi_bins']} |"
        )
    lines += ["", "## Categoricas (proporcao na referencia e na janela, e PSI sobre as categorias)", ""]
    for name, column in CATEGORICAL_VARIABLES.items():
        ref_prop = category_proportions(reference.categorical[name])
        win_prop = category_proportions([str(r[column]) for r in rows if r.get(column) is not None])
        psi = report.psi[name]
        lines += [f"**{name}** (PSI {fmt(psi.value)}{f', {psi.reason}' if psi.reason else ''})", "", "| categoria | referencia | janela |", "|---|---|---|"]
        for category in sorted(set(ref_prop) | set(win_prop)):
            lines.append(f"| {category} | {ref_prop.get(category, 0.0):.3f} | {win_prop.get(category, 0.0):.3f} |")
        lines.append("")
    lines += [
        "## Notas",
        "",
        f"- {PSI_NOTE}.",
        "- Com poucas imagens por janela o PSI e inflado pelo acaso (cerca de 3 imagens por bin com 33 imagens contra 132 de referencia); nao ajustamos limiares para fazer o controle passar.",
        "- p-valores ajustados por Benjamini-Hochberg (FDR) sobre as variaveis numericas; o tamanho de efeito (D ou PSI) vale mais que o p.",
        "",
    ]
    (out_dir / "drift.md").write_text("\n".join(lines), encoding="utf-8")
    with (out_dir / "drift.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def plot_channel_histograms(out_dir: Path, reference: Reference, rows: Sequence[Mapping[str, object]], report: DriftReport) -> None:
    hists = window_histograms(rows)
    fig = Figure(figsize=(11, 7))
    axes = fig.subplots(2, 2, sharex=True)
    for ax, (label, channel) in zip(axes.flat, CHANNEL_LABELS.items()):
        ref = np.asarray(reference.hist[channel], dtype=np.float64)
        win = np.asarray(hists[channel], dtype=np.float64)
        ax.plot(ref / ref.sum(), color="#888888", label="referencia")
        ax.plot(win / win.sum(), color=CHANNEL_COLORS[label], label="janela")
        ch = report.channels[label]
        ax.set_title(f"{label}: D={fmt(ch.d)} ({ch.level.name})")
        ax.set_xlabel("valor 0-255")
        ax.set_ylabel("fracao do total (peso igual por imagem)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "hist_canais.png", dpi=110)


def plot_numeric(out_dir: Path, reference: Reference, rows: Sequence[Mapping[str, object]]) -> None:
    names = [n for n in PLOTTED if numeric_column(rows, NUMERIC_VARIABLES[n]) and any(v is not None for v in reference.numeric[n])]
    cols = 4
    nrows = math.ceil(len(names) / cols)
    fig = Figure(figsize=(16, 3.2 * nrows))
    axes = fig.subplots(nrows, cols)
    for ax, name in zip(axes.flat, names):
        ref_values = [v for v in reference.numeric[name] if v is not None]
        win_values = numeric_column(rows, NUMERIC_VARIABLES[name])
        bins = np.histogram_bin_edges(np.concatenate([ref_values, win_values]), bins=15)
        ax.hist(ref_values, bins=bins, density=True, alpha=0.5, color="#888888", label="referencia")
        ax.hist(win_values, bins=bins, density=True, alpha=0.5, color="#1f77b4", label="janela")
        ax.set_title(name, fontsize=9)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    list(axes.flat)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "hist_variaveis.png", dpi=100)

    fig = Figure(figsize=(16, 3.2 * nrows))
    axes = fig.subplots(nrows, cols)
    for ax, name in zip(axes.flat, names):
        ref_values = [v for v in reference.numeric[name] if v is not None]
        win_values = numeric_column(rows, NUMERIC_VARIABLES[name])
        ax.boxplot([ref_values, win_values], tick_labels=["referencia", "janela"])
        ax.set_title(name, fontsize=9)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "boxplot_variaveis.png", dpi=100)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", type=Path, default=None, help="padrao: SERVING_LOG_DIR ou ./serving-logs")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--last", type=int, default=DEFAULT_WINDOW_IMAGES, help="usa as N imagens mais recentes (ignorado com --since/--until)")
    parser.add_argument("--since", default=None, help="ISO 8601; com --until limita a janela por tempo")
    parser.add_argument("--until", default=None)
    parser.add_argument("--label", default="janela", help="nome do relatorio: reports/<label>/")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    try:
        reference = load_reference(args.artifacts_dir)
    except ReferenceMissing as exc:
        print(exc, file=sys.stderr)
        return 2
    except KeyError as exc:
        print(f"coluna obrigatoria ausente na referencia: {exc}", file=sys.stderr)
        return 2
    reader = LogReader(args.log_dir if args.log_dir is not None else log_dir())
    reader.refresh()
    rows = window_rows(reader, args.since, args.until, args.last)
    if not rows:
        print("nenhuma imagem com variaveis de entrada na janela", file=sys.stderr)
        return 1
    try:
        check_columns(reference, rows)
    except MissingColumns as exc:
        print(exc, file=sys.stderr)
        return 2

    report = compute_drift(reference, rows)
    table = variable_table(reference, rows, report)
    out_dir = args.out_dir / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    write_report(out_dir, args.label, reference, rows, report, table)
    plot_channel_histograms(out_dir, reference, rows, report)
    plot_numeric(out_dir, reference, rows)

    print(f"{args.label}: {report.n_window} imagens na janela, {report.n_reference} na referencia")
    for label, ch in report.channels.items():
        print(f"  {label}: D={fmt(ch.d)} p={fmt(ch.p, 4)} {ch.level.name}")
    print(f"  pior nivel: {report.worst.name}")
    print(f"relatorio em {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
