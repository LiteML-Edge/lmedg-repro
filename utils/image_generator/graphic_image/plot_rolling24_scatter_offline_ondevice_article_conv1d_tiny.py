#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Rolling-24 scatter plots from an Excel workbook.

The scatter points are read from:
  - Predictions_rolling24_Conv1D_Tiny_Python
  - Predictions_rolling24_Conv1D_Tiny_Firmware_Replay

The metric boxes are read from:
  - Metrics_rolling24_Conv1D_Tiny_Python
  - Metrics_rolling24_Conv1D_Tiny_Firmware_Replay

MAE, RMSE, and R2 are not recalculated from the plotted points.
R2 is read as a coefficient from the spreadsheet and displayed as a percentage in the metric box.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _set_ieee_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 7,
        "axes.titlesize": 6,
        "axes.labelsize": 7,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "mathtext.fontset": "dejavuserif",
        "figure.dpi": 100,
        "savefig.dpi": 600,
    })


def _is_empty(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    return str(x).strip().lower() in ("", "nan", "none")


def _coerce_float(x) -> float:
    if _is_empty(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _find_title_cell(raw: pd.DataFrame, title: str) -> tuple[int, int]:
    target = title.strip().lower()
    for r in range(raw.shape[0]):
        for c in range(raw.shape[1]):
            value = raw.iat[r, c]
            if not _is_empty(value) and str(value).strip().lower() == target:
                return r, c

    pattern = re.compile(re.escape(title), re.IGNORECASE)
    for r in range(raw.shape[0]):
        for c in range(raw.shape[1]):
            value = raw.iat[r, c]
            if not _is_empty(value) and pattern.search(str(value)):
                return r, c

    raise ValueError(f"Block title not found: {title}")


def _extract_table_block(raw: pd.DataFrame, title: str) -> pd.DataFrame:
    title_row, title_col = _find_title_cell(raw, title)

    header_row = None
    for r in range(title_row + 1, raw.shape[0]):
        if not _is_empty(raw.iat[r, title_col]):
            header_row = r
            break
    if header_row is None:
        raise ValueError(f"Header row not found after title: {title}")

    cols = []
    c = title_col
    while c < raw.shape[1] and not _is_empty(raw.iat[header_row, c]):
        cols.append(c)
        c += 1

    if not cols:
        raise ValueError(f"No header columns found for block: {title}")

    headers = [str(raw.iat[header_row, c]).strip() for c in cols]
    rows = []
    for r in range(header_row + 1, raw.shape[0]):
        values = [raw.iat[r, c] for c in cols]
        if all(_is_empty(v) for v in values):
            break
        rows.append(values)

    df = pd.DataFrame(rows, columns=headers)
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        if col.lower() != "datetime_end":
            df[col] = df[col].map(_coerce_float)

    return df


def _read_single_sheet_blocks(
    excel_path: Path,
    sheet_name: str,
    offline_prediction_title: str,
    firmware_prediction_title: str,
    offline_metrics_title: str,
    firmware_metrics_title: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None, engine="openpyxl")
    offline_pred = _extract_table_block(raw, offline_prediction_title)
    firmware_pred = _extract_table_block(raw, firmware_prediction_title)
    offline_metrics = _extract_table_block(raw, offline_metrics_title)
    firmware_metrics = _extract_table_block(raw, firmware_metrics_title)
    return offline_pred, firmware_pred, offline_metrics, firmware_metrics


def _read_sheet_table(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if col.lower() != "datetime_end":
            df[col] = df[col].map(_coerce_float)
    return df


def _list_sheets(excel_path: Path) -> list[str]:
    return pd.ExcelFile(excel_path, engine="openpyxl").sheet_names


def _get_col(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    raise KeyError(f"Missing column. Tried {names}. Available columns: {list(df.columns)}")


def _first_value(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        raise KeyError(f"Missing metric column '{column}'. Available columns: {list(df.columns)}")
    values = df[column].dropna()
    if values.empty:
        raise ValueError(f"Metric column '{column}' is empty.")
    return float(values.iloc[0])


def _metrics_for_variable(metrics_df: pd.DataFrame, variable: str) -> dict[str, float]:
    variable = variable.upper().strip()
    if variable == "T":
        return {
            "N": _first_value(metrics_df, "N"),
            "MAE": _first_value(metrics_df, "MAE_T"),
            "RMSE": _first_value(metrics_df, "RMSE_T"),
            "R2": _first_value(metrics_df, "R2_T"),
        }
    if variable == "H":
        return {
            "N": _first_value(metrics_df, "N"),
            "MAE": _first_value(metrics_df, "MAE_H"),
            "RMSE": _first_value(metrics_df, "RMSE_H"),
            "R2": _first_value(metrics_df, "R2_H"),
        }
    raise ValueError("variable must be 'T' or 'H'.")


def _nice_limits(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    lo = float(min(gt.min(), pred.min()))
    hi = float(max(gt.max(), pred.max()))
    rng = hi - lo
    pad = 0.04 * rng if rng > 0 else 0.5
    return lo - pad, hi + pad


def _panel_label(ax, text: str) -> None:
    ax.text(
        0.02, 1.01, text,
        transform=ax.transAxes,
        va="bottom", ha="left",
        fontsize=7,
        fontweight="normal",
        clip_on=False,
    )


def _scatter(
    ax,
    gt,
    pred,
    unit: str,
    *,
    metrics: dict[str, float],
    show_grid: bool,
    panel: str | None,
    metric_decimals: int,
) -> None:
    gt = np.asarray(gt, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(gt) & np.isfinite(pred)
    gt = gt[mask]
    pred = pred[mask]

    n_points = int(gt.size)
    n_metric = int(round(float(metrics["N"])))
    mae = float(metrics["MAE"])
    rmse = float(metrics["RMSE"])
    r2 = float(metrics["R2"])
    r2_percent = 100.0 * r2

    if n_points != n_metric:
        print(f"Warning: plotted points N={n_points}, metrics N={n_metric}. Using metrics from spreadsheet.")

    ax.scatter(gt, pred, s=10, alpha=0.9, linewidths=0.0)

    if n_points > 0:
        lo, hi = _nice_limits(gt, pred)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="black")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    if show_grid:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.30)
    else:
        ax.grid(False)
    ax.set_xlabel(f"Ground truth ({unit})")
    ax.set_ylabel(f"Prediction ({unit})")
    ax.set_aspect("equal", adjustable="box")

    metric_fmt = f"{{:.{metric_decimals}f}}"
    txt = (
        f"N = {n_metric}\n"
        f"MAE = {metric_fmt.format(mae)} {unit}\n"
        f"RMSE = {metric_fmt.format(rmse)} {unit}\n"
        #f"R² = {r2_percent:.2f}%"
        f"R² = {r2:.4f}"
    )

    ax.text(
        0.03, 0.97, txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6,
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor="white",
            edgecolor="black",
            linewidth=0.6,
        ),
    )

    if panel:
        _panel_label(ax, panel)


def make_figure(
    offline_pred: pd.DataFrame,
    firmware_pred: pd.DataFrame,
    offline_metrics: pd.DataFrame,
    firmware_metrics: pd.DataFrame,
    variable: str,
    out_prefix: Path,
    *,
    dpi: int = 600,
    panel_headers: bool = True,
    outer_box: bool = False,
    no_grid: bool = False,
    metric_decimals: int = 4,
) -> tuple[Path, Path]:
    variable = variable.upper().strip()

    if variable == "T":
        gt_off = _get_col(offline_pred, ["Tin", "T_in"])
        pr_off = _get_col(offline_pred, ["Tp", "T_p", "Tpred"])
        gt_dev = _get_col(firmware_pred, ["Tin", "T_in"])
        pr_dev = _get_col(firmware_pred, ["Tp", "T_p", "Tpred"])
        unit = "°C"
        suffix = "T_in"
    elif variable == "H":
        gt_off = _get_col(offline_pred, ["Hin", "H_in"])
        pr_off = _get_col(offline_pred, ["Hp", "H_p", "Hpred"])
        gt_dev = _get_col(firmware_pred, ["Hin", "H_in"])
        pr_dev = _get_col(firmware_pred, ["Hp", "H_p", "Hpred"])
        unit = "%"
        suffix = "H_in"
    else:
        raise ValueError("variable must be 'T' or 'H'.")

    off_metrics = _metrics_for_variable(offline_metrics, variable)
    dev_metrics = _metrics_for_variable(firmware_metrics, variable)

    _set_ieee_style()

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 5.8), constrained_layout=True)

    model_name = ""

    panels = (
        f"(a) Offline (Python)",
        f"(b) On-device (Firmware Replay)",
    ) if panel_headers else (None, None)

    #model_name = "Conv1D_Tiny"

    #panels = (
    #    f"(a) {model_name} — Offline (Python)",
    #    f"(b) {model_name} — On-device (FW Replay)",
    #) if panel_headers else (None, None)

    _scatter(
        axes[0], gt_off, pr_off, unit,
        metrics=off_metrics,
        show_grid=(not no_grid),
        panel=panels[0],
        metric_decimals=metric_decimals,
    )
    _scatter(
        axes[1], gt_dev, pr_dev, unit,
        metrics=dev_metrics,
        show_grid=(not no_grid),
        panel=panels[1],
        metric_decimals=metric_decimals,
    )

    if outer_box:
        fig.add_artist(Rectangle(
            (0.01, 0.01), 0.98, 0.98,
            transform=fig.transFigure,
            fill=False,
            edgecolor="black",
            linewidth=0.8,
        ))

    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"Loaded variable: {variable}")
    print(
        f"Offline: points={len(gt_off.dropna())}, N={int(off_metrics['N'])}, "
        f"MAE={off_metrics['MAE']:.4f}, RMSE={off_metrics['RMSE']:.4f}, "
        f"R2={off_metrics['R2']:.4f} ({100.0 * off_metrics['R2']:.2f}%)"
    )
    print(
        f"Firmware: points={len(gt_dev.dropna())}, N={int(dev_metrics['N'])}, "
        f"MAE={dev_metrics['MAE']:.4f}, RMSE={dev_metrics['RMSE']:.4f}, "
        f"R2={dev_metrics['R2']:.4f} ({100.0 * dev_metrics['R2']:.2f}%)"
    )

    return png_path, pdf_path

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate Rolling-24 scatter plots using prediction rows and spreadsheet metric rows."
    )

    p.add_argument("--excel", type=str, required=True, help="Excel workbook path (.xlsx).")
    p.add_argument("--sheet", type=str, default=None, help="Sheet containing all four blocks. Default: first sheet.")

    p.add_argument("--offline_title", type=str, default="Predictions_rolling24_Conv1D_Tiny_Python")
    p.add_argument("--ondevice_title", type=str, default="Predictions_rolling24_Conv1D_Tiny_Firmware_Replay")
    p.add_argument("--offline_metrics_title", type=str, default="Metrics_rolling24_Conv1D_Tiny_Python")
    p.add_argument("--ondevice_metrics_title", type=str, default="Metrics_rolling24_Conv1D_Tiny_Firmware_Replay")

    p.add_argument("--offline_sheet", type=str, default=None)
    p.add_argument("--ondevice_sheet", type=str, default=None)
    p.add_argument("--offline_metrics_sheet", type=str, default=None)
    p.add_argument("--ondevice_metrics_sheet", type=str, default=None)

    p.add_argument("--variable", type=str, choices=["T", "H"], default="T")
    p.add_argument("--outdir", type=str, default=".")
    p.add_argument("--basename", type=str, default=None)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--metric_decimals", type=int, default=4)

    p.add_argument("--no_panel_labels", action="store_true")
    p.add_argument("--outer_box", action="store_true")
    p.add_argument("--no_grid", action="store_true")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    excel = Path(args.excel)
    if not excel.exists():
        raise FileNotFoundError(excel)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.offline_sheet or args.ondevice_sheet or args.offline_metrics_sheet or args.ondevice_metrics_sheet:
        missing = [
            name for name, value in {
                "--offline_sheet": args.offline_sheet,
                "--ondevice_sheet": args.ondevice_sheet,
                "--offline_metrics_sheet": args.offline_metrics_sheet,
                "--ondevice_metrics_sheet": args.ondevice_metrics_sheet,
            }.items()
            if not value
        ]
        if missing:
            raise SystemExit("Two-sheet/four-sheet mode requires all sheet arguments: " + ", ".join(missing))

        offline_pred = _read_sheet_table(excel, args.offline_sheet)
        firmware_pred = _read_sheet_table(excel, args.ondevice_sheet)
        offline_metrics = _read_sheet_table(excel, args.offline_metrics_sheet)
        firmware_metrics = _read_sheet_table(excel, args.ondevice_metrics_sheet)
    else:
        sheets = _list_sheets(excel)
        sheet = args.sheet if args.sheet else sheets[0]
        try:
            offline_pred, firmware_pred, offline_metrics, firmware_metrics = _read_single_sheet_blocks(
                excel,
                sheet,
                args.offline_title,
                args.ondevice_title,
                args.offline_metrics_title,
                args.ondevice_metrics_title,
            )
        except Exception as exc:
            raise SystemExit(
                f"{exc}\n\nAvailable sheets in '{excel.name}': {sheets}\n"
                f"Use --sheet if the target data is not in the first worksheet."
            ) from exc

    if args.basename:
        base = args.basename
    else:
        base = f"rolling24_scatter_offline_ondevice_{'T_in' if args.variable == 'T' else 'H_in'}_Conv1D_Tiny"

    png_path, pdf_path = make_figure(
        offline_pred,
        firmware_pred,
        offline_metrics,
        firmware_metrics,
        args.variable,
        outdir / base,
        dpi=args.dpi,
        panel_headers=(not args.no_panel_labels),
        outer_box=args.outer_box,
        no_grid=args.no_grid,
        metric_decimals=args.metric_decimals,
    )

    print(f"Saved:\n  {png_path}\n  {pdf_path}")


if __name__ == "__main__":
    main()
