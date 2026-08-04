#!/usr/bin/env python3
"""
Generate paper-ready LiteML-Edge tables from workbook folders and firmware logs.

Default execution is bootstrap-oriented and does not require path arguments when
this script is placed under:

    LiteML/utils/table_generator/

Expected default artifact locations relative to the LiteML repository root:
- utils/workbook_mlp
- utils/workbook_lstm
- utils/workbook_Conv1D_Tiny
- firmwares/environment_mlp/PlatfIO_mlp/logs
- firmwares/environment_lstm/PlatfIO_lstm/logs
- firmwares/environment_Conv1D_Tiny/PlatfIO_Conv1D_Tiny/logs

The generated outputs are aligned with the manuscript structure:
- replay conformance table by model and hardware from workbook comparison artifacts
- energy/latency table by model and hardware with separate Replay and Field values
- memory table with separate minimum free heap values for Replay and Field
- IDLE table with separate Replay and Field baseline values

The generated LaTeX fragments use booktabs-style rules and manuscript-oriented
single-column tabular layouts sized to \\columnwidth. The parent manuscript should load the packages:
- booktabs
- multirow
- siunitx

Calculation summary
-------------------
Prediction/model-I/O values are read from hardware-specific Excel workbooks.
Replay and Field deployment values are parsed from firmware logs for every detected hardware target.
For each model, hardware target, and mode, the newest valid log is selected by file modification time.

Per-log aggregates are computed as follows:
- Invoke energy mean: arithmetic mean of E_inference_window(ΔE_total)
- Event energy mean: arithmetic mean of E_inference_pipeline(ΔE_total)
- Invoke time mean: arithmetic mean of t_inference
- Event time mean: arithmetic mean of t_inference_pipeline
- IDLE voltage/current/power means: arithmetic means over all [BENCH] IDLE rows
- Flash/Arena/Total footprint: values from the last [MEM] row in the log
- Minimum free heap: minimum free-heap value parsed from matching [BENCH] rows
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from openpyxl import Workbook, load_workbook

MODEL_ORDER = ["MLP", "LSTM", "Conv1D Tiny"]
HARDWARE_ORDER = ["WEMOS LOLIN32", "NUCLEO-F411RE"]
MODE_ORDER = ["Replay", "Field"]

MODEL_CANONICAL = {
    "mlp": "MLP",
    "lstm": "LSTM",
    "conv1d_tiny": "Conv1D Tiny",
    "conv1d tiny": "Conv1D Tiny",
    "conv1d": "Conv1D Tiny",
    "con1d_tiny": "Conv1D Tiny",
    "con1d tiny": "Conv1D Tiny",
    "con1d": "Conv1D Tiny",
}

PREDICTION_PREFIX = "predictions_metrics_vs_log_comparison"
MODEL_IO_PREFIX = "model_io_comparison"

# Signed values are accepted because INA219 logs may contain negative current or
# power when the shunt polarity is inverted. Both the micro sign (µ) and plain
# ASCII "u" are accepted for energy units.
NUMBER_RE = r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"

PWR_RE = re.compile(
    rf"\[PWR\] infer \| E_inference_window\(ΔE_total\)=({NUMBER_RE})[µu]Wh .*?"
    rf"\| E_inference_pipeline\(ΔE_total\)=({NUMBER_RE})[µu]Wh .*?"
    rf"\| t_inference=({NUMBER_RE})ms \| t_inference_pipeline=({NUMBER_RE})ms",
    re.UNICODE,
)
IDLE_RE = re.compile(
    rf"\[BENCH\] IDLE .*?\| V_bus=({NUMBER_RE})V I_bus=({NUMBER_RE})mA P_bus=({NUMBER_RE})mW",
    re.UNICODE,
)
MEM_RE = re.compile(
    rf"\[MEM\] Model=({NUMBER_RE})kB \(FLASH\) \| Arena=({NUMBER_RE})kB \(RAM\) \| Total≈({NUMBER_RE})kB",
    re.UNICODE,
)
HEAP_RE = re.compile(
    rf"\[BENCH\] .*?heap=({NUMBER_RE})kB/({NUMBER_RE})kB "
    rf"\(min=({NUMBER_RE})kB, biggest=({NUMBER_RE})kB\) \| arena=({NUMBER_RE})kB",
    re.UNICODE,
)

# Mode and hardware are intentionally detected from the log body only. The log
# filename is never used to classify Replay/Field or the target platform.
MODE_VALUE_RE = re.compile(r"\bLITEML_MODE\b\s*(?:[:=|]\s*)?([0-9]+)\b", re.IGNORECASE)
MODE_TEXT_PATTERNS = [
    re.compile(r"\bLITEML_MODE\b\s*(?:[:=|]\s*)?(REPLAY|FIELD)\b", re.IGNORECASE),
    re.compile(r"\[MODE\]\s*(REPLAY|FIELD)\b", re.IGNORECASE),
    re.compile(r"\bMODE\s*[:=]\s*(REPLAY|FIELD)\b", re.IGNORECASE),
    re.compile(r"\b(REPLAY|FIELD)\s+MODE\b", re.IGNORECASE),
]

HARDWARE_PATTERNS: list[tuple[str, tuple[re.Pattern[str], ...]]] = [
    (
        "NUCLEO-F411RE",
        (
            re.compile(r"\bLITEML_TARGET_STM32F411RE\s*[:=]\s*1\b", re.IGNORECASE),
            re.compile(r"\bSTM32F411RE\b", re.IGNORECASE),
            re.compile(r"\bNUCLEO[-_ /]*F411RE\b", re.IGNORECASE),
            re.compile(r"\bSTM32[-_ ]?F411RE\b", re.IGNORECASE),
        ),
    ),
    (
        "WEMOS LOLIN32",
        (
            re.compile(r"\bLITEML_TARGET_ESP32\s*[:=]\s*1\b", re.IGNORECASE),
            re.compile(r"\bLOLIN32\b", re.IGNORECASE),
            re.compile(r"\bWEMOS[-_ ]+LOLIN32\b", re.IGNORECASE),
            re.compile(r"\bESP32\b", re.IGNORECASE),
            re.compile(r"\bWROOM[-_ ]?32\b", re.IGNORECASE),
        ),
    ),
]

MODEL_BODY_PATTERNS = [
    re.compile(r"\btype\s*[:=]\s*(MLP|LSTM|CONV1D[_ -]?TINY|CONV1D)\b", re.IGNORECASE),
    re.compile(r"\bmodel\s*[:=]\s*(MLP|LSTM|CONV1D[_ -]?TINY|CONV1D)\b", re.IGNORECASE),
    re.compile(r"\barch\s*[:=]\s*(MLP|LSTM|CONV1D[_ -]?TINY|CONV1D)\b", re.IGNORECASE),
]


@dataclass
class BundleRecord:
    model: str
    workbook_root: Path | None
    log_root: Path | None
    prediction_workbooks: dict[str, Path]
    model_io_workbooks: dict[str, Path]
    log_files: list[Path]


class ExtractionError(RuntimeError):
    pass


def normalize_model_name(text: str) -> str | None:
    """Return the canonical model label inferred from text."""
    key = text.lower().replace("-", "_")
    for token, canonical in MODEL_CANONICAL.items():
        if token in key:
            return canonical
    return None


def read_log_text(path: Path) -> str:
    """Read a firmware log once with tolerant UTF-8 decoding."""
    return path.read_text(encoding="utf-8", errors="ignore")


def detect_log_mode_from_text(text: str) -> str | None:
    """Detect Replay or Field exclusively from identifiers inside the log body."""
    numeric = MODE_VALUE_RE.search(text)
    if numeric:
        return {"1": "Replay", "2": "Field"}.get(numeric.group(1))

    for pattern in MODE_TEXT_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).title()
    return None


def detect_log_hardware_from_text(text: str) -> str | None:
    """Detect the hardware target exclusively from identifiers inside the log body."""
    matches: list[str] = []
    for hardware, patterns in HARDWARE_PATTERNS:
        if any(pattern.search(text) for pattern in patterns):
            matches.append(hardware)

    if len(matches) == 1:
        return matches[0]
    if not matches:
        return None

    # Prefer an explicit target macro when generic platform names from toolchain
    # banners make more than one family appear in the same log.
    if re.search(r"\bLITEML_TARGET_STM32F411RE\s*[:=]\s*1\b", text, re.IGNORECASE):
        return "NUCLEO-F411RE"
    if re.search(r"\bLITEML_TARGET_ESP32\s*[:=]\s*1\b", text, re.IGNORECASE):
        return "WEMOS LOLIN32"
    return None


def detect_log_model_from_text(text: str) -> str | None:
    """Detect the model from explicit metadata, prioritizing type over model/arch."""
    for pattern in MODEL_BODY_PATTERNS:
        match = pattern.search(text)
        if match:
            return normalize_model_name(match.group(1))
    return None


def inspect_log_identity(path: Path) -> dict[str, Any]:
    """Return model, hardware, and mode detected from the log contents."""
    text = read_log_text(path)
    return {
        "path": path,
        "model": detect_log_model_from_text(text),
        "hardware": detect_log_hardware_from_text(text),
        "mode": detect_log_mode_from_text(text),
    }


def parse_log_order_key(path: Path) -> tuple[int, str]:
    """Sort logs by file modification time, independently of the filename."""
    try:
        modified_ns = path.stat().st_mtime_ns
    except OSError:
        modified_ns = 0
    return (modified_ns, str(path))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExtractionError(message)


def script_repo_root() -> Path:
    """Infer the LiteML repository root from the script location."""
    script_path = Path(__file__).resolve()
    try:
        return script_path.parents[2]
    except IndexError as exc:
        raise ExtractionError(
            "Unable to infer the LiteML repository root from the script location."
        ) from exc


def default_workbook_dirs(repo_root: Path) -> dict[str, Path]:
    """Return the default workbook folders expected for each supported model."""
    return {
        "MLP": repo_root / "utils" / "workbook_mlp",
        "LSTM": repo_root / "utils" / "workbook_lstm",
        "Conv1D Tiny": repo_root / "utils" / "workbook_Conv1D_Tiny",
    }


def default_log_dirs(repo_root: Path) -> dict[str, list[Path]]:
    """Return the default firmware log folders expected for each supported model."""
    return {
        "MLP": [
            repo_root / "firmwares" / "environment_mlp" / "PlatfIO_mlp" / "logs",
        ],
        "LSTM": [
            repo_root / "firmwares" / "environment_lstm" / "PlatfIO_lstm" / "logs",
        ],
        "Conv1D Tiny": [
            repo_root / "firmwares" / "environment_Conv1D_Tiny" / "PlatfIO_Conv1D_Tiny" / "logs",
        ],
    }


def detect_workbook_hardware_from_path(path: Path, workbook_root: Path | None = None) -> str | None:
    """Detect the target hardware from workbook directory or filename tokens."""
    candidate = path
    if workbook_root is not None:
        try:
            candidate = path.relative_to(workbook_root)
        except ValueError:
            candidate = path

    normalized = "/".join(
        part.lower().replace("-", "_").replace(" ", "_")
        for part in candidate.parts
    )

    stm32_match = re.search(
        r"(?:^|/|_)(?:stm32f411re|stm32_f411re|nucleo_f411re)(?:/|_|$)",
        normalized,
    )
    esp32_match = re.search(
        r"(?:^|/|_)(?:esp32|lolin32|wemos_lolin32)(?:/|_|$)",
        normalized,
    )

    if stm32_match and not esp32_match:
        return "NUCLEO-F411RE"
    if esp32_match and not stm32_match:
        return "WEMOS LOLIN32"
    return None


def keep_newest_workbook(
    selected: dict[str, Path],
    hardware: str,
    candidate: Path,
) -> None:
    """Keep the newest workbook for one hardware target."""
    current = selected.get(hardware)
    if current is None or parse_log_order_key(candidate) > parse_log_order_key(current):
        selected[hardware] = candidate


def scan_model_workbooks(
    model: str,
    workbook_root: Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Find the newest prediction and model-I/O workbooks for each hardware target."""
    del model  # Model ownership is defined by workbook_root.
    if not workbook_root.exists():
        return {}, {}

    prediction_workbooks: dict[str, Path] = {}
    model_io_workbooks: dict[str, Path] = {}

    for path in workbook_root.rglob("*.xlsx"):
        hardware = detect_workbook_hardware_from_path(path, workbook_root)
        if hardware is None:
            continue

        lower_name = path.name.lower()
        if PREDICTION_PREFIX in lower_name:
            keep_newest_workbook(prediction_workbooks, hardware, path)
        elif MODEL_IO_PREFIX in lower_name:
            keep_newest_workbook(model_io_workbooks, hardware, path)

    return prediction_workbooks, model_io_workbooks


def scan_model_logs(log_roots: list[Path]) -> list[Path]:
    """Collect every .log file available under the configured log folders."""
    log_files: list[Path] = []
    for log_root in log_roots:
        if not log_root.exists():
            continue
        for path in log_root.rglob("*.log"):
            if path not in log_files:
                log_files.append(path)
    return log_files


def discover_default_bundles(repo_root: Path) -> dict[str, BundleRecord]:
    """Build model records from the standard LiteML workbook and log directories."""
    bundles: dict[str, BundleRecord] = {}
    workbook_dirs = default_workbook_dirs(repo_root)
    log_dirs = default_log_dirs(repo_root)

    for model in MODEL_ORDER:
        workbook_root = workbook_dirs[model]
        prediction_workbooks, model_io_workbooks = scan_model_workbooks(model, workbook_root)
        log_root_candidates = log_dirs[model]
        log_files = scan_model_logs(log_root_candidates)

        bundles[model] = BundleRecord(
            model=model,
            workbook_root=workbook_root,
            log_root=next((p for p in log_root_candidates if p.exists()), log_root_candidates[0]),
            prediction_workbooks=prediction_workbooks,
            model_io_workbooks=model_io_workbooks,
            log_files=log_files,
        )

    return bundles


def discover_fallback_bundles(search_root: Path) -> dict[str, BundleRecord]:
    """Search the repository tree for additional matching workbooks and logs.

    This fallback is used only to fill missing artifacts when the default folder map
    does not provide a complete set for a given model.
    """
    bundles: dict[str, BundleRecord] = {}

    for path in search_root.rglob("*"):
        if not path.is_file():
            continue

        model = normalize_model_name(str(path))
        if model is None:
            continue

        record = bundles.get(model)
        if record is None:
            record = BundleRecord(
                model=model,
                workbook_root=path.parent,
                log_root=path.parent,
                prediction_workbooks={},
                model_io_workbooks={},
                log_files=[],
            )
            bundles[model] = record

        lower_name = path.name.lower()
        if lower_name.endswith(".xlsx"):
            hardware = detect_workbook_hardware_from_path(path, search_root)
            if hardware is None:
                continue
            if PREDICTION_PREFIX in lower_name:
                keep_newest_workbook(record.prediction_workbooks, hardware, path)
            elif MODEL_IO_PREFIX in lower_name:
                keep_newest_workbook(record.model_io_workbooks, hardware, path)
        elif lower_name.endswith(".log") and path not in record.log_files:
            record.log_files.append(path)

    return bundles


def merge_bundle_maps(primary: dict[str, BundleRecord], fallback: dict[str, BundleRecord]) -> dict[str, BundleRecord]:
    """Merge the default artifact map with the fallback discoveries."""
    merged = dict(primary)
    for model, fallback_record in fallback.items():
        if model not in merged:
            merged[model] = fallback_record
            continue

        record = merged[model]
        for hardware, path in fallback_record.prediction_workbooks.items():
            if hardware not in record.prediction_workbooks:
                record.prediction_workbooks[hardware] = path
        for hardware, path in fallback_record.model_io_workbooks.items():
            if hardware not in record.model_io_workbooks:
                record.model_io_workbooks[hardware] = path
        for log_path in fallback_record.log_files:
            if log_path not in record.log_files:
                record.log_files.append(log_path)

    return merged


def read_rows(path: Path, sheet_name: str) -> list[tuple[Any, ...]]:
    """Read all rows from one workbook sheet using data-only values."""
    workbook = load_workbook(path, read_only=True, data_only=True)
    if sheet_name not in workbook.sheetnames:
        return []
    worksheet = workbook[sheet_name]
    return list(worksheet.iter_rows(values_only=True))


def read_key_value_sheet(path: Path, sheet_name: str) -> dict[str, Any]:
    """Read a two-column Metric/Value sheet into a dictionary."""
    rows = read_rows(path, sheet_name)
    if not rows:
        return {}
    header = rows[0]
    if len(header) < 2 or header[0] != "Metric" or header[1] != "Value":
        return {}
    output: dict[str, Any] = {}
    for row in rows[1:]:
        if not row or row[0] in (None, ""):
            continue
        output[str(row[0]).strip()] = row[1]
    return output


def read_first_table(path: Path, sheet_name: str) -> list[dict[str, Any]]:
    """Read the first non-empty tabular region from a sheet into row dictionaries."""
    rows = read_rows(path, sheet_name)
    if not rows:
        return []
    header: list[str] | None = None
    start_idx = 0
    for idx, row in enumerate(rows):
        if row and row[0] not in (None, ""):
            header = [str(cell).strip() if cell is not None else "" for cell in row]
            start_idx = idx + 1
            break
    if header is None:
        return []
    table: list[dict[str, Any]] = []
    for row in rows[start_idx:]:
        if not row or all(cell in (None, "") for cell in row):
            continue
        item: dict[str, Any] = {}
        for idx, key in enumerate(header):
            if not key:
                continue
            item[key] = row[idx] if idx < len(row) else None
        table.append(item)
    return table


def detect_log_mode(path: Path) -> str | None:
    """Compatibility wrapper that classifies mode from the log body only."""
    return detect_log_mode_from_text(read_log_text(path))


def choose_latest_logs_by_hardware_mode(
    log_files: list[Path],
    expected_model: str,
) -> tuple[dict[str, dict[str, Path]], list[dict[str, Any]]]:
    """Select the newest Replay and Field log for each detected hardware target.

    Classification is based only on identifiers inside each file. The filename
    does not participate in model, mode, or hardware detection.
    """
    candidates: dict[str, dict[str, list[Path]]] = {}
    inventory: list[dict[str, Any]] = []

    for path in log_files:
        identity = inspect_log_identity(path)
        detected_model = identity["model"]
        hardware = identity["hardware"]
        mode = identity["mode"]

        status = "candidate"
        if hardware is None and mode is None:
            status = "ignored: hardware and mode not detected"
        elif hardware is None:
            status = "ignored: hardware not detected"
        elif mode is None:
            status = "ignored: mode not detected"
        else:
            # The containing model bundle remains authoritative. Internal model
            # metadata is recorded for auditing but is not used to reject a log,
            # because legacy firmware logs may contain an inconsistent arch/type tag.
            candidates.setdefault(hardware, {}).setdefault(mode, []).append(path)
            if detected_model is not None and detected_model != expected_model:
                status = f"candidate: model tag differs ({detected_model})"

        inventory.append(
            {
                "Expected model": expected_model,
                "Detected model": detected_model or "UNKNOWN",
                "Hardware": hardware or "UNKNOWN",
                "Mode": mode or "UNKNOWN",
                "Log file": path.name,
                "Log path": str(path),
                "Classification": status,
            }
        )

    selected: dict[str, dict[str, Path]] = {}
    for hardware, by_mode in candidates.items():
        selected[hardware] = {}
        for mode, paths in by_mode.items():
            selected[hardware][mode] = sorted(paths, key=parse_log_order_key)[-1]

    for row in inventory:
        hardware = row["Hardware"]
        mode = row["Mode"]
        path = Path(row["Log path"])
        if hardware in selected and mode in selected[hardware]:
            if path == selected[hardware][mode]:
                row["Classification"] = "selected"
            elif row["Classification"] == "candidate":
                row["Classification"] = "not selected: older duplicate"

    return selected, inventory


def parse_prediction_artifacts(workbook: Path) -> dict[str, Any]:
    """Extract prediction agreement fields from the prediction workbook.

    Returned values are sourced as follows:
    - summary: direct key-value export from the Summary sheet
    - latest_match: the Overview row named
      "Latest valid rolling24 metrics at 4 decimals", with Summary fallback
    - prediction_line: the Overview row named
      "Prediction agreement at 2 decimals"
    """
    summary = read_key_value_sheet(workbook, "Summary")
    latest_metrics = read_first_table(workbook, "Latest_Metrics")
    overview = read_first_table(workbook, "Overview")

    latest_match = None
    prediction_line = None
    for row in overview:
        block = row.get("Block")
        if block == "Latest valid rolling24 metrics at 4 decimals":
            latest_match = row.get("Matched_Rows")
        if block == "Prediction agreement at 2 decimals":
            prediction_line = row.get("Matched_Rows")

    return {
        "summary": summary,
        "latest_metrics": latest_metrics,
        "latest_match": latest_match or summary.get("Latest metrics rounded matches"),
        "prediction_line": prediction_line,
    }


def parse_model_io_artifacts(workbook: Path) -> dict[str, Any]:
    """Extract stage-wise comparison status rows and tolerance metadata from the model-I/O workbook."""
    overview = read_first_table(workbook, "Overview")
    tolerance = read_first_table(workbook, "Tolerance_Protocol")
    by_block = {str(row.get("Block")): row for row in overview if row.get("Block")}
    return {"overview": overview, "tolerance": tolerance, "by_block": by_block}


def parse_log_metrics(path: Path) -> dict[str, Any]:
    """Parse aggregate energy, timing, memory, heap, and IDLE statistics from one firmware log."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    power_rows = PWR_RE.findall(text)
    idle_rows = IDLE_RE.findall(text)
    memory_rows = MEM_RE.findall(text)
    heap_rows = HEAP_RE.findall(text)

    require(power_rows, f"No [PWR] inference rows found in {path.name}.")
    require(idle_rows, f"No [BENCH] IDLE rows found in {path.name}.")
    require(memory_rows, f"No [MEM] rows found in {path.name}.")
    require(heap_rows, f"No heap rows found in {path.name}.")

    invoke_energy_uwh = [float(row[0]) for row in power_rows]
    pipeline_energy_uwh = [float(row[1]) for row in power_rows]
    invoke_time_ms = [float(row[2]) for row in power_rows]
    pipeline_time_ms = [float(row[3]) for row in power_rows]

    idle_voltage_v = [float(row[0]) for row in idle_rows]
    idle_current_ma = [float(row[1]) for row in idle_rows]
    idle_power_mw = [float(row[2]) for row in idle_rows]

    model_flash_kb, arena_kb, total_kb = [float(value) for value in memory_rows[-1]]
    free_heap_kb = [float(row[2]) for row in heap_rows]
    reserved_heap_kb = [float(row[1]) for row in heap_rows]

    def mean(values: Iterable[float]) -> float:
        values_list = list(values)
        return sum(values_list) / len(values_list)

    return {
        "log_file": path.name,
        "invoke_events": len(invoke_energy_uwh),
        "idle_samples": len(idle_voltage_v),
        "invoke_energy_uwh_mean": mean(invoke_energy_uwh),
        "pipeline_energy_uwh_mean": mean(pipeline_energy_uwh),
        "invoke_time_ms_mean": mean(invoke_time_ms),
        "pipeline_time_ms_mean": mean(pipeline_time_ms),
        "idle_voltage_v_mean": mean(idle_voltage_v),
        "idle_current_ma_mean": mean(idle_current_ma),
        "idle_power_mw_mean": mean(idle_power_mw),
        "model_flash_kb": model_flash_kb,
        "arena_kb": arena_kb,
        "total_footprint_kb": total_kb,
        "min_free_heap_kb": min(free_heap_kb),
        "max_heap_pool_kb": max(reserved_heap_kb),
    }


def normalize_prediction_agreement(value: Any) -> str:
    """Normalize prediction agreement text to the manuscript style."""
    text = str(value).strip()
    return re.sub(r"\s*\|\s*", "; ", text)

def latex_escape(text: Any) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def round_half_up(value: float, digits: int) -> float:
    quantum = Decimal("1").scaleb(-digits)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def format_float(value: Any, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def portable_path(value: object, repo_root: Path) -> object:
    """Return repository-relative POSIX paths for generated reviewer-facing records."""
    if not isinstance(value, str) or not value:
        return value
    try:
        path = Path(value)
    except (TypeError, ValueError):
        return value
    if not path.is_absolute():
        return value.replace("\\", "/")
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.name


def portable_rows(rows: list[dict[str, Any]], repo_root: Path) -> list[dict[str, Any]]:
    return [
        {key: portable_path(value, repo_root) for key, value in row.items()}
        for row in rows
    ]


def round_numeric_rows(
    rows: list[dict[str, Any]],
    digits: int = 6,
) -> list[dict[str, Any]]:
    """Limit exported floating-point records without changing internal calculations."""
    return [
        {
            key: round(value, digits) if isinstance(value, float) else value
            for key, value in row.items()
        }
        for row in rows
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_workbook(path: Path, sheets: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    default = workbook.active
    workbook.remove(default)
    for sheet_name, rows in sheets.items():
        worksheet = workbook.create_sheet(sheet_name[:31])
        if not rows:
            worksheet["A1"] = "No rows"
            continue
        headers: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in headers:
                    headers.append(key)
        for column, header in enumerate(headers, start=1):
            worksheet.cell(row=1, column=column, value=header)
        for row_idx, row in enumerate(rows, start=2):
            for column, header in enumerate(headers, start=1):
                worksheet.cell(row=row_idx, column=column, value=row.get(header))
    workbook.save(path)


def sort_model_rows(model_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model_order = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    hardware_order = {name: idx for idx, name in enumerate(HARDWARE_ORDER)}
    return sorted(
        model_rows,
        key=lambda row: (
            model_order.get(str(row.get("Model")), 999),
            hardware_order.get(str(row.get("Hardware")), 999),
            str(row.get("Hardware", "")),
        ),
    )


def build_replay_conformance_rows(model_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in sort_model_rows(model_rows):
        rows.append(
            {
                "Model": item["Model"],
                "Hardware": item["Hardware"],
                "Shared data pipeline": "MATCH",
                "Critical input (x*, p*)": item["Critical_Input_Status"],
                "Raw tensor dump": item["Raw_Tensor_Status"],
                "Immediate raw output (o_raw)": item["Decoded_Raw_Status"],
                "Final prediction (y*)": item["Postprocessed_Status"],
                "Predictions (2 decimals)": item["Predictions_2dp"],
                "Latest metrics (4 decimals)": item["Latest_Metrics_4dp"],
            }
        )
    return rows


def build_energy_rows(model_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in sort_model_rows(model_rows):
        rows.append(
            {
                "Model": item["Model"],
                "Hardware": item["Hardware"],
                "Replay energy (uWh)": round_half_up(item["Replay_Invoke_Energy_uWh_Mean"], 3),
                "Replay time (ms)": round(item["Replay_Invoke_Time_ms_Mean"], 2),
                "Field energy (uWh)": round_half_up(item["Field_Invoke_Energy_uWh_Mean"], 3),
                "Field time (ms)": round(item["Field_Invoke_Time_ms_Mean"], 2),
                "Replay pipeline energy (uWh)": round_half_up(item["Replay_Event_Energy_uWh_Mean"], 3),
                "Replay pipeline time (ms)": round(item["Replay_Event_Time_ms_Mean"], 2),
                "Field pipeline energy (uWh)": round_half_up(item["Field_Event_Energy_uWh_Mean"], 3),
                "Field pipeline time (ms)": round(item["Field_Event_Time_ms_Mean"], 2),
                "Replay log": item["Replay_Log_File"],
                "Field log": item["Field_Log_File"],
            }
        )
    return rows


def build_memory_rows(model_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in sort_model_rows(model_rows):
        rows.append(
            {
                "Model": item["Model"],
                "Hardware": item["Hardware"],
                "Model flash (kB)": round(item["Model_Flash_kB"], 2),
                "Tensor arena (kB)": round(item["Tensor_Arena_kB"], 2),
                "Min free heap Replay (kB)": round(item["Replay_Minimum_Free_Heap_kB"], 1),
                "Min free heap Field (kB)": round(item["Field_Minimum_Free_Heap_kB"], 1),
                "Replay flash (kB)": round(item["Replay_Model_Flash_kB"], 2),
                "Field flash (kB)": round(item["Field_Model_Flash_kB"], 2),
                "Replay arena (kB)": round(item["Replay_Tensor_Arena_kB"], 2),
                "Field arena (kB)": round(item["Field_Tensor_Arena_kB"], 2),
                "Memory consistency": item["Memory_Consistency"],
                "Replay log": item["Replay_Log_File"],
                "Field log": item["Field_Log_File"],
            }
        )
    return rows


def build_idle_rows(model_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in sort_model_rows(model_rows):
        rows.append(
            {
                "Model": item["Model"],
                "Hardware": item["Hardware"],
                "Replay idle current (mA)": round(item["Replay_Idle_Current_mA_Mean"], 2),
                "Replay idle power (mW)": round(item["Replay_Idle_Power_mW_Mean"], 2),
                "Field idle current (mA)": round(item["Field_Idle_Current_mA_Mean"], 2),
                "Field idle power (mW)": round(item["Field_Idle_Power_mW_Mean"], 2),
                "Replay idle voltage (V)": round(item["Replay_Idle_Voltage_V_Mean"], 3),
                "Field idle voltage (V)": round(item["Field_Idle_Voltage_V_Mean"], 3),
                "Replay log": item["Replay_Log_File"],
                "Field log": item["Field_Log_File"],
            }
        )
    return rows


def group_rows_by_model(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group already sorted paper-table rows by model while preserving model order."""
    grouped: list[tuple[str, list[dict[str, Any]]]] = []
    current_model: str | None = None
    current_rows: list[dict[str, Any]] = []

    for row in sort_model_rows(rows):
        model = str(row["Model"])
        if current_model is None:
            current_model = model
        if model != current_model:
            grouped.append((current_model, current_rows))
            current_model = model
            current_rows = []
        current_rows.append(row)

    if current_model is not None:
        grouped.append((current_model, current_rows))
    return grouped


def latex_status_symbol(value: Any) -> str:
    """Render conformance status text using the manuscript symbols."""
    normalized = str(value).strip().upper()
    if normalized == "MATCH":
        return r"$\checkmark$"
    if normalized in {"MISMATCH", "DIFFER", "DIFFERS"}:
        return r"$\times$"
    return latex_escape(value)


def latex_shortstack(value: Any) -> str:
    """Render semicolon- or pipe-separated content as a compact shortstack."""
    parts = [part.strip() for part in re.split(r"\s*(?:;|\|)\s*", str(value)) if part.strip()]
    if not parts:
        return ""
    escaped = [latex_escape(part) for part in parts]
    if len(escaped) == 1:
        return escaped[0]
    return r"\shortstack{" + r"\\".join(escaped) + "}"


def latex_prediction_stack(value: Any) -> str:
    """Render prediction agreement as centered T/H matched-total lines."""
    text = str(value).strip()
    t_match = re.search(r"\bT\s*[:=]?\s*(\d+)(?:\s*/\s*(\d+))?", text, re.IGNORECASE)
    h_match = re.search(r"\bH\s*[:=]?\s*(\d+)(?:\s*/\s*(\d+))?", text, re.IGNORECASE)
    if t_match and h_match:
        t_value = t_match.group(1)
        h_value = h_match.group(1)
        t_total = t_match.group(2)
        h_total = h_match.group(2)
        t_entry = rf"T\,{t_value}/{t_total}" if t_total else rf"T\,{t_value}"
        h_entry = rf"H\,{h_value}/{h_total}" if h_total else rf"H\,{h_value}"
        return rf"\makecell[c]{{{t_entry}\\{h_entry}}}"
    return latex_shortstack(value)


def latex_metric_checks_stack(value: Any) -> str:
    """Render Rolling-24 metric agreement as a dynamically derived ratio."""
    text = str(value).strip()

    ratio = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", text)
    if ratio:
        return f"{ratio.group(1)}/{ratio.group(2)}"

    match_count = re.search(r"\b(\d+)\s+match(?:es)?\b", text, re.IGNORECASE)
    differs_count = re.search(r"\b(\d+)\s+differ(?:s)?\b", text, re.IGNORECASE)
    if match_count and differs_count:
        matches = int(match_count.group(1))
        differs = int(differs_count.group(1))
        return f"{matches}/{matches + differs}"

    return latex_shortstack(value)


def latex_conformance_model_cell(value: Any) -> str:
    """Render the model cell using the manuscript line break for Conv1D Tiny."""
    text = str(value).strip()
    if text == "Conv1D Tiny":
        return r"\makecell[c]{Conv1D\\Tiny}"
    return latex_escape(text)


def collapse_identical_conformance_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse hardware rows only when every displayed conformance field is identical.

    When hardware results differ, the hardware name is appended to the model label so
    no result is silently discarded even though the compact one-column layout omits a
    dedicated Hardware column.
    """
    displayed_fields = (
        "Shared data pipeline",
        "Critical input (x*, p*)",
        "Raw tensor dump",
        "Immediate raw output (o_raw)",
        "Final prediction (y*)",
        "Predictions (2 decimals)",
        "Latest metrics (4 decimals)",
    )
    collapsed: list[dict[str, Any]] = []

    for model, model_rows in group_rows_by_model(rows):
        first = model_rows[0]
        first_signature = tuple(str(first[field]) for field in displayed_fields)
        all_identical = all(
            tuple(str(row[field]) for field in displayed_fields) == first_signature
            for row in model_rows[1:]
        )

        if all_identical:
            collapsed.append(dict(first))
            continue

        for row in model_rows:
            expanded = dict(row)
            expanded["Model"] = f"{model} ({row['Hardware']})"
            collapsed.append(expanded)

    return collapsed


def render_replay_conformance_tex(rows: list[dict[str, Any]]) -> str:
    """Render only the Replay conformance table in the manuscript tabularx pattern."""
    compact_rows = collapse_identical_conformance_rows(rows)
    lines = [
        r"\newlength{\equivwidth}",
        r"\setlength{\equivwidth}{0.98\columnwidth}",
        "",
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Replay stage-wise conformance on both targets under Rolling-24 ($n{=}24$ for each model--target pair).}",
        r"\label{tab:equiv}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.8pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\renewcommand{\tabularxcolumn}[1]{>{\centering\arraybackslash}m{#1}}",
        "",
        r"\newcommand{\theadfit}[1]{{\fontsize{6.0}{6.5}\selectfont\bfseries #1}}",
        r"\newcommand{\hdrstrut}{\rule{0pt}{2.6ex}}",
        r"\newcommand{\hcellone}[1]{\makecell[c]{\hdrstrut\theadfit{#1}}}",
        r"\newcommand{\hcelltwo}[2]{\makecell[c]{\hdrstrut\theadfit{#1}\\\theadfit{#2}}}",
        r"\newcommand{\hcellthree}[3]{\makecell[c]{\hdrstrut\theadfit{#1}\\\theadfit{#2}\\{\fontsize{5.4}{5.9}\selectfont\bfseries #3}}}",
        "",
        r"\begin{tabularx}{\equivwidth}{",
        r">{\centering\arraybackslash}m{1.05cm}",
        r"X X X X X X X}",
        r"\toprule",
        r"\midrule",
        r"\hcellone{Model} &",
        r"\hcelltwo{Shared}{pipe.} &",
        r"\hcellthree{Crit.}{input}{$(x^*,p^*)$} &",
        r"\hcelltwo{Raw}{dump} &",
        r"\hcelltwo{Dec.}{$o_{\mathrm{raw}}$} &",
        r"\hcelltwo{Post.}{$y^*$} &",
        r"\hcelltwo{Pred.}{(T/H)} &",
        r"\hcelltwo{Metric}{checks} \\",
        r"\midrule",
    ]

    for row in compact_rows:
        lines.extend(
            [
                f"{latex_conformance_model_cell(row['Model'])} &",
                f"{latex_status_symbol(row['Shared data pipeline'])} & "
                f"{latex_status_symbol(row['Critical input (x*, p*)'])} & "
                f"{latex_status_symbol(row['Raw tensor dump'])} & "
                f"{latex_status_symbol(row['Immediate raw output (o_raw)'])} & "
                f"{latex_status_symbol(row['Final prediction (y*)'])} &",
                (
                    f"{latex_prediction_stack(row['Predictions (2 decimals)'])} & "
                    f"{latex_metric_checks_stack(row['Latest metrics (4 decimals)'])}"
                    + r" \\"
                ),
            ]
        )

    lines.extend(
        [
            r"\midrule",
            r"\bottomrule",
            r"\end{tabularx}",
            "",
            r"\vspace{1pt}",
            r"\footnotesize $\checkmark$ = MATCH; $\times$ = MISMATCH.",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_energy_tex(rows: list[dict[str, Any]]) -> str:
    """Render the per-inference energy and latency table in the final manuscript layout."""
    lines = [
        r"\begin{table}[!b]",
        r"\centering",
        r"\caption{PER-INFERENCE ENERGY AND LATENCY ($n=24$) UNDER ROLLING-24.}",
        r"\label{tab:infer_energy}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        "",
        r"\begin{tabular*}{\columnwidth}{",
        r"@{\extracolsep{\fill}}",
        r"c",
        r"c",
        r"S[",
        r"    table-column-width=1.02cm,",
        r"    table-format=1.3,",
        r"    table-number-alignment=center",
        r"]",
        r"S[",
        r"    table-column-width=1.02cm,",
        r"    table-format=2.2,",
        r"    table-number-alignment=center",
        r"]",
        r"S[",
        r"    table-column-width=1.02cm,",
        r"    table-format=1.3,",
        r"    table-number-alignment=center",
        r"]",
        r"S[",
        r"    table-column-width=1.02cm,",
        r"    table-format=2.2,",
        r"    table-number-alignment=center",
        r"]",
        r"@{}}",
        r"\toprule",
        r"\midrule",
        r"\multirow[c]{3}{*}[-0.45ex]{\textbf{Model}} &",
        r"\multirow[c]{3}{*}[-0.45ex]{\textbf{Hardware}} &",
        r"\multicolumn{2}{c}{\textbf{Replay}} &",
        r"\multicolumn{2}{c}{\textbf{Field}} \\",
        r"\cmidrule(lr){3-4}",
        r"\cmidrule(lr){5-6}",
        r"& &",
        r"\multicolumn{1}{c}{$\boldsymbol{\overline{\Delta E}}$} &",
        r"\multicolumn{1}{c}{$\boldsymbol{\overline{t}}$} &",
        r"\multicolumn{1}{c}{$\boldsymbol{\overline{\Delta E}}$} &",
        r"\multicolumn{1}{c}{$\boldsymbol{\overline{t}}$} \\",
        r"& &",
        r"\multicolumn{1}{c}{\textbf{(\si{\micro\watt\hour})}} &",
        r"\multicolumn{1}{c}{\textbf{(\si{\milli\second})}} &",
        r"\multicolumn{1}{c}{\textbf{(\si{\micro\watt\hour})}} &",
        r"\multicolumn{1}{c}{\textbf{(\si{\milli\second})}} \\",
        r"\midrule",
    ]

    for model, model_rows in group_rows_by_model(rows):
        span = len(model_rows)
        for row_idx, row in enumerate(model_rows):
            model_cell = (
                f"\\multirow[c]{{{span}}}{{*}}{{{latex_escape(model)}}}"
                if row_idx == 0
                else ""
            )
            lines.append(
                f"{model_cell} & {latex_escape(row['Hardware'])} & "
                f"{format_float(row['Replay energy (uWh)'], 3)} & "
                f"{format_float(row['Replay time (ms)'], 2)} & "
                f"{format_float(row['Field energy (uWh)'], 3)} & "
                f"{format_float(row['Field time (ms)'], 2)}"
                + r" \\"
            )

    lines.extend(
        [
            r"\midrule",
            r"\bottomrule",
            r"\end{tabular*}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"

def render_memory_tex(rows: list[dict[str, Any]]) -> str:
    """Render the model size, tensor arena, and heap-headroom table in the final layout."""
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{MODEL SIZE, TENSOR ARENA, AND HEAP HEADROOM.}",
        r"\label{tab:memory}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        "",
        r"\begin{tabular*}{\columnwidth}{",
        r"@{\extracolsep{\fill}}",
        r">{\centering\arraybackslash}m{1.37cm}",
        r">{\centering\arraybackslash}m{2.04cm}",
        r"S[",
        r"    table-column-width=1.02cm,",
        r"    table-format=2.2,",
        r"    table-number-alignment=center",
        r"]",
        r"S[",
        r"    table-column-width=1.02cm,",
        r"    table-format=1.2,",
        r"    table-number-alignment=center",
        r"]",
        r"S[",
        r"    table-column-width=1.02cm,",
        r"    table-format=3.1,",
        r"    table-number-alignment=center",
        r"]",
        r"S[",
        r"    table-column-width=1.02cm,",
        r"    table-format=3.1,",
        r"    table-number-alignment=center",
        r"]",
        r"@{}}",
        r"\toprule",
        r"\midrule",
        r"\multirow[c]{1.7}{*}{\textbf{Model}} &",
        r"\multirow[c]{1.7}{*}{\textbf{Hardware}} &",
        r"\multicolumn{1}{c}{",
        r"    \multirow[c]{2}{*}{",
        r"        \shortstack[c]{\textbf{Model size}\\\textbf{(kB)}}",
        r"    }",
        r"} &",
        r"\multicolumn{1}{c}{",
        r"    \multirow[c]{2}{*}{",
        r"        \shortstack[c]{\textbf{Arena}\\\textbf{(kB)}}",
        r"    }",
        r"} &",
        r"\multicolumn{2}{c}{",
        r"    \shortstack[c]{\textbf{Min. free heap}\\\textbf{(kB)}}",
        r"} \\",
        r"\cmidrule(lr){5-6}",
        r"& & & &",
        r"\multicolumn{1}{c}{\textbf{Replay}} &",
        r"\multicolumn{1}{c}{\textbf{Field}} \\",
        r"\midrule",
    ]

    for model, model_rows in group_rows_by_model(rows):
        span = len(model_rows)
        for row_idx, row in enumerate(model_rows):
            model_cell = (
                f"\\multirow[c]{{{span}}}{{*}}{{{latex_escape(model)}}}"
                if row_idx == 0
                else ""
            )
            lines.append(
                f"{model_cell} & {latex_escape(row['Hardware'])} & "
                f"{format_float(row['Model flash (kB)'], 2)} & "
                f"{format_float(row['Tensor arena (kB)'], 2)} & "
                f"{format_float(row['Min free heap Replay (kB)'], 1)} & "
                f"{format_float(row['Min free heap Field (kB)'], 1)}"
                + r" \\"
            )

    lines.extend(
        [
            r"\midrule",
            r"\bottomrule",
            r"\end{tabular*}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"

def render_idle_tex(rows: list[dict[str, Any]]) -> str:
    """Render the baseline IDLE current and power table in the final manuscript layout."""
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{BASELINE IDLE CURRENT AND POWER.}",
        r"\label{tab:idle}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        "",
        r"\begin{tabular*}{\columnwidth}{",
        r"@{\extracolsep{\fill}}",
        r">{\centering\arraybackslash}m{1.37cm}",
        r">{\centering\arraybackslash}m{2.04cm}",
        r"S[",
        r"    table-column-width=1.02cm,",
        r"    table-format=2.2,",
        r"    table-number-alignment=center",
        r"]",
        r"S[",
        r"    table-column-width=1.02cm,",
        r"    table-format=3.2,",
        r"    table-number-alignment=center",
        r"]",
        r"S[",
        r"    table-column-width=1.02cm,",
        r"    table-format=2.2,",
        r"    table-number-alignment=center",
        r"]",
        r"S[",
        r"    table-column-width=1.02cm,",
        r"    table-format=3.2,",
        r"    table-number-alignment=center",
        r"]",
        r"@{}}",
        r"\toprule",
        r"\midrule",
        r"\multirow[c]{3.3}{*}{\textbf{Model}} &",
        r"\multirow[c]{3.3}{*}{\textbf{Hardware}} &",
        r"\multicolumn{2}{c}{\textbf{Replay}} &",
        r"\multicolumn{2}{c}{\textbf{Field}} \\",
        r"\cmidrule(lr){3-4}",
        r"\cmidrule(lr){5-6}",
        r"& &",
        r"\multicolumn{1}{c}{\(\mathbf{I}_{\mathrm{mean}}\)} &",
        r"\multicolumn{1}{c}{\(\mathbf{P}_{\mathrm{mean}}\)} &",
        r"\multicolumn{1}{c}{\(\mathbf{I}_{\mathrm{mean}}\)} &",
        r"\multicolumn{1}{c}{\(\mathbf{P}_{\mathrm{mean}}\)} \\",
        r"& &",
        r"\multicolumn{1}{c}{\textbf{(mA)}} &",
        r"\multicolumn{1}{c}{\textbf{(mW)}} &",
        r"\multicolumn{1}{c}{\textbf{(mA)}} &",
        r"\multicolumn{1}{c}{\textbf{(mW)}} \\",
        r"\midrule",
    ]

    for model, model_rows in group_rows_by_model(rows):
        span = len(model_rows)
        for row_idx, row in enumerate(model_rows):
            model_cell = (
                f"\\multirow[c]{{{span}}}{{*}}{{{latex_escape(model)}}}"
                if row_idx == 0
                else ""
            )
            lines.append(
                f"{model_cell} & {latex_escape(row['Hardware'])} & "
                f"{format_float(row['Replay idle current (mA)'], 2)} & "
                f"{format_float(row['Replay idle power (mW)'], 2)} & "
                f"{format_float(row['Field idle current (mA)'], 2)} & "
                f"{format_float(row['Field idle power (mW)'], 2)}"
                + r" \\"
            )

    lines.extend(
        [
            r"\midrule",
            r"\bottomrule",
            r"\end{tabular*}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"

def assemble_model_records(bundle: BundleRecord) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create one consolidated record per model and detected hardware target."""
    selected_logs, inventory = choose_latest_logs_by_hardware_mode(bundle.log_files, bundle.model)
    require(selected_logs, f"No logs with detectable hardware/mode identifiers were found for {bundle.model}.")

    for hardware, by_mode in selected_logs.items():
        missing_modes = [mode for mode in MODE_ORDER if mode not in by_mode]
        require(
            not missing_modes,
            f"Missing {', '.join(missing_modes)} log(s) for {bundle.model} on {hardware}. "
            "Mode and hardware must be declared inside the log body.",
        )
        require(
            hardware in bundle.prediction_workbooks,
            f"Prediction workbook missing for {bundle.model} on {hardware}.",
        )
        require(
            hardware in bundle.model_io_workbooks,
            f"Model I/O workbook missing for {bundle.model} on {hardware}.",
        )

    records: list[dict[str, Any]] = []
    hardware_names = sorted(
        selected_logs,
        key=lambda name: (HARDWARE_ORDER.index(name) if name in HARDWARE_ORDER else 999, name),
    )

    for hardware in hardware_names:
        prediction_workbook = bundle.prediction_workbooks[hardware]
        model_io_workbook = bundle.model_io_workbooks[hardware]

        detected_prediction_hardware = detect_workbook_hardware_from_path(
            prediction_workbook,
            bundle.workbook_root,
        )
        detected_model_io_hardware = detect_workbook_hardware_from_path(
            model_io_workbook,
            bundle.workbook_root,
        )
        require(
            detected_prediction_hardware == hardware,
            f"Prediction workbook hardware mismatch for {bundle.model}: "
            f"row={hardware}, workbook={prediction_workbook}.",
        )
        require(
            detected_model_io_hardware == hardware,
            f"Model I/O workbook hardware mismatch for {bundle.model}: "
            f"row={hardware}, workbook={model_io_workbook}.",
        )

        prediction = parse_prediction_artifacts(prediction_workbook)
        model_io = parse_model_io_artifacts(model_io_workbook)
        blocks = model_io["by_block"]
        critical = blocks.get("Input / critical tensor columns", {})
        raw_tensor = blocks.get("Postprocess / raw tensor dump columns", {})
        raw_output = blocks.get("Postprocess / raw output columns", {})
        final_prediction = blocks.get("Postprocess / final prediction columns", {})
        prediction_summary = prediction["summary"]

        replay_path = selected_logs[hardware]["Replay"]
        field_path = selected_logs[hardware]["Field"]
        replay_metrics = parse_log_metrics(replay_path)
        field_metrics = parse_log_metrics(field_path)

        flash_equal = abs(replay_metrics["model_flash_kb"] - field_metrics["model_flash_kb"]) <= 0.01
        arena_equal = abs(replay_metrics["arena_kb"] - field_metrics["arena_kb"]) <= 0.01
        memory_consistency = "MATCH" if flash_equal and arena_equal else "DIFFER"

        records.append(
            {
                "Model": bundle.model,
                "Hardware": hardware,
                "Workbook_Root": str(bundle.workbook_root) if bundle.workbook_root else "",
                "Log_Root": str(bundle.log_root) if bundle.log_root else "",
                "Prediction_Workbook": str(prediction_workbook),
                "Model_IO_Workbook": str(model_io_workbook),
                "Replay_Log_File": replay_path.name,
                "Replay_Log_Path": str(replay_path),
                "Field_Log_File": field_path.name,
                "Field_Log_Path": str(field_path),
                "Critical_Input_Status": critical.get("Status", "UNKNOWN"),
                "Raw_Tensor_Status": raw_tensor.get("Status", "UNKNOWN"),
                "Decoded_Raw_Status": raw_output.get("Status", "UNKNOWN"),
                "Postprocessed_Status": final_prediction.get("Status", "UNKNOWN"),
                "Predictions_2dp": normalize_prediction_agreement(
                    prediction.get("prediction_line") or prediction_summary.get("Prediction status", "UNKNOWN")
                ),
                "Latest_Metrics_4dp": prediction.get("latest_match")
                or prediction_summary.get("Latest metrics rounded matches", "UNKNOWN"),
                "Prediction_T_Matches": prediction_summary.get("Prediction T matches"),
                "Prediction_H_Matches": prediction_summary.get("Prediction H matches"),
                "Replay_Invoke_Energy_uWh_Mean": replay_metrics["invoke_energy_uwh_mean"],
                "Replay_Event_Energy_uWh_Mean": replay_metrics["pipeline_energy_uwh_mean"],
                "Replay_Invoke_Time_ms_Mean": replay_metrics["invoke_time_ms_mean"],
                "Replay_Event_Time_ms_Mean": replay_metrics["pipeline_time_ms_mean"],
                "Replay_Minimum_Free_Heap_kB": replay_metrics["min_free_heap_kb"],
                "Replay_Idle_Voltage_V_Mean": replay_metrics["idle_voltage_v_mean"],
                "Replay_Idle_Current_mA_Mean": replay_metrics["idle_current_ma_mean"],
                "Replay_Idle_Power_mW_Mean": replay_metrics["idle_power_mw_mean"],
                "Replay_Model_Flash_kB": replay_metrics["model_flash_kb"],
                "Replay_Tensor_Arena_kB": replay_metrics["arena_kb"],
                "Replay_Total_Footprint_kB": replay_metrics["total_footprint_kb"],
                "Field_Invoke_Energy_uWh_Mean": field_metrics["invoke_energy_uwh_mean"],
                "Field_Event_Energy_uWh_Mean": field_metrics["pipeline_energy_uwh_mean"],
                "Field_Invoke_Time_ms_Mean": field_metrics["invoke_time_ms_mean"],
                "Field_Event_Time_ms_Mean": field_metrics["pipeline_time_ms_mean"],
                "Field_Minimum_Free_Heap_kB": field_metrics["min_free_heap_kb"],
                "Field_Idle_Voltage_V_Mean": field_metrics["idle_voltage_v_mean"],
                "Field_Idle_Current_mA_Mean": field_metrics["idle_current_ma_mean"],
                "Field_Idle_Power_mW_Mean": field_metrics["idle_power_mw_mean"],
                "Field_Model_Flash_kB": field_metrics["model_flash_kb"],
                "Field_Tensor_Arena_kB": field_metrics["arena_kb"],
                "Field_Total_Footprint_kB": field_metrics["total_footprint_kb"],
                # Keep the manuscript-compatible single footprint columns. Field is
                # used as before; both mode-specific values remain available above.
                "Model_Flash_kB": field_metrics["model_flash_kb"],
                "Tensor_Arena_kB": field_metrics["arena_kb"],
                "Total_Footprint_kB": field_metrics["total_footprint_kb"],
                "Memory_Consistency": memory_consistency,
            }
        )

    return records, inventory


def main() -> int:
    """Resolve artifact locations, parse the inputs, and write all output tables."""
    inferred_repo_root = script_repo_root()

    parser = argparse.ArgumentParser(
        description="Generate LiteML-Edge paper tables from default workbook and log directories."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=inferred_repo_root,
        help="LiteML repository root. When omitted, the script infers it from its own location.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "out_tables",
        help="Directory where generated files will be written.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    default_bundles = discover_default_bundles(repo_root)
    fallback_bundles = discover_fallback_bundles(repo_root)
    bundles = merge_bundle_maps(default_bundles, fallback_bundles)

    available_models = [
        model for model in MODEL_ORDER
        if bundles.get(model)
        and bundles[model].prediction_workbooks
        and bundles[model].model_io_workbooks
        and bundles[model].log_files
    ]
    require(available_models, f"No valid LiteML-Edge workbook/log sets were found under {repo_root}.")

    model_rows: list[dict[str, Any]] = []
    log_inventory_rows: list[dict[str, Any]] = []
    for model in available_models:
        records, inventory = assemble_model_records(bundles[model])
        model_rows.extend(records)
        log_inventory_rows.extend(inventory)

    replay_rows = build_replay_conformance_rows(model_rows)
    energy_rows = build_energy_rows(model_rows)
    memory_rows = build_memory_rows(model_rows)
    idle_rows = build_idle_rows(model_rows)

    # Keep reviewer-facing manifests portable across extraction locations.
    model_rows = round_numeric_rows(
        portable_rows(model_rows, repo_root),
        digits=6,
    )
    log_inventory_rows = portable_rows(log_inventory_rows, repo_root)

    write_csv(output_dir / "paper_model_sources.csv", model_rows)
    write_csv(output_dir / "log_classification_inventory.csv", log_inventory_rows)
    write_csv(output_dir / "table_replay_conformance.csv", replay_rows)
    write_csv(output_dir / "table_infer_energy.csv", energy_rows)
    write_csv(output_dir / "table_memory.csv", memory_rows)
    write_csv(output_dir / "table_idle.csv", idle_rows)

    write_workbook(
        output_dir / "liteml_edge_paper_tables.xlsx",
        {
            "model_sources": model_rows,
            "log_inventory": log_inventory_rows,
            "replay_conformance": replay_rows,
            "infer_energy": energy_rows,
            "memory": memory_rows,
            "idle": idle_rows,
        },
    )

    (output_dir / "table_replay_conformance.tex").write_text(render_replay_conformance_tex(replay_rows), encoding="utf-8")
    (output_dir / "table_infer_energy.tex").write_text(render_energy_tex(energy_rows), encoding="utf-8")
    (output_dir / "table_memory.tex").write_text(render_memory_tex(memory_rows), encoding="utf-8")
    (output_dir / "table_idle.tex").write_text(render_idle_tex(idle_rows), encoding="utf-8")

    manifest = {
        "repo_root": ".",
        "models_found": list(dict.fromkeys(row["Model"] for row in model_rows)),
        "hardware_found": list(dict.fromkeys(row["Hardware"] for row in model_rows)),
        "classification_rule": (
            "model, hardware, and mode are detected from log contents; filenames are ignored; "
            "prediction and model-I/O workbooks are independently selected for each hardware target "
            "from directory or filename tokens"
        ),
        "default_workbook_dirs": {k: portable_path(str(v), repo_root) for k, v in default_workbook_dirs(repo_root).items()},
        "default_log_dirs": {k: [portable_path(str(p), repo_root) for p in v] for k, v in default_log_dirs(repo_root).items()},
        "outputs": [
            "paper_model_sources.csv",
            "log_classification_inventory.csv",
            "table_replay_conformance.csv",
            "table_infer_energy.csv",
            "table_memory.csv",
            "table_idle.csv",
            "liteml_edge_paper_tables.xlsx",
            "table_replay_conformance.tex",
            "table_infer_energy.tex",
            "table_memory.tex",
            "table_idle.tex",
        ],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"LiteML-Edge paper tables generated in: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExtractionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
