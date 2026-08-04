# Conv1D_Tiny audit workbooks by platform

This package contains two scripts for validating **Replay** logs from the Conv1D_Tiny firmware and generating separate workbooks for each hardware platform:

- `compare_model_io_to_log_Conv1D_Tiny.py`
- `compare_predictions_metrics_to_log_Conv1D_Tiny.py`

Recognized platforms:

- `esp32` — Lolin32 / ESP-WROOM-32
- `stm32f411re` — NUCLEO-F411RE / STM32F411RE

> **Important:** these two scripts accept **Replay** logs only. Field mode does not contain the blocks required for a 1:1 comparison with the Python references.

---

## 1. Purpose of each script

### `compare_model_io_to_log_Conv1D_Tiny.py`

Audits the internal stages of the Python ↔ firmware pipeline:

- preprocessing;
- input tensor;
- raw output tensor;
- semantic output;
- post-processing;
- final prediction;
- exact, bitwise, and tolerance-based agreement.

Minimum blocks expected in the log:

```text
[DBG_REPLAY_CSV]
[DBG_MODEL_IN_CSV]
[DBG_MODEL_OUT_CSV]
```

Tolerances preserved from the original script:

```text
Model input:                  2e-6
Raw output:                   2e-6
Semantic output:              2e-6
Final output:                 2e-5
Semantic debug:               2e-5
```

### `compare_predictions_metrics_to_log_Conv1D_Tiny.py`

Audits the final Rolling-24 stage:

- prediction alignment by `datetime_end`;
- temperature and humidity ground truth;
- temperature and humidity predictions;
- global MAE, RMSE, and R²;
- per-variable MAE, RMSE, and R²;
- maximum differences and result classification.

Minimum blocks expected in the log:

```text
[DBG_REPLAY_CSV]
[HOUR] METRICS
```

Preserved default configuration:

```text
Spreadsheet rows:             25 through 48
Predictions:                  comparison with 2 decimal places
Metrics:                      comparison with 4 decimal places
```

---

## 2. Requirements

- Python 3.10 or later;
- `pandas`;
- `openpyxl`;
- project structure containing `utils/global_utils/paths_Conv1D_Tiny.py`.

Install the libraries:

```bash
python -m pip install pandas openpyxl
```

Run the scripts from the project root, where the `utils` directory can be located.

In PowerShell, the root can also be set explicitly:

```powershell
$env:RUNNER_PROJECT_ROOT = (Get-Location).Path
```

On Linux or macOS:

```bash
export RUNNER_PROJECT_ROOT="$(pwd)"
```

---

## 3. Expected project structure

Default structure used when paths are not provided through arguments. The scripts search recursively under `firmwares/environment_Conv1D_Tiny` to locate compatible logs:

```text
<PROJECT_ROOT>/
├── utils/
│   ├── global_utils/
│   │   └── paths_Conv1D_Tiny.py
│   └── workbook_Conv1D_Tiny/
└── firmwares/
    └── environment_Conv1D_Tiny/
        └── PlatfIO_Conv1D_Tiny/
            └── logs/
                ├── device-monitor-*.log
                ├── device-monitor-*.txt
                └── device-monitor-*.zip
```

The `.zip` files must contain at least one `.log` or `.txt` file.

---

## 4. Platform and mode identification

The scripts search the log for lines such as:

```text
[PLATFORM] Lolin32 / ESP-WROOM-32
[MODE] LITEML_MODE=1 -> REPLAY (evaluation contract)
```

or:

```text
[PLATFORM] NUCLEO-F411RE / STM32F411RE
[MODE] LITEML_MODE=1 -> REPLAY (evaluation contract)
```

When `--platform auto` is used, the platform is obtained from the `[PLATFORM]` line.

When `--mode auto` is used, the mode is detected from the log. Execution continues only when the detected mode is `REPLAY`.

---

## 5. Recommended commands

### Generate both workbooks for ESP32 and STM32F411RE

Model I/O:

```bash
python compare_model_io_to_log_Conv1D_Tiny.py --platform all --mode replay
```

Predictions and metrics:

```bash
python compare_predictions_metrics_to_log_Conv1D_Tiny.py --platform all --mode replay
```

### Use explicit directories

```bash
python compare_model_io_to_log_Conv1D_Tiny.py \
  --platform all \
  --mode replay \
  --log-dir firmwares/environment_Conv1D_Tiny/PlatfIO_Conv1D_Tiny/logs \
  --out-dir utils/workbook_Conv1D_Tiny
```

```bash
python compare_predictions_metrics_to_log_Conv1D_Tiny.py \
  --platform all \
  --mode replay \
  --log-dir firmwares/environment_Conv1D_Tiny/PlatfIO_Conv1D_Tiny/logs \
  --out-dir utils/workbook_Conv1D_Tiny
```

In PowerShell, use the backtick to split commands across lines:

```powershell
python .\compare_model_io_to_log_Conv1D_Tiny.py `
  --platform all `
  --mode replay `
  --log-dir .\firmwares\environment_Conv1D_Tiny\PlatfIO_Conv1D_Tiny\logs `
  --out-dir .\utils\workbook_Conv1D_Tiny
```

### Generate only for ESP32

```bash
python compare_model_io_to_log_Conv1D_Tiny.py --platform esp32 --mode replay
```

```bash
python compare_predictions_metrics_to_log_Conv1D_Tiny.py --platform esp32 --mode replay
```

### Generate only for STM32F411RE

```bash
python compare_model_io_to_log_Conv1D_Tiny.py --platform stm32f411re --mode replay
```

```bash
python compare_predictions_metrics_to_log_Conv1D_Tiny.py --platform stm32f411re --mode replay
```

### Process a specific log

Use this form for one platform only:

```bash
python compare_model_io_to_log_Conv1D_Tiny.py \
  --platform stm32f411re \
  --mode replay \
  --log device-monitor-260724-154440.zip
```

```bash
python compare_predictions_metrics_to_log_Conv1D_Tiny.py \
  --platform stm32f411re \
  --mode replay \
  --log device-monitor-260724-154440.zip
```

The script verifies that the platform and mode found in the file match the provided arguments.

---

## 6. Generated outputs

When `--out` is not used, files are automatically organized by platform and mode:

```text
utils/workbook_Conv1D_Tiny/
├── esp32/
│   └── replay/
│       ├── model_io_comparison_Conv1D_Tiny_esp32_replay.xlsx
│       └── predictions_metrics_vs_log_comparison_Conv1D_Tiny_esp32_replay.xlsx
└── stm32f411re/
    └── replay/
        ├── model_io_comparison_Conv1D_Tiny_stm32f411re_replay.xlsx
        └── predictions_metrics_vs_log_comparison_Conv1D_Tiny_stm32f411re_replay.xlsx
```

When the destination workbook is open in Excel, the script creates an alternative name containing the date and time instead of overwriting a locked file.

---

## 7. Model I/O script arguments

```text
--log PATH
    Specific .log, .txt, or .zip file.

--log-dir DIRECTORY
    Directory searched recursively for device-monitor-* logs.

--out PATH.xlsx
    Exact workbook path. Allowed for one platform only.

--out-dir DIRECTORY
    Root directory where <platform>/replay/ subdirectories will be created.

--platform {auto,all,esp32,stm32f411re}
    auto: detects the platform from the selected log.
    all: generates one workbook for ESP32 and another for STM32F411RE.
    esp32: accepts ESP32 logs only.
    stm32f411re: accepts STM32F411RE logs only.

--mode {auto,replay}
    replay: explicitly requires a Replay log.
    auto: detects the mode but rejects Field.

-h, --help
    Displays command-line help.
```

Help:

```bash
python compare_model_io_to_log_Conv1D_Tiny.py --help
```

> The model I/O script imports `utils.global_utils.paths_Conv1D_Tiny` during initialization. Therefore, even `--help` must be executed inside the project structure or with `RUNNER_PROJECT_ROOT` configured.

---

## 8. Predictions and metrics script arguments

In addition to the common arguments, this script accepts:

```text
--pred-xlsx PATH.xlsx
    Python Rolling-24 prediction spreadsheet.

--metrics-xlsx PATH.xlsx
    Python Rolling-24 metrics spreadsheet.

--excel-row-start N
    First included Excel row. Default: 25.

--excel-row-end N
    Last included Excel row. Default: 48.

--round-decimals N
    Decimal places used for prediction comparison. Default: 2.

--metrics-round-decimals N
    Decimal places used for metric comparison. Default: 4.
```

Full help:

```bash
python compare_predictions_metrics_to_log_Conv1D_Tiny.py --help
```

Example with explicit references:

```bash
python compare_predictions_metrics_to_log_Conv1D_Tiny.py \
  --platform stm32f411re \
  --mode replay \
  --pred-xlsx environment_quantized_predictions_rolling24_Conv1D_Tiny.xlsx \
  --metrics-xlsx environment_quantized_metrics_rolling24_Conv1D_Tiny.xlsx \
  --log device-monitor-260724-154440.zip \
  --out stm32_predictions_metrics.xlsx
```

---

## 9. Rules for `--platform all`

With `--platform all`, the script runs once for each platform.

Do not combine `--platform all` with:

```text
--log
--out
```

These arguments represent a single file and cannot serve both platforms simultaneously.

Use:

```text
--log-dir
--out-dir
```

Correct example:

```bash
python compare_model_io_to_log_Conv1D_Tiny.py \
  --platform all \
  --log-dir ./logs \
  --out-dir ./workbooks
```

---

## 10. How the log is selected automatically

When `--log` is not provided, the script:

1. recursively searches for `device-monitor-*` files with `.log`, `.txt`, or `.zip` extensions;
2. orders candidates by the date and time in the filename and by modification time;
3. checks the platform, mode, and required blocks;
4. selects the most recent compatible log;
5. skips incompatible candidates and reports the reasons when no valid file is found.

---

## 11. Common errors

### `ModuleNotFoundError: No module named 'utils'`

The script was started outside the project root.

PowerShell solution:

```powershell
cd C:\path\to\project
$env:RUNNER_PROJECT_ROOT = (Get-Location).Path
python .\compare_model_io_to_log_Conv1D_Tiny.py --platform esp32
```

### Field log rejected

Expected message when the file is not Replay:

```text
FIELD logs do not provide the required 1:1 replay blocks
```

This is intentional. Both scripts in this package are Replay auditors.

### Incompatible platform

Example: `--platform esp32` was requested, but the log reports STM32F411RE.

Correct the argument or select another file.

### Missing log block

Enable the required firmware debug blocks and generate a new complete Replay log.

### Workbook open in Excel

Close the file before running the script again. When it remains open, the script generates another filename with a timestamp.

### No compatible log found

Confirm:

- the filename starts with `device-monitor-`;
- the extension is `.log`, `.txt`, or `.zip`;
- the `[PLATFORM]` line is present;
- the mode is Replay;
- the required blocks for the corresponding script are present.

---

## 12. Recommended execution flow

1. Compile the firmware for the desired platform in Replay mode.
2. Generate one complete log per platform.
3. Place the logs in the configured directory.
4. Close older workbooks that are open in Excel.
5. Run the model I/O auditor first.
6. Run the predictions and metrics auditor afterward.
7. Review the summary sheets before interpreting detailed differences.
8. Preserve the ESP32 and STM32F411RE workbooks separately as experimental evidence.

Complete recommended command:

```bash
python compare_model_io_to_log_Conv1D_Tiny.py --platform all --mode replay
python compare_predictions_metrics_to_log_Conv1D_Tiny.py --platform all --mode replay
```

---

## 13. Scope and limitation

The generated workbooks allow each platform to be compared against the same Python reference and verify stage-by-stage equivalence.

They do not convert platform-specific ESP32 and STM32 memory metrics into directly equivalent measurements. ESP32 free heap and STM32 heap-to-stack gap must remain separately identified in the scientific analysis.
