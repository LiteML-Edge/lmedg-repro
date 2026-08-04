# Evidence Traceability

> **Document role:** Use this file after [`REVIEWER_START_HERE.md`](../REVIEWER_START_HERE.md) to locate the exact artifacts supporting each model, target, execution mode, table, figure, and validation result. It is an evidence map, not an execution guide. Commands for regenerating verification outputs are in [`REVIEWER_VERIFICATION_GUIDE.md`](REVIEWER_VERIFICATION_GUIDE.md).

## How to use this map

1. Select the manuscript claim or experiment component under review.
2. Follow the corresponding contract, host-reference, firmware-log, workbook, table, figure, or environment path below.
3. Use `docs/EVIDENCE_MANIFEST.csv` for one-row-per-log mapping and `docs/ARTIFACT_INVENTORY.csv` for packaged-artifact presence and size.
4. Return to the reviewer guide before executing regeneration or hardware commands.


## Contract-to-evidence traceability

| Model | Formal contract | Host references | Firmware and hardware evidence |
|---|---|---|---|
| MLP | `contracts/mlp_contract.json` | `metrics/environment_mlp/quantized_model/run/quantization_metrics_results/` | `firmwares/environment_mlp/PlatfIO_mlp/` |
| Conv1D Tiny | `contracts/conv1d_tiny_contract.json` | `metrics/environment_Conv1D_Tiny/quantized_model/run/quantization_metrics_results/` | `firmwares/environment_Conv1D_Tiny/PlatfIO_Conv1D_Tiny/` |
| LSTM | `contracts/lstm_contract.json` | `metrics/environment_lstm/quantized_model/run/quantization_metrics_results/` | `firmwares/environment_lstm/PlatfIO_lstm/` |

The common decision thresholds are frozen in `contracts/tolerance_policy.json`. The `artifacts` object in each model contract provides finer-grained paths to the exact model and header files.

## Experimental evidence matrix

| Model | ESP32 REPLAY | ESP32 FIELD | STM32 REPLAY | STM32 FIELD | Host references |
|---|---|---|---|---|---|
| MLP | `firmwares/environment_mlp/PlatfIO_mlp/logs/device-monitor-260725-090646.log` | `firmwares/environment_mlp/PlatfIO_mlp/logs/device-monitor-260725-123210.log` | `firmwares/environment_mlp/PlatfIO_mlp/logs/device-monitor-260727-170448.log` | `firmwares/environment_mlp/PlatfIO_mlp/logs/device-monitor-260727-205824.log` | `metrics/environment_mlp/quantized_model/run/` |
| Conv1D Tiny | `firmwares/environment_Conv1D_Tiny/PlatfIO_Conv1D_Tiny/logs/device-monitor-260725-111032.log` | `firmwares/environment_Conv1D_Tiny/PlatfIO_Conv1D_Tiny/logs/device-monitor-260725-152233.log` | `firmwares/environment_Conv1D_Tiny/PlatfIO_Conv1D_Tiny/logs/device-monitor-260727-185942.log` | `firmwares/environment_Conv1D_Tiny/PlatfIO_Conv1D_Tiny/logs/device-monitor-260727-233942.log` | `metrics/environment_Conv1D_Tiny/quantized_model/run/` |
| LSTM | `firmwares/environment_lstm/PlatfIO_lstm/logs/device-monitor-260725-100758.log` | `firmwares/environment_lstm/PlatfIO_lstm/logs/device-monitor-260725-142952.log` | `firmwares/environment_lstm/PlatfIO_lstm/logs/device-monitor-260727-180657.log` | `firmwares/environment_lstm/PlatfIO_lstm/logs/device-monitor-260728-070726.log` | `metrics/environment_lstm/quantized_model/run/` |

## Stage-wise evidence

For every model, the quantized metrics run contains preprocessing references, model-input references, decoded and raw-output references, final prediction tables, and Rolling-24 metrics. The two comparison scripts align those references with the selected REPLAY log and create human-readable workbooks.

## Tolerance-sensitivity evidence

`validation/tolerance_sensitivity/tolerance_sensitivity.xlsx` and `.csv` trace each adopted stage threshold to the observed maximum difference in the six frozen model-I/O workbooks. The evidence evaluates numeric thresholds at 1x, 10x, and 100x while retaining exact comparison for raw tensor bytes and metadata. MLP and Conv1D Tiny remain PASS, and LSTM remains FAIL, on both targets under all three factors. During this sensitivity analysis, no thresholds were fitted, replaced, or recalibrated; the analysis only applied 1x, 10x, and 100x factors to the previously adopted engineering acceptance bounds.

## Rolling-24 `n=24` scope evidence

`docs/N24_SCOPE_NOTE.md` provides the manuscript-aligned interpretation. The three machine-readable contracts record `window_steps = 24`, `chronological_replay_rows = 47`, and `reported_valid_events = 24`. The two LSTM REPLAY workbook folders under `utils/workbook_lstm/` preserve the exact `3/24 (12.5%)` classification on WEMOS LOLIN32 and NUCLEO-F411RE. Consecutive events share input history and are not used as independent observations for population inference; each event remains a distinct prediction and contract comparison.

## Pipeline dependency evidence

The six root YAML files declare `pio_build` as dependent on both `pio_pull_scalers` and `pio_pull_rolling24`. `docs/PIPELINE_DEPENDENCY_VALIDATION.md` provides the verification command and expected topological-plan pattern.

## Tables

Run:

```powershell
python utils/table_generator/generate_liteml_edge_paper_tables.py --repo-root . --output-dir reviewer_outputs/tables
```

The generator reads the workbook folders and firmware logs and exports paper-ready LaTeX, CSV, and XLSX material. `liteml_edge_paper_tables.xlsx` is the primary human-readable output because it preserves numeric cell types independently of regional spreadsheet settings. CSV files are auxiliary machine-readable exports using commas as field delimiters and periods as decimal separators. Detailed floating-point source records are limited to six decimal places; manuscript-table exports retain their table-specific rounding.

## Figures

The packaged article workbook is `utils/image_generator/graphic_image/data_conv1d_tiny.xlsx`. Generate both published-variable panels with:

```powershell
python utils/image_generator/graphic_image/plot_rolling24_scatter_offline_ondevice_article_conv1d_tiny.py --excel utils/image_generator/graphic_image/data_conv1d_tiny.xlsx --variable T --outdir reviewer_outputs/figures --basename rolling24_T --dpi 600
python utils/image_generator/graphic_image/plot_rolling24_scatter_offline_ondevice_article_conv1d_tiny.py --excel utils/image_generator/graphic_image/data_conv1d_tiny.xlsx --variable H --outdir reviewer_outputs/figures --basename rolling24_H --dpi 600
```

## Related-work evidence

`related_work/table_I/` contains the final comparison matrix, criterion-level evidence log, DOI/source index, and combined audit workbook used to support Table I. See `docs/RELATED_WORK_EVIDENCE.md` for the interpretation rule and reviewer verification path.

## Hardware schematics

`hardware_schematics/wemos_lolin32/` and `hardware_schematics/nucleo_f411re/` contain PDF schematics, editable Visio sources, and GPIO workbooks for the two evaluated boards. See `docs/HARDWARE_SCHEMATICS.md` for cross-checking against the firmware pin definitions and build targets.

## Inventory files

- `docs/EVIDENCE_MANIFEST.csv`: one row per hardware log.
- `docs/ARTIFACT_INVENTORY.csv`: datasets, models, headers, metrics, figures, and workbooks.
- `docs/PLATFORMIO_LIBRARY_MANIFEST.csv`: packaged firmware libraries and versions.
## Environment evidence

`environment_reports/README.md` is the reviewer-facing index for two evidence groups. `environment_reports/final_builds/` contains the six successful final REPLAY compilation records for three models and two hardware targets. `environment_reports/system reports/` records the Windows 11 host, Python virtual environment, PlatformIO Core, exact `espressif32` and `ststm32` platform versions, frameworks, tools, toolchains, managed libraries, and project-local TensorFlow/sensor libraries.

The six build records correspond to the REPLAY firmware configurations used for stage-wise conformance verification. FIELD configurations use the same PlatformIO projects and build procedure, with the acquisition mode selected by a compile-time macro.


## Return to the reviewer route

Return to [`REVIEWER_START_HERE.md`](../REVIEWER_START_HERE.md) after locating the required artifacts. Use [`REVIEWER_VERIFICATION_GUIDE.md`](REVIEWER_VERIFICATION_GUIDE.md) for regeneration commands and result interpretation.
