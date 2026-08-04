# Reviewer Verification Guide

> **Document role:** This guide verifies the retained submission evidence and regenerates reviewer-facing outputs without requiring retraining. Start with [`REVIEWER_START_HERE.md`](../REVIEWER_START_HERE.md) for scope, evidence boundaries, and the recommended inspection order. Use [`docs/END_TO_END_REPRODUCTION.md`](END_TO_END_REPRODUCTION.md) only when a fresh dataset-to-device reconstruction is required.

## Verification boundary

Sections 1 through 3 can be completed without connected hardware and without retraining. Section 4 covers firmware build and device execution. Section 5 redirects to the full reconstruction guide. Newly generated material should be written to `reviewer_outputs/` so that frozen submission evidence remains distinguishable from fresh reviewer outputs.


Use [`REVIEWER_START_HERE.md`](../REVIEWER_START_HERE.md) to locate the evidence associated with each review concern.

## Required mode before a complete hardware pipeline

For a new hardware run intended to reproduce the formal Python--firmware 1:1 comparison, set the corresponding firmware project's `include/config.h` to:

```cpp
#define LITEML_MODE LITEML_MODE_REPLAY
```

The root YAML pipelines do not switch this macro automatically. FIELD mode is reserved for live-sensor execution and deployment-cost evidence.

## 1. Inspect the packaged evidence

The main evidence sets are:

- `contracts/`: model contracts and common tolerance policy;
- `validation/tolerance_sensitivity/`: tolerance analysis at `1x`, `10x`, and `100x`;
- `docs/N24_SCOPE_NOTE.md`: manuscript-aligned `n=24` interpretation;
- `metrics/*/quantized_model/run/`: host-side reference outputs;
- `firmwares/*/*/logs/`: REPLAY and FIELD hardware logs;
- `utils/workbook_*/`: retained comparison workbooks and comparison scripts;
- `hardware_schematics/`: board wiring and GPIO workbooks;
- `environment_reports/final_builds/`: six final REPLAY compilation records;
- `environment_reports/system reports/`: host, Python, PlatformIO, toolchain, package, and local-library records;
- `related_work/table_I/`: evidence supporting Table I.

The exact model-to-log mapping is recorded in `docs/EVIDENCE_TRACEABILITY.md` and `docs/EVIDENCE_MANIFEST.csv`.

### Inspect the retained build and system reports

Open `environment_reports/README.md` as the reviewer-facing index. The exact build-record mapping and inspection procedure are documented in `environment_reports/final_builds/README.md`; the role of every host and software report is documented in `environment_reports/system reports/README.md`.

The six build records correspond to the REPLAY firmware configurations used for stage-wise conformance verification. FIELD configurations use the same PlatformIO projects and build procedure, with the acquisition mode selected by a compile-time macro.

The build records are compilation evidence. The twelve runtime serial logs remain under the three firmware projects and are mapped in `docs/EVIDENCE_MANIFEST.csv`.

## 2. Regenerate the Python--firmware validation workbooks

Create the immediate model-I/O workbooks:

```powershell
python utils/workbook_mlp/compare_model_io_to_log_mlp.py --log-dir firmwares/environment_mlp/PlatfIO_mlp/logs --out-dir reviewer_outputs/mlp/model_io --platform all --mode replay
python utils/workbook_Conv1D_Tiny/compare_model_io_to_log_Conv1D_Tiny.py --log-dir firmwares/environment_Conv1D_Tiny/PlatfIO_Conv1D_Tiny/logs --out-dir reviewer_outputs/conv1d_tiny/model_io --platform all --mode replay
python utils/workbook_lstm/compare_model_io_to_log_lstm.py --log-dir firmwares/environment_lstm/PlatfIO_lstm/logs --out-dir reviewer_outputs/lstm/model_io --platform all --mode replay
```

Create the final Rolling-24 prediction and metric workbooks:

```powershell
python utils/workbook_mlp/compare_predictions_metrics_to_log_mlp.py --log-dir firmwares/environment_mlp/PlatfIO_mlp/logs --out-dir reviewer_outputs/mlp/predictions_metrics --platform all --mode replay
python utils/workbook_Conv1D_Tiny/compare_predictions_metrics_to_log_Conv1D_Tiny.py --log-dir firmwares/environment_Conv1D_Tiny/PlatfIO_Conv1D_Tiny/logs --out-dir reviewer_outputs/conv1d_tiny/predictions_metrics --platform all --mode replay
python utils/workbook_lstm/compare_predictions_metrics_to_log_lstm.py --log-dir firmwares/environment_lstm/PlatfIO_lstm/logs --out-dir reviewer_outputs/lstm/predictions_metrics --platform all --mode replay
```

These commands use the retained host references and REPLAY logs and produce one output per model and hardware target.

## 3. Regenerate manuscript tables and figures

Tables:

```powershell
python utils/table_generator/generate_liteml_edge_paper_tables.py --repo-root . --output-dir reviewer_outputs/tables
```

For human inspection, open `reviewer_outputs/tables/liteml_edge_paper_tables.xlsx`. It preserves numeric cell types independently of regional spreadsheet settings. The CSV files are auxiliary machine-readable exports that use commas as field delimiters and periods as decimal separators. Detailed floating-point source records are limited to six decimal places; manuscript-table exports retain their table-specific rounding. In locale-sensitive spreadsheet software, import CSV files with an English/US numeric locale instead of opening them directly.

Rolling-24 figures:

```powershell
python utils/image_generator/graphic_image/plot_rolling24_scatter_offline_ondevice_article_conv1d_tiny.py --excel utils/image_generator/graphic_image/data_conv1d_tiny.xlsx --variable T --outdir reviewer_outputs/figures --basename rolling24_T --dpi 600
python utils/image_generator/graphic_image/plot_rolling24_scatter_offline_ondevice_article_conv1d_tiny.py --excel utils/image_generator/graphic_image/data_conv1d_tiny.xlsx --variable H --outdir reviewer_outputs/figures --basename rolling24_H --dpi 600
```
## 4. Build and execute firmware from the retained headers

Follow `docs/FIRMWARE_BUILD_AND_HARDWARE.md`. For formal 1:1 reproduction, confirm REPLAY in `include/config.h` before building the required PlatformIO environment. The six retained REPLAY compilation records can be inspected first under `environment_reports/final_builds/`. Use FIELD only when collecting live-sensor and deployment-cost evidence. Upload the firmware, capture the serial output, and run the comparison commands above.

## 5. Reconstruct the experiment from source data

Follow `docs/END_TO_END_REPRODUCTION.md`. It documents the model-specific dataset, baseline, pruning, quantization, header export, firmware synchronization, build, upload, logging, validation, table, and figure stages.

This guide covers the technical verification and reproduction material contained in this pack.


## Next document

- Return to [`REVIEWER_START_HERE.md`](../REVIEWER_START_HERE.md) to continue the recommended review sequence.
- Open [`EVIDENCE_TRACEABILITY.md`](EVIDENCE_TRACEABILITY.md) to locate exact supporting files.
- Open [`END_TO_END_REPRODUCTION.md`](END_TO_END_REPRODUCTION.md) only for a fresh reconstruction from source data.
