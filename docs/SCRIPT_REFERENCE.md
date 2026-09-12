# Script Reference

All invocations below are executed from the repository root. The training scripts do not expose command-line parameters; run them directly. CLI utilities expose `--help`, and the captured help text is stored in `docs/cli_help/`.

## Dataset, training, conversion, and export scripts

| Model | Stage | Script | Invocation | Primary output |
|---|---|---|---|---|
| MLP | dataset | `datasets/environment_mlp/environment_dataset_mlp.py` | `python datasets/environment_mlp/environment_dataset_mlp.py` | datasets/environment_mlp/environment_dataset_mlp.csv |
| MLP | baseline | `trainings/environment_mlp/base_model/environment_base_model_mlp.py` | `python trainings/environment_mlp/base_model/environment_base_model_mlp.py` | versioned `.keras`, scaler, metrics, and figures |
| MLP | pruning | `trainings/environment_mlp/pruned_model/environment_pruned_model_mlp.py` | `python trainings/environment_mlp/pruned_model/environment_pruned_model_mlp.py` | versioned pruned `.keras` and metrics |
| MLP | quantization | `trainings/environment_mlp/quantized_model/environment_quantized_model_mlp.py` | `python trainings/environment_mlp/quantized_model/environment_quantized_model_mlp.py` | TFLite model, replay/reference evidence, metrics, and figures |
| MLP | model header | `trainings/environment_mlp/header_generator/header_generator_mlp.py` | `python trainings/environment_mlp/header_generator/header_generator_mlp.py` | `environment_model_data_<model>.h` |
| MLP | scaler header | `trainings/environment_mlp/scalers_exporter/scale_vector_exporter_mlp.py` | `python trainings/environment_mlp/scalers_exporter/scale_vector_exporter_mlp.py` | `scalers_exported_<model>.h` |
| Conv1D Tiny | dataset | `datasets/environment_Conv1D_Tiny/environment_dataset_Conv1D_Tiny.py` | `python datasets/environment_Conv1D_Tiny/environment_dataset_Conv1D_Tiny.py` | datasets/environment_Conv1D_Tiny/environment_dataset_Conv1D_Tiny.csv |
| Conv1D Tiny | baseline | `trainings/environment_Conv1D_Tiny/base_model/environment_base_model_Conv1D_Tiny.py` | `python trainings/environment_Conv1D_Tiny/base_model/environment_base_model_Conv1D_Tiny.py` | versioned `.keras`, scaler, metrics, and figures |
| Conv1D Tiny | pruning | `trainings/environment_Conv1D_Tiny/pruned_model/environment_pruned_model_Conv1D_Tiny.py` | `python trainings/environment_Conv1D_Tiny/pruned_model/environment_pruned_model_Conv1D_Tiny.py` | versioned pruned `.keras` and metrics |
| Conv1D Tiny | quantization | `trainings/environment_Conv1D_Tiny/quantized_model/environment_quantized_model_Conv1D_Tiny.py` | `python trainings/environment_Conv1D_Tiny/quantized_model/environment_quantized_model_Conv1D_Tiny.py` | TFLite model, replay/reference evidence, metrics, and figures |
| Conv1D Tiny | model header | `trainings/environment_Conv1D_Tiny/header_generator/header_generator_Conv1D_Tiny.py` | `python trainings/environment_Conv1D_Tiny/header_generator/header_generator_Conv1D_Tiny.py` | `environment_model_data_<model>.h` |
| Conv1D Tiny | scaler header | `trainings/environment_Conv1D_Tiny/scalers_exporter/scale_vector_exporter_Conv1D_Tiny.py` | `python trainings/environment_Conv1D_Tiny/scalers_exporter/scale_vector_exporter_Conv1D_Tiny.py` | `scalers_exported_<model>.h` |
| LSTM | dataset | `datasets/environment_lstm/environment_dataset_lstm.py` | `python datasets/environment_lstm/environment_dataset_lstm.py` | datasets/environment_lstm/environment_dataset_lstm.csv |
| LSTM | baseline | `trainings/environment_lstm/base_model/environment_base_model_lstm.py` | `python trainings/environment_lstm/base_model/environment_base_model_lstm.py` | versioned `.keras`, scaler, metrics, and figures |
| LSTM | pruning | `trainings/environment_lstm/pruned_model/environment_pruned_model_lstm.py` | `python trainings/environment_lstm/pruned_model/environment_pruned_model_lstm.py` | versioned pruned `.keras` and metrics |
| LSTM | quantization | `trainings/environment_lstm/quantized_model/environment_quantized_model_lstm.py` | `python trainings/environment_lstm/quantized_model/environment_quantized_model_lstm.py` | TFLite model, replay/reference evidence, metrics, and figures |
| LSTM | model header | `trainings/environment_lstm/header_generator/header_generator_lstm.py` | `python trainings/environment_lstm/header_generator/header_generator_lstm.py` | `environment_model_data_<model>.h` |
| LSTM | scaler header | `trainings/environment_lstm/scalers_exporter/scale_vector_exporter_lstm.py` | `python trainings/environment_lstm/scalers_exporter/scale_vector_exporter_lstm.py` | `scalers_exported_<model>.h` |

## Pipeline and PlatformIO utilities

| Script | Purpose | Main interface |
|---|---|---|
| `runner.py` | Loads a YAML DAG and executes selected stages. | `--pipeline`, `--only`, `--from`, `--to`, `--after`, `--before`, `--list`, `--dry-run`, `--keep-going` |
| `utils/global_utils/pio_pull_headers.py` | Copies generated model/scaler/replay headers into firmware `include/` and optionally stores snapshots. | `--project-dir`, `--runs-base`, `--selector`, `--header-source`, `--header-dst`, `--versioning` |
| `utils/global_utils/pio_build.py` | Calls `platformio run` for a firmware project. | `--project-dir`, `--selector`, `--extra` |
| `utils/global_utils/pio_upload.py` | Calls PlatformIO upload. | `--project-dir`, `--selector`, `--port`, `--extra` |
| `utils/global_utils/pio_monitor.py` | Opens serial monitor, waits for the port, and records timestamped logs. | `-d/--project-dir`, `-e/--env`, `--port`, `--baud`, `--wait`, `--extra`, `--no-log` |
| `utils/global_utils/report_env.py` | Records installed Python packages in Markdown. | `--output` |
| `utils/global_utils/make_dirs_tree.py` | Creates a directory tree from a text plan. | `--root`, `--plan`, `--dry-run`, `--verbose` |

## Validation tools

| Model | Immediate I/O comparison | Final prediction/metric comparison |
|---|---|---|
| MLP | `utils/workbook_mlp/compare_model_io_to_log_mlp.py` | `utils/workbook_mlp/compare_predictions_metrics_to_log_mlp.py` |
| Conv1D Tiny | `utils/workbook_Conv1D_Tiny/compare_model_io_to_log_Conv1D_Tiny.py` | `utils/workbook_Conv1D_Tiny/compare_predictions_metrics_to_log_Conv1D_Tiny.py` |
| LSTM | `utils/workbook_lstm/compare_model_io_to_log_lstm.py` | `utils/workbook_lstm/compare_predictions_metrics_to_log_lstm.py` |

Common comparison options: `--log`, `--log-dir`, `--out`, `--out-dir`, `--platform {auto,all,esp32,stm32f411re}`, and `--mode {auto,replay}`. The final comparison tools also expose spreadsheet-row and rounding controls.

## Manuscript-output tools

| Script | Purpose | Invocation |
|---|---|---|
| `utils/table_generator/generate_liteml_edge_paper_tables.py` | Regenerates manuscript tables from workbook and log directories. | `python ... --repo-root . --output-dir reviewer_outputs/tables` |
| `utils/image_generator/graphic_image/plot_rolling24_scatter_offline_ondevice_article_conv1d_tiny.py` | Generates Rolling-24 article scatter panels. | Run with `--excel`, optional sheet arguments, `--variable {T,H}`, `--outdir`, and `--dpi`. |

## Internal modules

`utils/global_utils/paths_*.py`, `versioning.py`, and `global_seed.py` are imported by the executable scripts. They define repository paths, run resolution/versioning, and reproducibility seeds; they are not standalone commands.

## Complete script inventory

The table below covers all project-owned Python files. Third-party library sources under firmware `lib/`, `.pio/libdeps/`, and frozen `run/` folders are intentionally excluded.

| Path | Category | `--help` | Role |
|---|---|---:|---|
| `datasets/environment_Conv1D_Tiny/environment_dataset_Conv1D_Tiny.py` | dataset | no | Generates the processed environmental dataset for Conv1D Tiny |
| `datasets/environment_lstm/environment_dataset_lstm.py` | dataset | no | Generates the processed environmental dataset for LSTM |
| `datasets/environment_mlp/environment_dataset_mlp.py` | dataset | no | Generates the processed environmental dataset for MLP |
| `runner.py` | pipeline | yes | YAML pipeline orchestrator |
| `trainings/environment_Conv1D_Tiny/base_model/environment_base_model_Conv1D_Tiny.py` | training | no | Trains and evaluates the baseline model and writes a versioned run |
| `trainings/environment_Conv1D_Tiny/header_generator/header_generator_Conv1D_Tiny.py` | export | no | Converts the selected TFLite model into a C/C++ model-data header |
| `trainings/environment_Conv1D_Tiny/pruned_model/environment_pruned_model_Conv1D_Tiny.py` | training | no | Loads the latest baseline run, prunes/fine-tunes, evaluates, and writes a versioned run |
| `trainings/environment_Conv1D_Tiny/quantized_model/environment_quantized_model_Conv1D_Tiny.py` | training | no | Quantizes/converts the model and exports TFLite, Rolling-24, Replay, tensor, and metric evidence |
| `trainings/environment_Conv1D_Tiny/scalers_exporter/scale_vector_exporter_Conv1D_Tiny.py` | export | no | Exports fitted scaler constants and feature order to a firmware header |
| `trainings/environment_lstm/base_model/environment_base_model_lstm.py` | training | no | Trains and evaluates the baseline model and writes a versioned run |
| `trainings/environment_lstm/header_generator/header_generator_lstm.py` | export | no | Converts the selected TFLite model into a C/C++ model-data header |
| `trainings/environment_lstm/pruned_model/environment_pruned_model_lstm.py` | training | no | Loads the latest baseline run, prunes/fine-tunes, evaluates, and writes a versioned run |
| `trainings/environment_lstm/quantized_model/environment_quantized_model_lstm.py` | training | no | Quantizes/converts the model and exports TFLite, Rolling-24, Replay, tensor, and metric evidence |
| `trainings/environment_lstm/scalers_exporter/scale_vector_exporter_lstm.py` | export | no | Exports fitted scaler constants and feature order to a firmware header |
| `trainings/environment_mlp/base_model/environment_base_model_mlp.py` | training | no | Trains and evaluates the baseline model and writes a versioned run |
| `trainings/environment_mlp/header_generator/header_generator_mlp.py` | export | no | Converts the selected TFLite model into a C/C++ model-data header |
| `trainings/environment_mlp/pruned_model/environment_pruned_model_mlp.py` | training | no | Loads the latest baseline run, prunes/fine-tunes, evaluates, and writes a versioned run |
| `trainings/environment_mlp/quantized_model/environment_quantized_model_mlp.py` | training | no | Quantizes/converts the model and exports TFLite, Rolling-24, Replay, tensor, and metric evidence |
| `trainings/environment_mlp/scalers_exporter/scale_vector_exporter_mlp.py` | export | no | Exports fitted scaler constants and feature order to a firmware header |
| `utils/global_utils/global_seed.py` | internal | no | Defines shared deterministic seed settings imported by training scripts |
| `utils/global_utils/make_dirs_tree.py` | utility | yes | Creates a directory tree from a text plan |
| `utils/global_utils/paths_Conv1D_Tiny.py` | internal | no | Defines model-specific repository and versioned-run paths |
| `utils/global_utils/paths_lstm.py` | internal | no | Defines model-specific repository and versioned-run paths |
| `utils/global_utils/paths_mlp.py` | internal | no | Defines model-specific repository and versioned-run paths |
| `utils/global_utils/pio_build.py` | firmware-utility | yes | Builds a selected PlatformIO environment |
| `utils/global_utils/pio_monitor.py` | firmware-utility | yes | Opens the serial monitor and captures timestamped logs |
| `utils/global_utils/pio_pull_headers.py` | firmware-utility | yes | Copies generated model/scaler/Replay headers into a PlatformIO project |
| `utils/global_utils/pio_upload.py` | firmware-utility | yes | Uploads a selected PlatformIO environment to the board |
| `utils/global_utils/report_env.py` | utility | yes | Writes the installed Python package environment to Markdown |
| `utils/global_utils/versioning.py` | internal | no | Creates and resolves versioned run directories and latest pointers |
| `utils/image_generator/graphic_image/plot_rolling24_scatter_offline_ondevice_article_conv1d_tiny.py` | publication-output | yes | Generates the Conv1D Tiny Rolling-24 scatter figure |
| `utils/table_generator/generate_liteml_edge_paper_tables.py` | publication-output | yes | Generates manuscript tables in LaTeX, CSV, and XLSX formats |
| `utils/workbook_Conv1D_Tiny/compare_model_io_to_log_Conv1D_Tiny.py` | validation | yes | Generates stage-wise immediate model-I/O comparison workbooks from Python references and REPLAY logs |
| `utils/workbook_Conv1D_Tiny/compare_predictions_metrics_to_log_Conv1D_Tiny.py` | validation | yes | Generates final prediction and Rolling-24 metric comparison workbooks |
| `utils/workbook_lstm/compare_model_io_to_log_lstm.py` | validation | yes | Generates stage-wise immediate model-I/O comparison workbooks from Python references and REPLAY logs |
| `utils/workbook_lstm/compare_predictions_metrics_to_log_lstm.py` | validation | yes | Generates final prediction and Rolling-24 metric comparison workbooks |
| `utils/workbook_mlp/compare_model_io_to_log_mlp.py` | validation | yes | Generates stage-wise immediate model-I/O comparison workbooks from Python references and REPLAY logs |
| `utils/workbook_mlp/compare_predictions_metrics_to_log_mlp.py` | validation | yes | Generates final prediction and Rolling-24 metric comparison workbooks |

A machine-readable copy is available in `docs/SCRIPT_CATALOG.csv`. Exact help captures for the 15 command-line tools are stored in `docs/cli_help/`.
