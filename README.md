# LiteML-Edge - Verification and Reproduction Pack for This Revision

## Manuscript Information

**Title:** LiteML-Edge: A Contract-Centered Framework and Methodology for Replay-Driven 1:1 Python--Firmware Validation in TinyML  
**Submission ID:** 10820  
**Authors:** Charles Pereira dos Santos; Israel Gondres Torné; Carlos Maurício Seródio Figueiredo; Fábio de Sousa Cardoso  
**Journal:** IEEE Latin America Transactions  
**Hardware targets:** WEMOS LOLIN32 (ESP32) and NUCLEO-F411RE (STM32F411RE)  
**Evaluated models:** MLP, Conv1D Tiny, and LSTM

This repository contains the retained evidence and the complete reconstruction workflow described in the manuscript.

## Which file should be opened first?

The two root documents have different roles:

| Document | Role |
|---|---|
| [`REVIEWER_START_HERE.md`](REVIEWER_START_HERE.md) | **Canonical and self-contained reviewer guide.** Open this first for the recommended inspection order, claim-to-evidence navigation, verification commands, build records, environment reports, and reproduction routes. It does not assume that this README has already been read. |
| `README.md` | General repository overview, scope, structure, basic setup, and package-wide conventions. |

Reviewers assessing the manuscript should begin with **[`REVIEWER_START_HERE.md`](REVIEWER_START_HERE.md)**. This README remains the general repository landing page.

## What the pack supports

The pack supports three distinct activities:

1. **Inspect the submitted evidence without retraining or hardware.** Use the contracts, retained host references, REPLAY/FIELD serial logs, comparison workbooks, tolerance analysis, build records, and traceability files.
2. **Regenerate verification outputs from retained artifacts.** Recreate Python--firmware comparison workbooks, manuscript tables, and figures in a separate `reviewer_outputs/` directory.
3. **Reconstruct the experiment from source data.** Run dataset generation, baseline training, pruning, quantization, header export, firmware synchronization, build, upload, monitor, and post-run validation.

The recommended order for these activities is defined in [`REVIEWER_START_HERE.md`](REVIEWER_START_HERE.md).

## Experimental scope

| Model | Host artifact | WEMOS LOLIN32 | NUCLEO-F411RE |
|---|---|---:|---:|
| MLP | Full-INT8 TFLite | REPLAY + FIELD | REPLAY + FIELD |
| Conv1D Tiny | INT8 input / float32 output TFLite | REPLAY + FIELD | REPLAY + FIELD |
| LSTM | Hybrid TFLite with Select TF Ops | REPLAY + FIELD | REPLAY + FIELD |

The packaged workflow covers raw environmental data, processed datasets, baseline training, pruning, quantization, TFLite conversion, C-header generation, firmware build and upload, serial-log capture, Python--firmware comparison, and manuscript table/figure generation.

## Evidence status

- Checked-in `run/` directories and retained hardware logs are **frozen submission evidence**.
- Commands in the reviewer guides write newly generated material to `reviewer_outputs/` whenever practical, preventing confusion with retained evidence.
- The six files under `environment_reports/final_builds/` are **REPLAY compilation records**, not firmware binaries and not FIELD build records.
- FIELD uses the same PlatformIO projects and build procedure, but the acquisition mode is selected by a compile-time macro in each firmware project's `include/config.h`.
- The formal stage-wise Python--firmware conformance evidence is based on REPLAY. FIELD provides live-sensor and hardware-dependent deployment evidence.

## Repository structure

- `contracts/`: machine-readable `C=(S,K,W,Q,P,M)` contracts and the common tolerance policy.
- `datasets/`: source data, dataset generators, and processed datasets.
- `trainings/`: baseline, pruned, and quantized training stages with retained artifacts.
- `metrics/`: host-side references, Rolling-24 outputs, metrics, workbooks, and figures.
- `firmwares/`: PlatformIO projects for the three models and both MCU targets, including REPLAY and FIELD runtime logs.
- `validation/tolerance_sensitivity/`: `1x/10x/100x` tolerance-sensitivity analysis.
- `hardware_schematics/`: schematics and GPIO workbooks for both boards.
- `related_work/table_I/`: evidence supporting the representative-work comparison.
- `environment_reports/`: index for the six final REPLAY build records and captured system reports.
- `utils/`: validation, table, figure, and PlatformIO helper utilities.
- `docs/`: verification, traceability, reproduction, firmware, setup, and supporting technical documentation.

## Core reviewer documents

| Purpose | Document |
|---|---|
| Canonical reviewer route | [`REVIEWER_START_HERE.md`](REVIEWER_START_HERE.md) |
| Verify retained evidence and regenerate comparisons | [`docs/REVIEWER_VERIFICATION_GUIDE.md`](docs/REVIEWER_VERIFICATION_GUIDE.md) |
| Map claims and experiments to exact files | [`docs/EVIDENCE_TRACEABILITY.md`](docs/EVIDENCE_TRACEABILITY.md) |
| Reconstruct from source data, including exact retained training settings | [`docs/END_TO_END_REPRODUCTION.md`](docs/END_TO_END_REPRODUCTION.md#training-configuration-summary) |
| Build, upload, monitor, and inspect hardware procedures | [`docs/FIRMWARE_BUILD_AND_HARDWARE.md`](docs/FIRMWARE_BUILD_AND_HARDWARE.md) |
| Install and check dependencies | [`docs/SETUP_AND_DEPENDENCIES.md`](docs/SETUP_AND_DEPENDENCIES.md) |
| Inspect build and system reports | [`environment_reports/README.md`](environment_reports/README.md) |

## Reference environment

- Microsoft Windows 11 Home Single Language, 64-bit, build 26100
- Python 3.10.0
- PlatformIO Core 6.1.19
- Espressif 32 platform 6.10.0
- ST STM32 platform 19.7.0

Open [`environment_reports/README.md`](environment_reports/README.md) to access the six final REPLAY build records and the captured host, Python, PlatformIO, toolchain, package, and local-library reports.

## Basic setup

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-reproduction.txt
```

Check the command interface and pipeline plans without running training or hardware:

```powershell
python runner.py --help
python runner.py -p environment_esp32_mlp_pipeline.yaml --list
python utils/workbook_mlp/compare_model_io_to_log_mlp.py --help
```

## Firmware modes

For a formal Python--firmware 1:1 reproduction, set the corresponding firmware project to REPLAY in `include/config.h` before building:

```cpp
#define LITEML_MODE LITEML_MODE_REPLAY
```

For live-sensor execution:

```cpp
#define LITEML_MODE LITEML_MODE_FIELD
```

The six root YAML pipelines do not change this macro automatically. The PlatformIO environment selects WEMOS LOLIN32 or NUCLEO-F411RE.

## Windows extraction

Extract the archive to a short path such as `C:\LiteML`. The archive starts at the project root to reduce Windows path length while preserving all project-relative paths.

## Package integrity

This navigation revision changes documentation only. It does not modify experimental code, retained models, datasets, metrics, workbooks, serial logs, build records, or numerical results.
