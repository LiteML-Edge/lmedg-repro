# Reviewer Start Here

**This is the canonical and self-contained entry point for reviewing the LiteML-Edge verification and reproduction pack.** It is intentionally written so that a reviewer can open it before `README.md` and still understand the package scope, evidence boundaries, recommended inspection sequence, and available reproduction routes.

## 1. What this package contains

The package supports verification and reconstruction of three models—MLP, Conv1D Tiny, and LSTM—on two MCU targets—WEMOS LOLIN32 and NUCLEO-F411RE—under REPLAY and FIELD execution modes.

It contains:

- machine-readable validation contracts and tolerance policies;
- retained host-side references and model artifacts;
- twelve runtime serial logs: three models x two targets x REPLAY/FIELD;
- retained Python--firmware comparison workbooks;
- `1x/10x/100x` tolerance-sensitivity evidence;
- six final REPLAY compilation records;
- host, Python, PlatformIO, toolchain, package, and local-library reports;
- complete dataset-to-device pipelines and direct stage commands;
- electrical schematics, GPIO mappings, and instrumentation documentation;
- table, figure, and related-work evidence.

## 2. Evidence boundaries that must be read first

These points prevent over-interpretation of the packaged material:

1. **REPLAY provides the formal stage-wise Python--firmware conformance evidence.** FIELD provides live-sensor and hardware-dependent deployment evidence.
2. **The six retained build records cover REPLAY only.** FIELD uses the same PlatformIO projects and build procedure, but `LITEML_MODE` is selected by a compile-time macro in each project's `include/config.h`.
3. **The build records are logs, not binaries.** They prove successful compilation of the retained REPLAY configurations; they do not imply that one REPLAY binary also executes FIELD mode.
4. **The reported Rolling-24 evidence is bounded to `n=24` valid post-warm-up events per model--target pair.** See [`docs/N24_SCOPE_NOTE.md`](docs/N24_SCOPE_NOTE.md) before drawing broader statistical conclusions.
5. **Checked-in `run/` directories and retained logs are frozen submission evidence.** Newly generated reviewer outputs should be written to `reviewer_outputs/` and compared against the retained artifacts rather than replacing them.

## 3. Choose the appropriate review route

| Reviewer objective | Start with | Hardware required? | Retraining required? |
|---|---|---:|---:|
| Understand the claims, limits, and artifact map | This document, then [`docs/EVIDENCE_TRACEABILITY.md`](docs/EVIDENCE_TRACEABILITY.md) | No | No |
| Inspect the submitted evidence | [`docs/REVIEWER_VERIFICATION_GUIDE.md`](docs/REVIEWER_VERIFICATION_GUIDE.md), Section 1 | No | No |
| Regenerate Python--firmware comparisons | [`docs/REVIEWER_VERIFICATION_GUIDE.md`](docs/REVIEWER_VERIFICATION_GUIDE.md), Section 2 | No | No |
| Regenerate manuscript tables and figures | [`docs/REVIEWER_VERIFICATION_GUIDE.md`](docs/REVIEWER_VERIFICATION_GUIDE.md), Section 3 | No | No |
| Inspect successful compilation records | [`environment_reports/final_builds/README.md`](environment_reports/final_builds/README.md) | No | No |
| Rebuild the retained firmware configurations | [`docs/FIRMWARE_BUILD_AND_HARDWARE.md`](docs/FIRMWARE_BUILD_AND_HARDWARE.md) | No for build | No with `--no-upstream` |
| Upload, monitor, or collect a new hardware log | [`docs/FIRMWARE_BUILD_AND_HARDWARE.md`](docs/FIRMWARE_BUILD_AND_HARDWARE.md) | Yes | No with retained headers |
| Reconstruct all artifacts from source data | [`docs/SETUP_AND_DEPENDENCIES.md`](docs/SETUP_AND_DEPENDENCIES.md), then [`docs/END_TO_END_REPRODUCTION.md`](docs/END_TO_END_REPRODUCTION.md) | Required only for device stages | Yes |

## 4. Recommended reviewer sequence

The following order separates claim verification from optional full reconstruction.

### Step 1 — Confirm scope and limitations

Open:

- [`docs/N24_SCOPE_NOTE.md`](docs/N24_SCOPE_NOTE.md)
- [`docs/PORTABILITY_AND_LIMITATIONS.md`](docs/PORTABILITY_AND_LIMITATIONS.md)

Check the Rolling-24 event count, dependence between consecutive windows, evaluated dataset/domain, fixed window policy, model/runtime scope, and limits on generalization.

### Step 2 — Inspect the formal validation contract

Open:

- [`contracts/README.md`](contracts/README.md)
- [`docs/CONTRACT_C_MAPPING.md`](docs/CONTRACT_C_MAPPING.md)
- `contracts/mlp_contract.json`
- `contracts/conv1d_tiny_contract.json`
- `contracts/lstm_contract.json`
- `contracts/tolerance_policy.json`

Check `C=(S,K,W,Q,P,M)`, frozen artifact paths, feature/target order, window policy, quantization/tensor contract, preprocessing/post-processing, metrics, and decision thresholds.

### Step 3 — Map manuscript claims to exact evidence

Open [`docs/EVIDENCE_TRACEABILITY.md`](docs/EVIDENCE_TRACEABILITY.md), then use:

- `docs/EVIDENCE_MANIFEST.csv` for the twelve runtime hardware logs;
- `docs/ARTIFACT_INVENTORY.csv` for packaged artifacts;
- `docs/PLATFORMIO_BUILD_MATRIX.csv` for the six model--target firmware environments;
- `docs/PLATFORMIO_LIBRARY_MANIFEST.csv` for packaged firmware dependencies.

This step identifies the exact files supporting each model, target, mode, table, figure, and verification result.

### Step 4 — Inspect retained evidence before executing anything

Follow Section 1 of [`docs/REVIEWER_VERIFICATION_GUIDE.md`](docs/REVIEWER_VERIFICATION_GUIDE.md). Inspect the retained host references, serial logs, comparison workbooks, tolerance-sensitivity analysis, schematics, build records, and system reports.

For the compilation and environment evidence, start at [`environment_reports/README.md`](environment_reports/README.md).

### Step 5 — Regenerate verification outputs without retraining

Follow Sections 2 and 3 of [`docs/REVIEWER_VERIFICATION_GUIDE.md`](docs/REVIEWER_VERIFICATION_GUIDE.md). The commands regenerate:

- immediate model-I/O comparison workbooks;
- final Rolling-24 prediction and metric workbooks;
- manuscript tables;
- Rolling-24 figures.

Write fresh outputs to `reviewer_outputs/` and compare them with the retained evidence.

### Step 6 — Inspect or reproduce firmware compilation

Open [`docs/FIRMWARE_BUILD_AND_HARDWARE.md`](docs/FIRMWARE_BUILD_AND_HARDWARE.md).

First inspect the six retained REPLAY build records under `environment_reports/final_builds/`. To rebuild the checked-in headers without regenerating upstream artifacts, set REPLAY in `include/config.h` and run `pio_build` with `--no-upstream` as documented.

### Step 7 — Reconstruct the experiment from source data

Only after the retained evidence has been understood, open:

1. [`docs/SETUP_AND_DEPENDENCIES.md`](docs/SETUP_AND_DEPENDENCIES.md)
2. [`docs/END_TO_END_REPRODUCTION.md`](docs/END_TO_END_REPRODUCTION.md)

The end-to-end guide documents dataset generation, baseline training, pruning, quantization, header export, firmware synchronization, build, upload, monitor, validation, and output regeneration for all six model--target pipelines. Its [Training configuration summary](docs/END_TO_END_REPRODUCTION.md#training-configuration-summary) consolidates the exact retained architectures, optimizers, losses, epoch limits, batch sizes, callbacks, pruning schedules, representative-dataset sizes, and TFLite input/output contracts before the model-specific commands.

### Step 8 — Review specialized supporting evidence

Use the following only when the corresponding manuscript element is under review:

- instrumentation and deployment cost: [`docs/FIRMWARE_BUILD_AND_HARDWARE.md`](docs/FIRMWARE_BUILD_AND_HARDWARE.md);
- electrical connections: [`docs/HARDWARE_SCHEMATICS.md`](docs/HARDWARE_SCHEMATICS.md);
- tolerance robustness: `validation/tolerance_sensitivity/README.md`;
- related-work comparison: [`docs/RELATED_WORK_EVIDENCE.md`](docs/RELATED_WORK_EVIDENCE.md);
- pipeline dependency order: [`docs/PIPELINE_DEPENDENCY_VALIDATION.md`](docs/PIPELINE_DEPENDENCY_VALIDATION.md);

## 5. Review-concern index

| Review concern | Evidence location | What to check |
|---|---|---|
| Concrete validation contract | `contracts/` and `docs/CONTRACT_C_MAPPING.md` | Ordered schema, frozen constants, window policy, tensor contract, preprocessing/post-processing, metrics, and artifact paths for all models. |
| Tolerance selection | `validation/tolerance_sensitivity/tolerance_sensitivity.xlsx` | MLP and Conv1D Tiny remain conformant and LSTM remains nonconformant at `1x`, `10x`, and `100x`; exact byte and metadata checks are unchanged. |
| `n=24`, `3/24`, and scope | `docs/N24_SCOPE_NOTE.md` | Manuscript-aligned interpretation of the bounded Rolling-24 trace and limits on generalization. |
| LSTM stage-wise divergence | `utils/workbook_lstm/esp32/replay/`, `utils/workbook_lstm/stm32f411re/replay/`, and the corresponding logs | First divergent stage and the same `3/24 (12.5%)` classification on both targets. |
| Pack contents and reproduction path | `docs/REVIEWER_VERIFICATION_GUIDE.md`, `docs/END_TO_END_REPRODUCTION.md`, and `docs/EVIDENCE_TRACEABILITY.md` | Packaged-evidence verification, complete dataset-to-device reproduction flow, and claim-to-file traceability. |
| Complete automated flow | Six root `environment_*_pipeline.yaml` files and `docs/PIPELINE_DEPENDENCY_VALIDATION.md` | Dataset, training, pruning, quantization, header synchronization, build, upload, and monitor stages. |
| Final firmware compilation evidence | `environment_reports/final_builds/README.md` and `environment_reports/final_builds/` | Six successful REPLAY build records: three models x two hardware targets, including PlatformIO RAM/Flash summaries and final success status. |
| Reproducible software environment | `environment_reports/README.md` and `environment_reports/system reports/` | Windows 11 host, Python and PlatformIO versions, platforms, toolchains, managed libraries, and local libraries. |
| Second MCU architecture | `firmwares/`, `hardware_schematics/`, and `docs/HARDWARE_SCHEMATICS.md` | NUCLEO-F411RE projects, REPLAY/FIELD logs, wiring, and GPIO assignments for all three models. |
| Hardware execution evidence | `docs/EVIDENCE_MANIFEST.csv` and `firmwares/*/*/logs/` | Twelve serial logs: three models x two targets x REPLAY/FIELD. |
| Instrumentation and deployment cost | `docs/FIRMWARE_BUILD_AND_HARDWARE.md` and firmware benchmark/sensor sources | Latency, energy, memory, INA219, sensor acquisition, and serial logging procedures. |
| Related-work comparison | `related_work/table_I/` and `docs/RELATED_WORK_EVIDENCE.md` | Source index, criterion-level evidence, and final comparison matrix. |

## 6. Document roles

| Document | Use it for |
|---|---|
| [`README.md`](README.md) | General repository overview and basic setup. |
| `REVIEWER_START_HERE.md` | Canonical reviewer sequence and review-concern routing. |
| [`docs/REVIEWER_VERIFICATION_GUIDE.md`](docs/REVIEWER_VERIFICATION_GUIDE.md) | Inspect retained evidence and regenerate comparisons, tables, and figures. |
| [`docs/EVIDENCE_TRACEABILITY.md`](docs/EVIDENCE_TRACEABILITY.md) | Locate the exact evidence supporting each model, target, mode, table, and figure. |
| [`docs/END_TO_END_REPRODUCTION.md`](docs/END_TO_END_REPRODUCTION.md) | Fresh reconstruction from source data. |
| [`docs/FIRMWARE_BUILD_AND_HARDWARE.md`](docs/FIRMWARE_BUILD_AND_HARDWARE.md) | Build, upload, monitor, mode selection, hardware, and instrumentation. |
| [`environment_reports/README.md`](environment_reports/README.md) | Access compilation records and captured system reports. |

A reviewer who only needs to verify the submitted evidence can complete Steps 1 through 6 without retraining. Step 7 is the full independent reconstruction path.
