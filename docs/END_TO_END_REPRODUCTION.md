# End-to-End Reproduction

> **Document role:** This guide performs a fresh dataset-to-device reconstruction. It is not required for inspecting or regenerating the retained verification evidence. Reviewers should first read [`REVIEWER_START_HERE.md`](../REVIEWER_START_HERE.md), complete the retained-evidence path in [`REVIEWER_VERIFICATION_GUIDE.md`](REVIEWER_VERIFICATION_GUIDE.md), and use [`SETUP_AND_DEPENDENCIES.md`](SETUP_AND_DEPENDENCIES.md) before executing the commands below.

## Reproduction boundary

Host-only stages can be executed without hardware. Build stages require PlatformIO; upload, monitor, FIELD acquisition, and new device logs require the corresponding board and instrumentation. A full run can update local run pointers and generate new artifacts, so retain the checked-in evidence unchanged and store review outputs separately.


Run every command from the repository root. The safest independent reconstruction is model-by-model and stage-by-stage. The training scripts are direct-execution programs rather than parameterized CLIs. Their configuration is defined in the source and path modules included in the repository.

## Required firmware mode for 1:1 reproduction

Before executing a complete hardware pipeline for the formal Python--firmware comparison, open the corresponding firmware project's `include/config.h` and set:

```cpp
#define LITEML_MODE LITEML_MODE_REPLAY
```

The six root YAML pipelines synchronize artifacts, build, upload, and monitor the selected PlatformIO environment, but they do not change `LITEML_MODE` automatically. Use `LITEML_MODE_FIELD` only for live-sensor execution and hardware-dependent deployment-cost measurements. FIELD evidence complements REPLAY and does not replace REPLAY for 1:1 equivalence claims.

## Common source data

All three dataset generators read the Singapore temperature and relative-humidity files under `datasets/singapore_dataset/`, merge them by timestamp, compute indoor aggregates, and export a fixed-order environmental CSV.

## Fixed feature/target contract

Feature order: `T_out, H_out, T_in_lag1, H_in_lag1, T_out_lag1, H_out_lag1, T_in_lag2, H_in_lag2, sin_hour, cos_hour, weekday, month`.

Targets: residual `Delta T_in` and `Delta H_in` are learned; absolute `T_in` and `H_in` are reconstructed for physical-domain metrics. The temporal window is 24 steps.

## Training configuration summary

> **Source-of-truth rule:** The values below are consolidated from the retained canonical training scripts in this pack. This section is a reviewer-facing summary, not a second configuration layer. The scripts remain the executable source of truth. Epoch counts are maximum limits; `EarlyStopping` may terminate a stage earlier.

### Common data and preprocessing configuration

| Item | Retained configuration |
|---|---|
| Global random seed | `42` for Python/NumPy/TensorFlow through `set_global_seed(42)` |
| Temporal window | `24` consecutive steps |
| Input features | `12`, in the fixed order documented above |
| Prediction target | Residual `Delta T_in` and `Delta H_in`; absolute values are reconstructed from lag-1 values for physical-domain metrics |
| Dataset partition | Chronological `60%/20%/20%` train/validation/test split over the generated windows; no random split is used |
| Input scaling | `MinMaxScaler`, fitted only on the training partition after flattening the temporal axis per feature |
| Input-domain handling | Inputs are clipped to the training physical-domain minima/maxima before the MinMax transform, matching the firmware contract |
| Target scaling | A separate `MinMaxScaler` is fitted on the training residual targets; residual targets are not clipped |
| Validation | The middle temporal partition is passed explicitly as `validation_data`; the most recent partition is reserved for test evaluation |

### Baseline training

| Model | Retained architecture | Optimizer and loss | Maximum training and callbacks |
|---|---|---|---|
| MLP | Flattened `24 x 12 = 288` input; `Dense(16, ReLU, He-normal)`; `Dense(2, linear)` | Adam, learning rate `3e-3`; MSE; MAE metric | `500` epochs, batch `576`; ReduceLROnPlateau factor `0.5`, patience `10`, `min_delta=1e-4`, minimum learning rate `1e-6`; EarlyStopping patience `15`, restore best weights |
| Conv1D Tiny | `Input(24,12)`; Gaussian noise `0.003`; `SeparableConv1D(12, kernel=3, ReLU)`; `DepthwiseConv1D(kernel=3, ReLU)`; global average pooling; dropout `0.02`; independent `Dense(1)` temperature and humidity heads; L2 `1e-5` | Per-head Huber losses: `delta_T=0.25`, `delta_H=0.5`; loss weights `1.5/1.5`; MAE per head | Warm-up: `2` epochs with Adam `1e-5`; stage 1: up to `1000` epochs, batch `256`, cosine-restart AdamW/Adam schedule from `3e-4`, EarlyStopping patience `30`, EMA decay `0.995`; stage 2: up to `60` epochs, batch `256`, cosine-restart schedule from `1e-4`, EarlyStopping patience `10`, EMA decay `0.995` |
| LSTM | `Input(24,12)`; `LSTM(24)`; `Dense(16, ReLU)`; `Dense(2)` | Adam, learning rate `1e-4`, `clipnorm=1.0`; MSE; MAE metric | `50` epochs, batch `220`; ReduceLROnPlateau factor `0.5`, patience `8`, minimum learning rate `1e-6`; EarlyStopping patience `10`, restore best weights |

For Conv1D Tiny, the stage-1 cosine schedule uses `first_decay_steps = 5 x steps_per_epoch`, `t_mul=2.0`, `m_mul=0.8`, and a minimum ratio equivalent to `1e-5/3e-4`. The stage-2 schedule uses `first_decay_steps = 3 x steps_per_epoch`, the same restart multipliers, and a minimum ratio equivalent to `1e-6/1e-4`. AdamW uses `weight_decay=1e-5` and `clipnorm=1.0`; the script contains compatibility fallbacks to Adam implementations when AdamW is unavailable.

### Pruning and post-pruning recovery

| Model | Retained pruning schedule | Pruning-aware training | Recovery after `strip_pruning` |
|---|---|---|---|
| MLP | Constant sparsity `0.50`, beginning at step `0`, updated once per epoch | Up to `100` epochs, batch `512`; Adam `1e-3`; MSE; EarlyStopping patience `10` | `8` epochs, batch `512`; Adam `1e-5`; MSE |
| Conv1D Tiny | Polynomial schedules begin after `5 x steps_per_epoch` and end at epoch `100`. Rules coded by layer type: `Conv1D` `0.10 -> 0.40`; temperature head `0.08 -> 0.40`; humidity head not pruned; other Dense layers `0.12 -> 0.55` | Up to `100` epochs, batch `512`; Adam `3e-4`, `clipnorm=1.0`; Huber `0.25/0.5`, weights `1.5/1.5`; EarlyStopping patience `15`; ReduceLROnPlateau patience `5`, minimum learning rate `1e-5` | Recovery: up to `20` epochs with Adam `2e-4`, weights `1.0/2.0`, EarlyStopping patience `5`, ReduceLROnPlateau patience `2`, minimum learning rate `5e-6`; then up to `5` additional fine-tuning epochs with EarlyStopping patience `2` |
| LSTM | Polynomial sparsity `0.20 -> 0.50`; begins after `10 x steps_per_epoch` and ends at epoch `20` | Up to `20` epochs, batch `64`; Adam `1e-5`; MSE; EarlyStopping patience `20` | The wrapper is stripped and the resulting model is saved; no separate post-strip fine-tuning stage is executed in the retained script |

The Conv1D Tiny pruning table reports the schedule rules exactly as coded. The set of layers actually wrapped is determined by the Keras layer-type matching performed by `prune_clone_fn` in the retained script.

### Quantization and TFLite conversion

| Model | Training immediately before conversion | Representative dataset | TFLite conversion contract |
|---|---|---|---|
| MLP | Full-model QAT for up to `20` epochs, batch `512`; Adam `1e-6`; MSE; EarlyStopping patience `10` | Up to `2000` evenly spaced training samples, yielded in batches of `16` | `Optimize.DEFAULT`; `TFLITE_BUILTINS_INT8`; INT8 input and INT8 output |
| Conv1D Tiny | No QAT. The retained pruned model first receives recovery fine-tuning for up to `40` epochs, batch `128`; Adam `1e-5`; Huber `0.25/0.5`, weights `1.0/2.0`; EarlyStopping patience `12`; ReduceLROnPlateau patience `6` | First up to `500` normalized training windows, yielded one at a time | Post-training full-integer conversion with `Optimize.DEFAULT` and `TFLITE_BUILTINS_INT8`; INT8 input and float32 output |
| LSTM | QAT is applied only to Dense layers; LSTM/Bidirectional and normalization layers use a no-op quantization configuration. Up to `300` epochs, batch `256`; Adam `1e-5`; weighted Huber with humidity weight `1.4`; EarlyStopping patience `15`; ReduceLROnPlateau patience `10`, minimum learning rate `1e-6` | Up to `512` normalized training sequences, yielded one at a time | Hybrid conversion with INT8 built-ins, float built-ins, and `SELECT_TF_OPS`; TensorList lowering disabled; float32 input/output when an LSTM is present; fallback remains float32 I/O with built-ins plus `SELECT_TF_OPS` |

### Canonical scripts used for this summary

| Stage | MLP | Conv1D Tiny | LSTM |
|---|---|---|---|
| Baseline | `trainings/environment_mlp/base_model/environment_base_model_mlp.py` | `trainings/environment_Conv1D_Tiny/base_model/environment_base_model_Conv1D_Tiny.py` | `trainings/environment_lstm/base_model/environment_base_model_lstm.py` |
| Pruning | `trainings/environment_mlp/pruned_model/environment_pruned_model_mlp.py` | `trainings/environment_Conv1D_Tiny/pruned_model/environment_pruned_model_Conv1D_Tiny.py` | `trainings/environment_lstm/pruned_model/environment_pruned_model_lstm.py` |
| Quantization | `trainings/environment_mlp/quantized_model/environment_quantized_model_mlp.py` | `trainings/environment_Conv1D_Tiny/quantized_model/environment_quantized_model_Conv1D_Tiny.py` | `trainings/environment_lstm/quantized_model/environment_quantized_model_lstm.py` |

## MLP

### 1. Generate the processed dataset

```powershell
python datasets/environment_mlp/environment_dataset_mlp.py
```

Expected output: `datasets/environment_mlp/environment_dataset_mlp.csv`.

### 2. Train the baseline model

```powershell
python trainings/environment_mlp/base_model/environment_base_model_mlp.py
```

This creates a versioned baseline run containing the `.keras` model, fitted X/y scalers, metrics, workbooks, and figures. The exact retained architecture, optimizer, loss, epoch limit, batch size, and callbacks are summarized in [Training configuration summary](#training-configuration-summary).

### 3. Prune and fine-tune

```powershell
python trainings/environment_mlp/pruned_model/environment_pruned_model_mlp.py
```

The pruning stage resolves the latest baseline run and exports a pruned `.keras` artifact plus evaluation evidence.

### 4. Quantize, convert, and export Python references

```powershell
python trainings/environment_mlp/quantized_model/environment_quantized_model_mlp.py
```

Quantization path: QAT with full INT8 built-ins; INT8 input and INT8 output.

This stage creates the TFLite artifact, Rolling-24 prediction/metric tables, replay samples, preprocessing references, tensor input/output references, debug workbooks, and figures.

### 5. Convert the TFLite model to a C header

```powershell
python trainings/environment_mlp/header_generator/header_generator_mlp.py
```

### 6. Export scaler constants to a C header

```powershell
python trainings/environment_mlp/scalers_exporter/scale_vector_exporter_mlp.py
```

### 7. Synchronize generated headers into firmware

The YAML pipeline contains the exact `pio_pull_headers.py` arguments for this model. Review the plan before execution:

```powershell
python runner.py -p environment_esp32_mlp_pipeline.yaml --list
python runner.py -p environment_stm32_mlp_pipeline.yaml --list
```

To rebuild only through the host-side quantization stage:

```powershell
python runner.py -p environment_esp32_mlp_pipeline.yaml --to quantize_optimizing
```

Then run the header/scaler exporters and header-copy steps in the order shown above or execute the complete hardware pipeline when a board is connected.

## Conv1D Tiny

### 1. Generate the processed dataset

```powershell
python datasets/environment_Conv1D_Tiny/environment_dataset_Conv1D_Tiny.py
```

Expected output: `datasets/environment_Conv1D_Tiny/environment_dataset_Conv1D_Tiny.csv`.

### 2. Train the baseline model

```powershell
python trainings/environment_Conv1D_Tiny/base_model/environment_base_model_Conv1D_Tiny.py
```

This creates a versioned baseline run containing the `.keras` model, fitted X/y scalers, metrics, workbooks, and figures. The exact retained architecture, optimizer, loss, epoch limit, batch size, and callbacks are summarized in [Training configuration summary](#training-configuration-summary).

### 3. Prune and fine-tune

```powershell
python trainings/environment_Conv1D_Tiny/pruned_model/environment_pruned_model_Conv1D_Tiny.py
```

The pruning stage resolves the latest baseline run and exports a pruned `.keras` artifact plus evaluation evidence.

### 4. Quantize, convert, and export Python references

```powershell
python trainings/environment_Conv1D_Tiny/quantized_model/environment_quantized_model_Conv1D_Tiny.py
```

Quantization path: recovery fine-tuning of the retained pruned model followed by post-training full-integer conversion with INT8 input and float32 output; no Conv1D QAT stage is used.

This stage creates the TFLite artifact, Rolling-24 prediction/metric tables, replay samples, preprocessing references, tensor input/output references, debug workbooks, and figures.

### 5. Convert the TFLite model to a C header

```powershell
python trainings/environment_Conv1D_Tiny/header_generator/header_generator_Conv1D_Tiny.py
```

### 6. Export scaler constants to a C header

```powershell
python trainings/environment_Conv1D_Tiny/scalers_exporter/scale_vector_exporter_Conv1D_Tiny.py
```

### 7. Synchronize generated headers into firmware

The YAML pipeline contains the exact `pio_pull_headers.py` arguments for this model. Review the plan before execution:

```powershell
python runner.py -p environment_esp32_Conv1D_pipeline.yaml --list
python runner.py -p environment_stm32_Conv1D_pipeline.yaml --list
```

To rebuild only through the host-side quantization stage:

```powershell
python runner.py -p environment_esp32_Conv1D_pipeline.yaml --to quantize_optimizing
```

Then run the header/scaler exporters and header-copy steps in the order shown above or execute the complete hardware pipeline when a board is connected.

## LSTM

### 1. Generate the processed dataset

```powershell
python datasets/environment_lstm/environment_dataset_lstm.py
```

Expected output: `datasets/environment_lstm/environment_dataset_lstm.csv`.

### 2. Train the baseline model

```powershell
python trainings/environment_lstm/base_model/environment_base_model_lstm.py
```

This creates a versioned baseline run containing the `.keras` model, fitted X/y scalers, metrics, workbooks, and figures. The exact retained architecture, optimizer, loss, epoch limit, batch size, and callbacks are summarized in [Training configuration summary](#training-configuration-summary).

### 3. Prune and fine-tune

```powershell
python trainings/environment_lstm/pruned_model/environment_pruned_model_lstm.py
```

The pruning stage resolves the latest baseline run and exports a pruned `.keras` artifact plus evaluation evidence.

### 4. Quantize, convert, and export Python references

```powershell
python trainings/environment_lstm/quantized_model/environment_quantized_model_lstm.py
```

Quantization path: Hybrid TFLite path with SELECT_TF_OPS; float32 input/output and mixed internal quantization.

This stage creates the TFLite artifact, Rolling-24 prediction/metric tables, replay samples, preprocessing references, tensor input/output references, debug workbooks, and figures.

### 5. Convert the TFLite model to a C header

```powershell
python trainings/environment_lstm/header_generator/header_generator_lstm.py
```

### 6. Export scaler constants to a C header

```powershell
python trainings/environment_lstm/scalers_exporter/scale_vector_exporter_lstm.py
```

### 7. Synchronize generated headers into firmware

The YAML pipeline contains the exact `pio_pull_headers.py` arguments for this model. Review the plan before execution:

```powershell
python runner.py -p environment_esp32_lstm_pipeline.yaml --list
python runner.py -p environment_stm32_lstm_pipeline.yaml --list
```

To rebuild only through the host-side quantization stage:

```powershell
python runner.py -p environment_esp32_lstm_pipeline.yaml --to quantize_optimizing
```

Then run the header/scaler exporters and header-copy steps in the order shown above or execute the complete hardware pipeline when a board is connected.

## Complete pipelines

Each YAML pipeline contains dataset generation, baseline training, pruning, quantization, model/scaler/replay header synchronization, PlatformIO build, upload, and monitor stages. In all six pipelines, `pio_build` depends on both `pio_pull_scalers` and `pio_pull_rolling24`, so the build cannot start before both generated headers have been synchronized. See `docs/PIPELINE_DEPENDENCY_VALIDATION.md`.

```powershell
python runner.py -p environment_esp32_mlp_pipeline.yaml
python runner.py -p environment_stm32_mlp_pipeline.yaml
python runner.py -p environment_esp32_Conv1D_pipeline.yaml
python runner.py -p environment_stm32_Conv1D_pipeline.yaml
python runner.py -p environment_esp32_lstm_pipeline.yaml
python runner.py -p environment_stm32_lstm_pipeline.yaml
```

Use `--list` or `--dry-run` before a complete run. `--only`, `--from`, `--to`, `--after`, and `--before` are available as shown by `python runner.py --help`.

The retained final compilation records can be inspected under `environment_reports/final_builds/`, and the captured host and software environment can be inspected under `environment_reports/system reports/`. Start with `environment_reports/README.md`.

The six build records correspond to the REPLAY firmware configurations used for stage-wise conformance verification. FIELD configurations use the same PlatformIO projects and build procedure, with the acquisition mode selected by a compile-time macro.

## Frozen evidence versus fresh runs

The checked-in `run/` folders are frozen evidence. Some historical `latest.txt` and manifest fields contain absolute paths from the experiment workstation. A clean sequential execution updates the local run pointers. To validate the packaged evidence without retraining, use the comparison commands in `docs/REVIEWER_VERIFICATION_GUIDE.md`; they resolve the retained metrics and logs directly.


## Return to the reviewer route

After completing or inspecting the reconstruction workflow, return to [`REVIEWER_START_HERE.md`](../REVIEWER_START_HERE.md). Use [`REVIEWER_VERIFICATION_GUIDE.md`](REVIEWER_VERIFICATION_GUIDE.md) to compare fresh outputs with the retained evidence.
