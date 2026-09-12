# Formal LiteML-Edge Contract Artifacts

This directory contains the integral machine-readable contract artifacts used by the LiteML-Edge validation workflow.

The formal contract is:

`C = (S, K, W, Q, P, M)`

The three model-specific JSON files instantiate the same contract structure for MLP, Conv1D Tiny, and LSTM while preserving each model's actual tensor interface and quantization behavior.

## Contract elements

### S — Feature and semantic schema

`S` defines the ordered 12-feature schema, feature-vector shape, residual model targets, absolute prediction variables, raw Replay fields, causal `H_in` conditioning, lag semantics, and time-feature policy.

The fixed feature order is:

1. `T_out`
2. `H_out`
3. `T_in_lag1`
4. `H_in_lag1`
5. `T_out_lag1`
6. `H_out_lag1`
7. `T_in_lag2`
8. `H_in_lag2`
9. `sin_hour`
10. `cos_hour`
11. `weekday`
12. `month`

The model targets are residuals (`delta_T_in`, `delta_H_in`), while `T_in` and `H_in` are the reconstructed absolute predictions.

`H_in_raw` is processed by a causal EMA with `alpha=0.08` before `H_in` lag construction. Replay begins from the exported pre-block EMA state and advances that state through the two seed rows. Field initializes the EMA from its first valid `H_in` sample.

Replay time features are derived from the exported epoch. Field uses civil/device time when available; when unavailable, the firmware uses its millis-based synthetic-hour fallback for cyclic hour features and sets the fallback calendar values documented in the model contracts.

### K — Scaling and input-bound policy

`K` contains the frozen `X_MIN`, `X_MAX`, `DY_MIN`, and `DY_MAX` values used by the selected model artifact.

Input features use MinMax scaling to `[0,1]`.

The Python reference clips physical-domain features to `X_MIN/X_MAX` before MinMax scaling. Firmware performs the algebraically equivalent form by applying the same MinMax transform and then clamping the normalized value to `[0,1]`.

Residual targets are scaled with their frozen `DY_MIN/DY_MAX` values. The inverse target transform is applied after model output decoding and before absolute-state reconstruction.

The scaler parameters remain fixed during validation and deployment.

### W — Rolling-window and Replay policy

All evaluated instances use:

- `W=24`
- stride `1`
- two Replay seed rows
- 47 chronological Replay rows
- 49 raw rows in the exported `2+47` Replay artifact

The two seed rows advance the causal humidity EMA from the exported pre-block state and initialize lag history. They are not counted as chronological evaluation rows and are excluded from reported metrics.

Inference and metric aggregation require a complete 24-step window. Incomplete-window warm-up samples are excluded from stage-wise conformance counts and metric aggregation.

Replay is deterministic and header-driven. Field preserves the same post-acquisition rolling/window semantics with live sensing.

### Q — Tensor I/O and numeric representation

`Q` records the actual model-specific tensor contract.

- **MLP:** INT8 input and INT8 output; input shape `[1,288]`; one output tensor `[1,2]`; FULL_INT8 interface.
- **Conv1D Tiny:** INT8 input `[1,24,12]`; two float32 scalar outputs; hybrid INT8-to-float32 tensor interface.
- **LSTM:** float32 input `[1,24,12]`; float32 output `[1,2]`; hybrid internal deployment with float32 interface.

When an interface tensor is quantized, scale and zero-point are sourced from `TfLiteTensor.params` and recorded in the corresponding contract. Saturation to the representable integer range belongs to `Q`; it is not output clipping.

Converter/operator-set details such as LSTM `SELECT_TF_OPS`, and graph details such as Conv1D Q/DQ operators, are recorded under `implementation_notes`, outside the formal tensor I/O contract `Q`.

In every model, semantic output position/channel 0 is `delta_T_in` and position/channel 1 is `delta_H_in`.

### P — Post-processing

`P` contains only post-model operations:

1. apply the semantic output mapping defined in `Q`;
2. inverse-scale the residuals using `DY_MIN/DY_MAX` from `K`;
3. reconstruct absolute state from the lag-1 baseline:

   - `T_in = T_in_lag1 + delta_T_in`
   - `H_in = H_in_lag1 + delta_H_in`

The evaluated instances use no affine correction, no head gain, and no output clipping.

Firmware-only non-finite fallbacks are recorded under `implementation_notes` as safety guards. They are not part of the shared Python↔firmware post-processing contract `P` and must not be interpreted as clipping or calibration.

### M — Valid events, metrics, and acceptance policy

A valid event requires:

- a complete post-warm-up window;
- a successful model invocation;
- decoded outputs conforming to the tensor contract `Q`.

Only valid post-warm-up events update conformance counts and Rolling-24 aggregation.

The reported horizon is `n=24` valid events.

The final metric vector contains nine components:

- MAE, RMSE, and R2 for the aggregate output;
- MAE, RMSE, and R2 for `T_in`;
- MAE, RMSE, and R2 for `H_in`.

Final prediction agreement is evaluated at two decimal places, while Rolling-24 metric agreement is evaluated at four decimal places.

The stage-wise numerical acceptance thresholds are **not duplicated** in the model contracts. `contracts/tolerance_policy.json` is the single machine-readable source of those thresholds.

## Conformance checkpoints

The contracts expose the same validation path used in the manuscript:

- `x*`: preprocessed input;
- `p*`: effective tensor payload;
- `o_raw`: raw tensor bytes/metadata and decoded raw output;
- `y*`: mapped, inverse-scaled, reconstructed prediction;
- final predictions and Rolling-24 metrics.

## Artifact traceability

Each model contract contains an `artifacts` object pointing to its deployed `.tflite` model, firmware model header, exported scaler header, Replay header, host-reference evidence, stage-wise comparison workbooks, and selected firmware logs.

Paths in `artifacts` and `tolerance_policy_ref` are relative to the repository root.

The JSON files are declarative specifications. Detailed converter implementation, firmware execution code, logs, workbooks, and diagnostics remain in their referenced source artifacts and are not duplicated into the contract.

These contract corrections clarify the machine-readable specification only. They do not alter datasets, preprocessing behavior, model artifacts, firmware execution, Replay/Field evidence, tolerances, metrics, numerical results, or conclusions.
