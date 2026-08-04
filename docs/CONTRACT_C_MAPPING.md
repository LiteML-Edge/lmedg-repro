# Concrete LiteML-Edge Contract Mapping: `C = (S, K, W, Q, P, M)`

This document maps the contract tuple to the repository artifacts for the three evaluated models. The machine-readable specifications are available in:

- `contracts/mlp_contract.json`
- `contracts/conv1d_tiny_contract.json`
- `contracts/lstm_contract.json`
- `contracts/tolerance_policy.json`

The contract is not only a conceptual tuple. Each element is frozen, linked to an implementation artifact, and checked against Python and firmware evidence.

## 1. Common data schema `S`

All three evaluated models use the same ordered 12-feature schema:

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

Targets are `T_in` and `H_in`. Raw replay rows contain `epoch`, `T_out`, `H_out`, `T_in`, and `H_in_raw`. The ordering is normative: changing the order changes the contract even when the same variables are present.

Implementation evidence:

- dataset generators: `datasets/environment_<model>/`;
- processed datasets: `datasets/environment_<model>/*.csv`;
- replay headers: the path given by `artifacts.replay_header` in each model contract.

## 2. Frozen constants `K`

Each contract records the exact per-feature `X_MIN`, `X_MAX`, target-residual `DY_MIN`, `DY_MAX`, and feature ranges used by both host and firmware. These values are copied from the frozen experiment artifacts and are not re-fitted during replay verification.

Model-specific distinction:

- the MLP residual target bounds differ slightly in the last decimal places from the Conv1D Tiny and LSTM bounds;
- the scaler headers under each firmware project are therefore model-specific and must not be exchanged.

Implementation evidence:

- frozen scaler objects: `trainings/environment_<model>/base_model/run/`;
- generated scaler headers: the path given by `artifacts.scaler_header`.

## 3. Window and chronology `W`

The evaluated policy is identical across the three models:

- input window: 24 chronological steps;
- stride: 1;
- seed rows: 2;
- chronological replay rows: 47;
- total raw rows: 49;
- final reported valid events: 24;
- humidity EMA coefficient: `0.08`;
- initial humidity EMA state: `64.087463`.

The two seed rows initialize lag features and the causal humidity state and are not counted among the 47 chronological samples or the 24 reported events. A valid event is evaluated only after a complete 24-step window is available. Incomplete-window warm-up samples are excluded from stage-wise conformance counts and Rolling-24 metric aggregation. With `W=24`, stride 1, and `n=24`, the required chronological cardinality is `W + (n - 1) = 47`.

## 4. Tensor and quantization contract `Q`

| Model | Input | Output | Input shape | Output layout | Quantization style |
|---|---|---|---|---|---|
| MLP | INT8 | INT8 | `[1, 288]` | one tensor `[1, 2]` | `FULL_INT8` |
| Conv1D Tiny | INT8 | float32 | `[1, 24, 12]` | two scalar tensors `[1, 1]` | `HYBRID_INT8_TO_FLOAT32_WITH_QDQ` |
| LSTM | float32 | float32 | `[1, 24, 12]` | one tensor `[1, 2]` | `HYBRID_FLOAT_IO_WITH_SELECT_TF_OPS` |

This element prevents apparently compatible models from being treated as equivalent when their tensor shape, dtype, output layout, or conversion path differs.

Implementation evidence:

- TFLite models: `artifacts.tflite_model`;
- generated model headers: `artifacts.model_header`;
- tensor metadata and raw references: `artifacts.host_reference_evidence`.

## 5. Processing and reconstruction `P`

For every model:

1. the causal EMA is applied to raw `H_in` before the humidity lag features are constructed;
2. lag-1 and lag-2 features are constructed in chronological order;
3. each input feature is clipped to the frozen training bounds and scaled to `[0, 1]`;
4. the learned outputs represent residuals in the semantic order `delta_T_in`, `delta_H_in`;
5. model outputs are inverse-scaled;
6. absolute predictions are reconstructed as `T_in_lag1 + delta_T_in` and `H_in_lag1 + delta_H_in`;
7. no additional calibration or head gain is applied.

The comparison workbooks separate model-input construction, raw/decoded model output, semantic output, and final reconstructed predictions. A pass at the final metric stage does not override an earlier-stage mismatch.

## 6. Metrics and decisions `M`

Only the 24 complete-window, post-warm-up valid events enter conformance counts and Rolling-24 aggregation. Prediction agreement is assessed at two decimal places, and final Rolling-24 metrics are compared at four decimal places. The reported metrics are MAE, RMSE, and R-squared. Stage decisions use the common frozen policy in `contracts/tolerance_policy.json`:

| Stage | Criterion |
|---|---:|
| raw tensor bytes, integer payloads, metadata, and memory order | exact (`0.0`) |
| normalized/model input numeric values | absolute tolerance `2e-6` |
| decoded raw output | absolute tolerance `2e-6` |
| semantic output | absolute tolerance `2e-6` |
| final post-processed prediction | absolute tolerance `2e-5` |
| auxiliary semantic debug values | absolute tolerance `2e-5` |

The thresholds are applied uniformly across models and targets. Calibration/diagnostic workbook sheets report the observed numerical spread but do not overwrite the adopted pass/fail policy. This explicit separation addresses the risk of threshold fitting discussed in the revised manuscript.

## 7. Executable verification path

For a selected model:

1. inspect its JSON contract;
2. verify that every path in `artifacts` exists;
3. inspect the TFLite model, model header, scaler header, and replay header;
4. build and run the selected hardware target in REPLAY mode;
5. capture the serial log;
6. execute the model-specific immediate-I/O comparison script;
7. execute the final prediction/metric comparison script;
8. inspect the generated workbook stage summaries and mismatch rows.

Commands are provided in `docs/REVIEWER_VERIFICATION_GUIDE.md`, `docs/END_TO_END_REPRODUCTION.md`, and `docs/FIRMWARE_BUILD_AND_HARDWARE.md`.

## 8. Why three contract files are necessary

The three models share `S`, most of `K`, `W`, `P`, and `M`, but they do not share the same `Q`. They also have model-specific frozen artifacts and slight scaler-bound differences. Separate machine-readable contracts prevent accidental substitution and make the Python-to-firmware equivalence claim auditable at the model level.
