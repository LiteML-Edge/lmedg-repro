# Formal LiteML-Edge Contract Artifacts

This directory contains the machine-readable contracts used in the LiteML-Edge validation workflow.

## Files

- `mlp_contract.json`: machine-readable contract for the MLP deployment path.
- `conv1d_tiny_contract.json`: machine-readable contract for the Conv1D Tiny deployment path.
- `lstm_contract.json`: machine-readable contract for the LSTM deployment path.
- `tolerance_policy.json`: common stage-wise acceptance policy applied unchanged to all models and both hardware targets.

Each model file materializes the tuple `C = (S, K, W, Q, P, M)`:

- `S`: ordered feature, target, and raw replay schema;
- `K`: frozen scaling constants and ranges;
- `W`: window size, stride, replay chronology, seed state, warm-up exclusion, and valid-event policy;
- `Q`: input/output tensor shapes, dtypes, layouts, and quantization style;
- `P`: smoothing and lag order, clipping, scaling, residual semantics, output mapping, and reconstruction rules;
- `M`: valid-event aggregation, prediction/metric comparison precision, metrics, and stage-specific pass/fail criteria.

The `artifacts` object in each contract points to the exact model, firmware headers, host references, workbooks, and hardware logs in this package. Paths are relative to the repository root.

The three model JSON files define the contract rules used in this package and link them to the corresponding implementation and evidence artifacts.

## Interpretation rule

The JSON files are declarative specifications. Python references, firmware sources, replay headers, logs, and comparison workbooks show how each contract was applied. `tolerance_policy.json` contains the adopted stage-wise thresholds. Calibration and diagnostic sheets record observed deviations without changing those thresholds. The `1x/10x/100x` sensitivity analysis is under `validation/tolerance_sensitivity/`.
