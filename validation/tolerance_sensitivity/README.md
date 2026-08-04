# Tolerance-Sensitivity Evidence

This folder contains a reviewer-verifiable sensitivity analysis of the fixed stage-wise tolerances at **1x, 10x, and 100x**.

## Files

- `tolerance_sensitivity.xlsx`: summary, detailed stage-level evidence, formulas, method, and source workbook paths.
- `tolerance_sensitivity.csv`: machine-readable copy of the detailed evidence.

## Method

For numeric stages, the classification at factor `f` is:

`PASS(f)` when `observed maximum absolute difference <= f x adopted tolerance`, for `f in {1, 10, 100}`.

The raw tensor byte/metadata stage remains an exact comparison with zero tolerance, and its result is not relaxed by numerical scaling. Because a numerical maximum absolute difference is not applicable to this exact comparison, the corresponding field is reported as `N/A`. The mismatch count records the number of events with byte or metadata differences.

All observed values were read from the `Tolerance_Protocol` sheets of the six frozen model-I/O comparison workbooks under `utils/workbook_*/<target>/replay/`. During this sensitivity analysis, no thresholds were fitted, replaced, or recalibrated; the analysis only applied 1x, 10x, and 100x factors to the previously adopted engineering acceptance bounds.

## Result

- MLP remains PASS at 1x, 10x, and 100x on WEMOS LOLIN32 and NUCLEO-F411RE.
- Conv1D Tiny remains PASS at 1x, 10x, and 100x on both targets.
- LSTM remains FAIL at 1x, 10x, and 100x on both targets.
- For LSTM, the decoded/semantic maximum difference is `1.12704e-2`, which is `56.352x` above the already relaxed 100x threshold (`2e-4`).
- For LSTM, the final-prediction maximum difference is `2.276627e-2`, which is `11.383135x` above the 100x threshold (`2e-3`).
- The LSTM raw tensor byte/metadata stage contains exact mismatches in 3/24 events and therefore remains FAIL at 1x, 10x, and 100x.

The sensitivity result therefore supports the conclusion that the LSTM divergence is not created by an excessively strict numerical tolerance.
