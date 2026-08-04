# Rolling-24 Scope and Interpretation (`n=24`)

## Manuscript-aligned scope statement

The Rolling-24 analysis uses `n=24` valid chronological events as a bounded diagnostic trace, not as an independent population sample. The reported LSTM divergence of `3/24 (12.5%)` is the exact descriptive result for these 24 valid events. Consecutive Rolling-24 events share input history and are therefore not treated as statistically independent observations for population inference, although each event remains a distinct prediction and Python--firmware contract comparison.

Accordingly, the evidence establishes reproducibility within the evaluated environmental dataset, the `W=24` policy, the TensorFlow-to-TFLM deployment path, and the two evaluated MCU targets. It does not establish universal validity across datasets, input modalities, window sizes, runtimes, or hardware platforms.

## What the reviewer should verify

1. Confirm `window_steps = 24`, `chronological_replay_rows = 47`, and `reported_valid_events = 24` in the three files under `contracts/`.
2. Inspect the LSTM REPLAY workbooks under:
   - `utils/workbook_lstm/esp32/replay/`
   - `utils/workbook_lstm/stm32f411re/replay/`
3. Confirm the exact `3/24` stage-wise mismatch classification for both targets in the generated model-I/O comparison workbooks.
4. Inspect `validation/tolerance_sensitivity/tolerance_sensitivity.xlsx` to confirm that the LSTM classification remains unchanged at `1x`, `10x`, and `100x`.
5. Use the MLP and Conv1D Tiny workbooks in the corresponding `utils/workbook_*/*/replay/` folders as the conformant comparison cases.

## Interpretation boundary

This note documents the same scope used in the revised manuscript. It does not convert the 24-event diagnostic trace into a population-level statistical claim and does not broaden the manuscript conclusions.
