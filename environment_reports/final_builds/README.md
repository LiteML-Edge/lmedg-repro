# Final REPLAY Build Records

> **Navigation:** Return to [`REVIEWER_START_HERE.md`](../../REVIEWER_START_HERE.md) for the complete review route or [`environment_reports/README.md`](../README.md) for the parent report index. This directory contains compilation evidence only; runtime serial evidence is mapped in `docs/EVIDENCE_MANIFEST.csv`.


This directory contains the six retained final compilation records for the firmware configurations used in stage-wise conformance verification.

The six build records correspond to the REPLAY firmware configurations used for stage-wise conformance verification. FIELD configurations use the same PlatformIO projects and build procedure, with the acquisition mode selected by a compile-time macro.

## Scope

The records cover three models and two hardware targets:

| Model | Hardware target | Build record |
|---|---|---|
| MLP | WEMOS LOLIN32 | `mlp_lolin32_replay_build.txt` |
| MLP | NUCLEO-F411RE | `mlp_nucleo_f411re_replay_build.txt` |
| Conv1D Tiny | WEMOS LOLIN32 | `conv1d_tiny_lolin32_replay_build.txt` |
| Conv1D Tiny | NUCLEO-F411RE | `conv1d_tiny_nucleo_f411re_replay_build.txt` |
| LSTM | WEMOS LOLIN32 | `lstm_lolin32_build_replay.txt` |
| LSTM | NUCLEO-F411RE | `lstm_nucleo_f411re_replay_build.txt` |

Each retained record contains the runner execution plan, the exact PlatformIO environment selected by the YAML pipeline, RAM and Flash usage reported by PlatformIO, the PlatformIO `SUCCESS` result, and the final runner `OK` status.

## Interpretation

- These files demonstrate successful compilation of the six final REPLAY configurations retained in the revision pack.
- They do not replace the firmware source, generated headers, `platformio.ini`, YAML pipelines, or runtime serial logs.
- They are not FIELD build records and must not be described as evidence that a single REPLAY binary executes FIELD mode.
- FIELD is compiled from the same PlatformIO project and environment after changing `LITEML_MODE` in the corresponding `include/config.h` from `LITEML_MODE_REPLAY` to `LITEML_MODE_FIELD`.

## Reviewer inspection

The files were captured with Windows PowerShell `Tee-Object` and are stored as UTF-16 little-endian text. Editors such as Visual Studio Code and Windows Notepad detect this encoding automatically. In PowerShell, use:

```powershell
Get-ChildItem ".\environment_reports\final_builds\*.txt"
Get-Content ".\environment_reports\final_builds\mlp_lolin32_replay_build.txt"
```

A successful record should contain both `[SUCCESS]` from PlatformIO and `[DONE] Finished 'pio_build' -> OK` from `runner.py`.

## Reproduce the six builds

Before building, set the corresponding firmware project's `include/config.h` to:

```cpp
#define LITEML_MODE LITEML_MODE_REPLAY
```

Then run from the repository root:

```powershell
python runner.py -p environment_esp32_mlp_pipeline.yaml --only pio_build --no-upstream
python runner.py -p environment_stm32_mlp_pipeline.yaml --only pio_build --no-upstream
python runner.py -p environment_esp32_Conv1D_pipeline.yaml --only pio_build --no-upstream
python runner.py -p environment_stm32_Conv1D_pipeline.yaml --only pio_build --no-upstream
python runner.py -p environment_esp32_lstm_pipeline.yaml --only pio_build --no-upstream
python runner.py -p environment_stm32_lstm_pipeline.yaml --only pio_build --no-upstream
```

`--no-upstream` is appropriate here because the revision pack already contains the synchronized model, scaler, and Rolling-24 headers. Omit `--no-upstream` when the purpose is to regenerate all upstream artifacts before compilation.
