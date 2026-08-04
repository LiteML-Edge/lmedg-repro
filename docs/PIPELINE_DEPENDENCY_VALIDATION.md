# YAML Pipeline Dependency Validation

## Correction applied

The six pipeline YAML files were updated so that `pio_build` depends on both generated-header synchronization steps:

```yaml
- name: pio_build
  ...
  depends_on: ["pio_pull_scalers", "pio_pull_rolling24"]
```

This guarantees that the firmware build starts only after both the newly exported scaler header and the Rolling-24 replay header have been copied into the corresponding PlatformIO project.

The two copy steps may appear in either order in the topological plan because they are independent of each other. The required condition is that both finish before `pio_build`.

## Updated files

- `environment_esp32_mlp_pipeline.yaml`
- `environment_stm32_mlp_pipeline.yaml`
- `environment_esp32_Conv1D_pipeline.yaml`
- `environment_stm32_Conv1D_pipeline.yaml`
- `environment_esp32_lstm_pipeline.yaml`
- `environment_stm32_lstm_pipeline.yaml`

## Verification command

Run from the repository root:

```powershell
python runner.py -p environment_esp32_mlp_pipeline.yaml --list
```

Repeat with the other five YAML files. In every plan, the relevant final sequence is:

```text
pio_pull_rolling24
pio_pull_scalers
pio_build  deps=['pio_pull_scalers', 'pio_pull_rolling24']
pio_upload
pio_monitor
```

The relative order of `pio_pull_rolling24` and `pio_pull_scalers` is not significant; `pio_build` is blocked until both dependencies are complete.

## Scope of the change

Only the `depends_on` entry of `pio_build` was changed in the six YAML files. No Python script, firmware source, model, dataset, header, log, workbook, or experimental result was modified.
