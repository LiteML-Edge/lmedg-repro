# Firmware Build, Upload, Hardware, and Instrumentation

> **Document role:** This is the specialized guide for firmware mode selection, compilation, upload, serial monitoring, hardware wiring, and instrumentation. Begin with [`REVIEWER_START_HERE.md`](../REVIEWER_START_HERE.md) for the overall review sequence. Use [`REVIEWER_VERIFICATION_GUIDE.md`](REVIEWER_VERIFICATION_GUIDE.md) for host-side evidence verification and [`END_TO_END_REPRODUCTION.md`](END_TO_END_REPRODUCTION.md) for fresh reconstruction of upstream artifacts.

## Firmware evidence boundary

The six retained files under `environment_reports/final_builds/` are successful REPLAY compilation records. They are not binaries and are not FIELD build records. REPLAY and FIELD use the same projects and PlatformIO environments, but the compile-time `LITEML_MODE` macro selects the acquisition path.


## Mode selection

Each model project uses one ESP32 environment and one STM32 environment. REPLAY/FIELD is selected in that project’s `include/config.h`:

```cpp
#define LITEML_MODE LITEML_MODE_REPLAY
```

or

```cpp
#define LITEML_MODE LITEML_MODE_FIELD
```

REPLAY is the deterministic Python--firmware validation mode. FIELD reads the DHT22 sensors and applies the field-runtime behavior.

The six build records correspond to the REPLAY firmware configurations used for stage-wise conformance verification. FIELD configurations use the same PlatformIO projects and build procedure, with the acquisition mode selected by a compile-time macro.

## Build matrix

| Model | Project | ESP32 build | STM32 build |
|---|---|---|---|
| MLP | `firmwares/environment_mlp/PlatfIO_mlp` | `pio run -d firmwares/environment_mlp/PlatfIO_mlp -e lolin32_mlp` | `pio run -d firmwares/environment_mlp/PlatfIO_mlp -e nucleo_f411re_mlp` |
| Conv1D Tiny | `firmwares/environment_Conv1D_Tiny/PlatfIO_Conv1D_Tiny` | `pio run -d firmwares/environment_Conv1D_Tiny/PlatfIO_Conv1D_Tiny -e lolin32_Conv1D_Tiny` | `pio run -d firmwares/environment_Conv1D_Tiny/PlatfIO_Conv1D_Tiny -e nucleo_f411re_Conv1D_Tiny` |
| LSTM | `firmwares/environment_lstm/PlatfIO_lstm` | `pio run -d firmwares/environment_lstm/PlatfIO_lstm -e lolin32_lstm` | `pio run -d firmwares/environment_lstm/PlatfIO_lstm -e nucleo_f411re_lstm` |

## Retained final REPLAY build records

Reviewers can inspect the six final compilation records under `environment_reports/final_builds/`. Start with `environment_reports/final_builds/README.md`, which maps every model--target pair to its exact record and explains the UTF-16 text encoding produced by Windows PowerShell `Tee-Object`.

Each record contains the YAML-selected PlatformIO environment, RAM and Flash usage, PlatformIO `SUCCESS`, and the final runner `OK` status. These are compilation records, not firmware binaries and not runtime serial logs. The twelve REPLAY/FIELD runtime logs remain in the firmware projects and are indexed by `docs/EVIDENCE_MANIFEST.csv`.

The retained build scope is:

```text
3 models x 2 hardware targets x REPLAY = 6 final build records
```

FIELD uses the same project and build procedure but requires changing the compile-time `LITEML_MODE` macro before compiling.

## Upload

### MLP

```powershell
pio run -d firmwares/environment_mlp/PlatfIO_mlp -e lolin32_mlp -t upload
pio run -d firmwares/environment_mlp/PlatfIO_mlp -e nucleo_f411re_mlp -t upload
```

### Conv1D Tiny

```powershell
pio run -d firmwares/environment_Conv1D_Tiny/PlatfIO_Conv1D_Tiny -e lolin32_Conv1D_Tiny -t upload
pio run -d firmwares/environment_Conv1D_Tiny/PlatfIO_Conv1D_Tiny -e nucleo_f411re_Conv1D_Tiny -t upload
```

### LSTM

```powershell
pio run -d firmwares/environment_lstm/PlatfIO_lstm -e lolin32_lstm -t upload
pio run -d firmwares/environment_lstm/PlatfIO_lstm -e nucleo_f411re_lstm -t upload
```

## Serial monitoring and log capture

The wrapper adds timestamp and `log2file` filters unless `--no-log` is used. Example for MLP:

```powershell
python utils/global_utils/pio_monitor.py -d firmwares/environment_mlp/PlatfIO_mlp -e lolin32_mlp --port COM6 --baud 115200 --wait 20
python utils/global_utils/pio_monitor.py -d firmwares/environment_mlp/PlatfIO_mlp -e nucleo_f411re_mlp --port COM8 --baud 115200 --wait 20
```

Use the corresponding project and environment names from the build matrix for Conv1D Tiny and LSTM. Move the final log into that project’s `logs/` folder using a unique `device-monitor-YYMMDD-HHMMSS.log` name.

## YAML pipeline execution

Run all commands from the repository root. The six YAML files use `runner.py` to execute a dependency-aware pipeline containing dataset generation, baseline training, pruning, quantization, header generation and synchronization, PlatformIO build, upload, and serial monitoring.

### Preparation

Create and activate the recorded Python environment before running a pipeline:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-reproduction.txt
```

For formal Python--firmware 1:1 reproduction, open the corresponding firmware project's `include/config.h` and select Replay before build and upload:

```cpp
#define LITEML_MODE LITEML_MODE_REPLAY
```

Use `LITEML_MODE_FIELD` only for live-sensor execution and hardware-dependent deployment-cost acquisition. The YAML pipelines do not change this macro automatically. Confirm the local upload and serial ports before hardware execution; the packaged configuration uses `COM6` for WEMOS LOLIN32 and `COM8` for NUCLEO-F411RE.

### Inspect a pipeline without executing it

List the calculated dependency plan:

```powershell
python runner.py -p environment_esp32_mlp_pipeline.yaml --list
```

Print the commands without executing them:

```powershell
python runner.py -p environment_esp32_mlp_pipeline.yaml --dry-run
```

Replace the YAML filename with any of the six files listed below.

### Compile the six retained REPLAY configurations

After confirming `LITEML_MODE_REPLAY` in the corresponding `include/config.h`, compile the packaged firmware inputs without rerunning upstream generation stages:

```powershell
python runner.py -p environment_esp32_mlp_pipeline.yaml --only pio_build --no-upstream
python runner.py -p environment_stm32_mlp_pipeline.yaml --only pio_build --no-upstream
python runner.py -p environment_esp32_Conv1D_pipeline.yaml --only pio_build --no-upstream
python runner.py -p environment_stm32_Conv1D_pipeline.yaml --only pio_build --no-upstream
python runner.py -p environment_esp32_lstm_pipeline.yaml --only pio_build --no-upstream
python runner.py -p environment_stm32_lstm_pipeline.yaml --only pio_build --no-upstream
```

These commands reproduce the scope represented by `environment_reports/final_builds/`. Omit `--no-upstream` when the intended procedure is to regenerate the dataset, model, and synchronized headers before compilation.

### Execute a complete YAML pipeline

#### MLP

```powershell
python runner.py -p environment_esp32_mlp_pipeline.yaml
python runner.py -p environment_stm32_mlp_pipeline.yaml
```

#### Conv1D Tiny

```powershell
python runner.py -p environment_esp32_Conv1D_pipeline.yaml
python runner.py -p environment_stm32_Conv1D_pipeline.yaml
```

#### LSTM

```powershell
python runner.py -p environment_esp32_lstm_pipeline.yaml
python runner.py -p environment_stm32_lstm_pipeline.yaml
```

A complete hardware pipeline requires the corresponding board to be connected for `pio_upload` and `pio_monitor`. The serial-monitor step remains active until it is stopped by the user or the underlying monitor process exits.

### Pipeline step names and functions

All six YAML files use the same step names:

| Step | Function |
|---|---|
| `preprocess` | Generate the processed environmental dataset. |
| `base_training` | Train the baseline model and retain its versioned artifacts. |
| `prune_optimizing` | Prune and fine-tune the latest baseline model. |
| `quantize_optimizing` | Quantize/convert the model and generate host-side references and Rolling-24 evidence. |
| `header_generator` | Convert the generated TFLite model into a C header. |
| `pio_pull_model` | Copy the latest generated model header into the selected firmware project. |
| `scalers_exporter` | Export scaler constants as a C header. |
| `pio_pull_scalers` | Copy the latest scaler header into the selected firmware project. |
| `pio_pull_rolling24` | Copy the Rolling-24 replay header into the selected firmware project. |
| `pio_build` | Build the selected PlatformIO environment without automatic cleaning. |
| `pio_upload` | Upload the previously built firmware to the selected target. |
| `pio_monitor` | Open the configured serial monitor and capture the execution log. |

`pio_build` depends on both `pio_pull_scalers` and `pio_pull_rolling24`. The relative order of those two synchronization steps is not significant, but both must finish before the build starts.

### Execute one step with its upstream dependencies

By default, `--only` executes the selected step and all required upstream dependencies. Example:

```powershell
python runner.py -p environment_esp32_mlp_pipeline.yaml --only pio_build
```

This command reconstructs all prerequisites required by `pio_build`; it does not execute `pio_upload` or `pio_monitor`.

Multiple target steps may be requested together:

```powershell
python runner.py -p environment_esp32_mlp_pipeline.yaml --only pio_build pio_monitor
```

The runner includes the required dependency chain for both selected steps.

### Execute one step in isolation

Use `--no-upstream` only when the required input artifacts already exist:

```powershell
python runner.py -p environment_esp32_mlp_pipeline.yaml --only preprocess --no-upstream
python runner.py -p environment_esp32_mlp_pipeline.yaml --only base_training --no-upstream
python runner.py -p environment_esp32_mlp_pipeline.yaml --only prune_optimizing --no-upstream
python runner.py -p environment_esp32_mlp_pipeline.yaml --only quantize_optimizing --no-upstream
python runner.py -p environment_esp32_mlp_pipeline.yaml --only header_generator --no-upstream
python runner.py -p environment_esp32_mlp_pipeline.yaml --only pio_pull_model --no-upstream
python runner.py -p environment_esp32_mlp_pipeline.yaml --only scalers_exporter --no-upstream
python runner.py -p environment_esp32_mlp_pipeline.yaml --only pio_pull_scalers --no-upstream
python runner.py -p environment_esp32_mlp_pipeline.yaml --only pio_pull_rolling24 --no-upstream
python runner.py -p environment_esp32_mlp_pipeline.yaml --only pio_build --no-upstream
python runner.py -p environment_esp32_mlp_pipeline.yaml --only pio_upload --no-upstream
python runner.py -p environment_esp32_mlp_pipeline.yaml --only pio_monitor --no-upstream
```

The same commands apply to Conv1D Tiny, LSTM, and STM32 by replacing only the YAML filename. Isolated execution does not validate or create missing prerequisites. In particular:

- isolated `pio_pull_model`, `pio_pull_scalers`, or `pio_pull_rolling24` requires the corresponding generated header to exist;
- isolated `pio_build` requires all synchronized headers and project dependencies to be present;
- isolated `pio_upload` requires a valid firmware build and a connected target;
- isolated `pio_monitor` requires the correct serial port and firmware already running on the target.

### Build, upload, and monitor with the runner

The following isolated sequence uses the artifacts already present in the pack. Connect the selected board and confirm the configured upload and monitor port before running the upload and monitor stages.

WEMOS LOLIN32 MLP example:

```powershell
python runner.py -p environment_esp32_mlp_pipeline.yaml --only pio_build --no-upstream
python runner.py -p environment_esp32_mlp_pipeline.yaml --only pio_upload --no-upstream
python runner.py -p environment_esp32_mlp_pipeline.yaml --only pio_monitor --no-upstream
```

NUCLEO-F411RE MLP example:

```powershell
python runner.py -p environment_stm32_mlp_pipeline.yaml --only pio_build --no-upstream
python runner.py -p environment_stm32_mlp_pipeline.yaml --only pio_upload --no-upstream
python runner.py -p environment_stm32_mlp_pipeline.yaml --only pio_monitor --no-upstream
```

Use the corresponding Conv1D Tiny or LSTM YAML filename for those models. The `pio_monitor` stage is interactive and remains active until stopped by the user or until the underlying monitor process exits.

### Execute a range of steps

Run from a selected stage through the end:

```powershell
python runner.py -p environment_esp32_mlp_pipeline.yaml --from quantize_optimizing
```

Run from the beginning through a selected stage:

```powershell
python runner.py -p environment_esp32_mlp_pipeline.yaml --to quantize_optimizing
```

Run only the dependency range from one stage through another:

```powershell
python runner.py -p environment_esp32_mlp_pipeline.yaml --from header_generator --to pio_build
```

Run all downstream stages after a selected step, excluding that step:

```powershell
python runner.py -p environment_esp32_mlp_pipeline.yaml --after quantize_optimizing
```

Run all upstream stages before a selected step, excluding that step:

```powershell
python runner.py -p environment_esp32_mlp_pipeline.yaml --before pio_upload
```

Continue executing independent remaining stages after an error:

```powershell
python runner.py -p environment_esp32_mlp_pipeline.yaml --from base_training --keep-going
```

Use `--keep-going` for diagnostic collection, not for claiming a successful reproduction. Review every failed stage before using downstream outputs.

### Runner command reference

```powershell
python runner.py --help
```

The implemented selection options are `--only`, `--no-upstream`, `--from`, `--to`, `--after`, `--before`, `--list`, `--dry-run`, and `--keep-going`.

## Hardware targets and common protocol

- **WEMOS LOLIN32:** ESP32/Xtensa target operated at a fixed CPU clock of 240 MHz.
- **NUCLEO-F411RE:** STM32F411RE ARM Cortex-M4 target operated at a fixed CPU clock of 100 MHz.
- **Execution modes:** Replay and Field.
- **Rolling policy:** `W = 24` with unit stride.
- **Evaluation horizon:** `n = 24` valid post-warm-up events for each model--hardware pair and mode.
- **Run control:** Replay and Field use matched run lengths and the same event-boundary semantics.

## Pin and peripheral contract

| Signal | WEMOS LOLIN32 | NUCLEO-F411RE |
|---|---|---|
| I2C SDA (INA219/OLED bus) | GPIO 5 | Arduino `SDA` |
| I2C SCL (INA219/OLED bus) | GPIO 4 | Arduino `SCL` |
| DHT22 external | GPIO 16 | D6 |
| DHT22 internal | GPIO 25 | D7 |
| Serial | COM6 in experiment setup | COM8 in experiment setup |

## Electrical measurement configuration

Electrical quantities are acquired independently of model inference through an external INA219 conversion chain read by the MCU over I2C. The measurement boundary is the complete-board 5 V input rail on both targets.

| Item | Setting |
|---|---|
| Monitor | External INA219 current/power monitor |
| Calibration preset | 16 V / 400 mA |
| Shunt resistance | 0.1 ohm |
| Measured boundary | Complete-board 5 V input rail |
| Acquisition control | Bounded polling |
| Outlier handling | 3-sigma rejection |
| Smoothing | Exponential moving average before cumulative integration |
| Energy update | `E[k] = E[k-1] + P[k] * delta_t` |

No fixed acquisition rate is claimed. The implementation uses bounded polling and firmware timestamps to determine the elapsed interval used in each cumulative-energy update.

## Event boundaries and measurement windows

The firmware records electrical and timing state at four boundaries:

1. **event-pre:** immediately before the complete inference event;
2. **invoke-pre:** immediately before the model invocation;
3. **invoke-post:** immediately after the model invocation;
4. **event-post:** immediately after the complete inference event.

These snapshots delimit two nested windows:

- **Invocation window:** `invoke-pre` to `invoke-post`, used for model-invocation latency and per-inference energy.
- **Complete-event window:** `event-pre` to `event-post`, retained as supporting telemetry for the complete inference pipeline.

The per-inference energy reported in Table V is obtained from the cumulative counter as:

`Delta E = E_invoke-post - E_invoke-pre`

Mean energy and mean latency are calculated over the same `n = 24` valid post-warm-up events for each model--hardware pair and are reported separately for Replay and Field.

## Baseline IDLE measurements

Baseline current and power are computed from samples collected outside inference windows. They are reported separately from invocation-only energy so that complete-board idle behavior is not conflated with per-inference execution cost.

## Comparable functional scope

The cross-platform experiments adopted a comparable functional scope. Wi-Fi, Bluetooth, and OLED display output were disabled on the WEMOS LOLIN32 because these ESP32-specific functions were not part of the evaluated pipeline and had no corresponding hardware or workload in the evaluated NUCLEO-F411RE configuration. The corresponding optional code paths are excluded from the STM32 builds through PlatformIO build flags.

Both targets retained inference, serial logging, INA219-based electrical measurement, and the sensing functions required by each execution mode. The corresponding WEMOS LOLIN32 experiments were repeated under this controlled configuration for all three evaluated models. These controls reduce the influence of platform-specific peripherals on latency and energy measurements, but the reported results remain platform-specific comparative indicators rather than an absolute ranking of the complete capabilities of the two boards.

## Electrical schematics

Complete board-level diagrams are provided under `hardware_schematics/` in PDF and editable Visio formats, with one `GPIO.xlsx` workbook per board. Use `docs/HARDWARE_SCHEMATICS.md` for the reviewer verification sequence. The schematics support the pin table above and allow independent inspection of the DHT22, INA219, I2C, power, and serial connections.

## Logical time

`FAST_HOUR_TEST=1` maps one wall-clock minute to one logical hour. This allows the 24-event measurement window to complete without a 24-hour wait.

## Evidence logs

`docs/EVIDENCE_MANIFEST.csv` lists all 12 packaged logs: three models x two targets x REPLAY/FIELD.

## Implementation locations

Latency, memory, current, power, and energy instrumentation is implemented in `src/benchmark.cpp` and configured by `include/config.h`. Sensor behavior and filtering are implemented in `src/sensors.cpp`.

## Interpretation and limitations

The acquisition protocol supports internally consistent comparison across models, modes, and targets under fixed event boundaries. The measurements remain complete-board 5 V rail observations obtained with the stated INA219 configuration and firmware accounting; they are not presented as traceable laboratory-metrology references. Platform-dependent latency, memory, energy, current, and power are supporting deployment evidence and do not alter the contract-conformance decisions.


## Return to the reviewer route

Return to [`REVIEWER_START_HERE.md`](../REVIEWER_START_HERE.md) for the complete inspection order. Use [`EVIDENCE_TRACEABILITY.md`](EVIDENCE_TRACEABILITY.md) to map a new or retained firmware log to the corresponding host references and manuscript evidence.
