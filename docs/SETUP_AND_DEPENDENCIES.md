# Setup and Dependencies

> **Document role:** Use this file only when preparing a host for command execution or full reconstruction. Package inspection and retained-evidence review do not require installation. Start with [`REVIEWER_START_HERE.md`](../REVIEWER_START_HERE.md); then use [`REVIEWER_VERIFICATION_GUIDE.md`](REVIEWER_VERIFICATION_GUIDE.md) for verification or [`END_TO_END_REPRODUCTION.md`](END_TO_END_REPRODUCTION.md) for a fresh reconstruction.


## Recorded reference environment

- Microsoft Windows 11 Home Single Language, 64-bit, build 26100
- Python 3.10.0
- TensorFlow 2.14.0
- TensorFlow Model Optimization 0.7.5
- PlatformIO Core 6.1.19
- Espressif 32 platform 6.10.0
- ST STM32 platform 19.7.0
- Arduino frameworks and toolchains listed in `environment_reports/system reports/*_packages.txt`

The reviewer-facing index is `environment_reports/README.md`. The captured system environment is stored under `environment_reports/system reports/`: `packages_report.md` records the Python virtual environment; the three `*_packages.txt` files record PlatformIO-managed platforms, frameworks, tools, toolchains, and libraries; and the three `*_local_libraries.txt` files inventory project-local libraries such as TensorFlow Lite Micro, FlatBuffers, and sensor support. The six retained REPLAY build records are stored separately under `environment_reports/final_builds/`.

## Windows setup

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-reproduction.txt
```

## Linux/macOS host-only setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-reproduction.txt
```

Hardware upload protocols and serial-port names are host-specific.

## Training configuration reference

After the environment checks pass, use the [Training configuration summary](END_TO_END_REPRODUCTION.md#training-configuration-summary) before starting a fresh reconstruction. It consolidates the exact retained model architectures, optimizer/loss settings, maximum epochs, batch sizes, callbacks, pruning schedules, calibration sample counts, and TFLite I/O contracts. The canonical scripts listed in that section remain the executable source of truth.

## Sanity checks

```powershell
python --version
python -c "import tensorflow as tf; print(tf.__version__)"
python -m platformio --version
python runner.py --help
```

## Firmware dependencies

Each firmware project contains:

1. project-local libraries under `lib/`, including TensorFlow Lite Micro, FlatBuffers, DHT sensor support, and Adafruit Unified Sensor;
2. PlatformIO-managed library sources preserved under `.pio/libdeps`, including the exact INA219, communication, and display dependencies used in the validated firmware builds.

The recorded versions are listed in `environment_reports/system reports/`. The machine-readable packaged-library inventory is `docs/PLATFORMIO_LIBRARY_MANIFEST.csv`.

## PlatformIO identifiers

`espressif32` is the PlatformIO platform identifier for the ESP32 target family. `ststm32` is the identifier used by the NUCLEO-F411RE environments. The manuscript and experimental tables use the hardware names **WEMOS LOLIN32** and **NUCLEO-F411RE**.

## Serial ports

The retained `platformio.ini` files contain the experiment workstation ports:

- ESP32: `COM6`
- STM32: `COM8`

On another workstation, replace them with the local upload and monitor ports. Serial-port assignments are host settings and are not part of the model contract.
