# System Reports

> **Navigation:** Return to [`REVIEWER_START_HERE.md`](../../REVIEWER_START_HERE.md) for the complete review route or [`environment_reports/README.md`](../README.md) for the parent report index. These files describe the captured host and software environment; they are not experimental result files.


This subdirectory records the software and host environment used for the LiteML-Edge experiments. The reviewer-facing parent index is `environment_reports/README.md`.

## Host and core environment

- `windows_host_report.txt`: Windows 11 edition, build, CPU, and physical memory.
- `windows_system_info.txt`: operating-system details captured from Windows Management Instrumentation.
- `platformio_system_info.txt`: unedited PlatformIO system summary.
- `packages_report.md`: Python packages and versions from the project virtual environment.

The line `Platform: Windows-10` in `platformio_system_info.txt` is the platform-family label emitted by PlatformIO/Python. The host report identifies the actual operating system as **Microsoft Windows 11 Home Single Language, version 10.0.26100, 64-bit**.

## PlatformIO-managed packages

- `mlp_packages.txt`
- `conv1d_tiny_packages.txt`
- `lstm_packages.txt`

Each file records both firmware environments in the corresponding project, including the exact platform, framework, tool, toolchain, and PlatformIO-managed library versions. The captured platform versions are:

- Espressif 32 platform: `espressif32 6.10.0`;
- ST STM32 platform: `ststm32 19.7.0`;
- PlatformIO Core: `6.1.19`;
- Python: `3.10.0`.

## Project-local libraries

- `mlp_local_libraries.txt`
- `conv1d_tiny_local_libraries.txt`
- `lstm_local_libraries.txt`

These reports inventory libraries stored directly under each firmware project's `lib/` directory, including TensorFlow Lite Micro, FlatBuffers, DHT sensor support, and Adafruit Unified Sensor. They complement the PlatformIO-managed dependency reports rather than duplicate them.

Absolute paths appearing in the captured reports identify the experiment workstation at capture time. Reproduction commands use repository-relative paths documented in the root README and `docs/SETUP_AND_DEPENDENCIES.md`.
