# LiteML-Edge Firmware — LSTM

This PlatformIO project deploys the LSTM model on:

- WEMOS LOLIN32 / ESP-WROOM-32: `lolin32_lstm`
- NUCLEO-F411RE / STM32F411RE: `nucleo_f411re_lstm`

## Select the execution mode

Edit `include/config.h` before building:

```cpp
#define LITEML_MODE LITEML_MODE_REPLAY
```

for deterministic Python--firmware validation, or:

```cpp
#define LITEML_MODE LITEML_MODE_FIELD
```

for physical DHT22 acquisition.

## Build

```powershell
pio run -d firmwares/environment_lstm/PlatfIO_lstm -e lolin32_lstm
pio run -d firmwares/environment_lstm/PlatfIO_lstm -e nucleo_f411re_lstm
```

## Upload

```powershell
pio run -d firmwares/environment_lstm/PlatfIO_lstm -e lolin32_lstm -t upload
pio run -d firmwares/environment_lstm/PlatfIO_lstm -e nucleo_f411re_lstm -t upload
```

## Monitor

```powershell
python utils/global_utils/pio_monitor.py -d firmwares/environment_lstm/PlatfIO_lstm -e lolin32_lstm --port COM6 --baud 115200 --wait 20
python utils/global_utils/pio_monitor.py -d firmwares/environment_lstm/PlatfIO_lstm -e nucleo_f411re_lstm --port COM8 --baud 115200 --wait 20
```

## Included headers

- model byte array;
- scaler constants;
- 2-seed + 47-row replay dataset.

## Libraries

Local TFLite Micro, FlatBuffers, DHT, and sensor libraries are provided under `lib/`. The PlatformIO-managed library sources used in the validated firmware builds are preserved under `.pio/libdeps` to retain the tested dependency snapshot and protect the reproduction workflow against future dependency unavailability. Missing declared dependencies can also be installed automatically during a normal online build.

See the repository-level `docs/FIRMWARE_BUILD_AND_HARDWARE.md` for wiring, modes, instrumentation, and evidence-log instructions.
