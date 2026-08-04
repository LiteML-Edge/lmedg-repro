# Hardware Schematics and GPIO Verification

## Purpose

`hardware_schematics/` contains the board-level connection evidence for the two hardware targets evaluated in the revised manuscript:

- WEMOS LOLIN32 (ESP32);
- NUCLEO-F411RE (ARM Cortex-M4).

The schematics allow reviewers to verify the sensor, power-monitoring, serial, and I2C connections independently of the firmware source.

## Included formats

Each target contains:

- a PDF schematic for direct inspection;
- an editable Microsoft Visio (`.vsdx`) source;
- a `GPIO.xlsx` pin-assignment workbook.

## Locations

### WEMOS LOLIN32

`hardware_schematics/wemos_lolin32/`

- `LiteML_Edge Wemos Lolin32 Electrical Schematic.pdf`
- `LiteML_Edge Wemos Lolin32 Electrical Schematic.vsdx`
- `GPIO.xlsx`

### NUCLEO-F411RE

`hardware_schematics/nucleo_f411re/`

- `LiteML_Edge Nucleo-F411RE Electrical Schematic.pdf`
- `LiteML_Edge Nucleo-F411RE Electrical Schematic.vsdx`
- `GPIO.xlsx`

## Reviewer verification path

1. Open the PDF for the target board.
2. Compare the DHT22, INA219, I2C, power, and serial connections with the pin table in `docs/FIRMWARE_BUILD_AND_HARDWARE.md`.
3. Compare the pin definitions and target flags with the corresponding firmware project under `firmwares/`.
4. Use the editable Visio source and GPIO workbook when a detailed connection audit is required.

These files document the experimental wiring. They do not replace the firmware configuration, the PlatformIO environment, or the physical-board build and upload checks.
