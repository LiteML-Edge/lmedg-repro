/**
 * ============================================================================
 *  config.h
 *  LiteML-Edge Build, Platform, Benchmark, and Model Configuration
 * ============================================================================
 *
 *  Purpose
 *  -------
 *  This header centralizes the compile-time configuration of the LiteML-Edge
 *  firmware. It defines:
 *
 *  1. Platform and peripheral availability.
 *  2. Runtime operating mode selection (REPLAY vs FIELD).
 *  3. Sensor, display, and time synchronization options.
 *  4. Benchmark instrumentation settings (latency, memory, power).
 *  5. Logical-time behavior for accelerated experiments.
 *  6. Calibration constants and reproducibility guards.
 *  7. Active deployed model identity and associated artifact headers.
 *
 *  Operational rationale
 *  ---------------------
 *  LiteML-Edge supports two main firmware behaviors:
 *
 *  - REPLAY mode:
 *      deterministic execution aligned with exported host-side artifacts,
 *      supporting controlled validation and host-vs-MCU comparison.
 *
 *  - FIELD mode:
 *      real-sensor execution with additional robustness mechanisms suitable
 *      for operational deployment.
 *
 *  This separation allows the same firmware base to support both scientific
 *  reproducibility experiments and real embedded operation.
 *
 *  Reproducibility note
 *  --------------------
 *  The constants defined here are part of the firmware-side experimental
 *  contract. They should remain aligned with the corresponding Python
 *  preprocessing, quantization, replay, and evaluation scripts to preserve
 *  comparability across host and MCU environments.
 *
 *  IEEE LATAM readiness
 *  --------------------
 *  This file is organized to improve traceability, reproducibility, and
 *  artifact clarity, which are relevant for paper-oriented experimental
 *  reporting and supplementary material preparation.
 * ============================================================================
 */

#pragma once

// ============================================================================
// Automatic platform selection based on the PlatformIO environment
// ============================================================================

#if defined(ARDUINO_ARCH_ESP32)

  #define LITEML_TARGET_ESP32        1
  #define LITEML_TARGET_STM32F411RE  0

  #define LITEML_PLATFORM_NAME       "Lolin32 / ESP-WROOM-32"
  #define LITEML_PLATFORM_HAS_WIFI   1
  #define LITEML_PLATFORM_HAS_BLE    1

#elif defined(ARDUINO_ARCH_STM32)

  #define LITEML_TARGET_ESP32        0
  #define LITEML_TARGET_STM32F411RE  1

  #define LITEML_PLATFORM_NAME       "NUCLEO-F411RE / STM32F411RE"
  #define LITEML_PLATFORM_HAS_WIFI   0
  #define LITEML_PLATFORM_HAS_BLE    0

#else

  #error "Unsupported platform. Use lolin32_C or nucleo_f411re."

#endif

// ============================================================================
// Platform and core resources
// ============================================================================

// Backward compatibility with names used by the existing source files.
#define LITEML_PLATFORM_ESP32 \
        LITEML_TARGET_ESP32

#define LITEML_PLATFORM_STM32F411RE \
        LITEML_TARGET_STM32F411RE

#ifndef LITEML_CPU_MHZ
  #if LITEML_TARGET_ESP32
    /// Existing Lolin32 CPU frequency.
    #define LITEML_CPU_MHZ 240
  #elif LITEML_TARGET_STM32F411RE
    /// NUCLEO-F411RE clock configured by the official board variant.
    #define LITEML_CPU_MHZ 100
  #endif
#endif

// Primary peripheral policy
/// OLED boot policy during REPLAY mode: 0 = start disabled, 1 = start enabled.
// config.h
#ifndef OLED_IN_REPLAY
  #define OLED_IN_REPLAY 0
#endif

#ifndef OLED_IN_FIELD
  #define OLED_IN_FIELD  0
#endif

#ifndef USE_DHT22
  /// Enables physical DHT22 sensor support.
  #define USE_DHT22      1
#endif

#ifndef USE_OLED
  /// Enables OLED UI support.
  #define USE_OLED       1
#endif

#ifndef USE_WIFI_TIME
  /// Enables Wi-Fi/NTP-based time synchronization.
  #define USE_WIFI_TIME 0
#endif

#ifndef USE_BLUETOOTH
  #define USE_BLUETOOTH 0
#endif

#if LITEML_TARGET_STM32F411RE && USE_WIFI_TIME
  #error "USE_WIFI_TIME requires external Wi-Fi hardware on STM32F411RE."
#endif

#if LITEML_TARGET_STM32F411RE && USE_BLUETOOTH
  #error "USE_BLUETOOTH requires external Bluetooth hardware on STM32F411RE."
#endif

// ============================================================================
// Build mode selection: REPLAY (host-aligned) vs FIELD (robust sensor runtime)
// ============================================================================

/// Deterministic replay mode for host-vs-MCU validation.
#define LITEML_MODE_REPLAY 1

/// Field mode for real-sensor operation with additional robustness behavior.
#define LITEML_MODE_FIELD  2

// Select the active build mode here:
//   - LITEML_MODE_REPLAY : maximum alignment with host-side scripts
//   - LITEML_MODE_FIELD  : additional filtering/robustness for field use
#ifndef LITEML_MODE
  #define LITEML_MODE LITEML_MODE_REPLAY //LITEML_MODE_FIELD //LITEML_MODE_REPLAY
#endif

#if (LITEML_MODE == LITEML_MODE_REPLAY)
  /// Compile-time boolean indicating deterministic replay behavior.
  #define LITEML_REPLAY 1
#else
  /// Compile-time boolean indicating field/runtime behavior.
  #define LITEML_REPLAY 0
#endif

#ifndef METRICS_STRICT_TRAINING
  /// When enabled, metrics are considered valid only after the full window is filled.
  #define METRICS_STRICT_TRAINING 1
#endif

#ifndef FAST_HOUR_TEST
  /// Compresses logical time for accelerated testing: 1 minute = 1 logical hour.
  #define FAST_HOUR_TEST 1
#endif

#ifndef BENCH_EVENT_STRETCH_US
  /// Artificial event stretching in microseconds, used only for calibration procedures.
  #define BENCH_EVENT_STRETCH_US 0
#endif

// ============================================================================
// Benchmark instrumentation
// ============================================================================

#ifndef BENCH_ENABLE_LATENCY
  /// Enables latency measurement instrumentation.
  #define BENCH_ENABLE_LATENCY  1
#endif

#ifndef BENCH_ENABLE_MEMORY
  /// Enables memory usage instrumentation.
  #define BENCH_ENABLE_MEMORY   1
#endif

#ifndef BENCH_ENABLE_POWER
  /// Enables power/energy instrumentation.
  #define BENCH_ENABLE_POWER    1
#endif

// Standardized power measurement backends
/// No power backend.
#define BENCH_PWR_NONE     0

/// INA219-based power backend.
#define BENCH_PWR_INA219   1

#ifndef BENCH_PWR_BACKEND
  /// Default power backend used by the benchmark subsystem.
  #define BENCH_PWR_BACKEND BENCH_PWR_INA219
#endif

// ============================================================================
// Pinout and device configuration
// ============================================================================

#if USE_OLED || (BENCH_ENABLE_POWER && (BENCH_PWR_BACKEND == BENCH_PWR_INA219))
  #if LITEML_TARGET_ESP32
    #ifndef OLED_SDA_PIN
      /// Existing Lolin32 OLED/INA219 shared I2C SDA pin.
      #define OLED_SDA_PIN  5
    #endif
    #ifndef OLED_SCL_PIN
      /// Existing Lolin32 OLED/INA219 shared I2C SCL pin.
      #define OLED_SCL_PIN  4
    #endif
  #elif LITEML_TARGET_STM32F411RE
    #ifndef OLED_SDA_PIN
      /// NUCLEO-F411RE default Arduino I2C SDA pin.
      #define OLED_SDA_PIN  SDA
    #endif
    #ifndef OLED_SCL_PIN
      /// NUCLEO-F411RE default Arduino I2C SCL pin.
      #define OLED_SCL_PIN  SCL
    #endif
  #endif
#endif

#if USE_DHT22
  // Note: DHTTYPE = DHT22 is defined in <DHT.h>; sensors.cpp includes that header.
  #if LITEML_TARGET_ESP32
    #ifndef PIN_DHT_OUT
      /// Existing Lolin32 external/environment sensor data pin.
      #define PIN_DHT_OUT   16
    #endif
    #ifndef PIN_DHT_IN
      /// Existing Lolin32 internal/enclosure sensor data pin.
      #define PIN_DHT_IN    25
    #endif
  #elif LITEML_TARGET_STM32F411RE
    #ifndef PIN_DHT_OUT
      /// NUCLEO-F411RE Arduino-header pin proposed for the external DHT22.
      #define PIN_DHT_OUT   D6
    #endif
    #ifndef PIN_DHT_IN
      /// NUCLEO-F411RE Arduino-header pin proposed for the internal DHT22.
      #define PIN_DHT_IN    D7
    #endif
  #endif
  #ifndef DHTTYPE
    /// DHT sensor family used by the firmware.
    #define DHTTYPE       DHT22
  #endif
#endif

// ============================================================================
// NTP and timezone configuration
// ============================================================================

#if USE_WIFI_TIME
  #ifndef WIFI_SSID
    /// Wi-Fi SSID used for optional NTP synchronization.
    #define WIFI_SSID  "YOUR_WIFI_SSID"
  #endif

  #ifndef WIFI_PSK
    /// Wi-Fi password used for optional NTP synchronization.
    #define WIFI_PSK   "YOUR_WIFI_PASSWORD"
  #endif

  #ifndef WIFI_CONNECT_TIMEOUT_MS
    /// Wi-Fi connection timeout in milliseconds.
    #define WIFI_CONNECT_TIMEOUT_MS 10000
  #endif

  #ifndef CONFIG_GMT_OFFSET_SEC
    /// GMT offset in seconds for Manaus local time.
    #define CONFIG_GMT_OFFSET_SEC       (-4 * 3600)
    #ifdef LITEML_DEFINE_TZ_MACROS
      #define GMT_OFFSET_SEC CONFIG_GMT_OFFSET_SEC
    #endif
  #endif

  #ifndef CONFIG_DAYLIGHT_OFFSET_SEC
    /// Daylight saving offset in seconds.
    #define CONFIG_DAYLIGHT_OFFSET_SEC  0
    #ifdef LITEML_DEFINE_TZ_MACROS
      #define DAYLIGHT_OFFSET_SEC CONFIG_DAYLIGHT_OFFSET_SEC
    #endif
  #endif
#endif

// ============================================================================
// Logical-time configuration (fallback when NTP is not available)
// ============================================================================

#if FAST_HOUR_TEST
  /// Accelerated logical-hour period: 1 minute = 1 hour.
  #define HOUR_PERIOD_MS  60000UL //#define HOUR_PERIOD_MS  60000UL
#else
  /// Real-time hour period.
  #define HOUR_PERIOD_MS  3600000UL
#endif

#ifndef LITEML_REPLAY_HOUR_PERIOD_MS
  /// Backward-compatibility alias for older code.
  #define LITEML_REPLAY_HOUR_PERIOD_MS HOUR_PERIOD_MS
#endif

// ============================================================================
// Humidity calibration constants
// ============================================================================

#ifndef CAL_H_IN_OFFSET
  /// Fixed calibration offset for internal humidity.
  #define CAL_H_IN_OFFSET  (-2.5f)
#endif

#ifndef CAL_H_OUT_OFFSET
  /// Fixed calibration offset for external humidity.
  #define CAL_H_OUT_OFFSET (+2.5f)
#endif

#ifndef AUTO_H_BIAS
  /// Enables or disables automatic humidity bias correction.
  #define AUTO_H_BIAS 0
#endif

// ============================================================================
// INA219 calibration and reproducibility parameters
// ============================================================================

// Official Adafruit_INA219 calibration presets
/// INA219 preset: 32 V / 2 A.
#define BENCH_INA219_CALIB_32V_2A     0

/// INA219 preset: 32 V / 1 A.
#define BENCH_INA219_CALIB_32V_1A     1

/// INA219 preset: 16 V / 400 mA.
#define BENCH_INA219_CALIB_16V_400mA  2

#ifndef BENCH_INA219_CALIB
  /// Default INA219 calibration preset used in this build.
  #define BENCH_INA219_CALIB BENCH_INA219_CALIB_16V_400mA //BENCH_INA219_CALIB_32V_2A
#endif

#ifndef BENCH_INA219_SHUNT_OHMS
  /// Assumed INA219 shunt resistance (typical breakout value).
  #define BENCH_INA219_SHUNT_OHMS 0.1f
#endif

#ifndef BENCH_INA219_LOG_CALIB
  /// Enables calibration reporting during boot for reproducibility logging.
  #define BENCH_INA219_LOG_CALIB 1
#endif

// ============================================================================
// Consistency guards
// ============================================================================

// If power measurement is enabled, a valid backend must be selected.
#if BENCH_ENABLE_POWER
  #if (BENCH_PWR_BACKEND!=BENCH_PWR_NONE) && \
      (BENCH_PWR_BACKEND!=BENCH_PWR_INA219)
    #error "Invalid BENCH_PWR_BACKEND. Use NONE/INA219/INA226/SHUNT_ADC."
  #endif
#endif

// ============================================================================
// LiteML-Edge model and metric configuration (MCU-side contract)
// ============================================================================
//
// These parameters must mirror the corresponding Python scripts:
//
//   - environment_quantized_model.py
//   - environment_quantized_model_Conv1D_Tiny.py
//   - environment_quantized_model_lstm.py
//
// The objective is to keep the firmware-side model identity, artifact
// selection, and evaluation assumptions aligned with the host-side pipeline,
// supporting fair host-vs-MCU comparison and artifact reproducibility.
//

// Active model identifiers
/// MLP model identifier.
#define LITEML_MODEL_MLP            0

/// LSTM model identifier.
#define LITEML_MODEL_LSTM           1

/// Conv1D Tiny model identifier.
#define LITEML_MODEL_CONV1D_TINY    2

// Select the active deployed model for this build.
#ifndef LITEML_MODEL_ID
  #define LITEML_MODEL_ID LITEML_MODEL_MLP
#endif

#if   (LITEML_MODEL_ID == LITEML_MODEL_MLP)
  /// Human-readable model name.
  #define LITEML_MODEL_NAME            "MLP"
  /// Input sequence length expected by the deployed model.

  #elif (LITEML_MODEL_ID == LITEML_MODEL_LSTM)
  /// Human-readable model name.
  #define LITEML_MODEL_NAME            "LSTM"
  /// Input sequence length expected by the deployed model.
  #define LITEML_MODEL_INPUT_SEQ_LEN   24
 
#elif (LITEML_MODEL_ID == LITEML_MODEL_CONV1D_TINY)
 /// Human-readable model name.
  #define LITEML_MODEL_NAME            "CONV1D_TINY"
  /// Input sequence length expected by the deployed model.
  #define LITEML_MODEL_INPUT_SEQ_LEN   24
#else
  #error "Invalid LITEML_MODEL_ID"
#endif

// ============================================================================