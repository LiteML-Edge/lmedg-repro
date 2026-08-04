/*
 * ============================================================================
 *  sensors.h
 *  LiteML-Edge Public Sensor Interface
 * ============================================================================
 *
 *  Purpose
 *  -------
 *  This header defines the public interface of the LiteML-Edge sensor module.
 *  It provides the data structure and APIs used to:
 *
 *  - acquire filtered environmental readings,
 *  - expose lag-based features aligned with logical-hour snapshots,
 *  - provide time-derived features compatible with training,
 *  - signal runtime state for inference and control logic,
 *  - support REPLAY-based deterministic validation workflows.
 *
 *  Scope
 *  -----
 *  The sensor module is responsible for producing the feature packet consumed
 *  by the inference pipeline. Its output includes:
 *
 *  - current indoor / outdoor readings,
 *  - lag-1 and lag-2 values,
 *  - cyclical and calendar time features,
 *  - pipeline validity and warm-up state,
 *  - optional logical-hour and snapshot identifiers.
 *
 *  Design goals
 *  ------------
 *  - Keep the interface lightweight and portable.
 *  - Preserve compatibility with the LiteML-Edge temporal feature contract.
 *  - Support auditable MCU-side evaluation and IEEE LATAM-ready documentation.
 *
 *  Acquisition note
 *  ----------------
 *  FIELD mode may reinitialize the DHT backend after prolonged invalid reads.
 *  This is part of the acquisition layer only and must not be confused with
 *  unchanged physical values, which remain a valid operating case.
 * ============================================================================
 */
#pragma once

// If the DHT stops responding (invalid readings), force reinitialization.
// This must not be confused with "unchanged value" — the environment may
// legitimately remain stable.
#ifndef DHT_NO_READ_MS
  #define DHT_NO_READ_MS  15000UL
#endif

#include <Arduino.h>
#include "config.h"

// -----------------------------------------------------------------------------
// Optional extra fields in SensorData enabled via config.h:
//   #define SENSOR_DATA_HAS_HOUR_ID
//   #define SENSOR_DATA_HAS_SNAPSHOT_ID
// -----------------------------------------------------------------------------

/**
 * @brief Output structure produced by the sensor module.
 *
 * This structure contains filtered environmental readings, lag features,
 * time features, and status flags required by the LiteML-Edge runtime.
 */
typedef struct {
  // Raw acquisition readings before any contract-side conditioning
  float T_out_raw;   // Outdoor temperature raw (°C)
  float H_out_raw;   // Outdoor relative humidity raw (%RH)
  float T_in_raw;    // Indoor temperature raw (°C)
  float H_in_raw;    // Indoor relative humidity raw (%RH)

  // Smoothed / conditioned readings after the common acquisition contract
  float T_out_smooth;   // Outdoor temperature after conditioning (°C)
  float H_out_smooth;   // Outdoor relative humidity after conditioning (%RH)
  float T_in_smooth;    // Indoor temperature after conditioning (°C)
  float H_in_smooth;    // Indoor relative humidity after conditioning (%RH)

  // Filtered current readings exported to the inference pipeline
  float T_out;   // Outdoor temperature (°C)
  float H_out;   // Outdoor relative humidity (%RH)
  float T_in;    // Indoor temperature (°C)
  float H_in;    // Indoor relative humidity (%RH)

  // Lag-1 values (snapshot from the previous logical hour)
  float T_in_lag1;   // Indoor temperature lag-1 (°C)
  float H_in_lag1;   // Indoor humidity lag-1 (%RH)
  float T_out_lag1;  // Outdoor temperature lag-1 (°C)
  float H_out_lag1;  // Outdoor humidity lag-1 (%RH)

  // Lag-2 values (snapshot from two logical hours earlier)
  float T_in_lag2;   // Indoor temperature lag-2 (°C)
  float H_in_lag2;   // Indoor humidity lag-2 (%RH)

  // Time features (same convention as training)
  float sin_hour;          // sin(2π*h/24)
  float cos_hour;          // cos(2π*h/24)
  int   weekday_pandas;    // 0 = Mon ... 6 = Sun (pandas-compatible)
  int   month_1_12;        // Month in [1, 12]

  // Runtime state signals for the pipeline
  bool  valid;             // A valid sample exists in this iteration
  bool  have_snapshot;     // Required lag history exists (lag1 + lag2)
  uint8_t warmup_ok_count; // Number of stable readings accumulated (used by main)

  // Optional identifiers for one-shot gating at logical-hour boundaries
  #if defined(SENSOR_DATA_HAS_HOUR_ID)
    int32_t hour_id;       // Current logical-hour ID
  #endif
  #if defined(SENSOR_DATA_HAS_SNAPSHOT_ID)
    int32_t snapshot_id;   // Monotonic counter incremented at each rollover
  #endif

} SensorData;

// -----------------------------------------------------------------------------
// Public sensor-module API
// -----------------------------------------------------------------------------

/**
 * @brief Turns the OLED display off or places it in power-save mode.
 *
 * Effective only when USE_OLED is enabled.
 */
void sensors_ui_off();

/**
 * @brief Turns the OLED display on for temporary use.
 *
 * Effective only when USE_OLED is enabled.
 */
void sensors_ui_on();

/**
 * @brief Disables network resources such as Wi-Fi / BLE when active.
 */
void sensors_network_off();

/**
 * @brief Enables network resources when required by the runtime.
 */
void sensors_network_on();

/**
 * @brief Initializes the sensor module backends.
 * @return true if the base hardware path initialized successfully.
 *
 * Initialization may include:
 * - DHT22 or equivalent sensor backends,
 * - optional OLED setup,
 * - time setup via NTP or millis()-based fallback.
 *
 * The function may still return true when operating in degraded time mode.
 */
bool init_sensors();

/**
 * @brief Reads sensors, updates filtering state, lag features, and time features.
 * @return A filled SensorData structure.
 *
 * If no usable data is currently available, the returned structure will contain
 * valid = false. Once acquisition succeeds, REPLAY and FIELD follow the same
 * post-acquisition evaluation contract.
 */
SensorData read_sensors();

/**
 * @brief Indicates whether REPLAY mode reached end-of-header.
 * @return true when the replay header reached EOF in validation mode.
 */
bool sensors_replay_eof();

// -----------------------------------------------------------------------------
// Auxiliary state signals used by other modules (inference / benchmark / main)
// -----------------------------------------------------------------------------

/**
 * @brief Indicates whether civil time is available from NTP / RTC.
 * @return true if NTP / RTC is valid; false when fallback timing may be active.
 */
bool sensors_time_ready();

/**
 * @brief Indicates whether the required lag history is available.
 * @return true when both lag1 and lag2 exist for the evaluation contract.
 */
bool sensors_have_snapshot();

/**
 * @brief Returns the current logical-hour identifier.
 * @return Logical-hour ID based on NTP / RTC or millis() / FAST_HOUR_TEST.
 */
int32_t sensors_current_hour_id();

/**
 * @brief One-shot latch for logical-hour rollover.
 * @return true only once when the logical hour changes.
 */
bool sensors_hour_rollover();

/**
 * @brief Returns the lag-1 Tin / Hin baseline used by the evaluation contract.
 * @param[out] Tprev Previous indoor temperature.
 * @param[out] Hprev Previous indoor humidity.
 * @return true if the values are valid.
 *
 * Implemented in sensors.cpp.
 */
bool sensors_get_prev_T_H(float* Tprev, float* Hprev);

// -----------------------------------------------------------------------------
// REPLAY support
// -----------------------------------------------------------------------------

/**
 * @brief Returns the epoch of the current replay-header sample.
 * @return Replay-sample epoch, or 0 when unavailable.
 *
 * Useful for 1:1 logs, for example when printing a [TIME] line immediately
 * after an EVENT type=HOUR in REPLAY mode.
 */
time_t sensors_replay_epoch();