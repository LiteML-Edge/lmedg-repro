#pragma once
#include <stdint.h>
#include <stdbool.h>

#include "sensors.h"   // SensorData
#include "benchmark.h" // BenchPower

/*
 * ============================================================================
 *  inference.h
 *  LiteML-Edge Public Inference Interface
 * ============================================================================
 *
 *  Purpose
 *  -------
 *  This header defines the public inference API used by LiteML-Edge firmware.
 *  It exposes the initialization, runtime inference, and optional self-test
 *  interfaces required by the application layer.
 *
 *  Scope
 *  -----
 *  The API is designed to support embedded TinyML inference with an auditable
 *  and reproducible contract, including:
 *
 *  - model initialization
 *  - window-based inference execution
 *  - residual-to-absolute output reconstruction
 *  - latency reporting
 *  - optional self-test validation
 *
 *  LiteML-Edge contract
 *  --------------------
 *  The residual output bounds DY_MIN / DY_MAX exported by
 *  scalers_exported_*.h must correspond to the residual targets
 *  (ΔT_in, ΔH_in) obtained from scaler_y (data_min_ / data_max_),
 *  so that the inverse transformation performed on the MCU reproduces
 *  the training-side residual decoding contract.
 *
 *  Integration note
 *  ----------------
 *  Reconstruction of absolute Tin / Hin values may depend on lag-1 state.
 *  The function sensors_get_prev_T_H() can be implemented in sensors.cpp.
 *  If it is not available or does not return valid data, inference.cpp may
 *  fall back to the current Tin / Hin path, depending on the active mode.
 *
 *  Design goals
 *  ------------
 *  - Keep the interface lightweight and portable.
 *  - Preserve compatibility with the LiteML-Edge residual contract.
 *  - Support auditable runtime behavior for IEEE LATAM-style reporting.
 * ============================================================================
 */

#ifndef HAS_INFERENCE_SELFTEST
  #define HAS_INFERENCE_SELFTEST  1
#endif

// -----------------------------------------------------------------------------
// Debug I/O semantic contract
// -----------------------------------------------------------------------------
// These constants and records standardize the meaning of debug fields emitted by
// inference.cpp. The naming is stage-based rather than dtype-based, so the same
// contract applies to MLP, Conv1D, and LSTM models.

static constexpr int LITEML_DEBUG_NUM_FEATURES = 12;
static constexpr int LITEML_DEBUG_WINDOW_STEPS = 24;
static constexpr int LITEML_DEBUG_WINDOW_SIZE =
    LITEML_DEBUG_NUM_FEATURES * LITEML_DEBUG_WINDOW_STEPS;

enum FeatureIndex : int {
  F_TOUT = 0,
  F_HOUT = 1,
  F_TIN_LAG1 = 2,
  F_HIN_LAG1 = 3,
  F_TOUT_LAG1 = 4,
  F_HOUT_LAG1 = 5,
  F_TIN_LAG2 = 6,
  F_HIN_LAG2 = 7,
  F_SIN_HOUR = 8,
  F_COS_HOUR = 9,
  F_WEEKDAY = 10,
  F_MONTH = 11,
};

struct DebugInputRecord {
  int idx;
  uint32_t epoch;
  int step;

  float gt_Tin_true;
  float gt_Hin_true;

  float pre_raw_Tout;
  float pre_raw_Hout;
  float pre_raw_Tin;
  float pre_raw_Hin;

  float pre_smooth_Tout;
  float pre_smooth_Hout;
  float pre_smooth_Tin;
  float pre_smooth_Hin;

  float state_Tout_phys_raw;
  float state_Hout_phys_raw;
  float state_Tin_lag1_phys_raw;
  float state_Hin_lag1_phys_raw;
  float state_Tout_lag1_phys_raw;
  float state_Hout_lag1_phys_raw;
  float state_Tin_lag2_phys_raw;
  float state_Hin_lag2_phys_raw;
  float state_sin_hour;
  float state_cos_hour;
  float state_weekday;
  float state_month;

  float in_f_phys_raw[LITEML_DEBUG_NUM_FEATURES];
  float in_f_phys_clip[LITEML_DEBUG_NUM_FEATURES];
  float in_f_scaled[LITEML_DEBUG_NUM_FEATURES];
  float in_x_float[LITEML_DEBUG_NUM_FEATURES];
};

struct DebugOutputRecord {
  int idx;
  uint32_t epoch;

  float out_o0_tensor;
  float out_o1_tensor;
  float out_o0_float;
  float out_o1_float;

  float y_T_scaled;
  float y_H_scaled;

  float d_T_pred;
  float d_H_pred;

  float p_Tprev_phys;
  float p_Hprev_phys;
  float p_T_pred;
  float p_H_pred;
};

// -----------------------------------------------------------------------------
// Previous-state API for residual reconstruction
// -----------------------------------------------------------------------------
// This function provides access to previous Tin / Hin values (lag-1) used in
// residual reconstruction. It may be implemented in sensors.cpp.
// If not available, inference.cpp can fall back to current Tin / Hin behavior.
bool sensors_get_prev_T_H(float* Tprev, float* Hprev);

// -----------------------------------------------------------------------------
// Public inference API (used by main.cpp)
// -----------------------------------------------------------------------------

/**
 * @brief Result structure returned by one inference execution.
 *
 * Fields include execution status, latency, current measured inputs, and
 * reconstructed absolute predictions.
 */
struct InferResult {
  bool  ok;        // Pipeline status: model ready and tensors valid.
  bool  invoked;   // True only if Invoke() was actually executed.

  uint32_t us;          // Backward-compatible alias of pure Invoke() latency.
  uint32_t us_invoke;   // Pure model execution latency (Invoke only).
  uint32_t us_event;    // End-to-end inference event latency.

  bool  power_valid;            // True if power snapshots were captured.
  float energy_invoke_mWh;      // Energy delta around Invoke() window.
  float energy_event_mWh;       // Energy delta around the full inference event.
  BenchPower pwr_invoke_pre;    // Snapshot immediately before Invoke().
  BenchPower pwr_invoke_post;   // Snapshot immediately after Invoke().
  BenchPower pwr_event_pre;     // Snapshot before the inference event.
  BenchPower pwr_event_post;    // Snapshot after the inference event.

  float Tin;       // Current measured Tin input.
  float Hin;       // Current measured Hin input.
  float Tin_pred;  // Reconstructed absolute Tin prediction.
  float Hin_pred;  // Reconstructed absolute Hin prediction.
};

/**
 * @brief Result structure returned by the optional inference self-test.
 */
struct InferenceSelfTestResult {
  bool  ok;        // Self-test pass / fail status.
  float y0;        // First decoded output value.
  float y1;        // Second decoded output value.
};

/**
 * @brief Initializes the inference engine and internal runtime state.
 * @return true if initialization succeeds, false otherwise.
 */
bool init_inference();

/**
 * @brief Runs one inference step using the provided sensor data.
 * @param d Current sensor and derived temporal data.
 * @return InferResult containing execution status and predictions.
 */
InferResult run_inference(const SensorData& d);

#if defined(HAS_INFERENCE_SELFTEST)
/**
 * @brief Executes the optional embedded inference self-test.
 * @return InferenceSelfTestResult with decoded outputs and pass / fail status.
 */
InferenceSelfTestResult inference_selftest();
#endif