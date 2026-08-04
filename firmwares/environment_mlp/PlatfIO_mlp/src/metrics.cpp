/*
 * ============================================================================
 *  metrics.cpp
 *  LiteML-Edge Rolling Metrics Module
 * ============================================================================
 *
 *  Purpose
 *  -------
 *  This module computes rolling evaluation metrics for LiteML-Edge firmware,
 *  enabling direct comparison between embedded inference behavior and the
 *  training/evaluation protocol used offline.
 *
 *  Main responsibilities
 *  ---------------------
 *  1. Maintain a fixed-size sliding window of recent valid prediction samples.
 *  2. Compute per-target metrics for temperature and humidity.
 *  3. Compute global aggregated metrics aligned with the training script.
 *  4. Enforce auditable selection rules for which events are counted.
 *
 *  Evaluation contract
 *  -------------------
 *  The rolling window is intended to preserve "apples-to-apples" comparison
 *  with the training-side evaluation protocol.
 *
 *  The current implementation:
 *  - counts only samples for which a real Invoke() occurred,
 *  - counts only hourly rollover samples when is_rollover == true,
 *  - ignores non-hourly events,
 *  - ignores samples containing invalid numeric values,
 *  - computes metrics only over the active sliding window.
 *
 *  Metric definitions
 *  ------------------
 *  Per variable:
 *    - MAE  = mean absolute error
 *    - RMSE = root mean squared error
 *    - R²   = coefficient of determination
 *
 *  Global combined metric:
 *    - MAE  = average of MAE_T and MAE_H
 *    - RMSE = sqrt( (MSE_T + MSE_H) / 2 )
 *    - R²   = average of R²_T and R²_H
 *
 *  Numerical notes
 *  ---------------
 *  - R² is reported as undefined (NaN) when fewer than 2 samples exist.
 *  - R² is also reported as undefined when target variance in the window is
 *    too small, to avoid unstable or misleading values.
 *  - Strict-training mode may require a fully populated window before any
 *    metric is reported.
 *
 *  Important
 *  ---------
 *  This version standardizes the module for IEEE LATAM-ready readability and
 *  auditability while preserving the original functional behavior and data flow.
 * ============================================================================
 */

// metrics.cpp
#include "metrics.h"
#include <math.h>
#include <config.h>

// --------------------------------------------------------------
// Sliding (rolling) window to preserve "apples-to-apples"
// comparison with the training-side evaluation protocol.
// --------------------------------------------------------------
#ifndef METRICS_WINDOW_SIZE
#define METRICS_WINDOW_SIZE 24
#endif

/**
 * @brief Stores the metric contribution of one valid sample.
 *
 * Each entry keeps absolute error, squared error, target value, and squared
 * target value for both temperature and humidity.
 */
struct MetricEntry {
  bool   valid;
  float abs_T, sq_T, y_T, y_sq_T;
  float abs_H, sq_H, y_H, y_sq_H;
};

static MetricEntry   g_buf[METRICS_WINDOW_SIZE];
static int           g_idx_write = 0;
static unsigned long g_n_samples = 0;

// Rolling sums per variable.
static float g_sum_abs_T   = 0.0f;
static float g_sum_sqerr_T = 0.0f;
static float g_sum_y_T     = 0.0f;
static float g_sum_y_sq_T  = 0.0f;

static float g_sum_abs_H   = 0.0f;
static float g_sum_sqerr_H = 0.0f;
static float g_sum_y_H     = 0.0f;
static float g_sum_y_sq_H  = 0.0f;

/**
 * @brief Computes MAE, RMSE, and R² for a single target variable.
 * @param n          Number of valid samples.
 * @param sum_abs    Sum of absolute errors.
 * @param sum_sqerr  Sum of squared errors.
 * @param sum_y      Sum of target values.
 * @param sum_y_sq   Sum of squared target values.
 * @param[out] mae   Mean absolute error.
 * @param[out] rmse  Root mean squared error.
 * @param[out] r2    Coefficient of determination.
 *
 * Behavior:
 * - If n == 0, all outputs are NaN.
 * - If n < 2, MAE and RMSE are computed, but R² is reported as NaN.
 * - If target variance is too small, R² is reported as NaN.
 */
static inline void compute_single_metrics(unsigned long n,
                                          float sum_abs,
                                          float sum_sqerr,
                                          float sum_y,
                                          float sum_y_sq,
                                          float &mae,
                                          float &rmse,
                                          float &r2) {
  if (n == 0) {
    mae  = NAN;
    rmse = NAN;
    r2   = NAN;
    return;
  }

  // R² requires at least 2 samples to be interpretable.
  if (n < 2) {
    mae  = sum_abs / ((float)n);
    rmse = sqrtf(sum_sqerr / ((float)n));
    r2   = NAN;
    if (!isfinite(rmse)) rmse = NAN;
    return;
  }

  const float n_d = (float)n;

  // MAE
  mae = sum_abs / n_d;

  // RMSE
  const float mse = sum_sqerr / n_d;
  rmse = sqrtf(mse);

  // R² = 1 - SS_res / SS_tot
  // SS_res = sum_sqerr
  // SS_tot = sum((y_true - mean_y)^2) = sum_y_sq - (sum_y^2 / n)
  const float ss_res = sum_sqerr;
  const float ss_tot = sum_y_sq - (sum_y * sum_y) / n_d;

  if (ss_tot <= 1e-6f) {
    // Very low variance in the window makes R² unstable or misleading.
    // To stay aligned with the training-side interpretation, mark it undefined.
    r2 = NAN;
  } else {
    r2 = 1.0f - (ss_res / ss_tot);
  }

  if (!isfinite(rmse)) rmse = NAN;
  if (!isfinite(r2))   r2   = NAN;
}

/**
 * @brief Resets the full rolling-metrics state.
 *
 * Clears the circular buffer, resets the write pointer, resets the current
 * number of samples, and zeroes all rolling sums.
 */
void reset_metrics() {
  g_idx_write = 0;
  g_n_samples = 0;

  for (int i = 0; i < METRICS_WINDOW_SIZE; ++i) {
    g_buf[i].valid  = false;
    g_buf[i].abs_T  = 0.0f;
    g_buf[i].sq_T   = 0.0f;
    g_buf[i].y_T    = 0.0f;
    g_buf[i].y_sq_T = 0.0f;
    g_buf[i].abs_H  = 0.0f;
    g_buf[i].sq_H   = 0.0f;
    g_buf[i].y_H    = 0.0f;
    g_buf[i].y_sq_H = 0.0f;
  }

  g_sum_abs_T   = 0.0f;
  g_sum_sqerr_T = 0.0f;
  g_sum_y_T     = 0.0f;
  g_sum_y_sq_T  = 0.0f;

  g_sum_abs_H   = 0.0f;
  g_sum_sqerr_H = 0.0f;
  g_sum_y_H     = 0.0f;
  g_sum_y_sq_H  = 0.0f;
}

/**
 * @brief Updates the rolling window with one additional sample.
 * @param t_true      Ground-truth indoor temperature.
 * @param h_true      Ground-truth indoor humidity.
 * @param t_pred      Predicted indoor temperature.
 * @param h_pred      Predicted indoor humidity.
 * @param is_rollover True only for hourly rollover events.
 * @param invoked     True only if a real Invoke() occurred.
 *
 * Important:
 * - Only real inference events are counted.
 * - Only hourly samples are counted, to match the training dataset protocol.
 * - Non-hourly events are excluded.
 * - Any sample containing NaN is ignored.
 */
void update_metrics(float t_true, float h_true,
                    float t_pred, float h_pred,
                    bool is_rollover,
                    bool invoked) {
  // Absolute gating: only count samples when a real Invoke() occurred.
  if (!invoked) return;

  if (!is_rollover) {
    // Only logical-hour rollover samples enter the rolling window.
    // Non-hourly events are intentionally excluded by contract.
    return;
  }

  // NaN protection: ignore invalid samples.
  if (isnan(t_true) || isnan(h_true) || isnan(t_pred) || isnan(h_pred)) {
    return;
  }

  // Compute prediction errors.
  const float err_T = t_pred - t_true;
  const float err_H = h_pred - h_true;

  // Build the new entry.
  MetricEntry e{};
  e.valid   = true;

  e.abs_T   = fabsf(err_T);
  e.sq_T    = err_T * err_T;
  e.y_T     = t_true;
  e.y_sq_T  = t_true * t_true;

  e.abs_H   = fabsf(err_H);
  e.sq_H    = err_H * err_H;
  e.y_H     = h_true;
  e.y_sq_H  = h_true * h_true;

  // Remove the previous contribution, if the slot is already occupied.
  MetricEntry &old = g_buf[g_idx_write];
  if (old.valid) {
    g_sum_abs_T   -= old.abs_T;
    g_sum_sqerr_T -= old.sq_T;
    g_sum_y_T     -= old.y_T;
    g_sum_y_sq_T  -= old.y_sq_T;

    g_sum_abs_H   -= old.abs_H;
    g_sum_sqerr_H -= old.sq_H;
    g_sum_y_H     -= old.y_H;
    g_sum_y_sq_H  -= old.y_sq_H;
  }

  // Write the new entry into the circular buffer.
  g_buf[g_idx_write] = e;

  // Add the new contribution to the rolling sums.
  g_sum_abs_T   += e.abs_T;
  g_sum_sqerr_T += e.sq_T;
  g_sum_y_T     += e.y_T;
  g_sum_y_sq_T  += e.y_sq_T;

  g_sum_abs_H   += e.abs_H;
  g_sum_sqerr_H += e.sq_H;
  g_sum_y_H     += e.y_H;
  g_sum_y_sq_H  += e.y_sq_H;

  // Advance circular write index.
  g_idx_write = (g_idx_write + 1) % METRICS_WINDOW_SIZE;

  // Update the number of active samples in the window (saturates at window size).
  if (g_n_samples < (unsigned long)METRICS_WINDOW_SIZE) {
    g_n_samples += 1;
  }
}

/**
 * @brief Returns global combined metrics for temperature and humidity.
 * @param[out] mae  Combined MAE.
 * @param[out] rmse Combined RMSE.
 * @param[out] r2   Combined R².
 *
 * Aggregation follows the training-side script:
 * - MAE  = mean(MAE_T, MAE_H)
 * - RMSE = sqrt( (MSE_T + MSE_H) / 2 )
 * - R²   = mean(R²_T, R²_H)
 *
 * All values are computed strictly over the active rolling window.
 */
void get_metrics(float &mae, float &rmse, float &r2) {
#if METRICS_STRICT_TRAINING
  if (g_n_samples < METRICS_WINDOW_SIZE) { mae=NAN; rmse=NAN; r2=NAN; return; }
#endif

  if (g_n_samples == 0) {
    mae  = NAN;
    rmse = NAN;
    r2   = NAN;
    return;
  }

  float mae_T = 0.0f, rmse_T = 0.0f, r2_T = 0.0f;
  float mae_H = 0.0f, rmse_H = 0.0f, r2_H = 0.0f;

  // Per-variable metrics.
  compute_single_metrics(g_n_samples, g_sum_abs_T, g_sum_sqerr_T, g_sum_y_T, g_sum_y_sq_T,
                         mae_T, rmse_T, r2_T);

  compute_single_metrics(g_n_samples, g_sum_abs_H, g_sum_sqerr_H, g_sum_y_H, g_sum_y_sq_H,
                         mae_H, rmse_H, r2_H);

  mae = 0.5f * (mae_T + mae_H);

  const float mse_T = rmse_T * rmse_T;
  const float mse_H = rmse_H * rmse_H;
  rmse = sqrtf(0.5f * (mse_T + mse_H));

  r2 = 0.5f * (r2_T + r2_H);
}

/**
 * @brief Returns rolling metrics for temperature only.
 * @param[out] mae_T  Temperature MAE.
 * @param[out] rmse_T Temperature RMSE.
 * @param[out] r2_T   Temperature R².
 */
void get_metrics_T(float &mae_T, float &rmse_T, float &r2_T) {
#if METRICS_STRICT_TRAINING
  if (g_n_samples < METRICS_WINDOW_SIZE) { mae_T=NAN; rmse_T=NAN; r2_T=NAN; return; }
#endif
  compute_single_metrics(g_n_samples, g_sum_abs_T, g_sum_sqerr_T, g_sum_y_T, g_sum_y_sq_T,
                         mae_T, rmse_T, r2_T);
}

/**
 * @brief Returns rolling metrics for humidity only.
 * @param[out] mae_H  Humidity MAE.
 * @param[out] rmse_H Humidity RMSE.
 * @param[out] r2_H   Humidity R².
 */
void get_metrics_H(float &mae_H, float &rmse_H, float &r2_H) {
#if METRICS_STRICT_TRAINING
  if (g_n_samples < METRICS_WINDOW_SIZE) { mae_H=NAN; rmse_H=NAN; r2_H=NAN; return; }
#endif
  compute_single_metrics(g_n_samples, g_sum_abs_H, g_sum_sqerr_H, g_sum_y_H, g_sum_y_sq_H,
                         mae_H, rmse_H, r2_H);
}

/**
 * @brief Returns the number of valid samples currently stored in the rolling window.
 * @return Active sample count.
 */
unsigned long metrics_samples_total() { return g_n_samples; }

/**
 * @brief Returns the number of valid temperature samples in the rolling window.
 * @return Active sample count.
 *
 * In the current implementation, temperature and humidity share the same window
 * and valid-sample count.
 */
unsigned long metrics_samples_T()     { return g_n_samples; }

/**
 * @brief Returns the number of valid humidity samples in the rolling window.
 * @return Active sample count.
 *
 * In the current implementation, temperature and humidity share the same window
 * and valid-sample count.
 */
unsigned long metrics_samples_H()     { return g_n_samples; }