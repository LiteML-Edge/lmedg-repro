/*
 * ============================================================================
 *  metrics.h
 *  LiteML-Edge Public Rolling Metrics Interface
 * ============================================================================
 *
 *  Purpose
 *  -------
 *  This header defines the public API for rolling metric computation in the
 *  LiteML-Edge firmware. The interface supports evaluation of reconstructed
 *  absolute predictions against absolute ground-truth references.
 *
 *  Scope
 *  -----
 *  The metrics module is responsible for:
 *
 *  - resetting the rolling evaluation state
 *  - updating the rolling window with new valid samples
 *  - reporting combined and per-variable metrics
 *  - exposing the effective number of samples used in the current window
 *
 *  Evaluation contract
 *  -------------------
 *  The metrics are intended to remain aligned with the training-side protocol.
 *  In practice:
 *
 *  - reference values are absolute Tin / Hin values
 *  - predicted values are reconstructed absolute Tin_pred / Hin_pred values
 *  - only selected events should enter the rolling window
 *  - the active sample count is capped by the configured window size
 *
 *  Global metric definition
 *  ------------------------
 *  The combined metrics follow the same aggregation logic used in the training
 *  script:
 *
 *    - MAE  = mean(MAE_T, MAE_H)
 *    - RMSE = sqrt( (MSE_T + MSE_H) / 2 )
 *    - R²   = mean(R²_T, R²_H)
 *
 *  Windowing note
 *  --------------
 *  All metrics are computed over the active sliding window, whose maximum size
 *  is implementation-defined (default: 24 samples in metrics.cpp).
 *
 *  Design goals
 *  ------------
 *  - Keep the interface lightweight and portable.
 *  - Preserve compatibility with the LiteML-Edge evaluation contract.
 *  - Support IEEE LATAM-ready documentation and auditability.
 * ============================================================================
 */

// -----------------------------------------------------------------------------
// Rolling metrics control
// -----------------------------------------------------------------------------

#ifndef METRICS_H
#define METRICS_H

#include <stdint.h>

/**
 * @brief Resets all metric accumulators and internal rolling-window state.
 *
 * This reinitializes the sliding evaluation window.
 */
void reset_metrics();

/**
 * @brief Updates the rolling metrics state with one sample.
 * @param t_true      Reference indoor temperature in absolute scale.
 * @param h_true      Reference indoor humidity in absolute scale.
 * @param t_pred      Reconstructed predicted temperature in absolute scale.
 * @param h_pred      Reconstructed predicted humidity in absolute scale.
 * @param is_rollover True for EVENT type=HOUR (one logical hourly sample).
 *                    False for any non-hourly event.
 * @param invoked     True only when a real inference Invoke() occurred.
 *
 * Intended usage:
 * - HOUR events enter the rolling window.
 * - Non-hourly events are excluded to preserve alignment with the training protocol.
 */
void update_metrics(float t_true, float h_true,
                    float t_pred, float h_pred,
                    bool is_rollover,
                    bool invoked = true);

// -----------------------------------------------------------------------------
// Combined metrics
// -----------------------------------------------------------------------------

/**
 * @brief Returns combined rolling metrics for T_in and H_in together.
 * @param[out] mae  Combined mean absolute error.
 * @param[out] rmse Combined root mean squared error.
 * @param[out] r2   Combined coefficient of determination.
 *
 * Aggregation follows the training-side script:
 * - MAE  = mean(MAE_T, MAE_H)
 * - RMSE = sqrt( (MSE_T + MSE_H) / 2 )
 * - R²   = mean(R²_T, R²_H)
 *
 * All metrics are computed over the current sliding window, up to N samples
 * (default N = 24).
 */
void get_metrics(float &mae, float &rmse, float &r2);

// -----------------------------------------------------------------------------
// Per-variable metrics
// -----------------------------------------------------------------------------

/**
 * @brief Returns rolling metrics for temperature only (T_in).
 * @param[out] mae_T  Temperature mean absolute error.
 * @param[out] rmse_T Temperature root mean squared error.
 * @param[out] r2_T   Temperature coefficient of determination.
 */
void get_metrics_T(float &mae_T, float &rmse_T, float &r2_T);

/**
 * @brief Returns rolling metrics for humidity only (H_in).
 * @param[out] mae_H  Humidity mean absolute error.
 * @param[out] rmse_H Humidity root mean squared error.
 * @param[out] r2_H   Humidity coefficient of determination.
 */
void get_metrics_H(float &mae_H, float &rmse_H, float &r2_H);

// -----------------------------------------------------------------------------
// Effective sample counts
// -----------------------------------------------------------------------------

/**
 * @brief Returns the number of samples currently used in the rolling window.
 * @return Number of HOUR-event samples contributing to the current metrics.
 *
 * The maximum value equals the configured window size (for example, 24).
 * Before the window is full, the value corresponds to the number of valid
 * hourly samples already accumulated.
 */
unsigned long metrics_samples_total();

/**
 * @brief Returns the number of samples used in temperature metrics.
 * @return Effective sample count for T_in metrics.
 *
 * In the current implementation, this is equal to the global sample count.
 */
unsigned long metrics_samples_T();

/**
 * @brief Returns the number of samples used in humidity metrics.
 * @return Effective sample count for H_in metrics.
 *
 * In the current implementation, this is equal to the global sample count.
 */
unsigned long metrics_samples_H();

#endif // METRICS_H