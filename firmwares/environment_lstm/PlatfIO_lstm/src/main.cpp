/*
 * ============================================================================
 *  main.cpp
 *  LiteML-Edge Main Runtime Orchestrator
 * ============================================================================
 *
 *  Purpose
 *  -------
 *  This file coordinates the full LiteML-Edge embedded runtime, integrating:
 *
 *  1. Sensor acquisition and logical-hour event detection.
 *  2. Inference engine execution.
 *  3. Rolling metric computation.
 *  4. Benchmark telemetry for latency, memory, power, and energy.
 *  5. Replay-mode validation and field-mode operation.
 *
 *  Execution model
 *  ---------------
 *  The runtime is event-driven:
 *
 *  - Sensors are read continuously.
 *  - Warm-up and logical-hour readiness are enforced before inference.
 *  - Inference is executed only on logical-hour rollover events.
 *  - Metrics are updated only when the inference contract allows it.
 *  - Benchmark telemetry is emitted for both IDLE and inference paths.
 *
 *  Supported modes
 *  ---------------
 *  - REPLAY mode: deterministic 1:1 evaluation from exported header samples.
 *  - FIELD mode: physical-sensor evaluation using the same post-acquisition contract.
 *
 *  Reproducibility and auditability
 *  --------------------------------
 *  - Runtime mode is explicitly logged at boot.
 *  - Inference is gated by warm-up and snapshot readiness.
 *  - Replay EOF is explicitly latched and logged.
 *  - Energy during inference is derived from total-energy delta snapshots.
 *  - Rolling metrics are aligned with the training-side protocol.
 *
 *  Important
 *  ---------
 *  This version standardizes comments, headers, and user-visible log messages
 *  in English for IEEE LATAM-ready readability and auditability, while
 *  preserving the original functional behavior and runtime structure.
 * ============================================================================
 */

#include <Arduino.h>
#include <math.h>
#include <time.h>

// --- Baseline overrides for LIVE_FEATURES in original scale ---
#ifndef TRAIN_HOUR_ORIG_MSE
#define TRAIN_HOUR_ORIG_MSE NAN
#endif
#ifndef TRAIN_HOUR_ORIG_RMSE
#define TRAIN_HOUR_ORIG_RMSE NAN
#endif
#ifndef TRAIN_HOUR_ORIG_MAE
#define TRAIN_HOUR_ORIG_MAE NAN
#endif

#include "sensors.h"
#include "inference.h"
#include "metrics.h"
#include "benchmark.h"
#include "config.h"

uint32_t infer_us = 0;

/**
 * @brief Rounds a float to two decimal places for log printing.
 * @param x Input value.
 * @return Rounded value with two decimal places.
 */
static inline float round2f_print(float x) {
  return roundf(x * 100.0f) / 100.0f;
}

// ---- Platform-specific power helpers ----
#if LITEML_TARGET_ESP32
  #include "esp_sleep.h"
  #include "esp_pm.h"

  /**
   * @brief Sleeps for a given number of milliseconds.
   * @param ms Sleep duration in milliseconds.
   *
   * In fully offline mode (millis()-based fallback), light sleep is avoided
   * and delay() is used to reduce the risk of RTC watchdog issues when Wi-Fi
   * is unavailable. When NTP / RTC time is valid, light sleep is used.
   */
  static inline void sleep_ms(uint32_t ms){
    if (ms == 0) return;

    // Fully offline mode (millis() fallback): avoid light sleep.
    if (!sensors_time_ready()) {
      delay(ms);
      return;
    }

    // With NTP / RTC time available, light sleep can be used normally.
    esp_sleep_enable_timer_wakeup((uint64_t)ms * 1000ULL);
    esp_light_sleep_start();
  }

  /**
   * @brief Sets the CPU frequency in MHz.
   * @param mhz Target CPU frequency.
   */
  static inline void cpu_set_mhz(int mhz){ setCpuFrequencyMhz(mhz); }
#else
  /**
 * @brief Platform-neutral millisecond sleep fallback.
 * @param ms Sleep duration in milliseconds.
 *
 * On non-ESP32 targets, the runtime falls back to Arduino delay() while
 * preserving the same call site used by the ESP32-specific sleep helper.
 */
static inline void sleep_ms(uint32_t ms){ delay(ms); }
  /**
 * @brief No-op CPU-frequency helper for platforms without dynamic clock control.
 * @param mhz Requested CPU frequency in megahertz.
 *
 * The argument is intentionally ignored so the higher-level runtime can keep a
 * uniform initialization path across supported and unsupported MCUs.
 */
static inline void cpu_set_mhz(int mhz){ (void)mhz; }
#endif
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// Telemetry / backoff
// ------------------------------------------------------------------
static unsigned long last_ui = 0;   // Used only for periodic IDLE / WARMUP logs
static unsigned long last_heavy_log_ms = 0; // Avoid immediate IDLE telemetry bursts after heavy log blocks

static const uint16_t BACKOFF_BASE_MS = 50;
static const uint16_t BACKOFF_CAP_MS  = 400;

// ------------------------------------------------------------------
// Basic sanity checks
// ------------------------------------------------------------------
/**
 * @brief Performs a basic sanity check for temperature values.
 * @param t Temperature candidate in degrees Celsius.
 * @return true when the value is finite and falls within the accepted runtime range.
 */
static inline bool sane_T(float t){ return isfinite(t) && (t > -20.0f) && (t < 80.0f); }
/**
 * @brief Performs a basic sanity check for relative-humidity values.
 * @param h Humidity candidate in percent relative humidity.
 * @return true when the value is finite and lies within the physical 0-100 %RH range.
 */
static inline bool sane_H(float h){ return isfinite(h) && (h >= 0.0f)  && (h <= 100.0f); }

/**
 * @brief Checks whether one full sample is numerically valid.
 * @param Tin Measured indoor temperature.
 * @param Hin Measured indoor humidity.
 * @param Tp  Predicted indoor temperature.
 * @param Hp  Predicted indoor humidity.
 * @return true if all values are finite and within basic valid ranges.
 */
static inline bool sane_sample(float Tin, float Hin, float Tp, float Hp){
  return sane_T(Tin) && sane_H(Hin) && sane_T(Tp) && sane_H(Hp);
}

/**
 * @brief Runs the optional inference self-test at boot.
 */
static void maybe_selftest_inference(){
#if defined(HAS_INFERENCE_SELFTEST)
  InferenceSelfTestResult st = inference_selftest(); // {ok, y0, y1}
  if (!st.ok) {
    Serial.println("[SELFTEST] Invoke failed or output is invalid.");
  } else {
    Serial.printf("[SELFTEST] y0=%.2f y1=%.2f\n", st.y0, st.y1);
  }
#else
  Serial.println("[SELFTEST] skipped (define HAS_INFERENCE_SELFTEST in the inference module).");
#endif
}

/**
 * @brief Emits lightweight periodic IDLE telemetry.
 *
 * This helper is used only for periodic IDLE reporting and must not be used
 * to throttle inference logs.
 */
static inline void idle_telemetry(){
  const unsigned long now_ms = millis();

  // After heavy event / metric logging, leave breathing room for UART and I2C
  // instead of emitting an immediate extra IDLE benchmark line.
  if ((now_ms - last_heavy_log_ms) < 1500UL) return;

  if (now_ms - last_ui > 2000UL) {
    last_ui = now_ms;

    // Important watchdog / I2C hardening:
    // do NOT hammer INA219 from the hot loop. In idle / warm-up paths,
    // refresh the cached power sample only when an IDLE line will actually be emitted.
    #if BENCH_ENABLE_POWER
      bench_tick();
      yield();
    #endif

    BenchMemory mem{}; if (BENCH_ENABLE_MEMORY) mem = bench_memory_snapshot();
    BenchPower  pwr{};
    if (BENCH_ENABLE_POWER) {
      if (!bench_power_last(pwr)) {
        bench_power_poll(pwr);
      }
    }
    bench_print_line("IDLE", 0, mem, pwr);
    yield();
  }
}

/**
 * @brief Arduino setup entry point.
 *
 * Initializes serial, CPU frequency, sensors, inference, metrics, and optional
 * power benchmarking.
 */
void setup(){
  
  #if LITEML_TARGET_ESP32
  Serial.setTxBufferSize(2048);
  #endif
  
  Serial.begin(115200);
  delay(5000);
  Serial.flush();
  Serial.flush();
  Serial.println("===== BOOT START =====");

  cpu_set_mhz(LITEML_CPU_MHZ);

  Serial.printf("[PLATFORM] %s\n", LITEML_PLATFORM_NAME);
  #if LITEML_TARGET_ESP32
    Serial.printf("[CPU_SET_MHz] freq=%d MHz\n", LITEML_CPU_MHZ);
  #elif LITEML_TARGET_STM32F411RE
    Serial.printf("[CPU_SET_MHz] freq=%lu MHz\n",
                  (unsigned long)(F_CPU / 1000000UL));
  #endif
  Serial.println("[BOOT] LiteML-Edge starting...");
 
  // --- MODE: REPLAY vs FIELD under the same evaluation contract ---
  Serial.printf("[MODE] LITEML_MODE=%d -> %s\n",
              (int)LITEML_MODE,
              LITEML_REPLAY ? "REPLAY (evaluation contract)" : "FIELD (evaluation contract)");
 
  #if LITEML_REPLAY
  // In replay mode the dataset drives the pipeline,
  // so hardware peripherals can be disabled to save power.
  sensors_network_off();
  //btStop();
  #if LITEML_TARGET_ESP32
    Serial.println("[NETWORK] WiFi/BLE radios disabled.");
  #elif LITEML_TARGET_STM32F411RE
    Serial.println("[NETWORK] WiFi/BLE not present on the selected board.");
  #endif
  #endif
  
  init_sensors();

  Serial.println("[MODE] Shared post-acquisition contract: hourly rollover, lag snapshots, common feature packing, and lag-1 residual baseline.");

  //if (!init_sensors()){
  //  Serial.println("[BOOT] init_sensors() failed. Continuing in degraded mode.");
  //}

  if (!init_inference()){
    Serial.println("[BOOT] init_inference() failed. ML disabled.");
  } else {
    maybe_selftest_inference();
  }

  reset_metrics();

  if (BENCH_ENABLE_POWER) {
    if (!bench_power_begin()){
      Serial.println("[BENCH] power backend not ready (continuing without power telemetry).");
    }
  }

  Serial.println("[BOOT] ready.");
  Serial.printf("[METRICS] mode=%s\n", METRICS_STRICT_TRAINING ? "LIVE_FEATURES" : "ALWAYS(update every event)");
 }

/**
 * @brief Arduino main loop entry point.
 *
 * Runtime stages:
 * 1. Bookkeeping / energy integration.
 * 2. Sensor acquisition and validation.
 * 3. Warm-up and snapshot gating.
 * 4. Logical-hour rollover gating.
 * 5. Inference on logical-hour rollover.
 * 6. Metric update and benchmark logging.
 */
void loop(){
#if LITEML_REPLAY
  // REPLAY_VALIDATE:
  // - once EOF is latched, never call read_sensors() again,
  //   so post-EOF debug rows are not emitted repeatedly.
  // - if EOF becomes visible before sensor acquisition, freeze immediately.
  static bool replay_done = false;
  if (replay_done || sensors_replay_eof()) {
    if (!replay_done) {
      replay_done = true;
      Serial.println("[REPLAY] EOF detected — metrics frozen and execution finished (idle).");
      Serial.println();
    }
    delay(1000);
    return;
  }
#endif

  // Bookkeeping / energy integration.
  // NOTE: continuous INA219 polling from the hot loop can starve the shared I2C
  // path during FIELD warm-up. Idle polling is therefore performed only when an
  // IDLE telemetry line is actually emitted; inference energy remains sampled
  // explicitly around Invoke() inside run_inference().

  // -------- 1) Sensors --------
  SensorData d = read_sensors();

#if LITEML_REPLAY
  // If EOF was latched during this acquisition cycle, freeze immediately and
  // avoid any further processing in the same loop iteration.
  if (!replay_done && sensors_replay_eof()) {
    replay_done = true;
    Serial.println("[REPLAY] EOF detected — metrics frozen and execution finished (idle).");
    Serial.println();
    delay(1000);
    return;
  }
#endif

  // Fast validation
  static uint32_t consecutive_sensor_fail = 0;
  if (!d.valid || !sane_T(d.T_in) || !sane_H(d.H_in) || !sane_T(d.T_out) || !sane_H(d.H_out)) {
    consecutive_sensor_fail++;
    if (consecutive_sensor_fail % 25 == 1) {
      Serial.println("[SENSORS] invalid reading (waiting for stabilization)...");
    }
    uint16_t backoff = BACKOFF_BASE_MS * min<uint32_t>((1u << min<uint32_t>(consecutive_sensor_fail, 3u)), 8u);
    backoff = min<uint16_t>(backoff, BACKOFF_CAP_MS);
    delay(backoff);
    yield();
    return;
  }
  consecutive_sensor_fail = 0;

  // -------- 2) Warm-up / required lag history --------
  if (!d.have_snapshot || d.warmup_ok_count < 6) {
    if (millis() - last_ui > 1500) {
      last_ui = millis();

      if (!d.have_snapshot && d.warmup_ok_count < 6) {
        Serial.printf("[WARMUP] waiting=stabilization+lag_history | snapshot=%d warmup=%u/6  Tin=%.2f Hin=%.2f Tout=%.2f Hout=%.2f\n",
                      (int)d.have_snapshot, (unsigned)d.warmup_ok_count,
                      d.T_in, d.H_in, d.T_out, d.H_out);
      } else if (!d.have_snapshot) {
        Serial.printf("[WARMUP] waiting=lag_history(lag1+lag2) | snapshot=%d warmup=%u/6  Tin=%.2f Hin=%.2f Tout=%.2f Hout=%.2f\n",
                      (int)d.have_snapshot, (unsigned)d.warmup_ok_count,
                      d.T_in, d.H_in, d.T_out, d.H_out);
      } else {
        Serial.printf("[WARMUP] waiting=stabilization | snapshot=%d warmup=%u/6  Tin=%.2f Hin=%.2f Tout=%.2f Hout=%.2f\n",
                      (int)d.have_snapshot, (unsigned)d.warmup_ok_count,
                      d.T_in, d.H_in, d.T_out, d.H_out);
      }
      last_heavy_log_ms = last_ui;
    }
    idle_telemetry();
    sleep_ms(60);
    yield();
    return;
  }

  // -------- 3) Logical-hour event --------
  // IMPORTANT: no throttle here, to avoid missing the rollover latch.
  const bool is_rollover = sensors_hour_rollover();   // fires once at logical-hour transition

  if (!is_rollover) {
    // No relevant event: only lightweight telemetry + short sleep.
    idle_telemetry();
    sleep_ms(40);
    yield();
    return;
  }

  Serial.println();
  last_heavy_log_ms = millis();
  // Log the hourly event explicitly BEFORE inference, to guarantee visibility.
  Serial.printf("[EVENT] type=HOUR  Tin=%.2f Hin=%.2f Tout=%.2f Hout=%.2f\n",
                d.T_in, d.H_in, d.T_out, d.H_out);

  // REPLAY-only time trace: after EVENT type=HOUR, also log the logical
  // timestamp derived from the exported header epoch.
  if (LITEML_REPLAY && is_rollover) {
    const time_t ep = sensors_replay_epoch();
    if (ep > 0) {
      struct tm tm_info;
      gmtime_r(&ep, &tm_info); // dataset epoch is UTC; preserves 1:1 with header
      const int wday_pandas = (tm_info.tm_wday + 6) % 7; // 0=Mon..6=Sun
      const int mon_1_12    = tm_info.tm_mon + 1;
      Serial.printf("[TIME]  HOUR logical=%04d-%02d-%02d %02d:%02d:%02d | wday=%d mon=%d | epoch=%lu\n",
                    tm_info.tm_year + 1900, tm_info.tm_mon + 1, tm_info.tm_mday,
                    tm_info.tm_hour, tm_info.tm_min, tm_info.tm_sec,
                    wday_pandas, mon_1_12, (unsigned long)ep);
    }
  }
  
  // To stay aligned with Conv1D Tiny training (24-hour window), inference is
  // executed only on hourly rollover events.

  // -------- 4) Inference (hourly rollover only) --------
  auto r = run_inference(d);
  infer_us = r.us_invoke;

  // If no Invoke occurred (window warm-up or not ready), do not record power or metrics.
  if (!r.invoked) {
    delay(20);
    yield();
    return;
  }

#if BENCH_ENABLE_POWER
  if (r.power_valid) {
    const float E_invoke_uWh = r.energy_invoke_mWh * 1000.0f;
    const float E_event_uWh  = r.energy_event_mWh  * 1000.0f;
    const float t_invoke_ms  = (float)r.us_invoke / 1000.0f;
    const float t_event_ms   = (float)r.us_event  / 1000.0f;

    BenchPower pwr_total{};
    if (!bench_power_last(pwr_total)) {
      pwr_total = r.pwr_event_post;
    }
    const float Etot_mWh = (pwr_total.energy_mWh > 0.0f)
                         ? pwr_total.energy_mWh
                         : r.pwr_event_post.energy_mWh;
    
    Serial.println();
    Serial.printf(
     "[PWR] infer | "
     "E_inference_window(ΔE_total)=%.3fµWh (E0=%.6fmWh E1=%.6fmWh) | "
     "E_inference_pipeline(ΔE_total)=%.3fµWh (E0=%.6fmWh E1=%.6fmWh) | "
     "t_inference=%.3fms | t_inference_pipeline=%.3fms | E_total_accum=%.3fmWh\n",
     (float)E_invoke_uWh,
     (float)r.pwr_invoke_pre.energy_mWh,
     (float)r.pwr_invoke_post.energy_mWh,
     (float)E_event_uWh,
     (float)r.pwr_event_pre.energy_mWh,
     (float)r.pwr_event_post.energy_mWh,
     (float)t_invoke_ms,
     (float)t_event_ms,
     (float)Etot_mWh
    );
  }
#endif

if (!r.ok || !isfinite(r.Tin_pred) || !isfinite(r.Hin_pred)) {
    static uint32_t consecutive_infer_fail = 0;
    consecutive_infer_fail++;
    if (consecutive_infer_fail % 10 == 1) {
      Serial.println("[INFER] invalid Invoke / output (retrying)...");
    }
    uint16_t backoff = BACKOFF_BASE_MS * min<uint32_t>((1u << min<uint32_t>(consecutive_infer_fail, 3u)), 8u);
    backoff = min<uint16_t>(backoff, BACKOFF_CAP_MS);
    delay(backoff);
    yield();
    return;
  }

  // -------- 5) Metrics (training-compatible) --------
  // Apples-to-apples: do not update metrics during window warm-up if Invoke() did not run.
  if (r.us == 0) {
    Serial.println("[infer] WARMUP - window < 24 h (Invoke NOT executed) - metrics skipped");
    delay(1);
    yield();
    return;
  }

update_metrics(
  d.T_in,
  d.H_in,
  r.Tin_pred,
  r.Hin_pred,
  is_rollover,
  r.invoked
);
  float mae=0.0, rmse=0.0, r2=NAN;
  get_metrics(mae, rmse, r2);

  // -------- 6) Memory / power (optional) --------
  BenchMemory mem{}; if (BENCH_ENABLE_MEMORY) mem = bench_memory_snapshot();

  // Reuse the event-post snapshot so the benchmark line stays aligned with
  // the same inference event reported in the [PWR] infer line.
  BenchPower  pwr{};
  if (BENCH_ENABLE_POWER) {
    if (r.power_valid) pwr = r.pwr_event_post;
    else bench_power_last(pwr);
  }

  // -------- 7) Inference logs (NO UI rate-limit here) --------
  bench_print_line("INFER_HOURLY", r.us_invoke, mem, pwr);

  char r2buf[16];
  if (isnan(r2)) { strcpy(r2buf, "NA"); }
  else { snprintf(r2buf, sizeof(r2buf), "%.3f", r2); }

  const char* tag = "HOUR";

  // Line 1: physical values only (matches dissertation log style).
  float Tp_print = round2f_print(r.Tin_pred);
  float Hp_print = round2f_print(r.Hin_pred);

  Serial.printf("[%s] Tin=%.2f Hin=%.2f | Tp=%.2f Hp=%.2f\n",
                tag,
                d.T_in, d.H_in,
                Tp_print, Hp_print);

  // Line 2: derived metrics (MAE / RMSE / R²) + per-variable metrics + inference time
  float mae_T = 0.0, rmse_T = 0.0, r2_T = 0.0;
  float mae_H = 0.0, rmse_H = 0.0, r2_H = 0.0;
  get_metrics_T(mae_T, rmse_T, r2_T);
  get_metrics_H(mae_H, rmse_H, r2_H);

  // Number of actual samples used in the metrics
  unsigned long n_total = metrics_samples_total();
  unsigned long n_T     = metrics_samples_T();
  unsigned long n_H     = metrics_samples_H();

  Serial.printf("[%s] SAMPLES | N_ALL=%lu N_T=%lu N_H=%lu\n",
                tag,
                (unsigned long)n_total,
                (unsigned long)n_T,
                (unsigned long)n_H);

  char r2_buf[24];
  char r2T_buf[24];
  char r2H_buf[24];
  if (isnan(r2))   { snprintf(r2_buf,  sizeof(r2_buf),  "NA"); } else { snprintf(r2_buf,  sizeof(r2_buf),  "%.4f", r2); }
  if (isnan(r2_T)) { snprintf(r2T_buf, sizeof(r2T_buf), "NA"); } else { snprintf(r2T_buf, sizeof(r2T_buf), "%.4f", r2_T); }
  if (isnan(r2_H)) { snprintf(r2H_buf, sizeof(r2H_buf), "NA"); } else { snprintf(r2H_buf, sizeof(r2H_buf), "%.4f", r2_H); }

  Serial.printf("[%s] METRICS | MAE=%.4f RMSE=%.4f R2=%s | MAE_T=%.4f RMSE_T=%.4f R2_T=%s | MAE_H=%.4f RMSE_H=%.4f R2_H=%s\n",
                tag,
                mae, rmse, r2_buf,
                mae_T, rmse_T, r2T_buf,
                mae_H, rmse_H, r2H_buf);
  last_heavy_log_ms = millis();
  last_ui = last_heavy_log_ms;
  // Small breathing space to avoid flooding, without hiding events.
  delay(1);
  yield();
}