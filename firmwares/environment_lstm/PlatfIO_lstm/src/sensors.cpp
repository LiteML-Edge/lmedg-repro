/* * ============================================================================
 *  sensors.cpp
 *  LiteML-Edge Sensor Acquisition and Time/State Management Module
 * ============================================================================
 *
 *  Purpose
 *  -------
 *  This module implements the sensor-side runtime support for LiteML-Edge,
 *  including:
 *
 *  1. Environmental data acquisition from DHT22 sensors.
 *  2. Optional OLED user-interface support.
 *  3. Optional Wi-Fi / NTP time synchronization.
 *  4. Replay-mode sample feeding from exported header datasets.
 *  5. Contract-preserving acquisition, anti-stall recovery, and sanity validation.
 *  6. Logical-hour snapshot management for lag-based feature generation.
 *
 *  Design rationale
 *  ----------------
 *  The implementation is designed to support both:
 *
 *  - FIELD mode: real sensors, filtering, recovery, and optional NTP time.
 *  - REPLAY mode: deterministic dataset-driven validation using exported samples.
 *
 *  This structure allows LiteML-Edge to preserve an auditable contract between
 *  offline evaluation and MCU-side execution, particularly for rolling-window
 *  and lag-based inference scenarios.
 *
 *  Main runtime responsibilities
 *  -----------------------------
 *  - Maintain acquisition robustness while preserving a common post-acquisition evaluation contract.
 *  - Provide time features (sin_hour, cos_hour, weekday, month).
 *  - Generate lag-1 and lag-2 snapshots aligned with logical-hour rollover.
 *  - Support replay-time derivation directly from dataset epoch values.
 *  - Offer acquisition health telemetry for DHT behavior and recovery actions.
 *
 *  Reproducibility and auditability
 *  --------------------------------
 *  - Replay mode uses header-exported samples only.
 *  - Logical-hour rollover is explicit and latched.
 *  - Anti-stall logic separates true lack of valid readings from naturally
 *    stable environments.
 *  - User-visible log messages are standardized in English.
 *
 *  Important
 *  ---------
 *  This version documents the unified evaluation contract used by both
 *  REPLAY and FIELD modes after acquisition, while preserving the original
 *  firmware structure and low-level acquisition safeguards.
 * ============================================================================
 */

#include "sensors.h"
#include "config.h"

#include <Arduino.h>
#include <math.h>
#include <time.h>
#include <limits.h>

#if (USE_OLED && LITEML_TARGET_ESP32) || \
    (BENCH_ENABLE_POWER && (BENCH_PWR_BACKEND == BENCH_PWR_INA219))
  #include <Wire.h>
#endif

#if USE_BLUETOOTH && LITEML_TARGET_ESP32
  #include "esp_bt.h"
#endif

#if defined(__GNUC__)
  #define MAYBE_UNUSED __attribute__((unused))
#else
  #define MAYBE_UNUSED
#endif

#if USE_DHT22
  #include <DHT.h>
  static DHT dht_out(PIN_DHT_OUT, DHTTYPE);
  static DHT dht_in (PIN_DHT_IN,  DHTTYPE);
#endif

#if USE_OLED && LITEML_TARGET_ESP32
  // OLED is supported only on the ESP32/Lolin32 platform.
  // STM32F411RE does not compile or link any display dependency.
  #include <SSD1306.h>
  static SSD1306 display(0x3c, OLED_SDA_PIN, OLED_SCL_PIN);

  // UI enabled/disabled at runtime
  static bool g_ui_enabled = true;
  // True only after display.init() succeeds in this boot
  static bool g_oled_initialized = false;

  /**
   * @brief Ensures the OLED controller is initialized without disturbing the shared I2C bus.
   * @return true when the OLED is ready for use.
   *
   * The I2C bus is intentionally kept available for INA219 even when the UI is disabled.
   * Therefore, OLED activation/deactivation must not deinitialize Wire.
   */
  static bool oled_ensure_initialized(){
    if (g_oled_initialized) return true;
    display.init();
    display.flipScreenVertically();
    display.clear();
    display.display();
    g_oled_initialized = true;
    return true;
  }
#endif

#if USE_WIFI_TIME && LITEML_TARGET_ESP32
  #include <WiFi.h>
#endif

#if LITEML_REPLAY
  // REPLAY raw 2+47: uses the header exported by the script, including
  // two seed rows (t-2, t-1) followed by the 47 real rows.
#if __has_include("environment_quantized_samples_replay_raw_2plus47_lstm.h")
  #include "environment_quantized_samples_replay_raw_2plus47_lstm.h"
#endif

  static uint16_t replay_idx = 0;
  static time_t   replay_epoch = 0;
  static bool     replay_seeded = false;

  // EOF latch (REPLAY validation): set when the header ends (no wrap).
  static bool replay_eof = false;

  // Cache: guarantees one header sample per logical "HOUR",
  // even if read_sensors() is called many times.
static bool     replay_have_cached = false;
static int32_t  replay_cached_hour_id = INT32_MIN;
static liteml_sample_raw_t replay_cached{};
static int32_t  replay_cached_raw_idx = -1;
static float    replay_cached_hin = NAN;
static float    replay_hin_ema_state = NAN;
static bool     replay_hin_ema_ready = false;
#endif

// =================== Evaluation-contract helpers ===================
// Debug: prints RAW vs filtered values (1 Hz). Enable with -DSENSORS_DEBUG_RAW=1.
#ifndef SENSORS_DEBUG_RAW
  #define SENSORS_DEBUG_RAW 0
#endif

#ifndef LITEML_DBG_SENSORS_EXACT
  #define LITEML_DBG_SENSORS_EXACT 1
#endif

#if USE_OLED && LITEML_TARGET_ESP32
  #ifndef OLED_REFRESH_MS
    #define OLED_REFRESH_MS 4000UL
  #endif
#endif

#if LITEML_DBG_SENSORS_EXACT
static bool    g_dbg_sensors_header_printed = false;
static bool    g_dbg_pre_raw_header_printed = false;
static bool    g_dbg_pre_smooth_header_printed = false;
static int     g_dbg_sensors_idx = 0;
static int32_t g_dbg_sensors_last_hour_id = INT32_MIN;

/**
 * @brief Prints the DBG_PRE_RAW_CSV header once per boot.
 *
 * This header defines the raw acquisition debug stream emitted before any
 * contract-side smoothing or lag construction is applied.
 */
static void dbg_pre_raw_print_header_once(){
  if (g_dbg_pre_raw_header_printed) return;
  g_dbg_pre_raw_header_printed = true;

  Serial.println(
    "[DBG_PRE_RAW_CSV] "
    "idx,mode,raw_idx,epoch,"
    "Tout_raw,Hout_raw,Tin_raw,Hin_raw"
  );
}

/**
 * @brief Prints the DBG_PRE_SMOOTH_CSV header once per boot.
 *
 * The corresponding rows expose the post-conditioning sensor values that feed
 * the shared acquisition contract before lag features are assembled.
 */
static void dbg_pre_smooth_print_header_once(){
  if (g_dbg_pre_smooth_header_printed) return;
  g_dbg_pre_smooth_header_printed = true;

  Serial.println(
    "[DBG_PRE_SMOOTH_CSV] "
    "idx,mode,raw_idx,epoch,"
    "Tout_smooth,Hout_smooth,Tin_smooth,Hin_smooth"
  );
}

/**
 * @brief Prints the DBG_SENSORS_CSV header once per boot.
 *
 * The schema captures the complete sensor-side packet exported to the rest of
 * the firmware, including lags, time features, and warm-up status flags.
 */
static void dbg_sensors_print_header_once(){
  if (g_dbg_sensors_header_printed) return;
  g_dbg_sensors_header_printed = true;

  Serial.println(
    "[DBG_SENSORS_CSV] "
    "idx,mode,raw_idx,epoch,"
    "Tout,Hout,Tin,Hin_raw,Hin_filtered,"
    "Tin_lag1,Hin_lag1,Tout_lag1,Hout_lag1,"
    "Tin_lag2,Hin_lag2,"
    "sin_hour,cos_hour,weekday,month,"
    "have_lag1,have_lag2,warmup"
  );
}

static void dbg_pre_raw_print_row(
    int dbg_idx,
    const char* mode,
    int32_t raw_idx,
    long epoch,
    float Tout_raw, float Hout_raw,
    float Tin_raw, float Hin_raw){

  dbg_pre_raw_print_header_once();

  Serial.printf(
    "[DBG_PRE_RAW_CSV] "
    "%d,%s,%ld,%ld,"
    "%.8f,%.8f,%.8f,%.8f\n",
    dbg_idx,
    mode,
    (long)raw_idx,
    epoch,
    Tout_raw, Hout_raw, Tin_raw, Hin_raw
  );
}

static void dbg_pre_smooth_print_row(
    int dbg_idx,
    const char* mode,
    int32_t raw_idx,
    long epoch,
    float Tout_smooth, float Hout_smooth,
    float Tin_smooth, float Hin_smooth){

  dbg_pre_smooth_print_header_once();

  Serial.printf(
    "[DBG_PRE_SMOOTH_CSV] "
    "%d,%s,%ld,%ld,"
    "%.8f,%.8f,%.8f,%.8f\n",
    dbg_idx,
    mode,
    (long)raw_idx,
    epoch,
    Tout_smooth, Hout_smooth, Tin_smooth, Hin_smooth
  );
}

static void dbg_sensors_print_row(
    int dbg_idx,
    const char* mode,
    int32_t raw_idx,
    long epoch,
    float Tout, float Hout,
    float Tin, float Hin_raw, float Hin_filtered,
    float Tin_lag1, float Hin_lag1,
    float Tout_lag1, float Hout_lag1,
    float Tin_lag2, float Hin_lag2,
    float sin_hour, float cos_hour,
    int weekday, int month,
    bool have_lag1, bool have_lag2,
    int warmup_count){

  dbg_sensors_print_header_once();
  
  Serial.printf(
    "[DBG_SENSORS_CSV] "
    "%d,%s,%ld,%ld,"
    "%.8f,%.8f,%.8f,%.8f,%.8f,"
    "%.8f,%.8f,%.8f,%.8f,"
    "%.8f,%.8f,"
    "%.8f,%.8f,%d,%d,"
    "%d,%d,%d\n",
    dbg_idx,
    mode,
    (long)raw_idx,
    epoch,
    Tout, Hout, Tin, Hin_raw, Hin_filtered,
    Tin_lag1, Hin_lag1, Tout_lag1, Hout_lag1,
    Tin_lag2, Hin_lag2,
    sin_hour, cos_hour, weekday, month,
    have_lag1 ? 1 : 0,
    have_lag2 ? 1 : 0,
    warmup_count
  );
}
#endif

// ============================================================================
// DHT robustness: anti-stall / reinit / optional power-cycle
// (DHT22 requires approximately 2 s between reads)
// ============================================================================
#ifndef DHT_MIN_INTERVAL_MS
  #define DHT_MIN_INTERVAL_MS 2100UL
#endif
#ifndef DHT_STALE_MS
  // Maximum time without a VALID DHT reading before attempting power-cycle/reinit.
  // This must not be confused with a stable environment, which can be normal.
  #define DHT_STALE_MS 15000UL          // 15 s without valid read => possible stall / I/O issue
#endif
#ifndef DHT_STALL_EPS_T
  #define DHT_STALL_EPS_T 0.05f
#endif
#ifndef DHT_STALL_EPS_H
  #define DHT_STALL_EPS_H 0.10f
#endif
#ifndef DHT_REINIT_COOLDOWN_MS
  #define DHT_REINIT_COOLDOWN_MS 5000UL
#endif

// Optional power-cycle control pin
// (set the pin that switches DHT 3V3 via MOSFET; -1 disables it)
#ifndef DHT_IN_PWR_PIN
  #define DHT_IN_PWR_PIN  -1
#endif
#ifndef DHT_OUT_PWR_PIN
  #define DHT_OUT_PWR_PIN -1
#endif
#ifndef DHT_POWER_CYCLE_OFF_MS
  #define DHT_POWER_CYCLE_OFF_MS 250UL
#endif

// ============================================================================
// Numeric helpers
// ============================================================================
/**
 * @brief Clamps a scalar to a closed numeric interval.
 * @param x Input value.
 * @param lo Lower bound.
 * @param hi Upper bound.
 * @return Value limited to the [lo, hi] interval.
 */
static inline float clampf(float x, float lo, float hi){
  return (x < lo) ? lo : (x > hi) ? hi : x;
}
/**
 * @brief Checks whether a temperature reading lies within the supported DHT-style range.
 * @param t Temperature value in degrees Celsius.
 * @return true when the reading is physically plausible for the acquisition layer.
 */
static inline bool saneTemp(float t){ return (t>-40.0f && t<85.0f); }
/**
 * @brief Checks whether a humidity reading lies within the physical 0-100 %RH range.
 * @param h Relative humidity value in percent.
 * @return true when the reading is physically plausible for the acquisition layer.
 */
static inline bool saneHum (float h){ return (h>=0.0f && h<=100.0f); }


// ==============================
// ============================================================================
// Internal state
// ============================================================================
static bool   time_ok=false;
static unsigned long millis_start=0;
static uint32_t g_fake_hour_base = 0;   // base of the synthetic hour counter (AUTO / fallback)
static uint32_t g_fake_hour_ms0  = 0;   // millis() corresponding to that base

static bool   have_hour_prev=false;
static float  hour_T_out_prev=0, hour_H_out_prev=0, hour_T_in_prev=0, hour_H_in_prev=0;
static bool   have_hour_prev2=false;
static float  hour_T_in_prev2=0, hour_H_in_prev2=0;
static float  hour_T_out_curr=0, hour_H_out_curr=0, hour_T_in_curr=0, hour_H_in_curr=0;
static int32_t last_hour_id=-1;

// Monotonic snapshot ID (debug / telemetry) + rollover latch.
static int32_t        snapshot_id_counter = 0;
static volatile bool  hour_changed_latch  = false;

// Warm-up / UI
static uint8_t  warmup_ok_count=0;
static unsigned long last_ui_ms = 0;

// -------- Anti-stall: freshness / reinit per channel --------
static float last_raw_t_in  = NAN, last_raw_h_in  = NAN;
static float last_raw_t_out = NAN, last_raw_h_out = NAN;

// Last instant with a valid DHT reading. This does not depend on variation:
// the environment may remain stable for minutes/hours and that is not a stall.
static unsigned long last_ok_in_ms  = 0;
static unsigned long last_ok_out_ms = 0;
static unsigned long last_reinit_in_ms  = 0;
static unsigned long last_reinit_out_ms = 0;
static uint16_t reinit_count_in=0, reinit_count_out=0;
static uint16_t power_cycle_in=0, power_cycle_out=0;

// -------- Rate-limited raw acquisition timestamps --------
static unsigned long last_read_ms_in  = 0;
static unsigned long last_read_ms_out = 0;

// -------- Internal health statistics --------
static uint32_t reads_ok_in=0,  reads_err_in=0;
static uint32_t reads_ok_out=0, reads_err_out=0;
static unsigned long last_health_log_ms=0;

// DHT read scheduler hardening:
// execute at most one physical DHT transaction per read_sensors() cycle.
// This avoids back-to-back long DHT critical sections on ESP32 while keeping
// the latest valid cache contract unchanged.
#if !LITEML_REPLAY
static uint8_t g_dht_rr_next = 0; // 0=OUT first, 1=IN first when both are due
#endif

// ============================================================================
// Time (NTP / RTC or fallback)
// ============================================================================
#if !LITEML_REPLAY && USE_WIFI_TIME && LITEML_TARGET_ESP32
/**
 * @brief Connects Wi-Fi and synchronizes time when needed.
 *
 * In AUTO mode, this function tries to obtain civil time via NTP.
 * If synchronization fails, the system falls back to a synthetic-hour counter
 * based on millis().
 */
static MAYBE_UNUSED void connect_time_if_needed() {
  if (time_ok) return;

  // ---- Wi-Fi on & connect ----
  sensors_network_on();

  const uint32_t WIFI_TOTAL_TIMEOUT_MS = 10000;    // 10 s for Wi-Fi connect
  const uint32_t WIFI_POLL_MS          = 250;
  uint32_t wifi_elapsed = 0;

  if (WiFi.status() != WL_CONNECTED) {
    Serial.printf("[SENSORS] Connecting Wi-Fi ssid=%s ...\n", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PSK);
  }
  while (WiFi.status() != WL_CONNECTED && wifi_elapsed < WIFI_TOTAL_TIMEOUT_MS) {
    delay(WIFI_POLL_MS);
    wifi_elapsed += WIFI_POLL_MS;
    Serial.print(".");
  }
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\n[SENSORS] Wi-Fi connection timed out. Turning Wi-Fi off and retrying later in offline mode.");
    // Turn Wi-Fi off to reduce power; a future call may retry.
    sensors_network_off();
    return;
  }
  Serial.printf("\n[SENSORS] Wi-Fi connected in %lu ms, IP=%s\n", (unsigned long)wifi_elapsed, WiFi.localIP().toString().c_str());

  // ---- NTP sync (blocking) ----
  // Manaus (AMT, UTC-4), no DST.
  const long GMT_OFFSET_SEC      = -4 * 3600;
  const int  DAYLIGHT_OFFSET_SEC = 0;

  configTime(GMT_OFFSET_SEC, DAYLIGHT_OFFSET_SEC,
             "pool.ntp.org", "time.google.com", "time.nist.gov");

  struct tm timeinfo;
  const uint32_t NTP_TOTAL_TIMEOUT_MS = 45000; // 45 s for NTP
  const uint32_t NTP_POLL_MS          = 300;
  uint32_t ntp_elapsed = 0;

  Serial.println("[SENSORS] Synchronizing NTP (blocking until success or timeout)...");
  while (ntp_elapsed < NTP_TOTAL_TIMEOUT_MS) {
    if (getLocalTime(&timeinfo, 1000)) { // allow 1 s for each internal attempt
      time_ok = true;
      time_t now = mktime(&timeinfo);
      Serial.printf("[SENSORS] NTP synchronized: %04d-%02d-%02d %02d:%02d:%02d (epoch=%ld)\n",
                   1900 + timeinfo.tm_year, 1 + timeinfo.tm_mon, timeinfo.tm_mday,
                   timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec, (long)now);
      break;
    }
    delay(NTP_POLL_MS);
    ntp_elapsed += NTP_POLL_MS;
    Serial.print("#");
  }

  if (!time_ok) {
    Serial.println("\n[SENSORS] NTP synchronization timed out — keeping Wi-Fi on for a future retry.");
    // AUTO mode: fall back to a synthetic-hour counter based on millis().
    g_fake_hour_base = 0;
    g_fake_hour_ms0  = millis();
    return; // keep Wi-Fi on; a future call will retry
  }

  // With synchronized time, align the synthetic base as well.
  g_fake_hour_base = 0;
  g_fake_hour_ms0  = millis();

  // Once time is in place, Wi-Fi can be turned off to save power.
  sensors_network_off();
}
#endif

/**
 * @brief Computes cyclical and calendar time features.
 * @param[out] sin_h Sine of hour-of-day.
 * @param[out] cos_h Cosine of hour-of-day.
 * @param[out] wday  Weekday in pandas convention (Monday = 0).
 * @param[out] mon   Month in [1, 12].
 */
static MAYBE_UNUSED void get_time_features(float& sin_h,float& cos_h,int& wday,int& mon){
#if LITEML_REPLAY
  // REPLAY 1:1: derive time features from dataset epoch (hour-aligned),
  // not from the ESP clock.
  if (replay_epoch > 0){
    struct tm tm_info; gmtime_r(&replay_epoch, &tm_info);
    float hour = tm_info.tm_hour + tm_info.tm_min/60.0f;
    sin_h = sinf(2.0f*(float)M_PI*hour/24.0f);
    cos_h = cosf(2.0f*(float)M_PI*hour/24.0f);
    wday  = (tm_info.tm_wday + 6)%7; // pandas: Monday = 0
    mon   = tm_info.tm_mon + 1;      // 1..12
    return;
  }
  // Fallback (should be rare before the first header sample).
#endif
  if (time_ok){
    time_t now; time(&now);
    struct tm tm_info; localtime_r(&now,&tm_info);
    float hour = tm_info.tm_hour + tm_info.tm_min/60.0f;
    sin_h = sinf(2.0f*(float)M_PI*hour/24.0f);
    cos_h = cosf(2.0f*(float)M_PI*hour/24.0f);
    wday  = (tm_info.tm_wday + 6)%7;
    mon   = tm_info.tm_mon + 1;
  } else {
    float hour = fmodf((millis()-millis_start)/3600000.0f,24.0f);
  #if FAST_HOUR_TEST
    hour = fmodf((millis()-millis_start)/(float)HOUR_PERIOD_MS*1.0f, 24.0f);
  #endif
    sin_h = sinf(2.0f*(float)M_PI*hour/24.0f);
    cos_h = cosf(2.0f*(float)M_PI*hour/24.0f);
    wday  = 0; mon = 1;
  }
}

/**
 * @brief Returns the current logical-hour identifier.
 * @return Logical-hour ID.
 *
 * In test mode, the hour is compressed according to HOUR_PERIOD_MS.
 * In AUTO mode:
 * - if time_ok == true, civil time is used;
 * - otherwise, a synthetic logical-hour counter based on millis() is used.
 */
int32_t sensors_current_hour_id(){
#if FAST_HOUR_TEST
  const uint32_t PERIOD = HOUR_PERIOD_MS;
  uint32_t base_ms = g_fake_hour_ms0 ? g_fake_hour_ms0 : millis_start;
  return (int32_t)(g_fake_hour_base + ((millis() - base_ms)/PERIOD));
#else
  if (time_ok){
    time_t now;
    time(&now);
    if (now > 0){
      return (int32_t)(now/3600);
    }
    // If RTC/NTP becomes invalid for any reason, fall back to synthetic mode.
    if (g_fake_hour_ms0 == 0){
      g_fake_hour_base = 0;
      g_fake_hour_ms0  = millis();
    }
  }
  uint32_t base_ms = g_fake_hour_ms0 ? g_fake_hour_ms0 : millis_start;
  return (int32_t)(g_fake_hour_base + ((millis() - base_ms)/3600000UL));
#endif
}

/**
 * @brief Reports whether replay mode has consumed the full exported header stream.
 * @return true once replay EOF has been latched; false otherwise.
 *
 * In FIELD mode, this function always returns false.
 */
bool sensors_replay_eof(){
#if LITEML_REPLAY
  return replay_eof;
#else
  return false;
#endif
}

/**
 * @brief Returns the epoch associated with the current replay sample.
 * @return Replay-sample epoch in seconds since the Unix epoch, or zero when unavailable.
 *
 * In FIELD mode, the function returns zero because no replay header is active.
 */
time_t sensors_replay_epoch(){
#if LITEML_REPLAY
  return replay_epoch;
#else
  return (time_t)0;
#endif
}

/**
 * @brief Reports whether civil time is currently available.
 * @return true when NTP / RTC time has been established; false when fallback timing is active.
 */
bool sensors_time_ready(){ return time_ok; }
/**
 * @brief Returns true when the evaluation contract has the required lag history.
 *
 * REPLAY normally starts with seeded lag1/lag2 from the exported header.
 * FIELD must accumulate two real hourly snapshots before the rolling contract
 * becomes ready. Therefore, readiness requires both lag1 and lag2.
 */
bool sensors_have_snapshot(){ return have_hour_prev && have_hour_prev2; }

// Common post-acquisition H_in EMA contract.
// In REPLAY, the alpha comes from the exported header.
// In FIELD, the same contract alpha is reused locally after acquisition.
#ifndef LITEML_CONTRACT_HIN_EMA_ALPHA
  #if defined(LITEML_HIN_EMA_ALPHA)
    #define LITEML_CONTRACT_HIN_EMA_ALPHA LITEML_HIN_EMA_ALPHA
  #else
    #define LITEML_CONTRACT_HIN_EMA_ALPHA 0.08f
  #endif
#endif


#if LITEML_REPLAY
/**
 * @brief Returns true when the replay header provides the 2+47 seed layout.
 * @return true when at least two seed rows and one real row are available.
 */
static inline bool replay_has_seed_rows(){
  return (LITEML_REPLAY_SEED_ROWS >= 2) && (LITEML_REPLAY_TOTAL_RAW_ROWS >= 3);
}

/**
 * @brief Validates one replay row before using it as seed/current sample.
 * @param row Replay row read from the exported header.
 * @return true when all fields are numerically valid and plausible.
 */
static inline bool replay_row_valid(const liteml_sample_raw_t& row){
  return saneTemp((float)row.T_out) &&
         saneHum ((float)row.H_out) &&
         saneTemp((float)row.T_in ) &&
         saneHum ((float)row.H_in_raw );
}

/**
 * @brief Applies the replay-side H_in EMA contract to one raw replay value.
 * @param h_in_raw Raw indoor-humidity value read from the exported replay header.
 * @return Filtered humidity value after one EMA update.
 *
 * The filter uses the exported alpha and initial state so that MCU-side replay
 * reproduces the same conditioning contract expected by offline validation.
 */
static inline float replay_filter_hin_from_header(float h_in_raw){
  const float alpha = clampf((float)LITEML_CONTRACT_HIN_EMA_ALPHA, 0.0f, 1.0f);

  if (!replay_hin_ema_ready){
    replay_hin_ema_state = (float)LITEML_HIN_EMA_PREV;
    replay_hin_ema_ready = true;
  }

  replay_hin_ema_state =
      alpha * h_in_raw + (1.0f - alpha) * replay_hin_ema_state;

  return replay_hin_ema_state;
}

/**
 * @brief Seeds lag history from the first two replay rows when available.
 *
 * Row 0 initializes lag2 (t-2) and row 1 initializes lag1 (t-1). The first
 * real replay sample then starts at raw index 2, preventing boot mirroring.
 */
static MAYBE_UNUSED void replay_seed_history_if_needed(){
  if (replay_seeded) return;
  if (!replay_has_seed_rows()) return;
  if (!replay_row_valid(LITEML_ROLL24_24_SAMPLES[0]) ||
      !replay_row_valid(LITEML_ROLL24_24_SAMPLES[1])) {
    Serial.println("[SENSORS] REPLAY: seed rows invalid — falling back to boot mirror.");
    return;
  }

  const float h_prev2_filt =
      replay_filter_hin_from_header((float)LITEML_ROLL24_24_SAMPLES[0].H_in_raw);

  const float h_prev1_filt =
      replay_filter_hin_from_header((float)LITEML_ROLL24_24_SAMPLES[1].H_in_raw);

  // lag1 (t-1): complete previous snapshot used by the model.
  hour_T_out_prev = (float)LITEML_ROLL24_24_SAMPLES[1].T_out;
  hour_H_out_prev = (float)LITEML_ROLL24_24_SAMPLES[1].H_out;
  hour_T_in_prev  = (float)LITEML_ROLL24_24_SAMPLES[1].T_in;
  hour_H_in_prev  = h_prev1_filt;
  have_hour_prev  = true;

  // lag2 (t-2): only indoor lag-2 is required by the current contract.
  hour_T_in_prev2 = (float)LITEML_ROLL24_24_SAMPLES[0].T_in;
  hour_H_in_prev2 = h_prev2_filt;
  have_hour_prev2 = true;

  replay_idx = (uint16_t)LITEML_REPLAY_SEED_ROWS;
  replay_seeded = true;

  Serial.printf("[SENSORS] REPLAY: seeded history from rows 0..1 with H_in EMA | first real raw idx=%u | real rows=%d\n",
                (unsigned)replay_idx, LITEML_REPLAY_REAL_ROWS);
}
#endif

#if !LITEML_REPLAY
static float field_hin_ema_state = NAN;
static bool  field_hin_ema_ready = false;
// Number of real logical-hour FIELD snapshots accumulated since boot.
// 0 -> none, 1 -> current only, 2 -> lag1 available, 3+ -> lag1 and lag2 available.
static uint8_t field_hour_seed_count = 0;
static uint8_t field_hour_seed_reported = 0xFF;

/**
 * @brief Emits progress telemetry while FIELD mode accumulates the initial lag seeds.
 *
 * The message is printed only when the number of collected logical-hour seeds
 * changes, preventing repetitive logs while preserving warm-up observability.
 */
static inline void maybe_log_field_seed_progress(){
  if (field_hour_seed_reported == field_hour_seed_count) return;
  field_hour_seed_reported = field_hour_seed_count;

  if (field_hour_seed_count == 0) return;

  if (field_hour_seed_count == 1) {
    Serial.println("[SENSORS] FIELD seed progress: current hour captured; waiting for lag1 and lag2.");
  } else if (field_hour_seed_count == 2) {
    Serial.println("[SENSORS] FIELD seed progress: lag1 ready; waiting one more logical hour for lag2.");
  } else {
    Serial.println("[SENSORS] FIELD seed progress: lag1 + lag2 ready; snapshot gate released.");
  }
}

/**
 * @brief Applies the same causal H_in EMA contract used in REPLAY/exported data.
 * @param h_in_raw Current raw indoor humidity sample.
 * @return Contract-aligned indoor humidity after the single causal EMA.
 *
 * FIELD has no exported t-1 EMA state. Therefore, the first valid FIELD sample
 * initializes the EMA state, and all subsequent samples follow the same causal
 * update equation used by the REPLAY/export contract.
 */
static inline float field_filter_hin_like_replay(float h_in_raw){
  const float alpha = clampf((float)LITEML_CONTRACT_HIN_EMA_ALPHA, 0.0f, 1.0f);

  if (!field_hin_ema_ready){
    field_hin_ema_state = h_in_raw;
    field_hin_ema_ready = true;
  } else {
    field_hin_ema_state = alpha * h_in_raw + (1.0f - alpha) * field_hin_ema_state;
  }
  return field_hin_ema_state;
}
#endif

/**
 * @brief One-shot public latch for logical-hour rollover.
 * @return true once when a logical-hour transition is detected.
 */
bool sensors_hour_rollover(){
  if (hour_changed_latch){
    hour_changed_latch = false;
    return true;
  }
  return false;
}

// ============================================================================
// Power-cycle helpers (optional)
// ============================================================================
/**
 * @brief Performs an optional power-cycle on a DHT supply-control pin.
 * @param pin Power-control pin. Negative values disable the operation.
 */
static inline void dht_power_cycle_pin(int pin){
#if USE_DHT22
  if (pin < 0) return;
  pinMode(pin, OUTPUT);
  digitalWrite(pin, LOW);
  delay(DHT_POWER_CYCLE_OFF_MS);
  digitalWrite(pin, HIGH);
  delay(50);
#endif
}

// ============================================================================
// Reinit helpers (anti-stall)
// ============================================================================
/**
 * @brief Attempts DHT IN reinitialization, optionally with power-cycle.
 * @param hard If true, perform power-cycle first when configured.
 */
static inline void maybe_reinit_dht_in(bool hard=false){
#if USE_DHT22
  unsigned long now = millis();
  if (now - last_reinit_in_ms >= DHT_REINIT_COOLDOWN_MS){
    if (hard && DHT_IN_PWR_PIN >= 0){
      dht_power_cycle_pin(DHT_IN_PWR_PIN);
      power_cycle_in++;
    }
    // Pulse the data pin.
    pinMode(PIN_DHT_IN, INPUT_PULLUP);
    delay(5);
    pinMode(PIN_DHT_IN, INPUT);
    delay(5);

    dht_in.begin();
    last_reinit_in_ms = now;
    reinit_count_in++;
    Serial.println(hard ? "[SENSORS] DHT IN power-cycle + reinit." : "[SENSORS] DHT IN reinit.");
  }
#endif
}

/**
 * @brief Attempts DHT OUT reinitialization, optionally with power-cycle.
 * @param hard If true, perform power-cycle first when configured.
 */
static inline void maybe_reinit_dht_out(bool hard=false){
#if USE_DHT22
  unsigned long now = millis();
  if (now - last_reinit_out_ms >= DHT_REINIT_COOLDOWN_MS){
    if (hard && DHT_OUT_PWR_PIN >= 0){
      dht_power_cycle_pin(DHT_OUT_PWR_PIN);
      power_cycle_out++;
    }
    pinMode(PIN_DHT_OUT, INPUT_PULLUP);
    delay(5);
    pinMode(PIN_DHT_OUT, INPUT);
    delay(5);

    dht_out.begin();
    last_reinit_out_ms = now;
    reinit_count_out++;
    Serial.println(hard ? "[SENSORS] DHT OUT power-cycle + reinit." : "[SENSORS] DHT OUT reinit.");
  }
#endif
}

// ============================================================================
// Snapshots per logical hour
// ============================================================================
#if !LITEML_REPLAY
/**
 * @brief Updates logical-hour snapshots in FIELD mode using the same canonical
 *        stream exported to the model.
 * @param tOut Outdoor temperature.
 * @param hOut Outdoor humidity.
 * @param tIn  Indoor temperature.
 * @param hIn  Indoor humidity after the single causal filter.
 *
 * This mirrors the REPLAY contract: the hour snapshots and the current sample
 * come from the same stream, so lag1/lag2 and current values stay coherent.
 */
static MAYBE_UNUSED void maybe_update_hour_snapshot_field(float tOut,float hOut,float tIn,float hIn){
  const int32_t hid = sensors_current_hour_id();
  if (hid != last_hour_id){
    if (last_hour_id == -1){
      // First real FIELD seed: keep only the current snapshot.
      // No lag is exposed yet.
      hour_T_out_curr=tOut; hour_H_out_curr=hOut;
      hour_T_in_curr =tIn;  hour_H_in_curr =hIn;
      field_hour_seed_count = 1;
    } else {
      // From the second valid FIELD hour onward, accumulate real lag history.
      if (field_hour_seed_count >= 2 && have_hour_prev){
        hour_T_in_prev2 = hour_T_in_prev;
        hour_H_in_prev2 = hour_H_in_prev;
        have_hour_prev2 = true;
      }

      if (field_hour_seed_count >= 1){
        hour_T_out_prev = hour_T_out_curr;
        hour_H_out_prev = hour_H_out_curr;
        hour_T_in_prev  = hour_T_in_curr;
        hour_H_in_prev  = hour_H_in_curr;
        have_hour_prev  = true;
      }

      hour_T_out_curr=tOut; hour_H_out_curr=hOut;
      hour_T_in_curr =tIn;  hour_H_in_curr =hIn;

      if (field_hour_seed_count < 3) field_hour_seed_count++;
    }

    maybe_log_field_seed_progress();

    last_hour_id = hid;
    snapshot_id_counter++;
    hour_changed_latch = true;
  }
}
#endif

// REPLAY 1:1: snapshot by hour using raw header values (no EMA / FMA).
#if LITEML_REPLAY
/**
 * @brief Updates logical-hour snapshots in REPLAY mode.
 * @param tOut Outdoor temperature.
 * @param hOut Outdoor humidity.
 * @param tIn  Indoor temperature.
 * @param hIn  Indoor humidity.
 */
static MAYBE_UNUSED void maybe_update_hour_snapshot_replay(float tOut,float hOut,float tIn,float hIn){
  const int32_t hid = sensors_current_hour_id();
  if (hid != last_hour_id){
    if (last_hour_id != -1){
      // Shift prev -> prev2 (lag2).
      if (have_hour_prev){
        hour_T_in_prev2 = hour_T_in_prev;
        hour_H_in_prev2 = hour_H_in_prev;
        have_hour_prev2 = true;
      }
      hour_T_out_prev=hour_T_out_curr; hour_H_out_prev=hour_H_out_curr;
      hour_T_in_prev =hour_T_in_curr;  hour_H_in_prev =hour_H_in_curr;
      have_hour_prev = true;
    }
    hour_T_out_curr=tOut; hour_H_out_curr=hOut;
    hour_T_in_curr =tIn;  hour_H_in_curr =hIn;

    if (last_hour_id == -1){
      const bool keep_seeded_history = replay_seeded && have_hour_prev && have_hour_prev2;
      if (!keep_seeded_history){
        hour_T_out_prev=hour_T_out_curr; hour_H_out_prev=hour_H_out_curr;
        hour_T_in_prev =hour_T_in_curr;  hour_H_in_prev =hour_H_in_curr;
        have_hour_prev = true;

        // Legacy fallback: when seed rows are unavailable, mirror lag1 into lag2.
        hour_T_in_prev2 = hour_T_in_prev;
        hour_H_in_prev2 = hour_H_in_prev;
        have_hour_prev2 = true;
      }
    }
    last_hour_id = hid;
    snapshot_id_counter++;
    hour_changed_latch = true;
  }
}
#endif

// ============================================================================
// DHT reading with per-channel rate limit (active FIELD acquisition path)
// ============================================================================
// ============================================================================
// RAW DHT reading (no median / no temporal filtering) — used in REPLAY 1:1 mode
// Keeps only the per-channel minimum rate limit to respect DHT22 timing.
// ============================================================================
#if !LITEML_REPLAY
/**
 * @brief Reads one raw DHT sample per channel, respecting rate limits.
 * @param[out] tOut Outdoor temperature.
 * @param[out] hOut Outdoor humidity.
 * @param[out] tIn  Indoor temperature.
 * @param[out] hIn  Indoor humidity.
 * @return true if at least one channel is valid.
 */
static MAYBE_UNUSED bool read_dht_raw_once(float& tOut, float& hOut, float& tIn, float& hIn){
#if USE_DHT22
  const unsigned long now = millis();

  auto read_out_once = [&](unsigned long ts){
    last_read_ms_out = ts;
    float to = dht_out.readTemperature();
    float ho = dht_out.readHumidity();
    if (isfinite(to) && saneTemp(to) && isfinite(ho) && saneHum(ho)) {
      reads_ok_out++;
      last_ok_out_ms = ts;
      last_raw_t_out = to;
      last_raw_h_out = ho;
    } else {
      reads_err_out++;
    }
  };

  auto read_in_once = [&](unsigned long ts){
    last_read_ms_in = ts;
    float ti = dht_in.readTemperature();
    float hi = dht_in.readHumidity();
    if (isfinite(ti) && saneTemp(ti) && isfinite(hi) && saneHum(hi)) {
      reads_ok_in++;
      last_ok_in_ms = ts;
      last_raw_t_in = ti;
      last_raw_h_in = hi;
    } else {
      reads_err_in++;
    }
  };

  // Important hardening: the DHT minimum interval must be enforced per ATTEMPT,
  // not only per successful read. Otherwise, a flaky sensor can be retried on
  // every loop iteration while its last valid cache is still NAN, which may
  // hammer the blocking DHT transaction path and trigger watchdog resets.
  const bool out_due = (now - last_read_ms_out >= DHT_MIN_INTERVAL_MS);
  const bool in_due  = (now - last_read_ms_in  >= DHT_MIN_INTERVAL_MS);

  // Execute at most one physical DHT transaction per loop cycle. When both
  // channels are due, alternate priority so neither side starves and the two
  // critical sections are split across different loop iterations.
  if (out_due && in_due) {
    if (g_dht_rr_next == 0) {
      read_out_once(now);
      g_dht_rr_next = 1;
    } else {
      read_in_once(now);
      g_dht_rr_next = 0;
    }
    yield();
  } else if (out_due) {
    read_out_once(now);
    g_dht_rr_next = 1;
    yield();
  } else if (in_due) {
    read_in_once(now);
    g_dht_rr_next = 0;
    yield();
  }

  // Return the latest valid RAW values (or NAN if still unavailable).
  tOut = last_raw_t_out; hOut = last_raw_h_out;
  tIn  = last_raw_t_in;  hIn  = last_raw_h_in;

  bool ok_out = isfinite(tOut) && isfinite(hOut) && saneTemp(tOut) && saneHum(hOut);
  bool ok_in  = isfinite(tIn ) && isfinite(hIn ) && saneTemp(tIn ) && saneHum(hIn );

  // True anti-stall logic: reinit only if the sensor remains too long
  // without a valid reading.
  if (last_ok_out_ms && (now - last_ok_out_ms > DHT_STALE_MS)) { maybe_reinit_dht_out(true); }
  if (last_ok_in_ms  && (now - last_ok_in_ms  > DHT_STALE_MS)) { maybe_reinit_dht_in (true); }

  // Health log every ~60 s.
  if (now - last_health_log_ms > 60000UL){
    last_health_log_ms = now;
    Serial.printf("[SENSORS] Health IN ok=%lu err=%lu reinit=%u pwr=%u | OUT ok=%lu err=%lu reinit=%u pwr=%u\n",
                  (unsigned long)reads_ok_in, (unsigned long)reads_err_in, reinit_count_in, power_cycle_in,
                  (unsigned long)reads_ok_out,(unsigned long)reads_err_out,reinit_count_out,power_cycle_out);
    yield();
  }

  return ok_out || ok_in;
#else
  (void)tOut; (void)hOut; (void)tIn; (void)hIn;
  return false;
#endif
}
#endif

#if LITEML_REPLAY
// ============================================================================
// Header (dataset) reading — used in REPLAY 1:1 mode
// One sample per event (logical hour). Header epoch drives time features.
// When the header ends, the index is not wrapped in validation mode.
// ============================================================================
/**
 * @brief Reads one replay sample from the exported header.
 * @param[out] tOut Outdoor temperature.
 * @param[out] hOut Outdoor humidity.
 * @param[out] tIn  Indoor temperature.
 * @param[out] hIn  Indoor humidity.
 * @return true if the replay sample is valid.
 */
static MAYBE_UNUSED bool read_replay_header_once(float& tOut, float& hOut, float& tIn, float& hIn){
  // IMPORTANT (REPLAY 1:1):
  // read_sensors() may be called multiple times between two logical HOUR events.
  // Therefore, the header advances only when the logical hour ID changes.
  // When 2+47 export is available, rows 0 and 1 are consumed only as lag seeds.
  if (LITEML_REPLAY_TOTAL_RAW_ROWS == 0) return false;

  replay_seed_history_if_needed();

  const int32_t hour_id = sensors_current_hour_id();
  if (!replay_have_cached || hour_id != replay_cached_hour_id){
    replay_cached_hour_id = hour_id;

    if (replay_idx >= LITEML_REPLAY_TOTAL_RAW_ROWS){
      if (!replay_eof){
        replay_eof = true;
        Serial.println("[SENSORS] REPLAY: end of raw header reached — EOF (validation mode). Replay stopped.");
      }
      return false;
    }
      const uint16_t raw_idx = replay_idx;
      replay_cached = LITEML_ROLL24_24_SAMPLES[replay_idx++];
      replay_cached_raw_idx = (int32_t)raw_idx;
      replay_epoch = (time_t)replay_cached.epoch;
      replay_cached_hin =
      replay_filter_hin_from_header((float)replay_cached.H_in_raw);
      replay_have_cached = true;
  }

  tOut = (float)replay_cached.T_out; hOut = (float)replay_cached.H_out;
  tIn  = (float)replay_cached.T_in;  hIn  = replay_cached_hin;

  return saneTemp(tOut) && saneHum(hOut) && saneTemp(tIn) && saneHum(hIn);
}
#endif


#if (USE_OLED && LITEML_TARGET_ESP32) || \
    (BENCH_ENABLE_POWER && (BENCH_PWR_BACKEND == BENCH_PWR_INA219))
/**
 * @brief Initializes the I2C bus used by the ESP32 OLED and/or INA219.
 *
 * OLED support is compiled only for ESP32. The STM32 branch remains available
 * solely for the INA219 backend when power benchmarking is enabled.
 */
static void liteml_i2c_begin() {
#if LITEML_TARGET_ESP32
  Wire.begin(OLED_SDA_PIN, OLED_SCL_PIN);
  Wire.setTimeout(50);
#elif LITEML_TARGET_STM32F411RE
  // STM32F411RE has no OLED path; use the board default I2C pins for INA219.
  Wire.begin();
#endif
}
#endif

// ============================================================================
// Initialization
// ============================================================================
/**
 * @brief Initializes sensors, optional UI, and time/replay state.
 * @return true on successful initialization path.
 */
bool init_sensors() {
#if USE_OLED && LITEML_TARGET_ESP32
  #if LITEML_REPLAY
    g_ui_enabled = (OLED_IN_REPLAY != 0);
  #else
    g_ui_enabled = (OLED_IN_FIELD != 0);
  #endif

  g_oled_initialized = false;
#endif

#if (USE_OLED && LITEML_TARGET_ESP32) || \
    (BENCH_ENABLE_POWER && (BENCH_PWR_BACKEND == BENCH_PWR_INA219))
  // I2C initialization for the ESP32 OLED and/or INA219.
  liteml_i2c_begin();
#endif

#if LITEML_TARGET_ESP32
  #if USE_OLED
    Serial.printf("[OLED] %s | mode=%s\n",
                  g_ui_enabled ? "enabled" : "disabled",
                  LITEML_REPLAY ? "REPLAY" : "FIELD");

    // Keep the shared I2C bus available for INA219 in all cases.
    // If the UI is enabled, fully initialize and show the boot screen.
    // If the UI is disabled, perform a one-time minimal initialization only
    // to force the panel physically off, then leave it untouched.
    if (g_ui_enabled) {
      if (oled_ensure_initialized()) {
        display.clear();
        display.display();
        display.displayOn();

        #if LITEML_REPLAY
          display.drawString(0, 0, "REPLAY 1:1 MODE");
        #else
          display.drawString(0, 0, "Initializing sensors...");
        #endif

        display.display();
      }
    } else {
      if (oled_ensure_initialized()) {
        display.clear();
        display.display();
        display.displayOff();
      }
    }
  #else
    Serial.println("[OLED] compile-time disabled.");
  #endif
#endif

bool ok_sensors = true;

#if USE_DHT22 && !LITEML_REPLAY
  // Enable optional power pins.
  if (DHT_IN_PWR_PIN  >= 0){ pinMode(DHT_IN_PWR_PIN , OUTPUT); digitalWrite(DHT_IN_PWR_PIN , HIGH); }
  if (DHT_OUT_PWR_PIN >= 0){ pinMode(DHT_OUT_PWR_PIN, OUTPUT); digitalWrite(DHT_OUT_PWR_PIN, HIGH); }

  dht_out.begin();
  dht_in.begin();

  delay(100);

  float t0 = dht_out.readTemperature();
  float t1 = dht_in.readTemperature();
  if (isnan(t0) || isnan(t1)) {
    Serial.println("[SENSORS] Warning: DHT22 has no valid reading yet (continuing; EMA will seed later).");
  } else {
    Serial.println("[SENSORS] DHT22 initialized successfully.");
  }
#endif

#if LITEML_REPLAY
  millis_start = millis();
  time_ok = false;
  replay_idx = 0;
  replay_epoch = 0;
  replay_seeded = false;
  replay_have_cached = false;
  replay_cached_hour_id = INT32_MIN;
  Serial.println("[SENSORS] REPLAY: using header epoch for time features.");
#elif USE_WIFI_TIME
  connect_time_if_needed();
  if (!time_ok) {
    Serial.println("[SENSORS] NTP not synchronized — falling back to millis().");
  } else {
    Serial.println("[SENSORS] Time synchronized via NTP.");
  }
#else
  millis_start = millis();
  Serial.println("[SENSORS] No NTP — using local millis-based clock.");
#endif

#if LITEML_REPLAY
  replay_idx = 0;
  replay_epoch = 0;
  replay_seeded = false;
  replay_eof = false;
  replay_have_cached = false;
  replay_cached_hour_id = INT32_MIN;
  replay_cached = {};
  replay_cached_raw_idx = -1;
  replay_cached_hin = NAN;
  replay_hin_ema_state = NAN;
  replay_hin_ema_ready = false;
#else
  field_hin_ema_state = NAN;
  field_hin_ema_ready = false;
  field_hour_seed_count = 0;
  field_hour_seed_reported = 0xFF;
#endif

  have_hour_prev = false;
  have_hour_prev2 = false;
  warmup_ok_count = 0;
  last_ui_ms = millis();
  last_hour_id = -1;
  snapshot_id_counter = 0;
  hour_changed_latch = false;

#if LITEML_DBG_SENSORS_EXACT
  g_dbg_sensors_header_printed = false;
  g_dbg_pre_raw_header_printed = false;
  g_dbg_pre_smooth_header_printed = false;
  g_dbg_sensors_idx = 0;
  g_dbg_sensors_last_hour_id = INT32_MIN;
#endif

  // Reset anti-stall state.
  last_raw_t_in  = last_raw_h_in  = NAN;
  last_raw_t_out = last_raw_h_out = NAN;
  last_ok_in_ms  = millis();
  last_ok_out_ms = millis();
  last_reinit_in_ms = last_reinit_out_ms = 0;
  reinit_count_in = reinit_count_out = 0;
  power_cycle_in = power_cycle_out = 0;

  // Reset rate-limited raw acquisition timestamps.
#if LITEML_TARGET_STM32F411RE
  last_read_ms_in = millis();
  last_read_ms_out = millis();
#else
  last_read_ms_in = 0;
  last_read_ms_out = 0;
#endif

  // Statistics.
  reads_ok_in = reads_ok_out = reads_err_in = reads_err_out = 0;
  last_health_log_ms = millis();

#if LITEML_REPLAY
  replay_seed_history_if_needed();
#endif

  Serial.println("[SENSORS] Initialization complete. Acquisition layer ready; unified evaluation contract armed.");
  return ok_sensors;
}

// ============================================================================
// Filtered read loop + lags / time features
// ============================================================================
// (REPLAY / FIELD) read_sensors() must be compiled in both modes.
/**
 * @brief Reads sensors or replay samples and produces a SensorData packet.
 * @return SensorData structure with current values, lags, time features, and status.
 */
SensorData read_sensors(){
  SensorData out{};
  out.valid=false;
  out.have_snapshot=sensors_have_snapshot();
  out.warmup_ok_count=warmup_ok_count;

  // 1) Source read.
  // REPLAY consumes the exported header.
  // FIELD consumes the live sensor stream, but after acquisition both modes
  // then follow the same post-acquisition evaluation contract.
  float tOut=0,hOut=0,tIn=0,hIn=0;
  float hIn_raw_dbg = NAN;
  long  epoch_dbg = 0L;
  int32_t raw_idx_dbg = -1;
#if LITEML_REPLAY
  const bool replay_ok = read_replay_header_once(tOut,hOut,tIn,hIn);
  if (!replay_ok){
    // REPLAY validation mode: once EOF is latched (or the current replay row
    // is unavailable/invalid), do not synthesize a sample from zero-initialized
    // locals. Return immediately and let the caller observe the EOF latch.
    return out;
  }
  if (replay_have_cached) {
    hIn_raw_dbg = (float)replay_cached.H_in_raw;
    epoch_dbg = (long)replay_epoch;
    raw_idx_dbg = replay_cached_raw_idx;
  } else {
    hIn_raw_dbg = hIn;
    epoch_dbg = (long)replay_epoch;
  }
#else
  read_dht_raw_once(tOut,hOut,tIn,hIn);
  hIn_raw_dbg = hIn;
  if (time_ok) {
    time_t now_dbg; time(&now_dbg);
    epoch_dbg = (long)now_dbg;
  }
#endif

  const bool ok_out_raw = saneTemp(tOut) && saneHum(hOut);
  const bool ok_in_raw  = saneTemp(tIn)  && saneHum(hIn);
  const bool ok_any_raw = ok_out_raw || ok_in_raw;
  const bool ok_both_raw = ok_out_raw && ok_in_raw;

  // Evaluation path: if the current source sample is invalid, do not create
  // synthetic model inputs. This keeps FIELD consistent with REPLAY.
  if (!ok_any_raw){ return out; }

#if LITEML_REPLAY
  // REPLAY 1:1: header values only.

  // Warm-up: count only when both channels are valid in the header.
  if (ok_both_raw && warmup_ok_count < 6) warmup_ok_count++;

  // Hour snapshot (lags) using the same canonical stream exported to the model.
  maybe_update_hour_snapshot_replay(tOut, hOut, tIn, hIn);
#else
  // FIELD evaluation contract: the common post-acquisition path must only run
  // when the complete physical sample is available. With the hardened DHT
  // scheduler, OUT and IN may become valid in different loop iterations; using
  // a partial sample here would poison the causal H_in EMA and advance the
  // snapshot contract with mixed / incomplete data.
  if (!ok_both_raw){
    return out;
  }

  // Keep T/H channels raw, and apply the single causal EMA only to H_in.
  hIn = field_filter_hin_like_replay(hIn);

  if (warmup_ok_count < 6) warmup_ok_count++;

  // Hour snapshot (lags) using the same canonical stream exported to the model.
  maybe_update_hour_snapshot_field(tOut, hOut, tIn, hIn);
#endif

  // 7) Output
  out.valid = ok_any_raw;

  out.T_out_raw = tOut;
  out.H_out_raw = hOut;
  out.T_in_raw  = tIn;
  out.H_in_raw  = hIn_raw_dbg;

  out.T_out_smooth = tOut;
  out.H_out_smooth = hOut;
  out.T_in_smooth  = tIn;
  out.H_in_smooth  = hIn;

  out.T_out = out.T_out_smooth; out.H_out = out.H_out_smooth;
  out.T_in  = out.T_in_smooth;  out.H_in  = out.H_in_smooth;

#if SENSORS_DEBUG_RAW && !LITEML_REPLAY
  // 1 Hz log: shows the last RAW value read by the DHT and the exported value
  // after the unified evaluation contract.
  static uint32_t dbg_last_ms = 0;
  uint32_t now_ms = millis();
  if (now_ms - dbg_last_ms >= 1000) {
    dbg_last_ms = now_ms;
    Serial.printf("[SENSORS] RAW vs FILT | IN T=%.3f->%.3f H=%.3f->%.3f | OUT T=%.3f->%.3f H=%.3f->%.3f | ok=%d\n",
                  last_raw_t_in,  out.T_in,  last_raw_h_in,  out.H_in,
                  last_raw_t_out, out.T_out, last_raw_h_out, out.H_out,
                  (int)ok_any_raw);
  }
#endif

  // Staleness diagnostics: indicates prolonged absence of valid sensor reads.
  if (!LITEML_REPLAY) {
    static unsigned long last_stale_log_ms = 0;
    const unsigned long nowms2 = millis();
    const bool out_stale = (last_ok_out_ms != 0) && ((nowms2 - last_ok_out_ms) > DHT_STALE_MS);
    const bool in_stale  = (last_ok_in_ms  != 0) && ((nowms2 - last_ok_in_ms ) > DHT_STALE_MS);
    if ((out_stale || in_stale) && (nowms2 - last_stale_log_ms > 5000UL)){
      last_stale_log_ms = nowms2;
      Serial.printf("[SENSORS] STALE INPUT IN=%d OUT=%d | ms_since_ok in=%lu out=%lu\n",
                    (int)in_stale, (int)out_stale,
                    (unsigned long)(nowms2 - last_ok_in_ms),
                    (unsigned long)(nowms2 - last_ok_out_ms));
    }
  }

  out.T_in_lag1  = have_hour_prev ? hour_T_in_prev  : out.T_in;
  out.H_in_lag1  = have_hour_prev ? hour_H_in_prev  : out.H_in;
  out.T_out_lag1 = have_hour_prev ? hour_T_out_prev : out.T_out;
  out.H_out_lag1 = have_hour_prev ? hour_H_out_prev : out.H_out;

  // lag2: two hours back (when unavailable, mirror lag1 for stability)
  out.T_in_lag2  = have_hour_prev2 ? hour_T_in_prev2 : out.T_in_lag1;
  out.H_in_lag2  = have_hour_prev2 ? hour_H_in_prev2 : out.H_in_lag1;

  get_time_features(out.sin_hour, out.cos_hour, out.weekday_pandas, out.month_1_12);
  out.have_snapshot   = sensors_have_snapshot();
  out.warmup_ok_count = warmup_ok_count;

#if LITEML_DBG_SENSORS_EXACT
  {
    const int32_t hid_dbg = last_hour_id;
    bool allow_dbg = true;
  #if LITEML_REPLAY
    allow_dbg = !replay_eof;
  #else
    allow_dbg = false;
  #endif
    if (allow_dbg && out.valid && hid_dbg != INT32_MIN && hid_dbg != g_dbg_sensors_last_hour_id) {
      g_dbg_sensors_last_hour_id = hid_dbg;
    #if LITEML_REPLAY
      const char* mode_dbg = "REPLAY";
    #else
      const char* mode_dbg = "FIELD";
    #endif
      const int dbg_idx = g_dbg_sensors_idx++;
      Serial.println();
      dbg_pre_raw_print_row(
        dbg_idx,
        mode_dbg,
        raw_idx_dbg,
        epoch_dbg,
        out.T_out_raw, out.H_out_raw,
        out.T_in_raw, out.H_in_raw
      );
      dbg_pre_smooth_print_row(
        dbg_idx,
        mode_dbg,
        raw_idx_dbg,
        epoch_dbg,
        out.T_out_smooth, out.H_out_smooth,
        out.T_in_smooth, out.H_in_smooth
      );
      dbg_sensors_print_row(
        dbg_idx,
        mode_dbg,
        raw_idx_dbg,
        epoch_dbg,
        out.T_out, out.H_out,
        out.T_in,
        hIn_raw_dbg,
        out.H_in,
        out.T_in_lag1, out.H_in_lag1,
        out.T_out_lag1, out.H_out_lag1,
        out.T_in_lag2, out.H_in_lag2,
        out.sin_hour, out.cos_hour,
        out.weekday_pandas, out.month_1_12,
        have_hour_prev, have_hour_prev2,
        warmup_ok_count
      );
    }
  }
#endif

#if defined(SENSOR_DATA_HAS_HOUR_ID)
  out.hour_id = last_hour_id;
#endif
#if defined(SENSOR_DATA_HAS_SNAPSHOT_ID)
  out.snapshot_id = snapshot_id_counter;
#endif

#if USE_OLED && LITEML_TARGET_ESP32 && !LITEML_REPLAY
  // Shared-bus hardening:
  // during early FIELD warm-up the loop is already busy with DHT reads,
  // warm-up/event serial logging, and optional INA219 polling. Postpone the
  // periodic OLED framebuffer transfer until the snapshot gate is ready so the
  // UI does not add avoidable I2C pressure in the reset-prone boot window.
  const bool oled_runtime_ready = out.have_snapshot && (out.warmup_ok_count >= 6);
  if (!oled_runtime_ready) {
    last_ui_ms = millis();
  } else if (g_ui_enabled && (millis() - last_ui_ms > OLED_REFRESH_MS)){
    if (oled_ensure_initialized()) {
      last_ui_ms = millis();
      display.clear();
      display.drawRect(2,2,124,60);

      // Avoid String concatenation inside the loop to reduce heap fragmentation.
      char l1[32], l2[32], l3[32], l4[32];
      snprintf(l1, sizeof(l1), "Tout:%.2f Hout:%.2f", out.T_out, out.H_out);
      snprintf(l2, sizeof(l2), "Tin :%.2f Hin :%.2f", out.T_in,  out.H_in);

      #if LITEML_REPLAY
        snprintf(l3, sizeof(l3), "mode:REPLAY eof:%d", (int)sensors_replay_eof());
        snprintf(l4, sizeof(l4), "lag:%s", have_hour_prev ? "ok" : "boot");
      #else
        snprintf(l3, sizeof(l3), "lag:%s", have_hour_prev ? "ok" : "boot");
        snprintf(l4, sizeof(l4), "RTC:%s", sensors_time_ready() ? "ok" : "fallback");
      #endif

      display.drawString(6, 6,  String(l1));
      display.drawString(6, 20, String(l2));
      display.drawString(6, 34, String(l3));
      display.drawString(6, 48, String(l4));
      display.display();
      yield();
    }
  }
#endif
  return out;
}


// ======= Exposed power-save helpers =======
/**
 * @brief Turns the OLED display off when supported.
 */
void sensors_ui_off(){
#if USE_OLED && LITEML_TARGET_ESP32
  g_ui_enabled = false;
  if (g_oled_initialized) {
    display.clear();
    display.display();
    display.displayOff();
  }
#endif
}

/**
 * @brief Re-enables the OLED user interface when it is available at compile time.
 *
 * The function restores panel visibility without reconfiguring the shared I2C
 * bus, preserving coexistence with other peripherals such as INA219.
 */
void sensors_ui_on(){
#if USE_OLED && LITEML_TARGET_ESP32
  g_ui_enabled = true;
  if (oled_ensure_initialized()) {
    display.displayOn();
    last_ui_ms = 0; // forces immediate refresh on the next read_sensors() cycle
  }
#endif
}

/**
 * @brief Enables network resources needed by the sensor/time layer.
 *
 * Centralizes Wi-Fi bring-up so NTP synchronization can use a consistent path.
 */
void sensors_network_on() {
  #if USE_WIFI_TIME && LITEML_TARGET_ESP32
    // Ensure Wi-Fi radio is on and in STA mode.
    if (WiFi.getMode() == WIFI_OFF) {
      WiFi.mode(WIFI_STA);
    } else if (WiFi.getMode() != WIFI_STA) {
      WiFi.mode(WIFI_STA);
    }
    WiFi.setSleep(false);

    // Connect if not already connected.
    if (WiFi.status() != WL_CONNECTED) {
      #ifdef WIFI_SSID
      #ifdef WIFI_PSK
        WiFi.begin(WIFI_SSID, WIFI_PSK);
      #else
        WiFi.begin(WIFI_SSID);
      #endif
      #endif

      unsigned long t0 = millis();
      const unsigned long timeout_ms = 10000UL; // 10 s safety limit
      while (WiFi.status() != WL_CONNECTED && (millis() - t0) < timeout_ms) {
        delay(100);
      }
    }
  #endif
}

/**
 * @brief Disables Wi-Fi and Bluetooth resources when possible.
 */
void sensors_network_off() {
#if USE_WIFI_TIME && LITEML_TARGET_ESP32

  if (WiFi.getMode() != WIFI_OFF) {
    WiFi.setSleep(true);
    WiFi.mode(WIFI_OFF);
  }

#endif

#if USE_BLUETOOTH && LITEML_TARGET_ESP32

  if (btStarted()) {
    btStop();
  }

  if (
    esp_bt_controller_get_status() ==
    ESP_BT_CONTROLLER_STATUS_ENABLED
  ) {
    esp_bt_controller_disable();
  }

#endif
}

static float g_last_Tin = NAN, g_last_Hin = NAN;
static bool  g_has_prev  = false;

/**
 * @brief Updates the external previous-value cache from current readings.
 * @param Tin Current indoor temperature.
 * @param Hin Current indoor humidity.
 */
void sensors_update_prev_from_current(float Tin, float Hin){
  g_last_Tin = Tin;
  g_last_Hin = Hin;
  g_has_prev = true;
}

/**
 * @brief Returns previous Tin / Hin values for residual reconstruction.
 * @param[out] Tprev Previous indoor temperature.
 * @param[out] Hprev Previous indoor humidity.
 * @return true when previous logical-hour values are available.
 *
 * To preserve 1:1 consistency with residual training, this function returns
 * the previous logical-hour snapshot updated by maybe_update_hour_snapshot().
 */
bool sensors_get_prev_T_H(float* Tprev, float* Hprev){
  if (!have_hour_prev) return false;
  if (Tprev) *Tprev = hour_T_in_prev;
  if (Hprev) *Hprev = hour_H_in_prev;
  return true;
}