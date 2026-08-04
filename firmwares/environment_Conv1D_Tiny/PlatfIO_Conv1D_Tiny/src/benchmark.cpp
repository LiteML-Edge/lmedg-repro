/*
 * ============================================================================
 *  benchmark.cpp
 *  LiteML-Edge Benchmark Support Module
 * ============================================================================
 *
 *  Purpose
 *  -------
 *  This module provides lightweight and auditable benchmarking utilities for
 *  embedded TinyML experiments, with emphasis on reproducibility and reporting
 *  quality compatible with IEEE LATAM style requirements.
 *
 *  Main responsibilities
 *  ---------------------
 *  1. Latency measurement in microseconds.
 *  2. Heap and TensorFlow Lite Micro arena memory inspection.
 *  3. Optional power acquisition through an abstract backend interface.
 *  4. Energy accumulation over time from sampled power measurements.
 *  5. Standardized benchmark line printing for serial logs.
 *
 *  Design notes
 *  ------------
 *  - The implementation preserves portability through conditional compilation.
 *  - ESP32 receives the most complete heap diagnostics.
 *  - AVR uses a conservative architecture-specific heap estimate.
 *  - Other architectures fall back to a safe zero-information memory report.
 *  - Power measurement is abstracted to allow backend replacement without
 *    changing the benchmark API.
 *  - If available at link time, TensorFlow Lite Micro arena usage is reported
 *    through tflm_arena_used_bytes().
 *
 *  Reproducibility and auditability
 *  --------------------------------
 *  - INA219 calibration is explicitly selected at compile time.
 *  - Serial logs expose the active calibration preset when enabled.
 *  - Outlier rejection and EMA smoothing are documented and localized.
 *  - The module avoids unsafe linker-symbol assumptions on unsupported targets.
 *
 *  Integration context
 *  -------------------
 *  This file is intended to support LiteML-Edge firmware evaluation pipelines,
 *  including latency, memory, power, and cumulative energy reporting during
 *  inference and system operation.
 *
 * ============================================================================
 */

#include "benchmark.h"
#include "config.h"

// Idle power telemetry can share the I2C bus with OLED updates.
// Polling the backend on every loop iteration is unnecessary for energy
// integration and may stress the bus / serial path enough to trigger long
// stalls on ESP32. Keep a bounded poll cadence and reuse the cached sample
// between polls.
#ifndef BENCH_POWER_POLL_MIN_MS
  #define BENCH_POWER_POLL_MIN_MS 500UL
#endif

#ifndef BENCH_I2C_FREQ_HZ
#define BENCH_I2C_FREQ_HZ 400000
#endif

#ifndef BENCH_I2C_USE_OLED_PINS
// Set to 1 to initialize Wire using the OLED pins.
#define BENCH_I2C_USE_OLED_PINS 0
#endif

// ======================= LATENCY =======================
static uint32_t _t0 = 0;

/**
 * @brief Starts a latency measurement window.
 *
 * Captures the current timestamp in microseconds so that the elapsed time
 * can later be computed with bench_latency_end_us().
 */
void bench_latency_begin(){ _t0 = micros(); }

/**
 * @brief Ends the latency measurement window.
 * @return Elapsed time in microseconds since bench_latency_begin().
 */
uint32_t bench_latency_end_us(){ return micros() - _t0; }

// ======================= MEMORY ========================
#if LITEML_TARGET_ESP32
  #include <esp_heap_caps.h>
#elif LITEML_TARGET_STM32F411RE
  #include <malloc.h>

  /*
   * STM32 runtime-memory diagnostics combine two sources:
   * - mallinfo().fordblks: free chunks already managed by newlib malloc;
   * - the live gap from _sbrk(0) to __get_MSP(): RAM not yet committed to
   *   either the dynamic heap or the active main stack.
   *
   * Their sum represents the memory currently available for future dynamic
   * allocations. The minimum of that same quantity is accumulated as the
   * STM32 free-heap low-water mark, matching the meaning of the ESP32 field.
   *
   * No ESP32 path is changed by this block.
   */
  extern "C" {
    extern char _end;
    void* _sbrk(int incr) __attribute__((weak));
  }

  static uint32_t g_stm32_min_free_heap = UINT32_MAX;

  static uint32_t bench_stm32_nonnegative_int(int value) {
    return (value > 0) ? static_cast<uint32_t>(value) : 0u;
  }

  static void bench_stm32_memory_snapshot(BenchMemory& m) {
    const uintptr_t heap_start = reinterpret_cast<uintptr_t>(&_end);

    uintptr_t heap_current = heap_start;
    if (_sbrk) {
      void* const current_break = _sbrk(0);
      if (current_break != reinterpret_cast<void*>(-1)) {
        const uintptr_t candidate = reinterpret_cast<uintptr_t>(current_break);
        if (candidate >= heap_start) {
          heap_current = candidate;
        }
      }
    }

    const uintptr_t stack_current = static_cast<uintptr_t>(__get_MSP());

    const uint32_t uncommitted_gap =
        (stack_current > heap_current)
            ? static_cast<uint32_t>(stack_current - heap_current)
            : 0u;

    const uint32_t heap_committed =
        (heap_current > heap_start)
            ? static_cast<uint32_t>(heap_current - heap_start)
            : 0u;

    /*
     * mallinfo() reports the state of the portion already committed to malloc.
     * fordblks is the sum of reusable free chunks; keepcost is the top free
     * chunk, which is contiguous with future _sbrk growth.
     */
    #if defined(__GNUC__)
      #pragma GCC diagnostic push
      #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    #endif
    const struct mallinfo allocator_info = mallinfo();
    #if defined(__GNUC__)
      #pragma GCC diagnostic pop
    #endif

    uint32_t allocator_free =
        bench_stm32_nonnegative_int(allocator_info.fordblks);
    if (allocator_free > heap_committed) {
      allocator_free = heap_committed;
    }

    uint32_t top_free =
        bench_stm32_nonnegative_int(allocator_info.keepcost);
    if (top_free > allocator_free) {
      top_free = allocator_free;
    }

    const uint32_t current_free_heap = allocator_free + uncommitted_gap;

    if (current_free_heap < g_stm32_min_free_heap) {
      g_stm32_min_free_heap = current_free_heap;
    }

    /*
     * STM32 semantics:
     * - free_heap: reusable malloc chunks + uncommitted heap-to-stack RAM;
     * - min_free_heap: smallest value of that same free-heap quantity;
     * - largest_free_block: top malloc chunk + adjacent uncommitted RAM;
     * - heap_used: committed heap bytes that are currently allocated;
     * - heap_total: allocated bytes + currently available free-heap bytes.
     *
     * This preserves heap_used == heap_total - free_heap and makes the
     * free_heap/min_free_heap fields semantically comparable with ESP32.
     */
    m.free_heap          = current_free_heap;
    m.min_free_heap      = (g_stm32_min_free_heap == UINT32_MAX)
                             ? current_free_heap
                             : g_stm32_min_free_heap;
    m.largest_free_block = top_free + uncommitted_gap;
    m.heap_used          = (heap_committed >= allocator_free)
                             ? (heap_committed - allocator_free)
                             : 0u;
    m.heap_total         = m.heap_used + m.free_heap;
  }
#endif

// Optional weak hook: if the inference module exposes tflm_arena_used_bytes(),
// we report it; otherwise, the value remains zero without breaking build/link.
extern "C" __attribute__((weak)) size_t tflm_arena_used_bytes(void);

/**
 * @brief Captures a memory usage snapshot for the current platform.
 * @return BenchMemory structure filled with heap and TFLM arena information.
 *
 * Platform behavior:
 * - ESP32: reports reliable absolute heap statistics.
 * - STM32F411RE: reports current/minimum free heap from newlib plus uncommitted RAM.
 * - AVR: estimates free heap using architecture-specific linker symbols.
 * - Other targets: safely returns zeroed heap information.
 */
BenchMemory bench_memory_snapshot(){
  BenchMemory m{};
#if LITEML_TARGET_ESP32
  // ESP32: reliable absolute values, independent of FreeRTOS APIs.
  m.free_heap          = heap_caps_get_free_size(MALLOC_CAP_8BIT);
  m.min_free_heap      = heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT);
  m.largest_free_block = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
  m.heap_total         = heap_caps_get_total_size(MALLOC_CAP_8BIT);
  m.heap_used          = (m.heap_total >= m.free_heap) ? (m.heap_total - m.free_heap) : 0;
#elif LITEML_TARGET_STM32F411RE
  // STM32F411RE: newlib free chunks + live uncommitted heap-to-stack RAM.
  bench_stm32_memory_snapshot(m);
#elif defined(ARDUINO_ARCH_AVR)
  // These symbols exist only on AVR; do NOT use them on other architectures.
  extern char _end;
  extern char *__brkval;
  char top;
  uint32_t heap_free = (uint32_t)(&top - (__brkval ? __brkval : &_end));
  m.free_heap          = heap_free;
  m.min_free_heap      = 0;
  m.largest_free_block = 0;
  m.heap_total         = 0;
  m.heap_used          = 0;
#else
  // Generic fallback without linker-symbol assumptions.
  m.free_heap          = 0;
  m.min_free_heap      = 0;
  m.largest_free_block = 0;
  m.heap_total         = 0;
  m.heap_used          = 0;
#endif

  // TensorFlow Lite Micro arena usage, if the symbol exists at link time.
  if (tflm_arena_used_bytes) {
    m.tflm_arena_used = tflm_arena_used_bytes();
  } else {
    m.tflm_arena_used = 0;
  }
  return m;
}

// ======================= POWER BACKENDS =================
// --- Abstract interface
class IPowerBackend {
public:
  virtual ~IPowerBackend(){}
  virtual bool begin() = 0;
  virtual bool read(float& v, float& i_mA, float& p_mW) = 0;
};

// --- INA219 backend
#if (BENCH_PWR_BACKEND == BENCH_PWR_INA219)
  #include <Wire.h>
  #include <Adafruit_INA219.h>

  static Adafruit_INA219 _ina219;

  /**
   * @brief INA219-based power backend.
   *
   * Provides voltage, current, and derived power measurements using the
   * Adafruit INA219 driver and an explicitly selected calibration preset.
   */
  class PwrINA219 : public IPowerBackend {
public:
  /**
   * @brief Initializes I2C and the INA219 sensor.
   * @return true on success, false otherwise.
   *
   * This method also applies an explicit INA219 calibration preset for
   * reproducibility and scientific traceability.
   */
  bool begin() override {
    // The shared I2C bus is already initialized in init_sensors() for both
    // OLED and INA219 paths. Reinitializing Wire here can transiently disturb
    // the live bus configuration during boot and later UI/power telemetry use.
    // Keep only the timeout hardening and let INA219 attach to the existing bus.
  #if LITEML_TARGET_ESP32
    Wire.setTimeOut(50);
  #endif

    if (!_ina219.begin()) return false;

    // Explicit calibration selection (reproducibility / paper-proof reporting).
  #if (BENCH_INA219_CALIB == BENCH_INA219_CALIB_32V_2A)
    _ina219.setCalibration_32V_2A();
  #elif (BENCH_INA219_CALIB == BENCH_INA219_CALIB_32V_1A)
    _ina219.setCalibration_32V_1A();
  #elif (BENCH_INA219_CALIB == BENCH_INA219_CALIB_16V_400mA)
    _ina219.setCalibration_16V_400mA();
  #else
    #error "Invalid BENCH_INA219_CALIB value (use 32V_2A, 32V_1A, or 16V_400mA)"
  #endif

  #if BENCH_INA219_LOG_CALIB
    Serial.printf("[BENCH][INA219] calib=");
    #if (BENCH_INA219_CALIB == BENCH_INA219_CALIB_32V_2A)
      Serial.printf("32V/2A");
    #elif (BENCH_INA219_CALIB == BENCH_INA219_CALIB_32V_1A)
      Serial.printf("32V/1A");
    #elif (BENCH_INA219_CALIB == BENCH_INA219_CALIB_16V_400mA)
      Serial.printf("16V/400mA");
    #endif
    Serial.printf(" | assumed_shunt=%.3f ohm\n", (float)BENCH_INA219_SHUNT_OHMS);
  #endif

    return true;
  }

  /**
   * @brief Reads bus voltage, current, and derived power.
   * @param[out] v     Bus voltage in volts.
   * @param[out] i_mA  Current in milliamperes.
   * @param[out] p_mW  Power in milliwatts.
   * @return true if the measurement is valid, false otherwise.
   *
   * Power is computed from bus voltage and measured current:
   *   P = V * I
   */
  bool read(float& v, float& i_mA, float& p_mW) override {
    // V = bus voltage on the load side.
    v = _ina219.getBusVoltage_V();
    yield();

    // I = current already calibrated according to the selected preset.
    i_mA = _ina219.getCurrent_mA();
    yield();

    if (!isfinite(v) || !isfinite(i_mA) || v <= 0.0f) return false;

    // Bus power in mW, consistent with reporting conventions.
    p_mW = v * (i_mA / 1000.0f) * 1000.0f; // V*A -> W -> mW
    return true;
  }
};
  static PwrINA219 backend;

// --- Null backend
#else
  /**
   * @brief Null power backend.
   *
   * Used when no physical power sensor backend is enabled. All measurements
   * are reported as zero while preserving the benchmark API.
   */
  class PwrNull : public IPowerBackend {
  public:
    bool begin() override { return true; }
    bool read(float& v, float& i_mA, float& p_mW) override {
      v=0; i_mA=0; p_mW=0; return true;
    }
  };
  static PwrNull backend;
#endif

// ======================= ENERGY INTEGRATION =============
static bool     _pwr_ready=false;
static float    _energy_mWh=0.0f;
static float    _ema_i_mA=0.0f;
static float    _ema_p_mW=0.0f;
static bool     _ema_init=false;
static float    _mu_i=0.0f, _s2_i=0.0f; static uint32_t _n_i=0;
static uint32_t _last_power_us=0;
static uint32_t _last_power_poll_ms=0;
static BenchPower _last_pwr{};

/**
 * @brief Fully clears the power-integration runtime state.
 * @param keep_last_sample If true, preserves the last instantaneous V/I/P values
 *                         and only resets the accumulated energy field.
 *
 * This helper is used to harden energy reset/reinitialization so that a new
 * accumulation window does not inherit stale dt, EMA state, or outlier
 * statistics from the previous window.
 */
static void bench_power_clear_runtime_(bool keep_last_sample){
  _energy_mWh = 0.0f;

  _ema_i_mA   = 0.0f;
  _ema_p_mW   = 0.0f;
  _ema_init   = false;

  _mu_i       = 0.0f;
  _s2_i       = 0.0f;
  _n_i        = 0;

  _last_power_us = micros();
  _last_power_poll_ms = millis();

  if (keep_last_sample) {
    _last_pwr.energy_mWh = 0.0f;
  } else {
    _last_pwr = {};
  }
}

/**
 * @brief Initializes the active power backend and resets energy accounting.
 * @return true if the backend is ready, false otherwise.
 */
bool bench_power_begin(){
  // Always start from a clean integration state, even if begin() fails.
  bench_power_clear_runtime_(false);

  _pwr_ready = backend.begin();
  if (!_pwr_ready) {
    // Keep the state fully reset so callers never inherit stale energy.
    bench_power_clear_runtime_(false);
    return false;
  }

  // Rebase the integration window after the backend is effectively ready.
  bench_power_clear_runtime_(false);
  return true;
}

/**
 * @brief Polls the power backend and updates cumulative energy.
 * @param[out] out Last valid power snapshot.
 * @return true if a new valid reading was acquired, false otherwise.
 *
 * Processing steps:
 * 1. Acquire raw voltage/current/power.
 * 2. Update running statistics for current.
 * 3. Reject strong outliers after warm-up using a 3-sigma rule.
 * 4. Smooth current and power using exponential moving average (EMA).
 * 5. Integrate power over elapsed time to accumulate energy in mWh.
 */
bool bench_power_poll(BenchPower& out){
  if (!_pwr_ready) { out = _last_pwr; return false; }

  const uint32_t now_us = micros();

  float v=0, i_mA=0, p_mW=0;
  if (!backend.read(v, i_mA, p_mW)) { out = _last_pwr; return false; }

  // Outlier rejection (3-sigma after warm-up) and EMA smoothing.
  // Update running mean/variance for current.
  _n_i++;
  float delta = i_mA - _mu_i;
  _mu_i += delta / (float)_n_i;
  _s2_i += delta * (i_mA - _mu_i);
  float sigma = (_n_i>1) ? sqrtf(_s2_i/(_n_i-1)) : 0.0f;

  bool reject = (_n_i>10 && fabsf(i_mA - _mu_i) > 3.0f * fmaxf(1.0f, sigma));
  if (reject) {
    // Keep the previous EMA as a conservative estimate.
    i_mA = _ema_init ? _ema_i_mA : i_mA;
    p_mW = v * (i_mA/1000.0f) * 1000.0f;
  }

  // EMA smoothing (alpha tuned for approximately 1-2 Hz sampling).
  const float A = 0.2f;
  if (!_ema_init){ _ema_i_mA = i_mA; _ema_p_mW = p_mW; _ema_init=true; }
  else { _ema_i_mA = A*i_mA + (1.0f-A)*_ema_i_mA; _ema_p_mW = A*p_mW + (1.0f-A)*_ema_p_mW; }
  i_mA = _ema_i_mA; p_mW = _ema_p_mW;

  uint32_t dt_us = now_us - _last_power_us;
  _last_power_us = now_us;

  // mW * us -> mWh by dividing by 3,600,000,000.
  _energy_mWh += (p_mW * (dt_us / 3600000000.0f));

  _last_pwr.voltage    = v;
  _last_pwr.current    = i_mA;
  _last_pwr.power      = p_mW;
  _last_pwr.energy_mWh = _energy_mWh;

  out = _last_pwr;
  return true;
}

/**
 * @brief Resets cumulative energy integration to zero.
 */
void bench_power_reset_energy(){
  // Harden reset semantics:
  // - clears accumulated energy;
  // - rebases the time origin to avoid integrating stale dt;
  // - clears EMA and outlier statistics so a new energy window starts clean.
  // Preserve the last instantaneous sample for observability.
  bench_power_clear_runtime_(true);
}

/**
 * @brief Returns the most recent cached power snapshot without forcing a new backend read.
 * @param[out] out Structure filled with the last stored voltage, current, power, and energy values.
 * @return true when the power backend has been initialized successfully and cached data are valid.
 *
 * This helper is intentionally side-effect free: it does not touch the sensor
 * backend, update EMA state, or advance energy integration. It is therefore
 * suitable for lightweight telemetry paths that need the last available power
 * state while avoiding extra I2C traffic.
 */
bool bench_power_last(BenchPower& out){
  out = _last_pwr;
  return _pwr_ready;
}

/**
 * @brief Optional periodic benchmark hook.
 *
 * Intended for loops or schedulers that want to force periodic power sampling.
 */
void bench_tick(){
  #if BENCH_ENABLE_POWER
    if (_pwr_ready) {
      const uint32_t now_ms = millis();
      if ((uint32_t)(now_ms - _last_power_poll_ms) >= (uint32_t)BENCH_POWER_POLL_MIN_MS) {
        BenchPower tmp{};
        yield();
        if (bench_power_poll(tmp)) {
          _last_power_poll_ms = now_ms;
        } else {
          // Even on a failed read, avoid hammering the I2C backend in a hot loop.
          _last_power_poll_ms = now_ms;
        }
        yield();
      }
    }
  #endif
}

// ======================= PRINT =========================
/**
 * @brief Prints a standardized benchmark line to the serial port.
 * @param tag  User-defined benchmark label.
 * @param us   Measured latency in microseconds.
 * @param mem  Memory snapshot.
 * @param pwr  Power snapshot.
 *
 * Output fields include latency, heap usage, TFLM arena, voltage, current,
 * instantaneous power, and cumulative energy.
 */
void bench_print_line(const char* tag,
                      uint32_t us,
                      const BenchMemory& mem,
                      const BenchPower& pwr){
  const float heap_used_kB   = mem.heap_used          / 1024.0f;
  const float heap_total_kB  = mem.heap_total         / 1024.0f;
  const float heap_min_kB    = mem.min_free_heap      / 1024.0f;
  const float heap_big_kB    = mem.largest_free_block / 1024.0f;
  const float arena_kB       = mem.tflm_arena_used    / 1024.0f;

  Serial.printf("[BENCH] %s | t_inference_us=%lu | heap=%.1fkB/%.1fkB (min=%.1fkB, biggest=%.1fkB) | arena=%.1fkB | "
                "V_bus=%.3fV I_bus=%.1fmA P_bus=%.1fmW E_total_accum=%.3fmWh\n",
    tag,
    (unsigned long)us,
    (float)heap_used_kB, (float)heap_total_kB,
    (float)heap_min_kB,  (float)heap_big_kB,
    (float)arena_kB,
    (float)pwr.voltage, (float)pwr.current, (float)pwr.power, (float)pwr.energy_mWh
  );
}