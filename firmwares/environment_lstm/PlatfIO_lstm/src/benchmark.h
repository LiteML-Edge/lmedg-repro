#pragma once
#include <Arduino.h>

/*
 * ============================================================================
 *  benchmark.h
 *  LiteML-Edge Benchmark Public Interface
 * ============================================================================
 *
 *  Purpose
 *  -------
 *  This header declares the public benchmarking API used by LiteML-Edge
 *  firmware for latency, memory, power, and energy monitoring.
 *
 *  Scope
 *  -----
 *  The interface is designed to support auditable embedded AI experiments,
 *  enabling standardized measurement and reporting of:
 *
 *  - Inference latency
 *  - Heap and TensorFlow Lite Micro arena usage
 *  - Voltage, current, and power acquisition
 *  - Cumulative energy integration
 *  - Consolidated serial benchmark logging
 *
 *  Design goals
 *  ------------
 *  - Keep the API lightweight and portable across MCUs.
 *  - Preserve compatibility with ESP32 and safe fallback behavior on other
 *    targets.
 *  - Provide benchmark structures suitable for reproducible TinyML evaluation.
 *  - Support IEEE LATAM-ready code documentation and auditability.
 *
 *  Notes
 *  -----
 *  - Memory metrics may vary depending on platform support.
 *  - Power and energy metrics depend on the enabled backend implementation.
 *  - This file documents the interface only; implementation details are located
 *    in benchmark.cpp.
 * ============================================================================
 */

// ======================= LATENCY =======================

/**
 * @brief Starts a latency measurement window.
 *
 * The elapsed interval can later be obtained with bench_latency_end_us().
 */
void bench_latency_begin();

/**
 * @brief Ends the latency measurement window.
 * @return Elapsed time in microseconds.
 */
uint32_t bench_latency_end_us();

// ======================= MEMORY ========================

/**
 * @brief Extended memory snapshot structure for absolute memory metrics.
 *
 * This structure is intended to support platform-aware memory reporting
 * without requiring FreeRTOS-specific APIs.
 */
typedef struct {
  uint32_t free_heap;           // Bytes currently available for dynamic allocation.
  uint32_t min_free_heap;       // Lowest observed value of the same free-heap metric.
  uint32_t largest_free_block;  // Largest known contiguous allocation region.
  uint32_t heap_total;          // Allocated bytes plus currently available heap bytes.
  uint32_t heap_used;           // Computed as heap_total - free_heap.
  uint32_t tflm_arena_used;     // Bytes used by the TFLM arena, if available.
} BenchMemory;

/**
 * @brief Captures a memory snapshot (heap / arena).
 * @return A BenchMemory structure containing the current memory state.
 *
 * Compatible with ESP32 and designed with a safe fallback for other MCUs.
 */
BenchMemory bench_memory_snapshot();

// ======================= POWER / ENERGY =================

/**
 * @brief Structure holding the latest electrical and energy measurements.
 */
typedef struct {
  float voltage;      // Voltage in volts (V).
  float current;      // Current in milliamperes (mA).
  float power;        // Instantaneous power in milliwatts (mW).
  float energy_mWh;   // Accumulated energy in mWh since the last reset.
} BenchPower;

/**
 * @brief Initializes the power measurement backend (INA219, ADC, etc.).
 * @return true if the backend was initialized successfully.
 */
bool bench_power_begin();

/**
 * @brief Acquires a power reading and updates cumulative energy integration.
 * @param[out] out Output structure containing V / I / P / accumulated E.
 * @return true if the reading was successful.
 */
bool bench_power_poll(BenchPower& out);

/**
 * @brief Resets the cumulative energy accumulator (mWh).
 */
void bench_power_reset_energy();

/**
 * @brief Returns the most recent valid power snapshot without polling.
 * @param[out] out Last stored power snapshot.
 * @return true if the backend is ready, false otherwise.
 */
bool bench_power_last(BenchPower& out);

/**
 * @brief Optional periodic tick hook.
 *
 * This function may be called periodically inside the main loop to update
 * power readings when the feature is enabled.
 */
void bench_tick();

// ======================= PRINT / LOG ====================

/**
 * @brief Prints a consolidated benchmark line.
 *
 * Example:
 * [BENCH] INFER | us=1234 | heap=217000/280000 (min=180000, big=120000)
 *         | arena=10240 | V=5.000V I=120.3mA P=601.5mW E=12.3mWh
 *
 * @param tag Benchmark label associated with the measured event.
 * @param us Measured latency in microseconds.
 * @param mem Memory snapshot structure.
 * @param pwr Power snapshot structure.
 */
void bench_print_line(const char* tag,
                      uint32_t us,
                      const BenchMemory& mem,
                      const BenchPower& pwr);