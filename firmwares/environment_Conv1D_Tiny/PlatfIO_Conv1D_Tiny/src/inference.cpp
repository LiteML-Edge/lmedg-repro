/*
 * ============================================================================
 *  inference.cpp
 *  LiteML-Edge Inference Runtime Module
 * ============================================================================
 *
 *  Purpose
 *  -------
 *  This module implements the embedded inference pipeline used by LiteML-Edge,
 *  supporting:
 *
 *  1. Sliding-window feature buffering for temporal models.
 *  2. Min-Max scaling using exported training-time scaler bounds.
 *  3. Quantized or float TensorFlow Lite Micro input/output handling.
 *  4. Support for either:
 *     - one concatenated output tensor, or
 *     - two separate output tensors (multi-head layout).
 *  5. Residual-space inverse transformation and absolute-state reconstruction.
 *  6. Warm-up handling for 24-step temporal inference.
 *  7. Common inference behavior for REPLAY and FIELD once the shared sensor contract is satisfied.
 *
 *  Design rationale
 *  ----------------
 *  The implementation is structured to make the inference contract auditable:
 *
 *  - Feature order is explicit and fixed.
 *  - Input normalization is deterministic and bounded.
 *  - Tensor quantization uses tensor-specific scale and zero-point values.
 *  - Output decoding respects the correct tensor for each prediction head.
 *  - Clamp events are summarized by feature for logging clarity.
 *  - Arena usage is exposed for benchmarking and memory audits.
 *
 *  Reproducibility and auditability
 *  --------------------------------
 *  - When available, exported scalers from training are used directly.
 *  - If no exported scaler header is present, deterministic fallback bounds
 *    are provided.
 *  - Serial logs expose tensor types, quantization parameters, scaler CRC,
 *    arena usage, and clamp summaries.
 *  - REPLAY and FIELD share the same model-facing contract once SensorData is assembled.
 *
 * ============================================================================
 */

// inference.cpp
// Correct support for 1 concatenated output OR 2 outputs (separate heads)
// + uses scale / zero-point from the correct tensor for each head

#include <Arduino.h>
#include <math.h>
#include <string.h>
#include <config.h>
#include "inference.h"

// ============================================================================
// Debug: exact replay validation log
// ----------------------------------------------------------------------------
// When enabled, prints one high-precision CSV line for each REAL inference.
// Intended for Python vs firmware validation of the final rolling24 window.
// ============================================================================
#ifndef LITEML_DBG_REPLAY_EXACT
#define LITEML_DBG_REPLAY_EXACT 1
#endif

#ifndef LITEML_DBG_REPLAY_EXACT_HEADER_EVERY_BOOT
#define LITEML_DBG_REPLAY_EXACT_HEADER_EVERY_BOOT 1
#endif

#ifndef LITEML_DBG_MODEL_IO
#define LITEML_DBG_MODEL_IO 1
#endif

#ifndef LITEML_DBG_MODEL_IO_HEADER_EVERY_BOOT
#define LITEML_DBG_MODEL_IO_HEADER_EVERY_BOOT 1
#endif

#ifndef LITEML_DBG_MODEL_IO_RAW_MAX_BYTES
#define LITEML_DBG_MODEL_IO_RAW_MAX_BYTES 32
#endif

#ifndef LITEML_DBG_MODEL_IO_RAW_MAX_DIMS
#define LITEML_DBG_MODEL_IO_RAW_MAX_DIMS 8
#endif

#ifndef LITEML_DBG_MODEL_IO_STABILITY
#define LITEML_DBG_MODEL_IO_STABILITY 1
#endif

#if __has_include("scalers_exported_Conv1D_Tiny.h")
  #include "scalers_exported_Conv1D_Tiny.h"
  #define HAS_SCALERS_EXPORT 1

#else
  constexpr int K_NUM_FEATURES = 12;
  static const float X_MIN[K_NUM_FEATURES] = {
    22.22f, 41.0f, -4.802f, -21.68f, 24.428f, 54.083f, 22.22f, 43.0f, -1.0f, -1.0f, 0.0f, 1.0f
  };
  static const float X_MAX[K_NUM_FEATURES] = {
    35.0f, 100.0f, 7.405f, 38.674f, 29.596f, 78.348f, 35.0f, 100.0f, 1.0f, 1.0f, 6.0f, 12.0f
  };
  static const float DY_MIN[2] = { -2.0f, -10.0f };
  static const float DY_MAX[2] = {  2.0f,  10.0f };
  // NOTE: Default ΔY bounds are provided in inference.h when the scalers
  // header does not define them.
#endif

// === Sliding temporal window
static constexpr int K_WINDOW_STEPS = 24;
static constexpr int K_WINDOW_SIZE  = K_WINDOW_STEPS * K_NUM_FEATURES;

// Ring buffer maintained in chronological order.
static float g_window[K_WINDOW_STEPS][K_NUM_FEATURES];
static int   g_window_head   = -1;
static int   g_window_filled = 0;

/**
 * @brief Resets the temporal sliding window state.
 *
 * Clears the full 24-step feature buffer and resets indexing state.
 */
static void window_reset(){
  g_window_head   = -1;
  g_window_filled = 0;
  for (int t = 0; t < K_WINDOW_STEPS; ++t){
    for (int j = 0; j < K_NUM_FEATURES; ++j){
      g_window[t][j] = 0.0f;
    }
  }
}

/**
 * @brief Pushes one feature vector into the temporal window.
 * @param f Feature vector for the current time step.
 *
 * The window behaves as a chronological ring buffer. Until it is full, samples
 * are appended; afterward, the oldest sample is overwritten.
 */
static void window_push(const float f[K_NUM_FEATURES]){
  if (g_window_filled < K_WINDOW_STEPS){
    g_window_head = g_window_filled;
    g_window_filled++;
  } else {
    g_window_head = (g_window_head + 1) % K_WINDOW_STEPS;
  }
  for (int j = 0; j < K_NUM_FEATURES; ++j){
    g_window[g_window_head][j] = f[j];
  }
}

/**
 * @brief Packs the 24-step window into a flat scaled input buffer.
 * @param[out] dst Destination buffer.
 * @param[in]  dst_len Destination buffer length.
 * @return true if the window is full and packing succeeds, false otherwise.
 *
 * Each feature is Min-Max scaled using X_MIN / X_MAX. Values outside the
 * training bounds are clamped into [0, 1]. Clamp events are aggregated by
 * feature name and logged in summarized form.
 */
static bool window_pack_scaled(float *dst, size_t dst_len){
  if (g_window_filled < K_WINDOW_STEPS) return false;
  if (dst_len < (size_t)K_WINDOW_SIZE) return false;

  static const char* FEATURE_NAMES[K_NUM_FEATURES] = {
    "T_out", "H_out", "T_in_lag1", "H_in_lag1",
    "T_out_lag1", "H_out_lag1", "T_in_lag2", "H_in_lag2",
    "sin_hour", "cos_hour", "weekday", "month"
  };

  int oldest = (g_window_filled < K_WINDOW_STEPS)
             ? 0
             : ((g_window_head + 1) % K_WINDOW_STEPS);

  // ------------------------------------------------------------------
  // Clamp logging behavior:
  // - preserves the original clamp behavior for every window element
  // - summarizes logs by feature name instead of repeating each timestep
  // - reports how many times each feature was clamped across the window
  // - stores one representative raw -> clamped example per feature
  // ------------------------------------------------------------------
  struct ClampSummary {
    bool  used = false;
    int   count = 0;
    bool  hit_min = false;
    bool  hit_max = false;
    float raw_min = 0.0f;
    float raw_max = 0.0f;
    float clamped_min = 0.0f;
    float clamped_max = 0.0f;
  };

  ClampSummary clamp_summary[K_NUM_FEATURES];
  int clamp_count_total = 0;

  size_t idx = 0;
  for (int t = 0; t < K_WINDOW_STEPS; ++t){
    int src = (oldest + t) % K_WINDOW_STEPS;
    for (int j = 0; j < K_NUM_FEATURES; ++j){
      const float x = g_window[src][j];
      const float mn = X_MIN[j];
      const float mx = X_MAX[j];
      const float den = (mx - mn);
      float xs = 0.0f;

      if (den > 1e-9f) {
        xs = (x - mn) / den;

        if (xs < 0.0f) {
          clamp_count_total++;

          ClampSummary& cs = clamp_summary[j];
          cs.used = true;
          cs.count++;
          cs.hit_min = true;
          cs.raw_min = x;
          cs.clamped_min = mn;

          xs = 0.0f;
        }

        if (xs > 1.0f) {
          clamp_count_total++;

          ClampSummary& cs = clamp_summary[j];
          cs.used = true;
          cs.count++;
          cs.hit_max = true;
          cs.raw_max = x;
          cs.clamped_max = mx;

          xs = 1.0f;
        }
      }

      dst[idx++] = xs;
    }
  }

  if (clamp_count_total > 0) {
    char clamp_log[512];
    clamp_log[0] = '\0';

    int feature_hits = 0;
    for (int j = 0; j < K_NUM_FEATURES; ++j) {
      if (!clamp_summary[j].used) continue;

      char tmp[96];
      tmp[0] = '\0';

      if (clamp_summary[j].hit_min && clamp_summary[j].hit_max) {
        snprintf(
          tmp, sizeof(tmp),
          "%s:min %.1f->%.1f (X_MIN), max %.1f->%.1f (X_MAX) (%dx)",
          FEATURE_NAMES[j],
          clamp_summary[j].raw_min, clamp_summary[j].clamped_min,
          clamp_summary[j].raw_max, clamp_summary[j].clamped_max,
          clamp_summary[j].count
        );
      } else if (clamp_summary[j].hit_min) {
        snprintf(
          tmp, sizeof(tmp),
          "%s:%.1f->%.1f (X_MIN) (%dx)",
          FEATURE_NAMES[j],
          clamp_summary[j].raw_min, clamp_summary[j].clamped_min,
          clamp_summary[j].count
        );
      } else {
        snprintf(
          tmp, sizeof(tmp),
          "%s:%.1f->%.1f (X_MAX) (%dx)",
          FEATURE_NAMES[j],
          clamp_summary[j].raw_max, clamp_summary[j].clamped_max,
          clamp_summary[j].count
        );
      }

      if (feature_hits == 0) {
        snprintf(clamp_log, sizeof(clamp_log), "%s", tmp);
      } else {
        strncat(clamp_log, " | ", sizeof(clamp_log) - strlen(clamp_log) - 1);
        strncat(clamp_log, tmp, sizeof(clamp_log) - strlen(clamp_log) - 1);
      }
      feature_hits++;
    }

    //Serial.printf("[CLAMP] n=%d | %s\n", clamp_count_total, clamp_log);
  }

  return true;
}

#include "inference.h"

// Snapshot of arena usage immediately after AllocateTensors().
static size_t g_arena_used_snapshot = 0;

static bool g_inference_ready = false;

/**
 * @brief Reports whether the inference engine is initialized and ready.
 * @return true if inference is ready, false otherwise.
 */
bool inference_ready() { return g_inference_ready; }

/**
 * @brief Returns the number of valid steps currently stored in the window.
 * @return Number of filled temporal steps.
 */
int inference_window_filled() { return (int)g_window_filled; }

#include "environment_model_data_Conv1D_Tiny.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"

namespace {

// ------------------------------------------------------------
// Model precision / quantization audit helpers (TFLM)
// ------------------------------------------------------------

/**
 * @brief Returns a human-readable name for a TensorFlow Lite tensor type.
 *
 * This helper is used for audit-oriented runtime logging, allowing the firmware
 * to print the numeric domain associated with input/output tensors in a readable
 * form (e.g., float32, int8, uint8).
 *
 * Supported mappings include the most relevant tensor types for TinyML
 * deployment on microcontrollers. Unknown or unsupported codes are reported
 * as "other".
 *
 * @param t TensorFlow Lite tensor type code.
 * @return Constant string describing the tensor type.
 */
static const char* tf_type_name(int t)
{
  switch (t) {
    case kTfLiteFloat32: return "float32";
    case kTfLiteFloat16: return "float16";
    case kTfLiteInt8:    return "int8";
    case kTfLiteInt16:   return "int16";
    case kTfLiteInt32:   return "int32";
    case kTfLiteUInt8:   return "uint8";
    case kTfLiteBool:    return "bool";
    default:             return "other";
  }
}

/**
 * @brief Classifies the model input/output precision at the tensor I/O boundary.
 *
 * This function inspects only the tensor type exposed at the model input and
 * output interfaces. It does not analyze internal operators or intermediate
 * tensors; therefore, it should be interpreted strictly as an I/O-level
 * precision descriptor.
 *
 * Typical outputs include:
 * - INT8_IO
 * - FLOAT32_IO
 * - FLOAT16_IO
 * - HYBRID_IO
 *
 * @param in_t Input tensor type.
 * @param out_t Output tensor type.
 * @return Constant string describing the I/O precision class.
 */
static const char* detect_model_precision_io(int in_t, int out_t)
{
  if (in_t == kTfLiteInt8    && out_t == kTfLiteInt8)    return "INT8_IO";
  if (in_t == kTfLiteFloat32 && out_t == kTfLiteFloat32) return "FLOAT32_IO";
  if (in_t == kTfLiteFloat16 && out_t == kTfLiteFloat16) return "FLOAT16_IO";
  return "HYBRID_IO";
}

/**
 * @brief Infers the model quantization style from graph structure and I/O types.
 *
 * This function combines:
 *  - input tensor type,
 *  - output tensor type, and
 *  - presence of Quantize / Dequantize operators in the graph
 *
 * to estimate the deployed quantization style for logging and firmware
 * auditability.
 *
 * Important:
 * This is a practical runtime classification heuristic. It is suitable for
 * embedded audit logs and deployment validation, but it is not intended to
 * replace full offline graph inspection.
 *
 * Possible classifications include:
 * - FULL_INT8
 * - FULL_FLOAT32
 * - FULL_FLOAT16
 * - HYBRID_FLOAT_IO
 * - HYBRID_INT8_IO_WITH_QDQ
 * - HYBRID_FLOAT32_TO_INT8
 * - HYBRID_INT8_TO_FLOAT32
 * - HYBRID_FLOAT16_TO_INT8
 * - HYBRID_INT8_TO_FLOAT16
 * - HYBRID_FLOAT32_TO_FLOAT16
 * - HYBRID_FLOAT16_TO_FLOAT32
 * - UINT8_IO
 * - HYBRID_UINT8
 * - UNKNOWN
 *
 * @param model Pointer to the loaded TFLite model.
 * @param in_t Input tensor type.
 * @param out_t Output tensor type.
 * @return Constant string describing the inferred quantization style.
 */
static const char* detect_model_quantization_style(const tflite::Model* model,
                                                   int in_t,
                                                   int out_t)
{
  if (!model || !model->subgraphs() || model->subgraphs()->size() == 0) {
    return "UNKNOWN";
  }

  const auto* sub = model->subgraphs()->Get(0);
  if (!sub || !sub->operators() || !model->operator_codes()) {
    return "UNKNOWN";
  }

  bool has_quantize   = false;
  bool has_dequantize = false;

  const auto* ops   = sub->operators();
  const auto* codes = model->operator_codes();

  for (unsigned i = 0; i < ops->size(); ++i) {
    const auto* op = ops->Get(i);
    if (!op) continue;

    const int opcode_index = op->opcode_index();
    if (opcode_index < 0 || opcode_index >= (int)codes->size()) continue;

    const auto* code = codes->Get(opcode_index);
    if (!code) continue;

    const auto builtin = code->builtin_code();

    if (builtin == tflite::BuiltinOperator_QUANTIZE)   has_quantize   = true;
    if (builtin == tflite::BuiltinOperator_DEQUANTIZE) has_dequantize = true;
  }

  // Pure or near-pure INT8 I/O
  if (in_t == kTfLiteInt8 && out_t == kTfLiteInt8) {
    if (!has_quantize && !has_dequantize) return "FULL_INT8";
    return "HYBRID_INT8_IO_WITH_QDQ";
  }

  // Pure or near-pure FLOAT32 I/O
  if (in_t == kTfLiteFloat32 && out_t == kTfLiteFloat32) {
    if (!has_quantize && !has_dequantize) return "FULL_FLOAT32";
    return "HYBRID_FLOAT_IO";
  }

  // Pure or near-pure FLOAT16 I/O
  if (in_t == kTfLiteFloat16 && out_t == kTfLiteFloat16) {
    if (!has_quantize && !has_dequantize) return "FULL_FLOAT16";
    return "HYBRID_FLOAT16_IO";
  }

  // Explicit mixed I/O patterns
  if (in_t == kTfLiteFloat32 && out_t == kTfLiteInt8) {
    return has_quantize || has_dequantize
      ? "HYBRID_FLOAT32_TO_INT8_WITH_QDQ"
      : "HYBRID_FLOAT32_TO_INT8";
  }

  if (in_t == kTfLiteInt8 && out_t == kTfLiteFloat32) {
    return has_quantize || has_dequantize
      ? "HYBRID_INT8_TO_FLOAT32_WITH_QDQ"
      : "HYBRID_INT8_TO_FLOAT32";
  }

  if (in_t == kTfLiteFloat16 && out_t == kTfLiteInt8) {
    return has_quantize || has_dequantize
      ? "HYBRID_FLOAT16_TO_INT8_WITH_QDQ"
      : "HYBRID_FLOAT16_TO_INT8";
  }

  if (in_t == kTfLiteInt8 && out_t == kTfLiteFloat16) {
    return has_quantize || has_dequantize
      ? "HYBRID_INT8_TO_FLOAT16_WITH_QDQ"
      : "HYBRID_INT8_TO_FLOAT16";
  }

  if (in_t == kTfLiteFloat32 && out_t == kTfLiteFloat16) {
    return "HYBRID_FLOAT32_TO_FLOAT16";
  }

  if (in_t == kTfLiteFloat16 && out_t == kTfLiteFloat32) {
    return "HYBRID_FLOAT16_TO_FLOAT32";
  }

  // Less common uint8-based cases
  if (in_t == kTfLiteUInt8 && out_t == kTfLiteUInt8) {
    return "UINT8_IO";
  }

  if (in_t == kTfLiteUInt8 || out_t == kTfLiteUInt8) {
    return "HYBRID_UINT8";
  }

  return "UNKNOWN";
}

/**
 * @brief Infers the structural family of the loaded neural network graph.
 *
 * This function inspects the operators present in the first TFLite subgraph
 * and classifies the model into a coarse architectural family suitable for
 * runtime telemetry and audit logs.
 *
 * The returned family is not intended to identify the exact training-side
 * model name. Instead, it reports the graph-level structural class, such as:
 * - LSTM
 * - CNN
 * - MLP
 * - UNKNOWN
 *
 * Current detection logic:
 * - Presence of UNIDIRECTIONAL_SEQUENCE_LSTM  -> LSTM
 * - Presence of CONV_2D or DEPTHWISE_CONV_2D -> CNN
 * - Presence of FULLY_CONNECTED only         -> MLP
 *
 * This makes the function particularly useful for validating whether the
 * embedded graph family matches the model family declared in config.h.
 *
 * @param model Pointer to the loaded TFLite model.
 * @return Constant string describing the inferred model family.
 */
static const char* detect_model_family(const tflite::Model* model)
{
  if (!model || !model->subgraphs() || model->subgraphs()->size() == 0) {
    return "UNKNOWN";
  }

  const auto* sub = model->subgraphs()->Get(0);
  if (!sub || !sub->operators() || !model->operator_codes()) {
    return "UNKNOWN";
  }

  bool has_lstm     = false;
  bool has_conv2d   = false;
  bool has_dwconv2d = false;
  bool has_fc       = false;

  const auto* ops   = sub->operators();
  const auto* codes = model->operator_codes();

  for (unsigned i = 0; i < ops->size(); ++i) {
    const auto* op = ops->Get(i);
    if (!op) continue;

    const int opcode_index = op->opcode_index();
    if (opcode_index < 0 || opcode_index >= (int)codes->size()) continue;

    const auto* code = codes->Get(opcode_index);
    if (!code) continue;

    const auto builtin = code->builtin_code();

    switch (builtin) {
      case tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM:
        has_lstm = true;
        break;

      case tflite::BuiltinOperator_CONV_2D:
        has_conv2d = true;
        break;

      case tflite::BuiltinOperator_DEPTHWISE_CONV_2D:
        has_dwconv2d = true;
        break;

      case tflite::BuiltinOperator_FULLY_CONNECTED:
        has_fc = true;
        break;

      default:
        break;
    }
  }

  if (has_lstm) return "LSTM";
  if (has_conv2d || has_dwconv2d) return "CNN";
  if (has_fc) return "MLP";

  return "UNKNOWN";
}

  constexpr int kTensorArenaSize = 16 * 1024;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;

  TfLiteTensor* input   = nullptr;
  TfLiteTensor* output  = nullptr;   // out0 (compatibility path)
  TfLiteTensor* output1 = nullptr;   // out1 (second head), if present

  tflite::MicroMutableOpResolver<22> op_resolver;

  // =========================================================
  // REPLAY 1:1: real sequential baseline (shift(1))
  // - avoids incorrect seed / lag1 behavior at the warm-up boundary
  //   (first real inference)
  // - FIELD mode remains untouched (kept in the #else block)
  // =========================================================
  #if LITEML_REPLAY
    static float replay_prev_Tin = 0.0f;
    static float replay_prev_Hin = 0.0f;
  #endif

  /**
   * @brief Applies inverse Min-Max transformation.
   * @param xs Scaled value.
   * @param mn Minimum bound.
   * @param mx Maximum bound.
   * @return Reconstructed raw value.
   */
  inline float minmax_inverse(float xs, float mn, float mx){
    return xs * (mx - mn) + mn;
  }

  /**
   * @brief Quantizes a float into int8 using tensor parameters.
   * @param x Float value.
   * @param s Tensor scale.
   * @param zp Tensor zero-point.
   * @return Quantized int8 value.
   */
  inline int8_t q_int8(float x, float s, int zp){
    const int32_t q = (int32_t)lroundf(x / s) + zp;
    if (q < -128) return -128;
    if (q >  127) return  127;
    return (int8_t)q;
  }

  /**
   * @brief Dequantizes an int8 value into float.
   * @param q Quantized int8 value.
   * @param s Tensor scale.
   * @param zp Tensor zero-point.
   * @return Dequantized float value.
   */
  inline float dq_int8(int8_t q, float s, int zp){
    return ((int32_t)q - zp) * s;
  }

  /**
   * @brief Computes CRC32 over a byte buffer.
   * @param data Input byte buffer.
   * @param len  Buffer length in bytes.
   * @return CRC32 checksum.
   */
  [[maybe_unused]] static uint32_t crc32_bytes(const uint8_t* data, size_t len){
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i=0;i<len;i++){
      uint32_t c = (uint32_t)data[i];
      crc ^= c;
      for (int k=0;k<8;k++){
        uint32_t mask = -(crc & 1u);
        crc = (crc >> 1) ^ (0xEDB88320u & mask);
      }
    }
    return ~crc;
  }

  /**
   * @brief Fills a feature vector with neutral zeros.
   * @param f Output feature vector.
   */
  void make_neutral_features(float f[K_NUM_FEATURES]){
    for(int j=0;j<K_NUM_FEATURES;++j) f[j]=0.0f;
  }

#if LITEML_DBG_MODEL_IO && LITEML_REPLAY
/**
 * @brief Returns the chronological start index of the current debug window.
 * @return Ring-buffer index corresponding to the oldest valid time step.
 *
 * This helper is used only by replay-side debug exporters so that flattened
 * debug traces preserve the same temporal order consumed by the model input.
 */
static inline int debug_window_oldest_index(){
    return (g_window_filled < K_WINDOW_STEPS)
         ? 0
         : ((g_window_head + 1) % K_WINDOW_STEPS);
  }

  /**
 * @brief Clamps one physical-domain feature to the exported training bounds.
 * @param feature_idx Feature position in the LiteML-Edge input contract.
 * @param x Raw physical-domain value before clipping.
 * @return Clipped value constrained to X_MIN / X_MAX for the selected feature.
 *
 * The function mirrors the training-side clamping rule used before Min-Max
 * normalization, allowing debug exports to expose both raw and clipped states.
 */
static inline float clamp_feature_phys(int feature_idx, float x){
    if (x < X_MIN[feature_idx]) return X_MIN[feature_idx];
    if (x > X_MAX[feature_idx]) return X_MAX[feature_idx];
    return x;
  }

  static void fill_debug_input_record(DebugInputRecord* rec,
                                      int idx,
                                      long epoch,
                                      int step,
                                      const SensorData& d,
                                      const float* x_scaled_window,
                                      const float* input_tensor_float_window){
    if (!rec) return;
    memset(rec, 0, sizeof(*rec));

    const int oldest = debug_window_oldest_index();
    const int src = (oldest + step) % K_WINDOW_STEPS;
    const int base = step * K_NUM_FEATURES;
    const float* raw = g_window[src];

    rec->idx = idx;
    rec->epoch = (uint32_t)epoch;
    rec->step = step;

    rec->gt_Tin_true = d.T_in;
    rec->gt_Hin_true = d.H_in;

    rec->pre_raw_Tout = d.T_out_raw;
    rec->pre_raw_Hout = d.H_out_raw;
    rec->pre_raw_Tin  = d.T_in_raw;
    rec->pre_raw_Hin  = d.H_in_raw;

    rec->pre_smooth_Tout = d.T_out_smooth;
    rec->pre_smooth_Hout = d.H_out_smooth;
    rec->pre_smooth_Tin  = d.T_in_smooth;
    rec->pre_smooth_Hin  = d.H_in_smooth;

    rec->state_Tout_phys_raw      = raw[F_TOUT];
    rec->state_Hout_phys_raw      = raw[F_HOUT];
    rec->state_Tin_lag1_phys_raw  = raw[F_TIN_LAG1];
    rec->state_Hin_lag1_phys_raw  = raw[F_HIN_LAG1];
    rec->state_Tout_lag1_phys_raw = raw[F_TOUT_LAG1];
    rec->state_Hout_lag1_phys_raw = raw[F_HOUT_LAG1];
    rec->state_Tin_lag2_phys_raw  = raw[F_TIN_LAG2];
    rec->state_Hin_lag2_phys_raw  = raw[F_HIN_LAG2];
    rec->state_sin_hour           = raw[F_SIN_HOUR];
    rec->state_cos_hour           = raw[F_COS_HOUR];
    rec->state_weekday            = raw[F_WEEKDAY];
    rec->state_month              = raw[F_MONTH];

    for (int j = 0; j < K_NUM_FEATURES; ++j){
      rec->in_f_phys_raw[j]  = raw[j];
      rec->in_f_phys_clip[j] = clamp_feature_phys(j, raw[j]);
      rec->in_f_scaled[j]    = x_scaled_window[base + j];
      rec->in_x_float[j]     = input_tensor_float_window[base + j];
    }
  }

  static void fill_debug_output_record(DebugOutputRecord* rec,
                                       int idx,
                                       long epoch,
                                       float out_o0_tensor,
                                       float out_o1_tensor,
                                       float out_o0_float,
                                       float out_o1_float,
                                       float y_T_scaled,
                                       float y_H_scaled,
                                       float d_T_pred,
                                       float d_H_pred,
                                       float p_Tprev_phys,
                                       float p_Hprev_phys,
                                       float p_T_pred,
                                       float p_H_pred){
    if (!rec) return;
    memset(rec, 0, sizeof(*rec));

    rec->idx = idx;
    rec->epoch = (uint32_t)epoch;
    rec->out_o0_tensor = out_o0_tensor;
    rec->out_o1_tensor = out_o1_tensor;
    rec->out_o0_float = out_o0_float;
    rec->out_o1_float = out_o1_float;
    rec->y_T_scaled = y_T_scaled;
    rec->y_H_scaled = y_H_scaled;
    rec->d_T_pred = d_T_pred;
    rec->d_H_pred = d_H_pred;
    rec->p_Tprev_phys = p_Tprev_phys;
    rec->p_Hprev_phys = p_Hprev_phys;
    rec->p_T_pred = p_T_pred;
    rec->p_H_pred = p_H_pred;
  }

  /**
 * @brief Reinterprets a float32 value as its raw IEEE-754 bit pattern.
 * @param value Floating-point value to serialize.
 * @return Unsigned 32-bit integer containing the exact binary representation.
 *
 * This helper supports bitwise debug traces for immediate model outputs, which
 * are useful when validating numerical identity between Python and firmware.
 */
static inline uint32_t float_to_u32_bits(float value){
    uint32_t bits = 0u;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
  }
#endif

#if LITEML_DBG_REPLAY_EXACT && LITEML_REPLAY
  static bool g_dbg_replay_header_printed = false;
  static int  g_dbg_replay_idx = 0;

  /**
 * @brief Prints the replay-exact CSV header once per boot.
 *
 * The header describes the high-precision replay validation stream emitted for
 * 1:1 Python-versus-firmware comparison on the final rolling window.
 */
static void dbg_replay_print_header_once(){
    if (g_dbg_replay_header_printed) return;
    g_dbg_replay_header_printed = true;

    Serial.println(
      "[DBG_REPLAY_CSV] "
      "idx,epoch,"
      "gt_Tin_true,gt_Hin_true,"
      "state_Tout_phys_raw,state_Hout_phys_raw,"
      "state_Tin_lag1_phys_raw,state_Hin_lag1_phys_raw,"
      "state_Tout_lag1_phys_raw,state_Hout_lag1_phys_raw,"
      "state_Tin_lag2_phys_raw,state_Hin_lag2_phys_raw,"
      "state_sin_hour,state_cos_hour,state_weekday,state_month,"
      "y_T_scaled,y_H_scaled,"
      "d_T_pred,d_H_pred,"
      "p_Tprev_phys,p_Hprev_phys,p_T_pred,p_H_pred"
    );
  }

  static void dbg_replay_print_row(long epoch,
                                   const SensorData& d,
                                   float y_T_scaled,
                                   float y_H_scaled,
                                   float d_T_pred,
                                   float d_H_pred,
                                   float p_Tprev_phys,
                                   float p_Hprev_phys,
                                   float p_T_pred,
                                   float p_H_pred) {
    dbg_replay_print_header_once();

    Serial.println();
    Serial.printf(
      "[DBG_REPLAY_CSV] "
      "%d,%ld,"
      "%.8f,%.8f,"
      "%.8f,%.8f,"
      "%.8f,%.8f,%.8f,%.8f,"
      "%.8f,%.8f,"
      "%.8f,%.8f,%d,%d,"
      "%.8f,%.8f,"
      "%.8f,%.8f,"
      "%.8f,%.8f,%.8f,%.8f\n",
      g_dbg_replay_idx++,
      epoch,
      d.T_in, d.H_in,
      d.T_out, d.H_out,
      d.T_in_lag1, d.H_in_lag1,
      d.T_out_lag1, d.H_out_lag1,
      d.T_in_lag2, d.H_in_lag2,
      d.sin_hour, d.cos_hour,
      d.weekday_pandas, d.month_1_12,
      y_T_scaled, y_H_scaled,
      d_T_pred, d_H_pred,
      p_Tprev_phys, p_Hprev_phys,
      p_T_pred, p_H_pred
    );
  }
#endif

#if LITEML_DBG_MODEL_IO && LITEML_REPLAY
  static bool g_dbg_model_in_header_printed = false;
  static bool g_dbg_model_out_header_printed = false;
  static bool g_dbg_model_out_bits_header_printed = false;
  static bool g_dbg_model_out_raw_header_printed = false;
  static bool g_dbg_model_out_stability_header_printed = false;
  static int  g_dbg_model_io_idx = 0;

  struct DebugTensorRawSnapshot {
    int type_code = -1;
    const char* type_name = "null";
    int bytes_total = -1;
    int bytes_dumped = 0;
    int dims_size = -1;
    uintptr_t data_ptr = 0u;
    int dims[LITEML_DBG_MODEL_IO_RAW_MAX_DIMS];
    uint8_t raw[LITEML_DBG_MODEL_IO_RAW_MAX_BYTES];
  };

  /**
 * @brief Captures metadata and a bounded raw-byte snapshot of one output tensor.
 * @param t Pointer to the TensorFlow Lite Micro tensor being inspected.
 * @param[out] snap Destination structure receiving type, shape, size, pointer, and raw bytes.
 *
 * The captured snapshot is later used for exact structural audits of the
 * immediate output buffer, including bytewise stability checks without a new
 * Invoke() call.
 */
static void dbg_tensor_raw_snapshot_capture(const TfLiteTensor* t, DebugTensorRawSnapshot* snap){
    if (!snap) return;

    snap->type_code = t ? (int)t->type : -1;
    snap->type_name = t ? tf_type_name(snap->type_code) : "null";
    snap->bytes_total = t ? (int)t->bytes : -1;
    snap->bytes_dumped = (snap->bytes_total > 0)
                        ? ((snap->bytes_total < LITEML_DBG_MODEL_IO_RAW_MAX_BYTES)
                           ? snap->bytes_total
                           : LITEML_DBG_MODEL_IO_RAW_MAX_BYTES)
                        : 0;
    snap->dims_size = (t && t->dims) ? (int)t->dims->size : -1;
    snap->data_ptr = (t && t->data.data)
                   ? (uintptr_t)(t->data.data)
                   : (uintptr_t)0u;

    for (int di = 0; di < LITEML_DBG_MODEL_IO_RAW_MAX_DIMS; ++di){
      snap->dims[di] = (t && t->dims && di < snap->dims_size)
                     ? (int)t->dims->data[di]
                     : -1;
    }

    for (int bi = 0; bi < LITEML_DBG_MODEL_IO_RAW_MAX_BYTES; ++bi){
      snap->raw[bi] = 0u;
    }

    if (t && t->data.data && snap->bytes_dumped > 0){
      memcpy(snap->raw, t->data.data, (size_t)snap->bytes_dumped);
    }
  }

  static bool dbg_tensor_raw_snapshot_same(const DebugTensorRawSnapshot& a,
                                           const DebugTensorRawSnapshot& b){
    if (a.type_code != b.type_code) return false;
    if (a.bytes_total != b.bytes_total) return false;
    if (a.bytes_dumped != b.bytes_dumped) return false;
    if (a.dims_size != b.dims_size) return false;

    const int dims_cmp = (a.dims_size > 0 && a.dims_size < LITEML_DBG_MODEL_IO_RAW_MAX_DIMS)
                       ? a.dims_size
                       : ((a.dims_size >= LITEML_DBG_MODEL_IO_RAW_MAX_DIMS)
                          ? LITEML_DBG_MODEL_IO_RAW_MAX_DIMS
                          : 0);
    for (int di = 0; di < dims_cmp; ++di){
      if (a.dims[di] != b.dims[di]) return false;
    }

    if (a.bytes_dumped > 0 && memcmp(a.raw, b.raw, (size_t)a.bytes_dumped) != 0) return false;
    return true;
  }

  /**
 * @brief Prints the DBG_MODEL_IN_CSV header once per boot.
 *
 * The emitted schema documents the staged input contract used in replay
 * validation, including raw physical values, clipped values, scaled values,
 * and the final tensor-facing float representation.
 */
static void dbg_model_in_print_header_once(){
    if (g_dbg_model_in_header_printed) return;
    g_dbg_model_in_header_printed = true;

    Serial.print("[DBG_MODEL_IN_CSV] idx,epoch,step,gt_Tin_true,gt_Hin_true,");
    Serial.print(
      "state_Tout_phys_raw,state_Hout_phys_raw,"
      "state_Tin_lag1_phys_raw,state_Hin_lag1_phys_raw,"
      "state_Tout_lag1_phys_raw,state_Hout_lag1_phys_raw,"
      "state_Tin_lag2_phys_raw,state_Hin_lag2_phys_raw,"
      "state_sin_hour,state_cos_hour,state_weekday,state_month"
    );
    for (int j = 0; j < K_NUM_FEATURES; ++j) {
      Serial.printf(",in_f%02d_phys_raw", j);
    }
    for (int j = 0; j < K_NUM_FEATURES; ++j) {
      Serial.printf(",in_f%02d_phys_clip", j);
    }
    for (int j = 0; j < K_NUM_FEATURES; ++j) {
      Serial.printf(",in_f%02d_scaled", j);
    }
    for (int j = 0; j < K_NUM_FEATURES; ++j) {
      Serial.printf(",in_x%02d_float", j);
    }
    Serial.println();
  }

  /**
 * @brief Prints the DBG_MODEL_OUT_CSV header once per boot.
 *
 * This header defines the semantic immediate-output trace used to compare
 * decoded tensor outputs, residual-domain values, and reconstructed absolute
 * predictions across Python and firmware.
 */
static void dbg_model_out_print_header_once(){
    if (g_dbg_model_out_header_printed) return;
    g_dbg_model_out_header_printed = true;

    Serial.println(
      "[DBG_MODEL_OUT_CSV] "
      "idx,epoch,"
      "out_o0_tensor,out_o1_tensor,"
      "out_o0_float,out_o1_float,"
      "y_T_scaled,y_H_scaled,"
      "d_T_pred,d_H_pred,"
      "p_Tprev_phys,p_Hprev_phys,p_T_pred,p_H_pred"
    );
  }

  /**
 * @brief Prints the DBG_MODEL_OUT_BITS_CSV header once per boot.
 *
 * The associated log stream exposes exact IEEE-754 bit patterns for decoded
 * output floats, enabling bitwise audits in addition to tolerance-based checks.
 */
static void dbg_model_out_bits_print_header_once(){
    if (g_dbg_model_out_bits_header_printed) return;
    g_dbg_model_out_bits_header_printed = true;

    Serial.println(
      "[DBG_MODEL_OUT_BITS_CSV] "
      "idx,epoch,"
      "out_o0_float,out_o1_float,"
      "out_o0_bits_hex,out_o1_bits_hex"
    );
  }
  /**
 * @brief Prints the DBG_MODEL_OUT_RAW_CSV header once per boot.
 *
 * This schema describes the raw tensor-buffer dump emitted immediately after
 * inference, including tensor metadata, dimensions, and a bounded byte sample.
 */
static void dbg_model_out_raw_print_header_once(){
    if (g_dbg_model_out_raw_header_printed) return;
    g_dbg_model_out_raw_header_printed = true;

    Serial.print(
      "[DBG_MODEL_OUT_RAW_CSV] "
      "idx,epoch,out_idx,type_code,type_name,bytes_total,bytes_dumped,dims_size"
    );
    for (int di = 0; di < LITEML_DBG_MODEL_IO_RAW_MAX_DIMS; ++di) {
      Serial.printf(",dim%02d", di);
    }
    for (int bi = 0; bi < LITEML_DBG_MODEL_IO_RAW_MAX_BYTES; ++bi) {
      Serial.printf(",b%02d_hex", bi);
    }
    Serial.println();
  }

  /**
 * @brief Prints the DBG_MODEL_OUT_STABILITY_CSV header once per boot.
 *
 * The stability stream records two consecutive raw-tensor captures without a
 * new Invoke() call so the firmware can audit pointer and byte stability.
 */
static void dbg_model_out_stability_print_header_once(){
    if (g_dbg_model_out_stability_header_printed) return;
    g_dbg_model_out_stability_header_printed = true;

    Serial.print(
      "[DBG_MODEL_OUT_STABILITY_CSV] "
      "idx,epoch,out_idx,"
      "type_a_code,type_b_code,type_a_name,type_b_name,"
      "bytes_a_total,bytes_b_total,bytes_a_dumped,bytes_b_dumped,"
      "dims_a_size,dims_b_size,ptr_a_hex,ptr_b_hex,ptr_equal,raw_equal"
    );
    for (int di = 0; di < LITEML_DBG_MODEL_IO_RAW_MAX_DIMS; ++di) {
      Serial.printf(",dim%02d_a,dim%02d_b", di, di);
    }
    for (int bi = 0; bi < LITEML_DBG_MODEL_IO_RAW_MAX_BYTES; ++bi) {
      Serial.printf(",b%02d_a_hex,b%02d_b_hex", bi, bi);
    }
    Serial.println();
  }

  static void dbg_model_out_raw_print_tensor_row(int idx,
                                                 long epoch,
                                                 int out_idx,
                                                 const TfLiteTensor* t){
    dbg_model_out_raw_print_header_once();

    const int type_code = t ? (int)t->type : -1;
    const char* type_name = t ? tf_type_name(type_code) : "null";
    const int bytes_total = t ? (int)t->bytes : -1;
    const int bytes_dumped = (bytes_total > 0)
                           ? ((bytes_total < LITEML_DBG_MODEL_IO_RAW_MAX_BYTES)
                              ? bytes_total
                              : LITEML_DBG_MODEL_IO_RAW_MAX_BYTES)
                           : 0;
    const int dims_size = (t && t->dims) ? (int)t->dims->size : -1;
    const uint8_t* raw = (t && t->data.data)
                       ? reinterpret_cast<const uint8_t*>(t->data.data)
                       : nullptr;

    Serial.printf(
      "[DBG_MODEL_OUT_RAW_CSV] %d,%ld,%d,%d,%s,%d,%d,%d",
      idx,
      epoch,
      out_idx,
      type_code,
      type_name,
      bytes_total,
      bytes_dumped,
      dims_size
    );

    for (int di = 0; di < LITEML_DBG_MODEL_IO_RAW_MAX_DIMS; ++di) {
      if (t && t->dims && di < dims_size) {
        Serial.printf(",%d", (int)t->dims->data[di]);
      } else {
        Serial.print(",");
      }
    }

    for (int bi = 0; bi < LITEML_DBG_MODEL_IO_RAW_MAX_BYTES; ++bi) {
      if (raw && bi < bytes_dumped) {
        Serial.printf(",0x%02X", (unsigned int)raw[bi]);
      } else {
        Serial.print(",");
      }
    }

    Serial.println();
  }

  /**
 * @brief Emits the raw-output tensor dump for all model outputs.
 * @param idx Replay/debug sample index associated with the inference event.
 * @param epoch Epoch timestamp associated with the same event.
 *
 * Each output tensor is serialized using the bounded raw snapshot contract so
 * external tools can verify shape, dtype, payload length, and dumped bytes.
 */
static void dbg_model_out_raw_print_all(int idx, long epoch){
    if (!interpreter) return;

    const int n_outputs = interpreter->outputs_size();
    for (int oi = 0; oi < n_outputs; ++oi){
      dbg_model_out_raw_print_tensor_row(idx, epoch, oi, interpreter->output(oi));
    }
  }


#if LITEML_DBG_MODEL_IO_STABILITY
  /**
 * @brief Emits raw-output stability records for all model outputs.
 * @param idx Replay/debug sample index associated with the inference event.
 * @param epoch Epoch timestamp associated with the same event.
 *
 * For each output tensor, the function captures two consecutive snapshots and
 * reports whether both the backing pointer and the raw bytes remain unchanged.
 */
static void dbg_model_out_stability_print_all(int idx, long epoch){
    if (!interpreter) return;

    dbg_model_out_stability_print_header_once();

    const int n_outputs = interpreter->outputs_size();
    for (int oi = 0; oi < n_outputs; ++oi){
      DebugTensorRawSnapshot snap_a{};
      DebugTensorRawSnapshot snap_b{};

      const TfLiteTensor* t_a = interpreter->output(oi);
      dbg_tensor_raw_snapshot_capture(t_a, &snap_a);

      const TfLiteTensor* t_b = interpreter->output(oi);
      dbg_tensor_raw_snapshot_capture(t_b, &snap_b);

      const int ptr_equal = (snap_a.data_ptr == snap_b.data_ptr) ? 1 : 0;
      const int raw_equal = dbg_tensor_raw_snapshot_same(snap_a, snap_b) ? 1 : 0;

      Serial.printf(
        "[DBG_MODEL_OUT_STABILITY_CSV] %d,%ld,%d,%d,%d,%s,%s,%d,%d,%d,%d,%d,%d,0x%08lX,0x%08lX,%d,%d",
        idx,
        epoch,
        oi,
        snap_a.type_code,
        snap_b.type_code,
        snap_a.type_name,
        snap_b.type_name,
        snap_a.bytes_total,
        snap_b.bytes_total,
        snap_a.bytes_dumped,
        snap_b.bytes_dumped,
        snap_a.dims_size,
        snap_b.dims_size,
        (unsigned long)snap_a.data_ptr,
        (unsigned long)snap_b.data_ptr,
        ptr_equal,
        raw_equal
      );

      for (int di = 0; di < LITEML_DBG_MODEL_IO_RAW_MAX_DIMS; ++di) {
        if (di < snap_a.dims_size && di < LITEML_DBG_MODEL_IO_RAW_MAX_DIMS) {
          Serial.printf(",%d", snap_a.dims[di]);
        } else {
          Serial.print(",");
        }
        if (di < snap_b.dims_size && di < LITEML_DBG_MODEL_IO_RAW_MAX_DIMS) {
          Serial.printf(",%d", snap_b.dims[di]);
        } else {
          Serial.print(",");
        }
      }

      for (int bi = 0; bi < LITEML_DBG_MODEL_IO_RAW_MAX_BYTES; ++bi) {
        if (bi < snap_a.bytes_dumped) {
          Serial.printf(",0x%02X", (unsigned int)snap_a.raw[bi]);
        } else {
          Serial.print(",");
        }
        if (bi < snap_b.bytes_dumped) {
          Serial.printf(",0x%02X", (unsigned int)snap_b.raw[bi]);
        } else {
          Serial.print(",");
        }
      }

      Serial.println();
    }
  }
#endif

  static void dbg_model_in_print_window(int idx,
                                        long epoch,
                                        const SensorData& d,
                                        const float* x_scaled_window,
                                        const float* input_tensor_float_window){
    dbg_model_in_print_header_once();

    for (int step = 0; step < K_WINDOW_STEPS; ++step){
      DebugInputRecord rec{};
      fill_debug_input_record(&rec,
                              idx,
                              epoch,
                              step,
                              d,
                              x_scaled_window,
                              input_tensor_float_window);

      Serial.printf(
        "[DBG_MODEL_IN_CSV] %d,%lu,%d,%.8f,%.8f,"
        "%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f",
        rec.idx,
        (unsigned long)rec.epoch,
        rec.step,
        rec.gt_Tin_true,
        rec.gt_Hin_true,
        rec.state_Tout_phys_raw,
        rec.state_Hout_phys_raw,
        rec.state_Tin_lag1_phys_raw,
        rec.state_Hin_lag1_phys_raw,
        rec.state_Tout_lag1_phys_raw,
        rec.state_Hout_lag1_phys_raw,
        rec.state_Tin_lag2_phys_raw,
        rec.state_Hin_lag2_phys_raw,
        rec.state_sin_hour,
        rec.state_cos_hour,
        rec.state_weekday,
        rec.state_month
      );

      for (int j = 0; j < K_NUM_FEATURES; ++j) Serial.printf(",%.8f", rec.in_f_phys_raw[j]);
      for (int j = 0; j < K_NUM_FEATURES; ++j) Serial.printf(",%.8f", rec.in_f_phys_clip[j]);
      for (int j = 0; j < K_NUM_FEATURES; ++j) Serial.printf(",%.8f", rec.in_f_scaled[j]);
      for (int j = 0; j < K_NUM_FEATURES; ++j) Serial.printf(",%.8f", rec.in_x_float[j]);
      Serial.println();
    }
  }

  /**
 * @brief Prints one semantic output-debug record.
 * @param rec Structured debug record containing raw, scaled, and reconstructed outputs.
 *
 * The emitted CSV row is designed for stage-wise comparison against the Python
 * reference workbook, from immediate decoded outputs through final predictions.
 */
static void dbg_model_out_print_row(const DebugOutputRecord& rec) {
    dbg_model_out_print_header_once();
    Serial.printf(
      "[DBG_MODEL_OUT_CSV] "
      "%d,%lu,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
      rec.idx,
      (unsigned long)rec.epoch,
      rec.out_o0_tensor,
      rec.out_o1_tensor,
      rec.out_o0_float,
      rec.out_o1_float,
      rec.y_T_scaled,
      rec.y_H_scaled,
      rec.d_T_pred,
      rec.d_H_pred,
      rec.p_Tprev_phys,
      rec.p_Hprev_phys,
      rec.p_T_pred,
      rec.p_H_pred
    );
}

  /**
 * @brief Prints one bitwise output-debug record.
 * @param rec Structured debug record containing decoded float outputs.
 *
 * The printed row exposes exact float bit patterns for the two semantic output
 * positions so that firmware logs can be audited at bit level when required.
 */
static void dbg_model_out_bits_print_row(const DebugOutputRecord& rec){
    dbg_model_out_bits_print_header_once();
    const uint32_t out_o0_bits = float_to_u32_bits(rec.out_o0_float);
    const uint32_t out_o1_bits = float_to_u32_bits(rec.out_o1_float);

    Serial.printf(
      "[DBG_MODEL_OUT_BITS_CSV] "
      "%d,%lu,%.8f,%.8f,0x%08lX,0x%08lX\n",
      rec.idx,
      (unsigned long)rec.epoch,
      rec.out_o0_float,
      rec.out_o1_float,
      (unsigned long)out_o0_bits,
      (unsigned long)out_o1_bits
    );
  }
#endif

}

/**
 * @brief Returns the model flatbuffer size in bytes.
 * @return Flatbuffer size in bytes.
 */
static size_t get_model_flatbuffer_bytes(){
  return (size_t)environment_model_len;
}

/**
 * @brief Initializes the TensorFlow Lite Micro inference engine.
 * @return true if initialization succeeds, false otherwise.
 *
 * This function:
 * - resets temporal state,
 * - loads the flatbuffer model,
 * - registers required operators,
 * - allocates tensors,
 * - stores arena usage,
 * - captures tensor handles,
 * - logs memory and quantization metadata.
 */
bool init_inference(){
  g_inference_ready = false;
  window_reset();

#if LITEML_DBG_REPLAY_EXACT && LITEML_REPLAY
  g_dbg_replay_header_printed = false;
  g_dbg_replay_idx = 0;
#endif
#if LITEML_DBG_MODEL_IO && LITEML_REPLAY
  g_dbg_model_in_header_printed = false;
  g_dbg_model_out_header_printed = false;
  g_dbg_model_out_bits_header_printed = false;
  g_dbg_model_out_raw_header_printed = false;
  g_dbg_model_out_stability_header_printed = false;
  g_dbg_model_io_idx = 0;
#endif

  // Reset replay sequential state (does NOT affect FIELD mode).
  #if LITEML_REPLAY
    replay_prev_Tin = 0.0f;
    replay_prev_Hin = 0.0f;
  #endif

  model = tflite::GetModel(environment_model);
  if (model->version() != TFLITE_SCHEMA_VERSION){
    error_reporter->Report("Incompatible schema");
    return false;
  }

  // === Registered operators ===
  op_resolver.AddUnidirectionalSequenceLSTM();
  op_resolver.AddConv2D();
  op_resolver.AddDepthwiseConv2D();
  op_resolver.AddFullyConnected();
  op_resolver.AddRelu();
  op_resolver.AddRelu6();
  op_resolver.AddQuantize();
  op_resolver.AddDequantize();

  op_resolver.AddMean();
  op_resolver.AddAveragePool2D();
  op_resolver.AddMaxPool2D();

  op_resolver.AddReshape();
  op_resolver.AddSqueeze();
  op_resolver.AddPack();
  op_resolver.AddExpandDims();
  op_resolver.AddShape();
  op_resolver.AddStridedSlice();
  op_resolver.AddPad();
  op_resolver.AddPadV2();

  op_resolver.AddAdd();
  op_resolver.AddSub();
  op_resolver.AddMul();

  static tflite::MicroInterpreter static_interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk){
    error_reporter->Report("AllocateTensors failed");
    return false;
  }
  g_arena_used_snapshot = interpreter->arena_used_bytes();

  input   = interpreter->input(0);
  output  = interpreter->output(0);
  output1 = (interpreter->outputs_size() > 1) ? interpreter->output(1) : nullptr;

  Serial.printf("[TFLM] outputs_size=%d | out0_type=%d out1_type=%d | out0_bytes=%d out1_bytes=%d\n",
    (int)interpreter->outputs_size(),
    (int)(output ? output->type : -1),
    (int)(output1 ? output1->type : -1),
    (int)(output ? output->bytes : -1),
    (int)(output1 ? output1->bytes : -1)
  );

  const size_t arena_used  = g_arena_used_snapshot;
  const size_t arena_res   = (size_t)kTensorArenaSize;
  const size_t model_bytes = get_model_flatbuffer_bytes();
  const float arena_used_kB = ((float)arena_used)/1024.0f;
  const float arena_res_kB  = ((float)arena_res)/1024.0f;
  const float model_kB      = ((float)model_bytes)/1024.0f;
  const float total_kB      = model_kB + arena_used_kB;

  Serial.printf("[ARENA] reserved=%luB used=%luB (%.1fkB/%.1fkB)\n",
                (unsigned long)arena_res,
                (unsigned long)arena_used,
                arena_used_kB, arena_res_kB);
  Serial.printf("[MEM] Model=%.2fkB (FLASH) | Arena=%.2fkB (RAM) | Total≈%.2fkB\n",
                model_kB, arena_used_kB, total_kB);

  Serial.printf("[quant] in: s=%.9f zp=%d | out0: s=%.9f zp=%d | out1: s=%.9f zp=%d | in_t=%d out0_t=%d out1_t=%d | in_lastdim=%d\n",
    input->params.scale, input->params.zero_point,
    output->params.scale, output->params.zero_point,
    output1 ? output1->params.scale : -1.0f,
    output1 ? output1->params.zero_point : -999,
    (int)input->type,
    (int)output->type,
    (int)(output1 ? output1->type : -1),
    (input->dims && input->dims->size>0) ? input->dims->data[input->dims->size-1] : -1
  );

  const int in_t   = (int)input->type;
  const int out0_t = (int)output->type;

  Serial.printf(
    "[MODEL] type=%s | arch=%s | io=%s->%s | precision_io=%s | quant_style=%s\n",
    LITEML_MODEL_NAME,
    detect_model_family(model),
    tf_type_name(in_t),
    tf_type_name(out0_t),
    detect_model_precision_io(in_t, out0_t),
    detect_model_quantization_style(model, in_t, out0_t)
  );

  #if defined(HAS_SCALERS_EXPORT)
  {
    uint32_t crc = 0u;
    crc ^= crc32_bytes(reinterpret_cast<const uint8_t*>(X_MIN), sizeof(float)*K_NUM_FEATURES);
    crc ^= crc32_bytes(reinterpret_cast<const uint8_t*>(X_MAX), sizeof(float)*K_NUM_FEATURES);
    crc ^= crc32_bytes(reinterpret_cast<const uint8_t*>(DY_MIN), sizeof(float)*2);
    crc ^= crc32_bytes(reinterpret_cast<const uint8_t*>(DY_MAX), sizeof(float)*2);
    Serial.printf("[SCALERS] crc32=0x%08lX K=%d | Y0=[%.3f,%.3f] Y1=[%.3f,%.3f]\n",
                (unsigned long)crc, K_NUM_FEATURES,
                DY_MIN[0], DY_MAX[0], DY_MIN[1], DY_MAX[1]);

    Serial.printf("[SCALERS] ΔY=[%.3f,%.3f]..[%.3f,%.3f]\n",
                DY_MIN[0], DY_MIN[1], DY_MAX[0], DY_MAX[1]);

    // --- NEW: confirm input scalers ---
    Serial.printf("[SCALERS] X0=[%.6f,%.6f] X1=[%.6f,%.6f]\n",
                X_MIN[0], X_MAX[0],
                X_MIN[1], X_MAX[1]);

    Serial.printf("[SCALERS] X2=[%.6f,%.6f] X3=[%.6f,%.6f]\n",
                X_MIN[2], X_MAX[2],
                X_MIN[3], X_MAX[3]);
}
  #endif

  g_inference_ready = true;
  return true;
}

/**
 * @brief Runs one inference step using the current sensor sample.
 * @param d Current sensor and temporal feature data.
 * @return InferResult with prediction fields, timing, and status flags.
 *
 * Pipeline:
 * 1. Build the current feature vector.
 * 2. Push it into the 24-step sliding window.
 * 3. Pack and scale the full window.
 * 4. Write the input tensor.
 * 5. Invoke the TFLM interpreter.
 * 6. Decode one-tensor or two-head outputs.
 * 7. Invert residual scaling and reconstruct absolute Tin / Hin.
 */
InferResult run_inference(const SensorData& d){
  InferResult r{};
  r.ok       = false;
  r.Tin      = d.T_in;
  r.Hin      = d.H_in;
  r.Tin_pred = 0.0f;
  r.Hin_pred = 0.0f;
  r.us       = 0;
  r.us_invoke = 0;
  r.us_event  = 0;
  r.invoked  = false;
  r.power_valid = false;
  r.energy_invoke_mWh = 0.0f;
  r.energy_event_mWh  = 0.0f;
  r.pwr_invoke_pre = {};
  r.pwr_invoke_post = {};
  r.pwr_event_pre = {};
  r.pwr_event_post = {};

  if (!g_inference_ready) return r;
  if (!interpreter || !input || !output){
    Serial.println("[infer] invalid interpreter/tensors");
    return r;
  }

  const uint32_t t0_event = micros();
#if BENCH_ENABLE_POWER
  if (bench_power_poll(r.pwr_event_pre)) {
    r.power_valid = true;
  }
#endif

  // --- 1) Build features for the current hour ---
  float f[K_NUM_FEATURES];
  const float sin_h = d.sin_hour;
  const float cos_h = d.cos_hour;
  const float wday  = d.weekday_pandas;
  const float mon   = d.month_1_12;

  // Features (12) — Option 3: without T_diff / H_diff + weekday / month
  // Order must be IDENTICAL to training:
  // [T_out, H_out, T_in_lag1, H_in_lag1, T_out_lag1, H_out_lag1,
  //  T_in_lag2, H_in_lag2, sin_hour, cos_hour, weekday, month]
  f[0]  = d.T_out;
  f[1]  = d.H_out;
  f[2]  = d.T_in_lag1;
  f[3]  = d.H_in_lag1;
  f[4]  = d.T_out_lag1;
  f[5]  = d.H_out_lag1;
  f[6]  = d.T_in_lag2;
  f[7]  = d.H_in_lag2;
  f[8]  = sin_h;
  f[9]  = cos_h;
  f[10] = wday;
  f[11] = mon;

  // --- 2) Update window ---
  window_push(f);

  // --- 3) Pack window ---
  float x_scaled_window[K_WINDOW_SIZE];
  if (!window_pack_scaled(x_scaled_window, K_WINDOW_SIZE)){
    Serial.printf("[infer] warmup: window %u/24 - (Invoke NOT executed)\n", (unsigned)g_window_filled);
    r.ok       = false;
    r.Tin      = d.T_in;
    r.Hin      = d.H_in;
    r.Tin_pred = d.T_in;
    r.Hin_pred = d.H_in;

    // Even during warm-up, update replay "previous" values to keep a clean sequence.
    #if LITEML_REPLAY
      replay_prev_Tin = d.T_in;
      replay_prev_Hin = d.H_in;
    #endif

    return r;
  }

  // --- 4) Copy to input tensor ---
#if LITEML_REPLAY
  const long epoch_dbg = (long)sensors_replay_epoch();
#endif

  if (input->type == kTfLiteInt8){
    const float s = input->params.scale;
    const int   z = input->params.zero_point;
    if (s <= 0.f){
      Serial.println("[infer] invalid input scale");
      return r;
    }
    for (int i = 0; i < K_WINDOW_SIZE; ++i){
      input->data.int8[i] = q_int8(x_scaled_window[i], s, z);
    }
  } else if (input->type == kTfLiteFloat32){
    memcpy(input->data.f, x_scaled_window, sizeof(float) * K_WINDOW_SIZE);
  } else {
    Serial.println("[infer] unsupported input type");
    return r;
  }

#if LITEML_DBG_MODEL_IO && LITEML_REPLAY
  float dbg_input_tensor_float_window[K_WINDOW_SIZE];
  bool  dbg_model_in_captured = false;

  if (input->type == kTfLiteInt8){
    for (int i = 0; i < K_WINDOW_SIZE; ++i){
      dbg_input_tensor_float_window[i] = (float)input->data.int8[i];
    }
  } else {
    memcpy(dbg_input_tensor_float_window, input->data.f, sizeof(dbg_input_tensor_float_window));
  }
  dbg_model_in_captured = true;
#elif LITEML_DBG_MODEL_IO
  /* model input debug remains disabled in FIELD */
#endif

#if BENCH_ENABLE_MEMORY && LITEML_TARGET_STM32F411RE
  // Update the observed STM32 minimum free-memory value while the local
  // run_inference() buffers are still active. This snapshot is taken before
  // Invoke() and therefore does not capture peak stack usage inside TFLM.
  // The ESP32 instrumentation path remains unchanged.
  (void)bench_memory_snapshot();
#endif

#if BENCH_ENABLE_POWER
  if (r.power_valid && !bench_power_poll(r.pwr_invoke_pre)) {
    r.power_valid = false;
  }
#endif

  // --- 5) Invoke ---
  const uint32_t t0 = micros();
  if (interpreter->Invoke() != kTfLiteOk){
    Serial.println("[infer] Invoke failed");
    return r;
  }
  const uint32_t t1 = micros();
  r.us = t1 - t0;
  r.us_invoke = r.us;
  r.invoked = true;

#if BENCH_ENABLE_POWER
  if (r.power_valid && bench_power_poll(r.pwr_invoke_post)) {
    r.energy_invoke_mWh = r.pwr_invoke_post.energy_mWh - r.pwr_invoke_pre.energy_mWh;
    if (r.energy_invoke_mWh < 0.0f) r.energy_invoke_mWh = 0.0f;
  } else {
    r.power_valid = false;
    r.energy_invoke_mWh = 0.0f;
  }
#endif

#if !LITEML_REPLAY
  Serial.println("[infer] REAL - Invoke executed (model ran)");
#endif

#if LITEML_DBG_MODEL_IO && LITEML_REPLAY
  if (r.invoked) {
    Serial.println();
    dbg_model_out_raw_print_all(g_dbg_model_io_idx, epoch_dbg);
#if LITEML_DBG_MODEL_IO_STABILITY
    Serial.println();
    dbg_model_out_stability_print_all(g_dbg_model_io_idx, epoch_dbg);
#endif
  }
#endif

  // --- 6) Output decoding: 1 concatenated tensor OR 2 separate heads ---
#if LITEML_DBG_MODEL_IO && LITEML_REPLAY
  float out_tensor_positional[2] = {0.0f, 0.0f};
#endif
  float out_float_positional[2] = {0.0f, 0.0f};
#if LITEML_REPLAY
  bool dbg_out_enabled = false;
  bool dbg_out_two_heads = false;
  int dbg_out_q0 = 0;
  int dbg_out_q1 = 0;
  float dbg_out_s0 = 0.0f;
  float dbg_out_s1 = 0.0f;
  int dbg_out_z0 = 0;
  int dbg_out_z1 = 0;
#endif

  TfLiteTensor* out0 = interpreter->output(0);
  TfLiteTensor* out1 = (interpreter->outputs_size() > 1) ? interpreter->output(1) : nullptr;

  auto read_q = [&](TfLiteTensor* t, int idx)->float {
    const float s = t->params.scale;
    const int   z = t->params.zero_point;
    if (s <= 0.f) return 0.0f;
    return dq_int8(t->data.int8[idx], s, z);
  };

  if (out1){
    // Two separate heads: each tensor contains one positional value.
    if (out0->type == kTfLiteInt8 && out1->type == kTfLiteInt8){
#if LITEML_DBG_MODEL_IO && LITEML_REPLAY
      out_tensor_positional[0] = (float)out0->data.int8[0];
      out_tensor_positional[1] = (float)out1->data.int8[0];
#endif
      out_float_positional[0] = read_q(out0, 0);
      out_float_positional[1] = read_q(out1, 0);
      #if LITEML_REPLAY
      dbg_out_enabled = true;
      dbg_out_two_heads = true;
      dbg_out_q0 = (int)out0->data.int8[0];
      dbg_out_q1 = (int)out1->data.int8[0];
      dbg_out_s0 = out0->params.scale;
      dbg_out_s1 = out1->params.scale;
      dbg_out_z0 = out0->params.zero_point;
      dbg_out_z1 = out1->params.zero_point;
      #endif
    } else if (out0->type == kTfLiteFloat32 && out1->type == kTfLiteFloat32){
#if LITEML_DBG_MODEL_IO && LITEML_REPLAY
      out_tensor_positional[0] = out0->data.f[0];
      out_tensor_positional[1] = out1->data.f[0];
#endif
      out_float_positional[0] = out0->data.f[0];
      out_float_positional[1] = out1->data.f[0];
    } else {
      Serial.println("[infer] unsupported / mixed output types (2 heads)");
      return r;
    }
  } else {
    // One tensor containing two positional values.
    if (out0->type == kTfLiteInt8){
      const float s = out0->params.scale;
      const int   z = out0->params.zero_point;
      if (s <= 0.f){
        Serial.println("[infer] invalid output scale");
        return r;
      }
#if LITEML_DBG_MODEL_IO && LITEML_REPLAY
      out_tensor_positional[0] = (float)out0->data.int8[0];
      out_tensor_positional[1] = (float)out0->data.int8[1];
#endif
      out_float_positional[0] = dq_int8(out0->data.int8[0], s, z);
      out_float_positional[1] = dq_int8(out0->data.int8[1], s, z);
      #if LITEML_REPLAY
      dbg_out_enabled = true;
      dbg_out_two_heads = false;
      dbg_out_q0 = (int)out0->data.int8[0];
      dbg_out_q1 = (int)out0->data.int8[1];
      dbg_out_s0 = s;
      dbg_out_s1 = s;
      dbg_out_z0 = z;
      dbg_out_z1 = z;
      #endif
    } else if (out0->type == kTfLiteFloat32){
#if LITEML_DBG_MODEL_IO && LITEML_REPLAY
      out_tensor_positional[0] = out0->data.f[0];
      out_tensor_positional[1] = out0->data.f[1];
#endif
      out_float_positional[0] = out0->data.f[0];
      out_float_positional[1] = out0->data.f[1];
    } else {
      Serial.println("[infer] unsupported output type");
      return r;
    }
  }

  // Current semantic mapping keeps positional output 0 -> T and 1 -> H.
  const float y_T_scaled = out_float_positional[0];
  const float y_H_scaled = out_float_positional[1];

  // --- 7) Post-processing: inverse Min-Max (raw residual Δ) + reconstruction ---
  float y_T_scaled_safe = y_T_scaled;
  float y_H_scaled_safe = y_H_scaled;

  if (!isfinite(y_T_scaled_safe)) y_T_scaled_safe = 0.0f;
  if (!isfinite(y_H_scaled_safe)) y_H_scaled_safe = 0.0f;
  float d_T_pred = minmax_inverse(y_T_scaled_safe, DY_MIN[0], DY_MAX[0]);  // ΔT_in (residual)
  float d_H_pred = minmax_inverse(y_H_scaled_safe, DY_MIN[1], DY_MAX[1]);  // ΔH_in (residual)

  float Tprev, Hprev;

  // Unified evaluation baseline for REPLAY and FIELD:
  // both modes reconstruct the absolute prediction from lag1.
  Tprev = d.T_in_lag1;
  Hprev = d.H_in_lag1;

  float Tin = Tprev + d_T_pred;
  float Hin = Hprev + d_H_pred;

  if (!isfinite(Tin)) Tin = d.T_in;
  if (!isfinite(Hin)) Hin = d.H_in;

  // Update replay "previous" values with the current stream sample
  // (does NOT affect FIELD mode).
  #if LITEML_REPLAY
    replay_prev_Tin = d.T_in;
    replay_prev_Hin = d.H_in;
  #endif

  r.Tin_pred = Tin;
  r.Hin_pred = Hin;
  r.ok       = true;
  r.us_event = micros() - t0_event;

#if BENCH_ENABLE_POWER
  if (r.power_valid) {
    if (bench_power_poll(r.pwr_event_post)) {
      r.energy_event_mWh = r.pwr_event_post.energy_mWh - r.pwr_event_pre.energy_mWh;
      if (r.energy_event_mWh < 0.0f) r.energy_event_mWh = 0.0f;
    } else {
      // Fallback: reuse the latest valid snapshot to avoid zeroed post-event logs.
      if (bench_power_last(r.pwr_event_post)) {
        r.energy_event_mWh = r.pwr_event_post.energy_mWh - r.pwr_event_pre.energy_mWh;
        if (r.energy_event_mWh < 0.0f) r.energy_event_mWh = 0.0f;
      } else {
        r.power_valid = false;
        r.energy_event_mWh = 0.0f;
      }
    }
  }
#endif

#if LITEML_REPLAY
#if LITEML_DBG_MODEL_IO
  if (r.invoked && dbg_model_in_captured) {
    Serial.println();
    dbg_model_in_print_window(
      g_dbg_model_io_idx,
      epoch_dbg,
      d,
      x_scaled_window,
      dbg_input_tensor_float_window
    );
  }
#endif
  
  Serial.println();
  Serial.println("[infer] REAL - Invoke executed (model ran)");

  if (dbg_out_enabled) {
    if (dbg_out_two_heads) {
      Serial.printf("[DBG_OUT] two_heads_int8 | out_o0_tensor=%d out_o1_tensor=%d | out_o0_scale=%.9f out_o0_zp=%d | out_o1_scale=%.9f out_o1_zp=%d | out_o0_float=%.6f out_o1_float=%.6f\n",
        dbg_out_q0, dbg_out_q1,
        dbg_out_s0, dbg_out_z0,
        dbg_out_s1, dbg_out_z1,
        out_float_positional[0], out_float_positional[1]
      );
    } else {
      Serial.printf("[DBG_OUT] one_tensor_int8 | out_o0_tensor=%d out_o1_tensor=%d | out_scale=%.9f out_zp=%d | out_o0_float=%.6f out_o1_float=%.6f\n",
        dbg_out_q0, dbg_out_q1,
        dbg_out_s0, dbg_out_z0,
        out_float_positional[0], out_float_positional[1]
      );
    }
  }

#if LITEML_DBG_MODEL_IO
  if (r.invoked) {
    DebugOutputRecord dbg_output_record{};
    fill_debug_output_record(
      &dbg_output_record,
      g_dbg_model_io_idx,
      epoch_dbg,
      out_tensor_positional[0],
      out_tensor_positional[1],
      out_float_positional[0],
      out_float_positional[1],
      y_T_scaled_safe,
      y_H_scaled_safe,
      d_T_pred,
      d_H_pred,
      Tprev,
      Hprev,
      Tin,
      Hin
    );
    dbg_model_out_print_row(dbg_output_record);
    dbg_model_out_bits_print_row(dbg_output_record);
    g_dbg_model_io_idx++;
  }
#endif

#if LITEML_DBG_REPLAY_EXACT
  if (r.invoked) {
    dbg_replay_print_row(
      epoch_dbg,
      d,
      y_T_scaled_safe,
      y_H_scaled_safe,
      d_T_pred,
      d_H_pred,
      Tprev,
      Hprev,
      Tin,
      Hin
    );
  }
#endif
#endif
  return r;
}

// ===== The remainder of the file stays equal to the original (SELFTEST etc.) =====
// The structure was preserved. If desired, the same replay reset / previous-value
// style can also be applied to selftest, although it is not required for replay 1:1.

/**
 * @brief Runs a lightweight inference self-test.
 * @return InferenceSelfTestResult with output values and pass/fail status.
 *
 * The self-test uses neutral zero features, performs a full inference pass,
 * decodes outputs, and checks whether the reconstructed values remain within
 * conservative expected bounds.
 */
InferenceSelfTestResult inference_selftest(){
  InferenceSelfTestResult out{}; out.ok=false;

  if (!g_inference_ready) { Serial.println("[SELFTEST] not ready"); return out; }
  if (!interpreter || !input || !output){
    Serial.println("[SELFTEST] invalid interpreter/tensors");
    return out;
  }

  float f[K_NUM_FEATURES];
  make_neutral_features(f);

  float x_scaled[K_WINDOW_SIZE];
  int idx = 0;
  for (int t = 0; t < K_WINDOW_STEPS; ++t){
    for (int j = 0; j < K_NUM_FEATURES; ++j){
      const float x   = f[j];
      const float mn  = X_MIN[j];
      const float mx  = X_MAX[j];
      const float den = (mx - mn);
      float xs = 0.0f;
      if (den > 1e-9f){
        xs = (x - mn) / den;
      }
      x_scaled[idx++] = xs;
    }
  }

  if (input->type == kTfLiteInt8){
    const float s = input->params.scale;
    const int   z = input->params.zero_point;
    if (s <= 0.f){ Serial.println("[SELFTEST] invalid input scale"); return out; }
    for (int i = 0; i < K_WINDOW_SIZE; ++i){
      input->data.int8[i] = q_int8(x_scaled[i], s, z);
    }
  } else if (input->type == kTfLiteFloat32){
    memcpy(input->data.f, x_scaled, sizeof(float) * K_WINDOW_SIZE);
  } else {
    Serial.println("[SELFTEST] unsupported input type");
    return out;
  }

  if (interpreter->Invoke() != kTfLiteOk){
    Serial.println("[SELFTEST] Invoke failed");
    return out;
  }

  // --- SELFTEST output: same handling for 1 tensor vs 2 heads ---
  float y_scaled[2];

  TfLiteTensor* out0 = interpreter->output(0);
  TfLiteTensor* out1 = (interpreter->outputs_size() > 1) ? interpreter->output(1) : nullptr;

  auto read_q = [&](TfLiteTensor* t, int idx)->float {
    const float s = t->params.scale;
    const int   z = t->params.zero_point;
    if (s <= 0.f) return 0.0f;
    return dq_int8(t->data.int8[idx], s, z);
  };

  if (out1){
    if (out0->type == kTfLiteInt8 && out1->type == kTfLiteInt8){
      y_scaled[0] = read_q(out0, 0);
      y_scaled[1] = read_q(out1, 0);
    } else if (out0->type == kTfLiteFloat32 && out1->type == kTfLiteFloat32){
      y_scaled[0] = out0->data.f[0];
      y_scaled[1] = out1->data.f[0];
    } else {
      Serial.println("[SELFTEST] unsupported / mixed output types (2 heads)");
      return out;
    }
  } else {
    if (out0->type == kTfLiteInt8){
      const float s = out0->params.scale;
      const int   z = out0->params.zero_point;
      if (s <= 0.f){ Serial.println("[SELFTEST] invalid output scale"); return out; }
      y_scaled[0] = dq_int8(out0->data.int8[0], s, z);
      y_scaled[1] = dq_int8(out0->data.int8[1], s, z);
    } else if (out0->type == kTfLiteFloat32){
      y_scaled[0] = out0->data.f[0];
      y_scaled[1] = out0->data.f[1];
    } else {
      Serial.println("[SELFTEST] unsupported output type");
      return out;
    }
  }

  float Tin = minmax_inverse(y_scaled[0], DY_MIN[0], DY_MAX[0]);
  float Hin = minmax_inverse(y_scaled[1], DY_MIN[1], DY_MAX[1]);

  if (!isfinite(Tin) || !isfinite(Hin)){
    Serial.println("[SELFTEST] non-finite output");
    return out;
  }

  const float margin_T = 0.5f * (DY_MAX[0] - DY_MIN[0]) + 0.5f;
  const float margin_H = 0.5f * (DY_MAX[1] - DY_MIN[1]) + 0.5f;

  const float min_T = DY_MIN[0] - margin_T;
  const float max_T = DY_MAX[0] + margin_T;
  const float min_H = DY_MIN[1] - margin_H;
  const float max_H = DY_MAX[1] + margin_H;

  bool in_range = (Tin >= min_T && Tin <= max_T &&
                   Hin >= min_H && Hin <= max_H);

  if (!in_range){
    Serial.printf("[SELFTEST] Tin=%.2f Hin=%.2f outside expected bounds (Y0=[%.3f,%.3f] Y1=[%.3f,%.3f])\n",
                  Tin, Hin, DY_MIN[0], DY_MAX[0], DY_MIN[1], DY_MAX[1]);
  }

  out.ok = in_range;
  out.y0 = Tin;
  out.y1 = Hin;

  Serial.printf("[SELFTEST] OK? %s  y0=%.3f  y1=%.3f  (in_scale=%.6f in_zp=%d  out0_scale=%.6f out0_zp=%d)\n",
    out.ok ? "YES" : "NO",
    (float)out.y0, (float)out.y1,
    (float)input->params.scale, (int)input->params.zero_point,
    (float)out0->params.scale, (int)out0->params.zero_point
  );

  return out;
}

/**
 * @brief Exposes TensorFlow Lite Micro arena usage for external modules.
 * @return Arena bytes used.
 *
 * Returns the post-allocation snapshot when available; otherwise queries the
 * interpreter directly if it exists.
 */
extern "C" size_t tflm_arena_used_bytes(void) {
  if (g_arena_used_snapshot > 0) return g_arena_used_snapshot;
  if (interpreter) return interpreter->arena_used_bytes();
  return 0;
}