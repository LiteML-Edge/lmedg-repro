#include "tensorflow/lite/core/api/error_reporter.h"
#include <cstdarg>

namespace tflite {

int ErrorReporter::Report(const char* format, ...) {
  va_list args;
  va_start(args, format);
  const int result = Report(format, args);
  va_end(args);
  return result;
}

}  // namespace tflite
