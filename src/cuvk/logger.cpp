#include <mutex>
#include <cstdio>
#include <strstream>
#include <ctime>
#include <fmt/format.h>
#include "cuvk/logger.hpp"

L_CUVK_BEGIN_
Logger LOG;

namespace detail {
  using namespace std::chrono;
  using namespace std::chrono_literals;
  void log_thread_main(Logger& logger) {
    for (;;) {
      if (logger._msgs.empty()) { continue; }
      {
        std::scoped_lock lk(logger._sync);
        while (!logger._msgs.empty()) {
          auto msg = logger._msgs.front();
          auto tm = msg.time;
          auto diff = tm - logger._start;
          auto micro = duration_cast<microseconds>(diff).count();
          auto milli = micro / 1000;
          micro %= 1000;

          std::fprintf(stdout, "% 8d.%03dms [%s] %s\n",
            (int)milli, (int)micro, msg.level, msg.msg.c_str());
          logger._msgs.pop();
        }
      }
#ifdef NDEBUG
      std::this_thread::sleep_for(100ms);
#endif // NDEBUG
    }
  }
}

Logger::Logger() :
  _sync(),
  _msgs(),
  _start(std::chrono::high_resolution_clock::now()),
  _th([this] { detail::log_thread_main(*this); }) {
  _th.detach();
}

const LogLevel Logger::DEBUG = "DEBUG";
const LogLevel Logger::TRACE = "TRACE";
const LogLevel Logger::INFO = "INFO";
const LogLevel Logger::WARNING = "WARNING";
const LogLevel Logger::ERROR = "ERROR";

VkCheck::VkCheck(const char* file, int line) : _file(file), _line(line) { }
VkResult VkCheck::operator<(int result) const {
  result = -result;
  if (result == VK_SUCCESS) return (VkResult)result;
  bool err = result < 0;
  const char* msg = nullptr;
  switch (result) {
  case VK_NOT_READY: msg = "not ready"; break;
  case VK_TIMEOUT: msg = "timeout"; break;
  case VK_EVENT_SET: msg = "event set"; break;
  case VK_EVENT_RESET: msg = "event reset"; break;
  case VK_INCOMPLETE: msg = "incomplete"; break;
  case VK_ERROR_OUT_OF_HOST_MEMORY: msg = "out of host memory"; break;
  case VK_ERROR_OUT_OF_DEVICE_MEMORY: msg = "out of device memory"; break;
  case VK_ERROR_INITIALIZATION_FAILED: msg = "initialization failed"; break;
  case VK_ERROR_DEVICE_LOST: msg = "device lost"; break;
  case VK_ERROR_MEMORY_MAP_FAILED: msg = "memory map failed"; break;
  case VK_ERROR_LAYER_NOT_PRESENT: msg = "layer not present"; break;
  case VK_ERROR_EXTENSION_NOT_PRESENT: msg = "extension not present"; break;
  case VK_ERROR_FEATURE_NOT_PRESENT: msg = "feature not present"; break;
  case VK_ERROR_INCOMPATIBLE_DRIVER: msg = "incompatible driver"; break;
  case VK_ERROR_TOO_MANY_OBJECTS: msg = "too many objects"; break;
  case VK_ERROR_FORMAT_NOT_SUPPORTED: msg = "format not supported"; break;
  case VK_ERROR_FRAGMENTED_POOL: msg = "fragmented pool"; break;
  case VK_ERROR_OUT_OF_POOL_MEMORY: msg = "out of pool memory"; break;
  case VK_ERROR_INVALID_EXTERNAL_HANDLE: msg = "invalid external handle"; break;
  }
  if (msg != nullptr) {
    if (err) {
      LOG.error(fmt::format("bad vulkan result: {}", msg));
    } else {
      LOG.warning(fmt::format("suspecious vulkan result: {}", msg));
    }
  } else {
    LOG.error("unknown error {}", result);
  }
  LOG.trace("See {}:{}", _file, _line);
  return (VkResult)result;
}

L_CUVK_END_
