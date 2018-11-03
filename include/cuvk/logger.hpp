#pragma once
#include <chrono>
#include <thread>
#include <mutex>
#include <string>
#include <queue>
#include "cuvk/comdef.hpp"
#include <vulkan/vulkan.h>
#include <fmt/format.h>

L_CUVK_BEGIN_

/*
 * Logger object. Logs are written to standard error.
 */
class Logger;
class VkCheck;

/*
 * Check vulkan result (error code) and panic when the result is not acceptable,
 * that an error has occurred.
 *
 * Use the macro like this.
 *
 * L_VK <- vk*();
 */
#define L_VK ::cuvk::VkCheck(__FILE__, __LINE__)
#undef ERROR // This symbol is a macro in `wingdi.h`.



namespace detail {
  void log_thread_main(Logger& logger);
}
using LogLevel = const char*;
class Logger {
private:
  friend void detail::log_thread_main(Logger& logger);
  struct LogMessage {
  public:
    LogLevel level;
    std::chrono::high_resolution_clock::time_point time;
    std::string msg;
  };

  std::chrono::high_resolution_clock::time_point _start;
  std::mutex _sync;
  std::queue<LogMessage> _msgs;
  std::thread _th;

public:
  static const LogLevel DEBUG;
  static const LogLevel TRACE;
  static const LogLevel INFO;
  static const LogLevel WARNING;
  static const LogLevel ERROR;

  Logger();
  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

  template<typename ... TArgs>
  void log(const LogLevel level,
    const std::string& msg,
    const TArgs& ... args) {
    std::scoped_lock lk(_sync);
    LogMessage m = {
      level,
      std::chrono::high_resolution_clock::now(),
      fmt::format(msg, args ...)
    };
    _msgs.emplace(m);
  }
  template<>
  void log(const LogLevel level,
    const std::string& msg) {
    std::scoped_lock lk(_sync);
    LogMessage m = { level, std::chrono::high_resolution_clock::now(), msg };
    _msgs.emplace(m);
  }
  template<typename ... TArgs>
  void debug(const std::string& msg, const TArgs& ... args) {
    log(DEBUG, msg, args ...);
  }
  template<typename ... TArgs>
  void trace(const std::string& msg, const TArgs& ... args) {
    log(TRACE, msg, args ...);
  }
  template<typename ... TArgs>
  void info(const std::string& msg, const TArgs& ... args) {
    log(INFO, msg, args ...);
  }
  template<typename ... TArgs>
  void warning(const std::string& msg, const TArgs& ... args) {
    log(WARNING, msg, args ...);
  }
  template<typename ... TArgs>
  void error(const std::string& msg, const TArgs& ... args) {
    log(ERROR, msg, args ...);
  }
};
class VkCheck {
private:
  const char* _file;
  int _line;
public:
  VkCheck(const char* file, int line);
  VkResult operator<(int result) const;
};

/*
 * Global default logger instance.
 */
extern Logger LOG;

L_CUVK_END_
