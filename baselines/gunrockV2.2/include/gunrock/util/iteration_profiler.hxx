#pragma once

#include <gunrock/compat/runtime_api.h>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define GUNROCK_HAS_NVTX 1
#elif __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#define GUNROCK_HAS_NVTX 1
#endif
#endif

#ifndef GUNROCK_HAS_NVTX
#define GUNROCK_HAS_NVTX 0
#endif

namespace gunrock {
namespace util {
namespace iteration_profiler {

inline void push_range(const std::string& name) {
#if GUNROCK_HAS_NVTX
  nvtxRangePushA(name.c_str());
#else
  (void)name;
#endif
}

inline void pop_range() {
#if GUNROCK_HAS_NVTX
  nvtxRangePop();
#endif
}

class range_t {
 public:
  explicit range_t(const std::string& name, bool enabled = true)
      : active_(enabled) {
    if (active_)
      push_range(name);
  }
  range_t(const range_t&) = delete;
  range_t& operator=(const range_t&) = delete;
  ~range_t() {
    if (active_)
      pop_range();
  }

  void pop() {
    if (active_) {
      pop_range();
      active_ = false;
    }
  }

 private:
  bool active_;
};

class event_timer_t {
 public:
  explicit event_timer_t(bool enabled = true) : enabled_(enabled) {
    if (enabled_) {
      hipEventCreate(&start_);
      hipEventCreate(&stop_);
    }
  }

  event_timer_t(const event_timer_t&) = delete;
  event_timer_t& operator=(const event_timer_t&) = delete;

  ~event_timer_t() {
    if (enabled_) {
      hipEventDestroy(start_);
      hipEventDestroy(stop_);
    }
  }

  void start(hipStream_t stream = 0) {
    if (enabled_)
      hipEventRecord(start_, stream);
  }

  float stop(hipStream_t stream = 0) {
    if (!enabled_)
      return 0.0f;
    hipEventRecord(stop_, stream);
    hipEventSynchronize(stop_);
    float elapsed_ms = 0.0f;
    hipEventElapsedTime(&elapsed_ms, start_, stop_);
    return elapsed_ms;
  }

 private:
  bool enabled_;
  hipEvent_t start_;
  hipEvent_t stop_;
};

struct row_t {
  std::string range_name;
  long long active_vertices;
  long long active_edges;
  long long input_frontier;
  long long output_frontier;
  float elapsed_ms;
};

class csv_writer_t {
 public:
  csv_writer_t() = default;

  csv_writer_t(const std::string& path,
               const std::string& algorithm,
               const std::string& dataset,
               long long source)
      : path_(path), algorithm_(algorithm), dataset_(dataset), source_(source) {
    if (!path_.empty()) {
      file_.open(path_);
      file_ << "algorithm,dataset,source,run,iteration,range_name,"
            << "active_vertices,active_edges,input_frontier,output_frontier,"
            << "elapsed_ms\n";
    }
  }

  csv_writer_t(const csv_writer_t&) = delete;
  csv_writer_t& operator=(const csv_writer_t&) = delete;

  bool enabled() const { return file_.is_open(); }

  void set_run(int run) { run_ = run; }

  void write(int iteration, const row_t& row) {
    if (!enabled())
      return;

    file_ << algorithm_ << ',' << dataset_ << ',' << source_ << ',' << run_
          << ',' << iteration << ',' << row.range_name << ','
          << row.active_vertices << ',' << row.active_edges << ','
          << row.input_frontier << ',' << row.output_frontier << ','
          << std::fixed << std::setprecision(6) << row.elapsed_ms << '\n';
  }

 private:
  std::string path_;
  std::string algorithm_;
  std::string dataset_;
  long long source_{-1};
  int run_{0};
  std::ofstream file_;
};

inline std::string make_range_name(const std::string& algorithm,
                                   int iteration,
                                   const std::string& suffix = "") {
  std::ostringstream out;
  out << "gunrock:" << algorithm << ":iter:" << iteration;
  if (!suffix.empty())
    out << ':' << suffix;
  return out.str();
}

}  // namespace iteration_profiler
}  // namespace util
}  // namespace gunrock
