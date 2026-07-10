#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace puercgp {
namespace detail {

inline void throw_if_cuda_error(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

class cuda_event_timer {
 public:
  cuda_event_timer() {
    throw_if_cuda_error(cudaEventCreate(&start_), "cudaEventCreate(start)");
    throw_if_cuda_error(cudaEventCreate(&stop_), "cudaEventCreate(stop)");
  }

  cuda_event_timer(const cuda_event_timer&) = delete;
  cuda_event_timer& operator=(const cuda_event_timer&) = delete;

  ~cuda_event_timer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void begin(cudaStream_t stream) {
    throw_if_cuda_error(cudaEventRecord(start_, stream), "cudaEventRecord");
  }

  float end(cudaStream_t stream) {
    throw_if_cuda_error(cudaEventRecord(stop_, stream), "cudaEventRecord");
    throw_if_cuda_error(cudaEventSynchronize(stop_), "cudaEventSynchronize");
    float ms = 0.0f;
    throw_if_cuda_error(cudaEventElapsedTime(&ms, start_, stop_),
                        "cudaEventElapsedTime");
    return ms;
  }

 private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

inline float elapsed_ms(std::chrono::high_resolution_clock::time_point start) {
  auto stop = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<float, std::milli>(stop - start).count();
}

template <typename function_t>
float timed_gpu(cudaStream_t stream, bool enabled, function_t&& function) {
  if (!enabled) {
    function();
    return 0.0f;
  }
  cuda_event_timer timer;
  timer.begin(stream);
  function();
  return timer.end(stream);
}

inline int grid_for(std::size_t work, int threads) {
  auto blocks = static_cast<int>((work + static_cast<std::size_t>(threads) - 1) /
                                 static_cast<std::size_t>(threads));
  return std::max(1, std::min(blocks, 65535));
}

}  // namespace detail
}  // namespace puercgp
