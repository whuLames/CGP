#pragma once

#include <cuda_runtime.h>

namespace puercgp {

struct execution_lane_t {
  cudaStream_t stream = nullptr;
  int priority = 0;
  int sm_quota = 0;
};

}  // namespace puercgp
