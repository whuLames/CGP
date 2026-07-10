#pragma once

#include <cstdint>
#include <limits>

#include <cuda_runtime.h>

namespace puercgp {
namespace algorithms {

using unified_value_t = float;

enum class algo_kind_t : std::uint8_t {
  bfs = 0,
  sssp = 1,
  wcc = 2
};

__host__ __device__ __forceinline__ constexpr unified_value_t
unified_infinity() {
  return std::numeric_limits<unified_value_t>::infinity();
}

__host__ __device__ __forceinline__ constexpr unified_value_t
unified_source_value() {
  return unified_value_t{0};
}

static_assert(sizeof(unified_value_t) == 4,
              "unified_value_t must be 4 bytes for atomicCAS reinterpret");
static_assert(sizeof(algo_kind_t) == 1,
              "algo_kind_t must be 1 byte for compact slot_kinds array");

}  // namespace algorithms
}  // namespace puercgp
