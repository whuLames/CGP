#pragma once

#include <cuda_runtime.h>

#include <puercgp/core/types.hxx>

namespace puercgp {
namespace detail {

__device__ __forceinline__ query_mask_t query_bit(int query_id) {
  return query_mask_t{1} << query_id;
}

__device__ __forceinline__ int mask_ffs(query_mask_t mask) {
  return __ffsll(static_cast<long long>(mask));
}

__device__ __forceinline__ int mask_popcount(query_mask_t mask) {
  return __popcll(static_cast<unsigned long long>(mask));
}

}  // namespace detail
}  // namespace puercgp
