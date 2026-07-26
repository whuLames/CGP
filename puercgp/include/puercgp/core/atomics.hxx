#pragma once

#include <type_traits>

#include <cuda_runtime.h>

#include <puercgp/core/mask.hxx>
#include <puercgp/core/types.hxx>

namespace puercgp {
namespace detail {

__device__ __forceinline__ query_mask_t atomic_or_query_mask(
    query_mask_t* address,
    query_mask_t value) {
  return static_cast<query_mask_t>(
      atomicOr(reinterpret_cast<unsigned long long*>(address),
               static_cast<unsigned long long>(value)));
}

template <typename vertex_t>
__device__ __forceinline__ void mark_next_shared_frontier(
    vertex_t vertex,
    query_mask_t improved,
    query_mask_t* next_frontier_mask,
    vertex_t* next_frontier_vertices,
    unsigned long long* next_unique_count,
    unsigned long long* next_pair_count) {
  if (improved == 0) {
    return;
  }
  query_mask_t old_next = atomic_or_query_mask(next_frontier_mask + vertex,
                                               improved);
  query_mask_t new_bits = improved & ~old_next;
  if (new_bits != 0) {
    atomicAdd(next_pair_count,
              static_cast<unsigned long long>(mask_popcount(new_bits)));
  }
  if (old_next == 0) {
    unsigned long long position = atomicAdd(next_unique_count, 1ULL);
    next_frontier_vertices[position] = vertex;
  }
}

template <typename vertex_t>
__device__ __forceinline__ void mark_next_shared_frontier_with_signal(
    vertex_t vertex,
    query_mask_t improved,
    query_mask_t* next_frontier_mask,
    vertex_t* next_frontier_vertices,
    unsigned long long* next_unique_count,
    unsigned long long* next_pair_count,
    query_mask_t* active_union) {
  if (improved == 0) {
    return;
  }
  atomic_or_query_mask(active_union, improved);
  query_mask_t old_next = atomic_or_query_mask(next_frontier_mask + vertex,
                                               improved);
  query_mask_t new_bits = improved & ~old_next;
  if (new_bits != 0) {
    atomicAdd(next_pair_count,
              static_cast<unsigned long long>(mask_popcount(new_bits)));
  }
  if (old_next == 0) {
    unsigned long long position = atomicAdd(next_unique_count, 1ULL);
    next_frontier_vertices[position] = vertex;
  }
}

template <typename value_t>
__device__ __forceinline__ value_t atomic_min_value(value_t* address,
                                                    value_t value) {
  if constexpr (std::is_same<value_t, int>::value) {
    return atomicMin(address, value);
  } else {
    int* address_as_int = reinterpret_cast<int*>(address);
    int old = *address_as_int;
    while (value < __int_as_float(old)) {
      int assumed = old;
      old = atomicCAS(address_as_int, assumed, __float_as_int(value));
      if (assumed == old) {
        return __int_as_float(assumed);
      }
    }
    return __int_as_float(old);
  }
}

template <typename value_t>
__device__ __forceinline__ value_t atomic_max_value(value_t* address,
                                                    value_t value) {
  if constexpr (std::is_same<value_t, int>::value) {
    return atomicMax(address, value);
  } else {
    int* address_as_int = reinterpret_cast<int*>(address);
    int old = *address_as_int;
    while (value > __int_as_float(old)) {
      int assumed = old;
      old = atomicCAS(address_as_int, assumed, __float_as_int(value));
      if (assumed == old) {
        return __int_as_float(assumed);
      }
    }
    return __int_as_float(old);
  }
}

template <typename Policy>
__device__ __forceinline__ typename Policy::value_type atomic_reduce_value(
    typename Policy::value_type* address,
    typename Policy::value_type value) {
  if constexpr (Policy::reduction == reduction_kind_t::minimum) {
    return atomic_min_value(address, value);
  } else {
    static_assert(Policy::reduction == reduction_kind_t::maximum,
                  "unsupported frontier reduction");
    return atomic_max_value(address, value);
  }
}

}  // namespace detail
}  // namespace puercgp
