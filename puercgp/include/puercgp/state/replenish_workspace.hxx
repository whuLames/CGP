#pragma once

#include <cstddef>

#include <thrust/device_vector.h>

#include <puercgp/algorithms/hybrid.hxx>

namespace puercgp {

class replenish_workspace {
 public:
  using value_type = algorithms::unified_value_t;

  void resize_final_buffer(std::size_t query_count, std::size_t vertex_count) {
    query_count_ = query_count;
    vertex_count_ = vertex_count;
    final_values_.resize(query_count_ * vertex_count_);
  }

  std::size_t query_count() const noexcept { return query_count_; }
  std::size_t vertex_count() const noexcept { return vertex_count_; }

  thrust::device_vector<value_type>& final_values_vector() noexcept {
    return final_values_;
  }

 private:
  std::size_t query_count_ = 0;
  std::size_t vertex_count_ = 0;
  thrust::device_vector<value_type> final_values_;
};

}  // namespace puercgp
