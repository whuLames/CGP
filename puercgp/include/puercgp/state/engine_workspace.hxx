#pragma once

#include <cstddef>

#include <thrust/device_vector.h>

#include <puercgp/core/types.hxx>

namespace puercgp {

template <typename vertex_t, typename value_t>
class engine_workspace {
 public:
  using vertex_type = vertex_t;
  using value_type = value_t;

  void resize(std::size_t vertex_count, std::size_t query_count) {
    vertex_count_ = vertex_count;
    query_count_ = query_count;
    const std::size_t value_count = vertex_count_ * query_count_;

    values_.resize(value_count);
    visited_mask_.resize(vertex_count_);
    frontier_mask_.resize(vertex_count_);
    next_frontier_mask_.resize(vertex_count_);
    frontier_vertices_.resize(vertex_count_);
    next_frontier_vertices_.resize(vertex_count_);
    current_unique_count_.resize(1);
    next_unique_count_.resize(1);
    next_pair_count_.resize(1);
    actual_degrees_.resize(vertex_count_);
    virtual_degrees_.resize(vertex_count_);
  }

  std::size_t vertex_count() const noexcept { return vertex_count_; }
  std::size_t query_count() const noexcept { return query_count_; }
  std::size_t value_count() const noexcept { return values_.size(); }

  thrust::device_vector<value_t>& values_vector() noexcept { return values_; }
  thrust::device_vector<query_mask_t>& visited_mask_vector() noexcept {
    return visited_mask_;
  }
  thrust::device_vector<query_mask_t>& frontier_mask_vector() noexcept {
    return frontier_mask_;
  }
  thrust::device_vector<query_mask_t>& next_frontier_mask_vector() noexcept {
    return next_frontier_mask_;
  }
  thrust::device_vector<vertex_t>& frontier_vertices_vector() noexcept {
    return frontier_vertices_;
  }
  thrust::device_vector<vertex_t>& next_frontier_vertices_vector() noexcept {
    return next_frontier_vertices_;
  }
  thrust::device_vector<unsigned long long>& current_unique_count_vector()
      noexcept {
    return current_unique_count_;
  }
  thrust::device_vector<unsigned long long>& next_unique_count_vector()
      noexcept {
    return next_unique_count_;
  }
  thrust::device_vector<unsigned long long>& next_pair_count_vector()
      noexcept {
    return next_pair_count_;
  }
  thrust::device_vector<unsigned long long>& actual_degrees_vector() noexcept {
    return actual_degrees_;
  }
  thrust::device_vector<unsigned long long>& virtual_degrees_vector() noexcept {
    return virtual_degrees_;
  }

  void swap_frontiers() {
    frontier_mask_.swap(next_frontier_mask_);
    frontier_vertices_.swap(next_frontier_vertices_);
  }

 private:
  std::size_t vertex_count_ = 0;
  std::size_t query_count_ = 0;
  thrust::device_vector<value_t> values_;
  thrust::device_vector<query_mask_t> visited_mask_;
  thrust::device_vector<query_mask_t> frontier_mask_;
  thrust::device_vector<query_mask_t> next_frontier_mask_;
  thrust::device_vector<vertex_t> frontier_vertices_;
  thrust::device_vector<vertex_t> next_frontier_vertices_;
  thrust::device_vector<unsigned long long> current_unique_count_;
  thrust::device_vector<unsigned long long> next_unique_count_;
  thrust::device_vector<unsigned long long> next_pair_count_;
  thrust::device_vector<unsigned long long> actual_degrees_;
  thrust::device_vector<unsigned long long> virtual_degrees_;
};

}  // namespace puercgp
