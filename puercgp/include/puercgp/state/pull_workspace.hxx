#pragma once

#include <cstddef>

#include <thrust/device_vector.h>

namespace puercgp {

class pull_workspace {
 public:
  void resize(std::size_t vertex_count) {
    vertex_count_ = vertex_count;
    unique_flags_.resize(vertex_count_);
    pair_counts_.resize(vertex_count_);
    unique_offsets_.resize(vertex_count_);
  }

  std::size_t vertex_count() const noexcept { return vertex_count_; }

  thrust::device_vector<unsigned long long>& unique_flags_vector() noexcept {
    return unique_flags_;
  }
  thrust::device_vector<unsigned long long>& pair_counts_vector() noexcept {
    return pair_counts_;
  }
  thrust::device_vector<unsigned long long>& unique_offsets_vector() noexcept {
    return unique_offsets_;
  }

 private:
  std::size_t vertex_count_ = 0;
  thrust::device_vector<unsigned long long> unique_flags_;
  thrust::device_vector<unsigned long long> pair_counts_;
  thrust::device_vector<unsigned long long> unique_offsets_;
};

}  // namespace puercgp
