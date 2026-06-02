#pragma once

#include <cstddef>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

namespace puercgp {

template <typename value_t>
class value_matrix {
 public:
  using value_type = value_t;

  void resize(std::size_t query_count, std::size_t vertex_count) {
    query_count_ = query_count;
    vertex_count_ = vertex_count;
    values_.resize(query_count_ * vertex_count_);
  }

  value_t* data() {
    return values_.empty() ? nullptr : thrust::raw_pointer_cast(values_.data());
  }

  const value_t* data() const {
    return values_.empty() ? nullptr : thrust::raw_pointer_cast(values_.data());
  }

  std::size_t query_count() const noexcept { return query_count_; }

  std::size_t vertex_count() const noexcept { return vertex_count_; }

  std::size_t size() const noexcept { return values_.size(); }

  thrust::device_vector<value_t>& values() noexcept { return values_; }

  const thrust::device_vector<value_t>& values() const noexcept {
    return values_;
  }

 private:
  std::size_t query_count_ = 0;
  std::size_t vertex_count_ = 0;
  thrust::device_vector<value_t> values_;
};

}  // namespace puercgp
