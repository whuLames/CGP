#pragma once

#include <cstddef>

#include <thrust/device_vector.h>

#include <puercgp/core/types.hxx>

namespace puercgp {

template <typename value_t> class rank_workspace {
public:
  void resize(std::size_t vertex_count, std::size_t query_count) {
    const std::size_t value_count = vertex_count * query_count;
    current_.resize(value_count);
    next_.resize(value_count);
    damping_factors_.resize(query_count);
    epsilons_.resize(query_count);
    sources_.resize(query_count);
    dangling_mass_.resize(query_count);
    changed_queries_.resize(1);
  }

  thrust::device_vector<value_t> &current_vector() noexcept { return current_; }
  thrust::device_vector<value_t> &next_vector() noexcept { return next_; }
  thrust::device_vector<value_t> &damping_factors_vector() noexcept {
    return damping_factors_;
  }
  thrust::device_vector<value_t> &epsilons_vector() noexcept {
    return epsilons_;
  }
  thrust::device_vector<int> &sources_vector() noexcept { return sources_; }
  thrust::device_vector<value_t> &dangling_mass_vector() noexcept {
    return dangling_mass_;
  }
  thrust::device_vector<query_mask_t> &changed_queries_vector() noexcept {
    return changed_queries_;
  }

  void swap_ranks() { current_.swap(next_); }

private:
  thrust::device_vector<value_t> current_;
  thrust::device_vector<value_t> next_;
  thrust::device_vector<value_t> damping_factors_;
  thrust::device_vector<value_t> epsilons_;
  thrust::device_vector<int> sources_;
  thrust::device_vector<value_t> dangling_mass_;
  thrust::device_vector<query_mask_t> changed_queries_;
};

} // namespace puercgp
