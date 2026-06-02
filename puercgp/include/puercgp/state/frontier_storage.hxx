#pragma once

#include <algorithm>
#include <cstddef>

#include <thrust/device_vector.h>

#include <puercgp/core/types.hxx>

namespace puercgp {

template <typename vertex_t>
struct frontier_item_t {
  vertex_t query_id = 0;
  vertex_t vertex = 0;
};

template <typename vertex_t, typename value_t>
class frontier_storage {
 public:
  using item_type = frontier_item_t<vertex_t>;
  using value_type = value_t;

  void resize(std::size_t query_count,
              std::size_t vertex_count,
              std::size_t edge_count = 0) {
    query_count_ = query_count;
    vertex_count_ = vertex_count;
    edge_count_ = edge_count;

    const std::size_t list_capacity =
        query_count_ * std::max<std::size_t>(vertex_count_, edge_count_);
    const std::size_t dense_capacity = query_count_ * vertex_count_;

    list_frontier_a_.resize(list_capacity);
    list_frontier_b_.resize(list_capacity);
    shared_vertices_a_.resize(vertex_count_);
    shared_vertices_b_.resize(vertex_count_);
    shared_masks_a_.resize(vertex_count_);
    shared_masks_b_.resize(vertex_count_);
    bitmap_a_.resize(dense_capacity);
    bitmap_b_.resize(dense_capacity);
    dense_a_.resize(dense_capacity);
    dense_b_.resize(dense_capacity);
    counts_.resize(2);
  }

  std::size_t query_count() const noexcept { return query_count_; }

  std::size_t vertex_count() const noexcept { return vertex_count_; }

  std::size_t edge_count() const noexcept { return edge_count_; }

  std::size_t list_capacity() const noexcept { return list_frontier_a_.size(); }

  std::size_t dense_capacity() const noexcept { return dense_a_.size(); }

  thrust::device_vector<item_type>& list_frontier_a() noexcept {
    return list_frontier_a_;
  }

  thrust::device_vector<item_type>& list_frontier_b() noexcept {
    return list_frontier_b_;
  }

  thrust::device_vector<vertex_t>& shared_vertices_a() noexcept {
    return shared_vertices_a_;
  }

  thrust::device_vector<vertex_t>& shared_vertices_b() noexcept {
    return shared_vertices_b_;
  }

  thrust::device_vector<query_mask_t>& shared_masks_a() noexcept {
    return shared_masks_a_;
  }

  thrust::device_vector<query_mask_t>& shared_masks_b() noexcept {
    return shared_masks_b_;
  }

  thrust::device_vector<unsigned char>& bitmap_a() noexcept { return bitmap_a_; }

  thrust::device_vector<unsigned char>& bitmap_b() noexcept { return bitmap_b_; }

  thrust::device_vector<value_t>& dense_a() noexcept { return dense_a_; }

  thrust::device_vector<value_t>& dense_b() noexcept { return dense_b_; }

  thrust::device_vector<std::size_t>& counts() noexcept { return counts_; }

 private:
  std::size_t query_count_ = 0;
  std::size_t vertex_count_ = 0;
  std::size_t edge_count_ = 0;
  thrust::device_vector<item_type> list_frontier_a_;
  thrust::device_vector<item_type> list_frontier_b_;
  thrust::device_vector<vertex_t> shared_vertices_a_;
  thrust::device_vector<vertex_t> shared_vertices_b_;
  thrust::device_vector<query_mask_t> shared_masks_a_;
  thrust::device_vector<query_mask_t> shared_masks_b_;
  thrust::device_vector<unsigned char> bitmap_a_;
  thrust::device_vector<unsigned char> bitmap_b_;
  thrust::device_vector<value_t> dense_a_;
  thrust::device_vector<value_t> dense_b_;
  thrust::device_vector<std::size_t> counts_;
};

}  // namespace puercgp
