#pragma once

#include <cstddef>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

namespace puercgp {

template <typename vertex_t, typename edge_t, typename weight_t = float>
struct csr_graph_view {
  using vertex_type = vertex_t;
  using edge_type = edge_t;
  using weight_type = weight_t;

  vertex_t number_of_vertices = 0;
  edge_t number_of_edges = 0;
  const edge_t* row_offsets = nullptr;
  const vertex_t* column_indices = nullptr;
  const weight_t* edge_weights = nullptr;
  const edge_t* pull_row_offsets = nullptr;
  const vertex_t* pull_column_indices = nullptr;
  const weight_t* pull_edge_weights = nullptr;

  __host__ __device__ vertex_t get_number_of_vertices() const {
    return number_of_vertices;
  }

  __host__ __device__ edge_t get_number_of_edges() const {
    return number_of_edges;
  }

  __host__ __device__ edge_t get_starting_edge(vertex_t vertex) const {
    return row_offsets[vertex];
  }

  __host__ __device__ vertex_t get_destination_vertex(edge_t edge) const {
    return column_indices[edge];
  }

  __host__ __device__ weight_t get_edge_weight(edge_t edge) const {
    return edge_weights == nullptr ? weight_t{1} : edge_weights[edge];
  }

  __host__ __device__ bool has_pull_adjacency() const {
    return pull_row_offsets != nullptr && pull_column_indices != nullptr;
  }

  __host__ __device__ edge_t get_starting_pull_edge(vertex_t vertex) const {
    return has_pull_adjacency() ? pull_row_offsets[vertex]
                                : get_starting_edge(vertex);
  }

  __host__ __device__ vertex_t get_pull_neighbor_vertex(edge_t edge) const {
    return has_pull_adjacency() ? pull_column_indices[edge]
                                : get_destination_vertex(edge);
  }

  __host__ __device__ weight_t get_pull_edge_weight(edge_t edge) const {
    if (!has_pull_adjacency()) {
      return get_edge_weight(edge);
    }
    return pull_edge_weights == nullptr ? weight_t{1} : pull_edge_weights[edge];
  }
};

template <typename vertex_t, typename edge_t, typename weight_t = float>
class csr_graph_storage {
 public:
  using view_type = csr_graph_view<vertex_t, edge_t, weight_t>;

  csr_graph_storage() = default;

  csr_graph_storage(vertex_t vertices,
                    thrust::host_vector<edge_t> row_offsets,
                    thrust::host_vector<vertex_t> column_indices,
                    thrust::host_vector<weight_t> edge_weights = {},
                    bool build_pull_adjacency = false)
      : vertices_(vertices), edges_(static_cast<edge_t>(column_indices.size())) {
    row_offsets_ = row_offsets;
    column_indices_ = column_indices;
    edge_weights_ = edge_weights;
    if (build_pull_adjacency) {
      build_pull_storage(row_offsets, column_indices, edge_weights);
    }
  }

  view_type view() const {
    return {vertices_,
            edges_,
            thrust::raw_pointer_cast(row_offsets_.data()),
            thrust::raw_pointer_cast(column_indices_.data()),
            edge_weights_.empty()
                ? nullptr
                : thrust::raw_pointer_cast(edge_weights_.data()),
            pull_row_offsets_.empty()
                ? nullptr
                : thrust::raw_pointer_cast(pull_row_offsets_.data()),
            pull_column_indices_.empty()
                ? nullptr
                : thrust::raw_pointer_cast(pull_column_indices_.data()),
            pull_edge_weights_.empty()
                ? nullptr
                : thrust::raw_pointer_cast(pull_edge_weights_.data())};
  }

 private:
  void build_pull_storage(
      const thrust::host_vector<edge_t>& row_offsets,
      const thrust::host_vector<vertex_t>& column_indices,
      const thrust::host_vector<weight_t>& edge_weights) {
    thrust::host_vector<edge_t> pull_row_offsets(
        static_cast<std::size_t>(vertices_) + 1, edge_t{0});
    for (edge_t edge = 0; edge < edges_; ++edge) {
      vertex_t dst = column_indices[static_cast<std::size_t>(edge)];
      ++pull_row_offsets[static_cast<std::size_t>(dst) + 1];
    }
    for (vertex_t vertex = 0; vertex < vertices_; ++vertex) {
      pull_row_offsets[static_cast<std::size_t>(vertex) + 1] +=
          pull_row_offsets[static_cast<std::size_t>(vertex)];
    }

    thrust::host_vector<vertex_t> pull_column_indices(
        static_cast<std::size_t>(edges_));
    thrust::host_vector<weight_t> pull_edge_weights;
    if (!edge_weights.empty()) {
      pull_edge_weights.resize(static_cast<std::size_t>(edges_));
    }

    thrust::host_vector<edge_t> cursor = pull_row_offsets;
    for (vertex_t src = 0; src < vertices_; ++src) {
      for (edge_t edge = row_offsets[static_cast<std::size_t>(src)];
           edge < row_offsets[static_cast<std::size_t>(src) + 1]; ++edge) {
        vertex_t dst = column_indices[static_cast<std::size_t>(edge)];
        edge_t position = cursor[static_cast<std::size_t>(dst)]++;
        pull_column_indices[static_cast<std::size_t>(position)] = src;
        if (!edge_weights.empty()) {
          pull_edge_weights[static_cast<std::size_t>(position)] =
              edge_weights[static_cast<std::size_t>(edge)];
        }
      }
    }

    pull_row_offsets_ = pull_row_offsets;
    pull_column_indices_ = pull_column_indices;
    pull_edge_weights_ = pull_edge_weights;
  }

  vertex_t vertices_ = 0;
  edge_t edges_ = 0;
  thrust::device_vector<edge_t> row_offsets_;
  thrust::device_vector<vertex_t> column_indices_;
  thrust::device_vector<weight_t> edge_weights_;
  thrust::device_vector<edge_t> pull_row_offsets_;
  thrust::device_vector<vertex_t> pull_column_indices_;
  thrust::device_vector<weight_t> pull_edge_weights_;
};

}  // namespace puercgp
