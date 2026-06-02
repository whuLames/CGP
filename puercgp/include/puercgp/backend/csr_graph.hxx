#pragma once

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
};

template <typename vertex_t, typename edge_t, typename weight_t = float>
class csr_graph_storage {
 public:
  using view_type = csr_graph_view<vertex_t, edge_t, weight_t>;

  csr_graph_storage() = default;

  csr_graph_storage(vertex_t vertices,
                    thrust::host_vector<edge_t> row_offsets,
                    thrust::host_vector<vertex_t> column_indices,
                    thrust::host_vector<weight_t> edge_weights = {})
      : vertices_(vertices),
        edges_(static_cast<edge_t>(column_indices.size())),
        row_offsets_(row_offsets),
        column_indices_(column_indices),
        edge_weights_(edge_weights) {}

  view_type view() const {
    return {vertices_,
            edges_,
            thrust::raw_pointer_cast(row_offsets_.data()),
            thrust::raw_pointer_cast(column_indices_.data()),
            edge_weights_.empty()
                ? nullptr
                : thrust::raw_pointer_cast(edge_weights_.data())};
  }

 private:
  vertex_t vertices_ = 0;
  edge_t edges_ = 0;
  thrust::device_vector<edge_t> row_offsets_;
  thrust::device_vector<vertex_t> column_indices_;
  thrust::device_vector<weight_t> edge_weights_;
};

}  // namespace puercgp
