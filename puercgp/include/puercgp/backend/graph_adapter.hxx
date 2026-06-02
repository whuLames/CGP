#pragma once

#include <stdexcept>

#include <cuda_runtime.h>

namespace puercgp {

template <typename graph_t>
class graph_adapter {
 public:
  using graph_type = graph_t;

  explicit graph_adapter(const graph_t& graph) : graph_(&graph) {}

  auto get_number_of_vertices() const {
    return graph_->get_number_of_vertices();
  }

  auto get_number_of_edges() const { return graph_->get_number_of_edges(); }

  template <typename vertex_t>
  auto get_starting_edge(vertex_t vertex) const {
    return graph_->get_starting_edge(vertex);
  }

  template <typename edge_t>
  auto get_destination_vertex(edge_t edge) const {
    return graph_->get_destination_vertex(edge);
  }

  template <typename edge_t>
  auto get_edge_weight(edge_t edge) const {
    return graph_->get_edge_weight(edge);
  }

  const graph_t& get() const noexcept { return *graph_; }

 private:
  const graph_t* graph_;
};

class execution_context {
 public:
  execution_context() { create_stream(); }

  explicit execution_context(cudaStream_t stream) : stream_(stream), owns_(false) {}

  execution_context(const execution_context&) = delete;
  execution_context& operator=(const execution_context&) = delete;

  execution_context(execution_context&& other) noexcept
      : stream_(other.stream_), owns_(other.owns_) {
    other.stream_ = nullptr;
    other.owns_ = false;
  }

  execution_context& operator=(execution_context&& other) noexcept {
    if (this != &other) {
      destroy_stream();
      stream_ = other.stream_;
      owns_ = other.owns_;
      other.stream_ = nullptr;
      other.owns_ = false;
    }
    return *this;
  }

  ~execution_context() { destroy_stream(); }

  cudaStream_t stream() const noexcept { return stream_; }

  void synchronize() const {
    auto status = cudaStreamSynchronize(stream_);
    if (status != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(status));
    }
  }

 private:
  void create_stream() {
    auto status = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (status != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(status));
    }
    owns_ = true;
  }

  void destroy_stream() {
    if (owns_ && stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  cudaStream_t stream_ = nullptr;
  bool owns_ = false;
};

}  // namespace puercgp
