#include "graphweft/graph.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int queries = 32;
constexpr int block_warps = 8;

void checked(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template<class T> class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t count) : count_(count) {
    if (count) checked(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
    other.data_ = nullptr;
    other.count_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (data_) cudaFree(data_);
      data_ = other.data_;
      count_ = other.count_;
      other.data_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }
  ~DeviceBuffer() { if (data_) cudaFree(data_); }
  T* get() const { return data_; }
  size_t size() const { return count_; }
  void upload(const std::vector<T>& values) {
    if (values.size() != count_) throw std::runtime_error("upload size mismatch");
    if (count_) checked(cudaMemcpy(data_, values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
  }
 private:
  T* data_ = nullptr;
  size_t count_ = 0;
};

// Each warp lane owns one of the 32 concurrent queries. Warps assigned to the
// same destination traverse disjoint incoming-edge streams, then reduce their
// per-query minima in shared memory. The vertex lists are static degree buckets
// built once, outside all timed regions.
template<int WarpsPerVertex>
__global__ void pull_bucket(const uint64_t* row, const uint32_t* col,
                            const float* weight, const float* old_values,
                            float* new_values, const uint32_t* vertices,
                            uint32_t vertex_count) {
  static_assert(block_warps % WarpsPerVertex == 0);
  constexpr int vertices_per_block = block_warps / WarpsPerVertex;
  __shared__ float partial[block_warps][queries];
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int local_vertex = warp / WarpsPerVertex;
  const int part = warp % WarpsPerVertex;
  const uint64_t item = uint64_t(blockIdx.x) * vertices_per_block + local_vertex;
  const bool valid = item < vertex_count;
  const uint32_t vertex = valid ? vertices[item] : 0;
  float best = valid ? old_values[size_t(vertex) * queries + lane] : INFINITY;
  if (valid) {
    const uint64_t begin = row[vertex], end = row[vertex + 1];
    for (uint64_t edge = begin + part; edge < end; edge += WarpsPerVertex) {
      const uint32_t source = col[edge];
      best = fminf(best, old_values[size_t(source) * queries + lane] + weight[edge]);
    }
  }
  partial[warp][lane] = best;
  __syncthreads();
  if (valid && part == 0) {
#pragma unroll
    for (int p = 1; p < WarpsPerVertex; ++p)
      best = fminf(best, partial[warp + p][lane]);
    new_values[size_t(vertex) * queries + lane] = best;
  }
}

template<int WarpsPerVertex>
void launch_bucket(const DeviceBuffer<uint64_t>& row,
                   const DeviceBuffer<uint32_t>& col,
                   const DeviceBuffer<float>& weight,
                   const DeviceBuffer<float>& old_values,
                   DeviceBuffer<float>& new_values,
                   const DeviceBuffer<uint32_t>& vertices,
                   cudaStream_t stream = nullptr) {
  if (!vertices.size()) return;
  constexpr int vertices_per_block = block_warps / WarpsPerVertex;
  const uint32_t blocks = uint32_t((vertices.size() + vertices_per_block - 1) /
                                   vertices_per_block);
  pull_bucket<WarpsPerVertex><<<blocks, block_warps * 32, 0, stream>>>(
      row.get(), col.get(), weight.get(), old_values.get(), new_values.get(),
      vertices.get(), uint32_t(vertices.size()));
  checked(cudaGetLastError());
}

struct Buckets {
  std::array<std::vector<uint32_t>, 4> vertices;
  std::array<uint64_t, 4> edges{};
};

struct Chunks {
  std::vector<uint64_t> vertex_offsets;
  std::vector<uint32_t> task_vertices;
  std::vector<uint64_t> task_begins;
};

Chunks make_chunks(const graphweft::HostGraph& graph, uint64_t edges_per_warp) {
  Chunks result;
  result.vertex_offsets.resize(size_t(graph.vertices) + 1);
  uint64_t tasks = 0;
  for (uint32_t vertex = 0; vertex < graph.vertices; ++vertex) {
    result.vertex_offsets[vertex] = tasks;
    const uint64_t begin = graph.incoming_row[vertex];
    const uint64_t end = graph.incoming_row[vertex + 1];
    tasks += (end - begin + edges_per_warp - 1) / edges_per_warp;
  }
  result.vertex_offsets[graph.vertices] = tasks;
  result.task_vertices.reserve(tasks);
  result.task_begins.reserve(tasks);
  for (uint32_t vertex = 0; vertex < graph.vertices; ++vertex) {
    const uint64_t begin = graph.incoming_row[vertex];
    const uint64_t end = graph.incoming_row[vertex + 1];
    for (uint64_t edge = begin; edge < end; edge += edges_per_warp) {
      result.task_vertices.push_back(vertex);
      result.task_begins.push_back(edge);
    }
  }
  return result;
}

// Strict edge tiling: every warp owns exactly one C-edge chunk (except the
// final tail chunk of a vertex). There is no rounding to a supported warp
// count and no per-vertex warp cap.
__global__ void pull_chunks(const uint64_t* row, const uint32_t* col,
                            const float* weight, const float* old_values,
                            const uint32_t* task_vertices,
                            const uint64_t* task_begins, uint64_t task_count,
                            uint64_t edges_per_warp, float* partial) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const uint64_t task = uint64_t(blockIdx.x) * block_warps + warp;
  if (task >= task_count) return;
  const uint32_t vertex = task_vertices[task];
  const uint64_t begin = task_begins[task];
  const uint64_t end = min(begin + edges_per_warp, row[vertex + 1]);
  float best = INFINITY;
  for (uint64_t edge = begin; edge < end; ++edge) {
    const uint32_t source = col[edge];
    best = fminf(best, old_values[size_t(source) * queries + lane] + weight[edge]);
  }
  partial[task * queries + lane] = best;
}

__global__ void reduce_chunks(const float* old_values, float* new_values,
                              const uint64_t* vertex_offsets,
                              const float* partial, uint32_t vertex_count) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const uint64_t vertex = uint64_t(blockIdx.x) * block_warps + warp;
  if (vertex >= vertex_count) return;
  float best = old_values[vertex * queries + lane];
  for (uint64_t task = vertex_offsets[vertex]; task < vertex_offsets[vertex + 1]; ++task)
    best = fminf(best, partial[task * queries + lane]);
  new_values[vertex * queries + lane] = best;
}

void launch_chunks(const DeviceBuffer<uint64_t>& row,
                   const DeviceBuffer<uint32_t>& col,
                   const DeviceBuffer<float>& weight,
                   const DeviceBuffer<float>& old_values,
                   DeviceBuffer<float>& new_values,
                   const DeviceBuffer<uint64_t>& vertex_offsets,
                   const DeviceBuffer<uint32_t>& task_vertices,
                   const DeviceBuffer<uint64_t>& task_begins,
                   uint64_t edges_per_warp,
                   DeviceBuffer<float>& partial,
                   uint32_t vertex_count,
                   cudaStream_t stream = nullptr) {
  if (task_vertices.size()) {
    const uint32_t blocks = uint32_t((task_vertices.size() + block_warps - 1) / block_warps);
    pull_chunks<<<blocks, block_warps * 32, 0, stream>>>(
        row.get(), col.get(), weight.get(), old_values.get(), task_vertices.get(),
        task_begins.get(), task_vertices.size(), edges_per_warp, partial.get());
    checked(cudaGetLastError());
  }
  const uint32_t reduce_blocks = (vertex_count + block_warps - 1) / block_warps;
  reduce_chunks<<<reduce_blocks, block_warps * 32, 0, stream>>>(
      old_values.get(), new_values.get(), vertex_offsets.get(), partial.get(), vertex_count);
  checked(cudaGetLastError());
}

Buckets make_buckets(const graphweft::HostGraph& graph, uint64_t edges_per_warp) {
  Buckets result;
  for (uint32_t vertex = 0; vertex < graph.vertices; ++vertex) {
    const uint64_t degree = graph.incoming_row[vertex + 1] - graph.incoming_row[vertex];
    // Round ceil(degree/C) up to the supported 1/2/4/8 warp shapes. Eight is
    // a cap: its warps stride over any remaining chunks of a hub vertex.
    const uint64_t wanted = degree ? (degree + edges_per_warp - 1) / edges_per_warp : 1;
    const int bucket = wanted <= 1 ? 0 : wanted <= 2 ? 1 : wanted <= 4 ? 2 : 3;
    result.vertices[bucket].push_back(vertex);
    result.edges[bucket] += degree;
  }
  return result;
}

std::vector<float> initial_values(uint32_t vertices) {
  std::vector<float> values(size_t(vertices) * queries);
  for (uint32_t vertex = 0; vertex < vertices; ++vertex)
    for (uint32_t query = 0; query < queries; ++query)
      values[size_t(vertex) * queries + query] =
          float((uint64_t(vertex) * 2654435761ULL + query * 97ULL) % 100003ULL);
  return values;
}

double quantile(std::vector<float> values, double q) {
  std::sort(values.begin(), values.end());
  const double position = q * double(values.size() - 1);
  const size_t lo = size_t(position), hi = std::min(lo + 1, values.size() - 1);
  return values[lo] + (position - lo) * (values[hi] - values[lo]);
}

template<class First, class Second>
std::array<double, 2> measure_pair(First&& first, Second&& second,
                                   int warmups, int repetitions) {
  cudaEvent_t start = nullptr, stop = nullptr;
  checked(cudaEventCreate(&start));
  checked(cudaEventCreate(&stop));
  for (int i = 0; i < warmups; ++i) { first(); second(); }
  checked(cudaDeviceSynchronize());
  std::array<std::vector<float>, 2> samples;
  samples[0].reserve(repetitions);
  samples[1].reserve(repetitions);
  auto one = [&](int index, auto&& launch) {
    checked(cudaEventRecord(start));
    launch();
    checked(cudaEventRecord(stop));
    checked(cudaEventSynchronize(stop));
    float milliseconds = 0;
    checked(cudaEventElapsedTime(&milliseconds, start, stop));
    samples[index].push_back(milliseconds);
  };
  for (int i = 0; i < repetitions; ++i) {
    if (i & 1) { one(1, second); one(0, first); }
    else { one(0, first); one(1, second); }
  }
  checked(cudaEventDestroy(start));
  checked(cudaEventDestroy(stop));
  return {quantile(std::move(samples[0]), 0.5),
          quantile(std::move(samples[1]), 0.5)};
}

void compare(const DeviceBuffer<float>& lhs, const DeviceBuffer<float>& rhs) {
  if (lhs.size() != rhs.size()) throw std::runtime_error("comparison size mismatch");
  std::vector<float> a(lhs.size()), b(rhs.size());
  checked(cudaMemcpy(a.data(), lhs.get(), a.size() * sizeof(float), cudaMemcpyDeviceToHost));
  checked(cudaMemcpy(b.data(), rhs.get(), b.size() * sizeof(float), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i])
      throw std::runtime_error("result mismatch at cell " + std::to_string(i));
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 4)
      throw std::invalid_argument("usage: graphweft_degree_pull GRAPH [GPU=0] [REPETITIONS=21]");
    const int gpu = argc >= 3 ? std::stoi(argv[2]) : 0;
    const int repetitions = argc >= 4 ? std::stoi(argv[3]) : 21;
    if (repetitions < 3) throw std::invalid_argument("REPETITIONS must be at least 3");
    checked(cudaSetDevice(gpu));
    const auto graph = graphweft::HostGraph::load(argv[1], true, true);
    if (graph.incoming_row.empty()) throw std::runtime_error("incoming CSR unavailable");
    std::cout << "graph=" << graph.identity << " V=" << graph.vertices
              << " E=" << graph.edges() << " Q=" << queries << " gpu=" << gpu << '\n';

    DeviceBuffer<uint64_t> row(graph.incoming_row.size()); row.upload(graph.incoming_row);
    DeviceBuffer<uint32_t> col(graph.incoming_col.size()); col.upload(graph.incoming_col);
    DeviceBuffer<float> weight(graph.incoming_weight.size()); weight.upload(graph.incoming_weight);
    auto host_old = initial_values(graph.vertices);
    DeviceBuffer<float> old_values(host_old.size()); old_values.upload(host_old);
    DeviceBuffer<float> baseline(host_old.size()), adaptive(host_old.size());
    std::vector<uint32_t> all(graph.vertices); std::iota(all.begin(), all.end(), 0);
    DeviceBuffer<uint32_t> all_vertices(all.size()); all_vertices.upload(all);

    auto baseline_launch = [&] {
      launch_bucket<1>(row, col, weight, old_values, baseline, all_vertices);
    };
    baseline_launch(); checked(cudaDeviceSynchronize());
    std::cout << std::fixed << std::setprecision(6);

    constexpr std::array<uint64_t, 7> capacities{32, 64, 128, 256, 512, 1024, 2048};
    for (const uint64_t capacity : capacities) {
      auto chunks = make_chunks(graph, capacity);
      DeviceBuffer<uint64_t> vertex_offsets(chunks.vertex_offsets.size());
      DeviceBuffer<uint32_t> task_vertices(chunks.task_vertices.size());
      DeviceBuffer<uint64_t> task_begins(chunks.task_begins.size());
      DeviceBuffer<float> partial(chunks.task_vertices.size() * queries);
      vertex_offsets.upload(chunks.vertex_offsets);
      task_vertices.upload(chunks.task_vertices);
      task_begins.upload(chunks.task_begins);
      auto adaptive_launch = [&] {
        launch_chunks(row, col, weight, old_values, adaptive, vertex_offsets,
                      task_vertices, task_begins, capacity, partial, graph.vertices);
      };
      const auto timing = measure_pair(baseline_launch, adaptive_launch, 5, repetitions);
      const double baseline_ms = timing[0], adaptive_ms = timing[1];
      adaptive_launch(); checked(cudaDeviceSynchronize());
      compare(baseline, adaptive);
      std::cout << "strict,C=" << capacity << ",baseline_ms=" << baseline_ms
                << ",strict_ms=" << adaptive_ms << ",speedup="
                << baseline_ms / adaptive_ms << ",warp_tasks="
                << chunks.task_vertices.size() << ",max_edges_per_warp="
                << capacity << ",exact=1\n";
    }
  } catch (const std::exception& error) {
    std::cerr << "degree-adaptive pull: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
