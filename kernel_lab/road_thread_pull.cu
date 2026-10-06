#include "graphweft/graph.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int queries = 32;
constexpr int threads = 256;

void checked(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template<class T> class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t count) : count_(count) {
    if (count_) checked(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)));
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
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

// Warp-Vertex baseline: one warp owns one destination and each lane owns one
// of the 32 queries. This isolates the mapping used by q32-w1 without engine
// or frontier overhead.
__global__ void warp_vertex_pull(const uint64_t* row, const uint32_t* col,
                                 const float* weight, const float* old_values,
                                 float* new_values, uint32_t vertex_count) {
  const uint64_t vertex = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if (vertex >= vertex_count) return;
  float best = old_values[vertex * queries + lane];
  for (uint64_t edge = row[vertex]; edge < row[vertex + 1]; ++edge) {
    const uint32_t source = col[edge];
    best = fminf(best, old_values[size_t(source) * queries + lane] + weight[edge]);
  }
  new_values[vertex * queries + lane] = best;
}

// True Thread-Vertex mapping. A thread owns K consecutive destinations and
// processes all 32 queries for each. QueryTile controls live accumulators;
// the tiny road adjacency list is reread for every query tile.
template<int VerticesPerThread, int QueryTile>
__global__ void thread_vertex_pull(const uint64_t* row, const uint32_t* col,
                                   const float* weight, const float* old_values,
                                   float* new_values, uint32_t vertex_count) {
  static_assert(queries % QueryTile == 0);
  const uint64_t first =
      (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) * VerticesPerThread;
#pragma unroll
  for (int item = 0; item < VerticesPerThread; ++item) {
    const uint64_t vertex = first + item;
    if (vertex >= vertex_count) continue;
    const uint64_t begin = row[vertex], end = row[vertex + 1];
    for (int query_base = 0; query_base < queries; query_base += QueryTile) {
      float best[QueryTile];
#pragma unroll
      for (int q = 0; q < QueryTile; ++q)
        best[q] = old_values[vertex * queries + query_base + q];
      for (uint64_t edge = begin; edge < end; ++edge) {
        const uint32_t source = col[edge];
        const float edge_weight = weight[edge];
#pragma unroll
        for (int q = 0; q < QueryTile; ++q)
          best[q] = fminf(best[q],
              old_values[size_t(source) * queries + query_base + q] + edge_weight);
      }
#pragma unroll
      for (int q = 0; q < QueryTile; ++q)
        new_values[vertex * queries + query_base + q] = best[q];
    }
  }
}

void launch_warp(const DeviceBuffer<uint64_t>& row,
                 const DeviceBuffer<uint32_t>& col,
                 const DeviceBuffer<float>& weight,
                 const DeviceBuffer<float>& old_values,
                 DeviceBuffer<float>& new_values, uint32_t vertices) {
  const uint64_t total_threads = uint64_t(vertices) * 32;
  warp_vertex_pull<<<uint32_t((total_threads + threads - 1) / threads), threads>>>(
      row.get(), col.get(), weight.get(), old_values.get(), new_values.get(), vertices);
  checked(cudaGetLastError());
}

template<int K, int QueryTile>
void launch_thread(const DeviceBuffer<uint64_t>& row,
                   const DeviceBuffer<uint32_t>& col,
                   const DeviceBuffer<float>& weight,
                   const DeviceBuffer<float>& old_values,
                   DeviceBuffer<float>& new_values, uint32_t vertices) {
  const uint64_t workers = (uint64_t(vertices) + K - 1) / K;
  thread_vertex_pull<K, QueryTile><<<uint32_t((workers + threads - 1) / threads), threads>>>(
      row.get(), col.get(), weight.get(), old_values.get(), new_values.get(), vertices);
  checked(cudaGetLastError());
}

std::vector<float> make_values(uint32_t vertices) {
  std::vector<float> values(size_t(vertices) * queries);
  for (uint32_t vertex = 0; vertex < vertices; ++vertex)
    for (uint32_t query = 0; query < queries; ++query)
      values[size_t(vertex) * queries + query] =
          float((uint64_t(vertex) * 2654435761ULL + query * 97ULL) % 100003ULL);
  return values;
}

double median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

template<class Baseline, class Candidate>
std::array<double, 2> measure_pair(Baseline&& baseline, Candidate&& candidate,
                                   int repetitions) {
  cudaEvent_t start = nullptr, stop = nullptr;
  checked(cudaEventCreate(&start)); checked(cudaEventCreate(&stop));
  for (int i = 0; i < 5; ++i) { baseline(); candidate(); }
  checked(cudaDeviceSynchronize());
  std::array<std::vector<float>, 2> samples;
  auto one = [&](int index, auto&& launch) {
    checked(cudaEventRecord(start)); launch(); checked(cudaEventRecord(stop));
    checked(cudaEventSynchronize(stop)); float ms = 0;
    checked(cudaEventElapsedTime(&ms, start, stop)); samples[index].push_back(ms);
  };
  for (int i = 0; i < repetitions; ++i) {
    if (i & 1) { one(1, candidate); one(0, baseline); }
    else { one(0, baseline); one(1, candidate); }
  }
  checked(cudaEventDestroy(start)); checked(cudaEventDestroy(stop));
  return {median(std::move(samples[0])), median(std::move(samples[1]))};
}

void verify(const DeviceBuffer<float>& reference, const DeviceBuffer<float>& candidate) {
  std::vector<float> a(reference.size()), b(candidate.size());
  checked(cudaMemcpy(a.data(), reference.get(), a.size() * sizeof(float), cudaMemcpyDeviceToHost));
  checked(cudaMemcpy(b.data(), candidate.get(), b.size() * sizeof(float), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i] != b[i]) throw std::runtime_error("result mismatch at cell " + std::to_string(i));
}

template<int K, int QueryTile>
void run_candidate(const std::string& name,
                   const DeviceBuffer<uint64_t>& row,
                   const DeviceBuffer<uint32_t>& col,
                   const DeviceBuffer<float>& weight,
                   const DeviceBuffer<float>& old_values,
                   DeviceBuffer<float>& baseline_output,
                   DeviceBuffer<float>& candidate_output,
                   uint32_t vertices, int repetitions) {
  auto baseline = [&] { launch_warp(row, col, weight, old_values, baseline_output, vertices); };
  auto candidate = [&] {
    launch_thread<K, QueryTile>(row, col, weight, old_values, candidate_output, vertices);
  };
  const auto times = measure_pair(baseline, candidate, repetitions);
  baseline(); candidate(); checked(cudaDeviceSynchronize());
  verify(baseline_output, candidate_output);
  std::cout << name << ",baseline_ms=" << times[0] << ",candidate_ms=" << times[1]
            << ",speedup=" << times[0] / times[1] << ",exact=1\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 4)
      throw std::invalid_argument("usage: graphweft_road_thread_pull GRAPH [GPU=0] [REPETITIONS=21]");
    const int gpu = argc >= 3 ? std::stoi(argv[2]) : 0;
    const int repetitions = argc >= 4 ? std::stoi(argv[3]) : 21;
    checked(cudaSetDevice(gpu));
    const auto graph = graphweft::HostGraph::load(argv[1], true, true);
    std::cout << std::fixed << std::setprecision(6)
              << "graph=" << graph.identity << " V=" << graph.vertices
              << " E=" << graph.edges() << " Q=" << queries << " gpu=" << gpu << '\n';
    DeviceBuffer<uint64_t> row(graph.incoming_row.size()); row.upload(graph.incoming_row);
    DeviceBuffer<uint32_t> col(graph.incoming_col.size()); col.upload(graph.incoming_col);
    DeviceBuffer<float> weight(graph.incoming_weight.size()); weight.upload(graph.incoming_weight);
    auto host_values = make_values(graph.vertices);
    DeviceBuffer<float> old_values(host_values.size()); old_values.upload(host_values);
    DeviceBuffer<float> baseline_output(host_values.size()), candidate_output(host_values.size());

    run_candidate<1,4>("TV-G1-QT4",row,col,weight,old_values,baseline_output,candidate_output,graph.vertices,repetitions);
    run_candidate<1,8>("TV-G1-QT8",row,col,weight,old_values,baseline_output,candidate_output,graph.vertices,repetitions);
    run_candidate<1,16>("TV-G1-QT16",row,col,weight,old_values,baseline_output,candidate_output,graph.vertices,repetitions);
    run_candidate<1,32>("TV-G1-QT32",row,col,weight,old_values,baseline_output,candidate_output,graph.vertices,repetitions);
    run_candidate<2,4>("TV-G2-QT4",row,col,weight,old_values,baseline_output,candidate_output,graph.vertices,repetitions);
    run_candidate<2,8>("TV-G2-QT8",row,col,weight,old_values,baseline_output,candidate_output,graph.vertices,repetitions);
    run_candidate<2,16>("TV-G2-QT16",row,col,weight,old_values,baseline_output,candidate_output,graph.vertices,repetitions);
    run_candidate<2,32>("TV-G2-QT32",row,col,weight,old_values,baseline_output,candidate_output,graph.vertices,repetitions);
    run_candidate<4,4>("TV-G4-QT4",row,col,weight,old_values,baseline_output,candidate_output,graph.vertices,repetitions);
    run_candidate<4,8>("TV-G4-QT8",row,col,weight,old_values,baseline_output,candidate_output,graph.vertices,repetitions);
    run_candidate<8,4>("TV-G8-QT4",row,col,weight,old_values,baseline_output,candidate_output,graph.vertices,repetitions);
    run_candidate<8,8>("TV-G8-QT8",row,col,weight,old_values,baseline_output,candidate_output,graph.vertices,repetitions);
  } catch (const std::exception& error) {
    std::cerr << "road thread pull: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
