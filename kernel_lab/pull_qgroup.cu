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
constexpr int warps_per_block = threads / 32;

void checked(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template<class T> class DeviceBuffer {
 public:
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

// q32 baseline: one warp per destination, one query per lane, one edge stream.
__global__ void pull_q32(const uint64_t* row, const uint32_t* col,
                         const float* weight, const float* old_values,
                         float* new_values, uint32_t vertex_count) {
  const uint64_t vertex = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const int query = threadIdx.x % 32;
  if (vertex >= vertex_count) return;
  float best = old_values[vertex * queries + query];
  for (uint64_t edge = row[vertex]; edge < row[vertex + 1]; ++edge) {
    const uint32_t source = col[edge];
    best = fminf(best, old_values[size_t(source) * queries + query] + weight[edge]);
  }
  new_values[vertex * queries + query] = best;
}

// One warp still owns exactly one destination. QueryLanes controls how the
// warp is split: 32/QueryLanes groups traverse disjoint incoming-edge streams,
// while each lane serially handles 32/QueryLanes queries. Group partials are
// reduced in shared memory before writing the destination's 32 values.
template<int QueryLanes>
__global__ void pull_qgroup(const uint64_t* row, const uint32_t* col,
                            const float* weight, const float* old_values,
                            float* new_values, uint32_t vertex_count) {
  static_assert(queries % QueryLanes == 0);
  constexpr int edge_groups = 32 / QueryLanes;
  constexpr int queries_per_lane = 32 / QueryLanes;
  __shared__ float partial[warps_per_block * 32 * queries_per_lane];
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int edge_group = lane / QueryLanes;
  const int query_lane = lane % QueryLanes;
  const uint64_t vertex = uint64_t(blockIdx.x) * warps_per_block + warp;
  const bool valid = vertex < vertex_count;
  float best[queries_per_lane];
#pragma unroll
  for (int k = 0; k < queries_per_lane; ++k) {
    const int query = k * QueryLanes + query_lane;
    best[k] = valid ? old_values[vertex * queries + query] : INFINITY;
  }
  if (valid) {
    const uint64_t begin = row[vertex], end = row[vertex + 1];
    for (uint64_t edge = begin + edge_group; edge < end; edge += edge_groups) {
      const uint32_t source = col[edge];
      const float edge_weight = weight[edge];
#pragma unroll
      for (int k = 0; k < queries_per_lane; ++k) {
        const int query = k * QueryLanes + query_lane;
        best[k] = fminf(best[k],
            old_values[size_t(source) * queries + query] + edge_weight);
      }
    }
  }
#pragma unroll
  for (int k = 0; k < queries_per_lane; ++k)
    partial[(warp * queries_per_lane + k) * 32 + lane] = best[k];
  __syncthreads();
  if (valid && edge_group == 0) {
#pragma unroll
    for (int k = 0; k < queries_per_lane; ++k) {
      float reduced = partial[(warp * queries_per_lane + k) * 32 + query_lane];
#pragma unroll
      for (int group = 1; group < edge_groups; ++group)
        reduced = fminf(reduced,
            partial[(warp * queries_per_lane + k) * 32 + group * QueryLanes + query_lane]);
      const int query = k * QueryLanes + query_lane;
      new_values[vertex * queries + query] = reduced;
    }
  }
}

void launch_q32(const DeviceBuffer<uint64_t>& row,
                const DeviceBuffer<uint32_t>& col,
                const DeviceBuffer<float>& weight,
                const DeviceBuffer<float>& old_values,
                DeviceBuffer<float>& new_values, uint32_t vertices) {
  const uint64_t total_threads = uint64_t(vertices) * 32;
  pull_q32<<<uint32_t((total_threads + threads - 1) / threads), threads>>>(
      row.get(), col.get(), weight.get(), old_values.get(), new_values.get(), vertices);
  checked(cudaGetLastError());
}

template<int QueryLanes>
void launch_qgroup(const DeviceBuffer<uint64_t>& row,
                   const DeviceBuffer<uint32_t>& col,
                   const DeviceBuffer<float>& weight,
                   const DeviceBuffer<float>& old_values,
                   DeviceBuffer<float>& new_values, uint32_t vertices) {
  pull_qgroup<QueryLanes><<<(vertices + warps_per_block - 1) / warps_per_block, threads>>>(
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

template<int Q>
void run_candidate(const DeviceBuffer<uint64_t>& row,
                   const DeviceBuffer<uint32_t>& col,
                   const DeviceBuffer<float>& weight,
                   const DeviceBuffer<float>& old_values,
                   DeviceBuffer<float>& baseline_output,
                   DeviceBuffer<float>& candidate_output,
                   uint32_t vertices, int repetitions) {
  auto baseline = [&] { launch_q32(row,col,weight,old_values,baseline_output,vertices); };
  auto candidate = [&] { launch_qgroup<Q>(row,col,weight,old_values,candidate_output,vertices); };
  const auto times = measure_pair(baseline,candidate,repetitions);
  baseline(); candidate(); checked(cudaDeviceSynchronize());
  verify(baseline_output,candidate_output);
  std::cout << "q" << Q << "_w1,baseline_ms=" << times[0]
            << ",candidate_ms=" << times[1] << ",speedup=" << times[0]/times[1]
            << ",edge_groups=" << 32/Q << ",queries_per_lane=" << 32/Q
            << ",exact=1\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 4)
      throw std::invalid_argument("usage: graphweft_pull_qgroup GRAPH [GPU=0] [REPETITIONS=21]");
    const int gpu = argc >= 3 ? std::stoi(argv[2]) : 0;
    const int repetitions = argc >= 4 ? std::stoi(argv[3]) : 21;
    checked(cudaSetDevice(gpu));
    const auto graph = graphweft::HostGraph::load(argv[1],true,true);
    std::cout << std::fixed << std::setprecision(6)
              << "graph=" << graph.identity << " V=" << graph.vertices
              << " E=" << graph.edges() << " Q=" << queries << " gpu=" << gpu << '\n';
    DeviceBuffer<uint64_t> row(graph.incoming_row.size()); row.upload(graph.incoming_row);
    DeviceBuffer<uint32_t> col(graph.incoming_col.size()); col.upload(graph.incoming_col);
    DeviceBuffer<float> weight(graph.incoming_weight.size()); weight.upload(graph.incoming_weight);
    auto host_values=make_values(graph.vertices);
    DeviceBuffer<float> old_values(host_values.size()); old_values.upload(host_values);
    DeviceBuffer<float> baseline_output(host_values.size()), candidate_output(host_values.size());
    run_candidate<16>(row,col,weight,old_values,baseline_output,candidate_output,graph.vertices,repetitions);
    run_candidate<8>(row,col,weight,old_values,baseline_output,candidate_output,graph.vertices,repetitions);
    run_candidate<4>(row,col,weight,old_values,baseline_output,candidate_output,graph.vertices,repetitions);
    run_candidate<2>(row,col,weight,old_values,baseline_output,candidate_output,graph.vertices,repetitions);
    run_candidate<1>(row,col,weight,old_values,baseline_output,candidate_output,graph.vertices,repetitions);
  } catch (const std::exception& error) {
    std::cerr << "pull qgroup: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
