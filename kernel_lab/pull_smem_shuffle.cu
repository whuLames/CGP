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

__global__ void init_values(float* values, size_t count) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint32_t vertex = uint32_t(i / queries);
  const uint32_t query = uint32_t(i % queries);
  values[i] = float((uint64_t(vertex) * 2654435761ULL + query * 97ULL) % 100003ULL);
}

// Current baseline: one warp per vertex, one query per lane. Edge metadata is
// read directly from global memory by every lane.
__global__ void pull_global_q32(const uint64_t* row, const uint32_t* col,
                                const float* weight, const float* old_values,
                                float* new_values, uint32_t vertex_count) {
  const uint64_t vertex = uint64_t(blockIdx.x) * warps_per_block + threadIdx.x / 32;
  const int query = threadIdx.x % 32;
  if (vertex >= vertex_count) return;
  float best = old_values[vertex * queries + query];
  for (uint64_t edge = row[vertex]; edge < row[vertex + 1]; ++edge) {
    const uint32_t source = col[edge];
    best = fminf(best, old_values[size_t(source) * queries + query] + weight[edge]);
  }
  new_values[vertex * queries + query] = best;
}

// Existing Q16/Q8 implementation: edge groups traverse disjoint streams and
// place partial minima in shared memory. A block barrier precedes reduction.
template<int QueryLanes>
__global__ void pull_global_shared_reduce(
    const uint64_t* row, const uint32_t* col, const float* weight,
    const float* old_values, float* new_values, uint32_t vertex_count) {
  static_assert(QueryLanes == 16 || QueryLanes == 8);
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
        best[k] = fminf(best[k], old_values[size_t(source) * queries + query] + edge_weight);
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
        reduced = fminf(reduced, partial[(warp * queries_per_lane + k) * 32 +
                                        group * QueryLanes + query_lane]);
      new_values[vertex * queries + k * QueryLanes + query_lane] = reduced;
    }
  }
}

// Same work decomposition as above, but group partials are exchanged inside
// the warp with shuffle instructions. No shared partial array or block barrier.
template<int QueryLanes>
__global__ void pull_global_shuffle_reduce(
    const uint64_t* row, const uint32_t* col, const float* weight,
    const float* old_values, float* new_values, uint32_t vertex_count) {
  static_assert(QueryLanes == 16 || QueryLanes == 8);
  constexpr int edge_groups = 32 / QueryLanes;
  constexpr int queries_per_lane = 32 / QueryLanes;
  const int lane = threadIdx.x % 32;
  const int edge_group = lane / QueryLanes;
  const int query_lane = lane % QueryLanes;
  const uint64_t vertex = uint64_t(blockIdx.x) * warps_per_block + threadIdx.x / 32;
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
        best[k] = fminf(best[k], old_values[size_t(source) * queries + query] + edge_weight);
      }
    }
  }
#pragma unroll
  for (int k = 0; k < queries_per_lane; ++k) {
    float reduced = best[k];
#pragma unroll
    for (int offset = 16; offset >= QueryLanes; offset >>= 1)
      reduced = fminf(reduced, __shfl_xor_sync(0xffffffffu, reduced, offset));
    if (valid && edge_group == 0)
      new_values[vertex * queries + k * QueryLanes + query_lane] = reduced;
  }
}

// GE-SpMM-style coalesced row caching. Every warp cooperatively loads a tile
// of 32 (source, weight) pairs once, then query lanes consume their portion
// from shared memory. Q16/Q8 retain the same edge-group decomposition and use
// shuffle reduction, making graph caching and reduction separately measurable.
template<int QueryLanes>
__global__ void pull_smem_shuffle_reduce(
    const uint64_t* row, const uint32_t* col, const float* weight,
    const float* old_values, float* new_values, uint32_t vertex_count) {
  static_assert(QueryLanes == 32 || QueryLanes == 16 || QueryLanes == 8);
  constexpr int queries_per_lane = 32 / QueryLanes;
  __shared__ uint32_t cached_source[warps_per_block * 32];
  __shared__ float cached_weight[warps_per_block * 32];
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int edge_group = lane / QueryLanes;
  const int query_lane = lane % QueryLanes;
  const int cache_base = warp * 32;
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
    for (uint64_t tile = begin; tile < end; tile += 32) {
      const uint64_t edge = tile + lane;
      if (edge < end) {
        cached_source[cache_base + lane] = col[edge];
        cached_weight[cache_base + lane] = weight[edge];
      }
      __syncwarp();
      const int group_offset = edge_group * QueryLanes;
      int group_count = int(end - tile) - group_offset;
      group_count = max(0, min(QueryLanes, group_count));
      for (int j = 0; j < group_count; ++j) {
        const uint32_t source = cached_source[cache_base + group_offset + j];
        const float edge_weight = cached_weight[cache_base + group_offset + j];
#pragma unroll
        for (int k = 0; k < queries_per_lane; ++k) {
          const int query = k * QueryLanes + query_lane;
          best[k] = fminf(best[k], old_values[size_t(source) * queries + query] + edge_weight);
        }
      }
      __syncwarp();
    }
  }
#pragma unroll
  for (int k = 0; k < queries_per_lane; ++k) {
    float reduced = best[k];
#pragma unroll
    for (int offset = 16; offset >= QueryLanes; offset >>= 1)
      reduced = fminf(reduced, __shfl_xor_sync(0xffffffffu, reduced, offset));
    if (valid && edge_group == 0)
      new_values[vertex * queries + k * QueryLanes + query_lane] = reduced;
  }
}

__global__ void compare_exact(const float* a, const float* b, size_t count,
                              unsigned int* mismatch) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count && a[i] != b[i]) atomicExch(mismatch, 1u);
}

template<class Kernel>
void launch_kernel(Kernel kernel, const DeviceBuffer<uint64_t>& row,
                   const DeviceBuffer<uint32_t>& col,
                   const DeviceBuffer<float>& weight,
                   const DeviceBuffer<float>& old_values,
                   DeviceBuffer<float>& output, uint32_t vertices) {
  kernel<<<(vertices + warps_per_block - 1) / warps_per_block, threads>>>(
      row.get(), col.get(), weight.get(), old_values.get(), output.get(), vertices);
  checked(cudaGetLastError());
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
    checked(cudaEventSynchronize(stop));
    float ms = 0; checked(cudaEventElapsedTime(&ms, start, stop));
    samples[index].push_back(ms);
  };
  for (int i = 0; i < repetitions; ++i) {
    if (i & 1) { one(1, candidate); one(0, baseline); }
    else { one(0, baseline); one(1, candidate); }
  }
  checked(cudaEventDestroy(start)); checked(cudaEventDestroy(stop));
  return {median(std::move(samples[0])), median(std::move(samples[1]))};
}

template<class Candidate>
void run_candidate(const char* name, Candidate&& candidate,
                   const DeviceBuffer<uint64_t>& row,
                   const DeviceBuffer<uint32_t>& col,
                   const DeviceBuffer<float>& weight,
                   const DeviceBuffer<float>& old_values,
                   DeviceBuffer<float>& baseline_output,
                   DeviceBuffer<float>& candidate_output,
                   DeviceBuffer<unsigned int>& mismatch,
                   uint32_t vertices, int repetitions) {
  auto baseline = [&] {
    launch_kernel(pull_global_q32, row, col, weight, old_values,
                  baseline_output, vertices);
  };
  auto launch_candidate = [&] { candidate(candidate_output); };
  const auto times = measure_pair(baseline, launch_candidate, repetitions);
  baseline(); launch_candidate();
  checked(cudaMemset(mismatch.get(), 0, sizeof(unsigned int)));
  const size_t cells = baseline_output.size();
  compare_exact<<<uint32_t((cells + threads - 1) / threads), threads>>>(
      baseline_output.get(), candidate_output.get(), cells, mismatch.get());
  checked(cudaGetLastError());
  unsigned int host_mismatch = 0;
  checked(cudaMemcpy(&host_mismatch, mismatch.get(), sizeof(host_mismatch),
                     cudaMemcpyDeviceToHost));
  if (host_mismatch) throw std::runtime_error(std::string(name) + " result mismatch");
  std::cout << "variant=" << name << ",baseline_ms=" << times[0]
            << ",candidate_ms=" << times[1]
            << ",speedup=" << times[0] / times[1] << ",exact=1\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 4)
      throw std::invalid_argument(
          "usage: graphweft_pull_smem_shuffle GRAPH [GPU=0] [REPETITIONS=21]");
    const int gpu = argc >= 3 ? std::stoi(argv[2]) : 0;
    const int repetitions = argc >= 4 ? std::stoi(argv[3]) : 21;
    checked(cudaSetDevice(gpu));
    const auto graph = graphweft::HostGraph::load(argv[1], true, true);
    std::cout << std::fixed << std::setprecision(6)
              << "graph=" << graph.identity << " V=" << graph.vertices
              << " E=" << graph.edges() << " D=" << queries
              << " mapping=one-warp-per-vertex gpu=" << gpu << '\n';
    DeviceBuffer<uint64_t> row(graph.incoming_row.size()); row.upload(graph.incoming_row);
    DeviceBuffer<uint32_t> col(graph.incoming_col.size()); col.upload(graph.incoming_col);
    DeviceBuffer<float> weight(graph.incoming_weight.size()); weight.upload(graph.incoming_weight);
    const size_t cells = size_t(graph.vertices) * queries;
    DeviceBuffer<float> old_values(cells), baseline_output(cells), candidate_output(cells);
    init_values<<<uint32_t((cells + threads - 1) / threads), threads>>>(old_values.get(), cells);
    checked(cudaGetLastError());
    DeviceBuffer<unsigned int> mismatch(1);

    auto shared16 = [&](DeviceBuffer<float>& out) {
      launch_kernel(pull_global_shared_reduce<16>, row,col,weight,old_values,out,graph.vertices);
    };
    auto shuffle16 = [&](DeviceBuffer<float>& out) {
      launch_kernel(pull_global_shuffle_reduce<16>, row,col,weight,old_values,out,graph.vertices);
    };
    auto shared8 = [&](DeviceBuffer<float>& out) {
      launch_kernel(pull_global_shared_reduce<8>, row,col,weight,old_values,out,graph.vertices);
    };
    auto shuffle8 = [&](DeviceBuffer<float>& out) {
      launch_kernel(pull_global_shuffle_reduce<8>, row,col,weight,old_values,out,graph.vertices);
    };
    auto smem32 = [&](DeviceBuffer<float>& out) {
      launch_kernel(pull_smem_shuffle_reduce<32>, row,col,weight,old_values,out,graph.vertices);
    };
    auto smem16 = [&](DeviceBuffer<float>& out) {
      launch_kernel(pull_smem_shuffle_reduce<16>, row,col,weight,old_values,out,graph.vertices);
    };
    auto smem8 = [&](DeviceBuffer<float>& out) {
      launch_kernel(pull_smem_shuffle_reduce<8>, row,col,weight,old_values,out,graph.vertices);
    };
    run_candidate("global-shared-q16",shared16,row,col,weight,old_values,
                  baseline_output,candidate_output,mismatch,graph.vertices,repetitions);
    run_candidate("global-shuffle-q16",shuffle16,row,col,weight,old_values,
                  baseline_output,candidate_output,mismatch,graph.vertices,repetitions);
    run_candidate("global-shared-q8",shared8,row,col,weight,old_values,
                  baseline_output,candidate_output,mismatch,graph.vertices,repetitions);
    run_candidate("global-shuffle-q8",shuffle8,row,col,weight,old_values,
                  baseline_output,candidate_output,mismatch,graph.vertices,repetitions);
    run_candidate("smem-q32",smem32,row,col,weight,old_values,
                  baseline_output,candidate_output,mismatch,graph.vertices,repetitions);
    run_candidate("smem-shuffle-q16",smem16,row,col,weight,old_values,
                  baseline_output,candidate_output,mismatch,graph.vertices,repetitions);
    run_candidate("smem-shuffle-q8",smem8,row,col,weight,old_values,
                  baseline_output,candidate_output,mismatch,graph.vertices,repetitions);
  } catch (const std::exception& error) {
    std::cerr << "pull smem/shuffle: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
