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
    checked(cudaMemcpy(data_, values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
  }
 private:
  T* data_ = nullptr;
  size_t count_ = 0;
};

template<bool Tiled>
__device__ __forceinline__ size_t value_index(uint32_t vertex,uint32_t query,
                                               uint32_t vertex_count,
                                               uint32_t query_count) {
  if constexpr(Tiled)
    return size_t(query >> 5) * (size_t(vertex_count) << 5) +
           (size_t(vertex) << 5) + (query & 31);
  return size_t(vertex) * query_count + query;
}

template<bool Tiled>
__global__ void initialize_values(float* values, uint64_t cells,
                                  uint32_t vertex_count,uint32_t query_count) {
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < cells;
       i += uint64_t(blockDim.x) * gridDim.x) {
    const uint64_t vertex = i / query_count;
    const uint32_t query = i % query_count;
    values[value_index<Tiled>(uint32_t(vertex),query,vertex_count,query_count)] =
        float((vertex * 2654435761ULL + query * 97ULL) % 100003ULL);
  }
}

template<int QueryLanes, bool Parallel, bool Shuffle, bool Tiled>
__global__ void global_pull(const uint64_t* row, const uint32_t* col,
                            const float* weight, const float* old_values,
                            float* new_values, uint32_t vertex_count,
                            uint32_t query_count) {
  static_assert(QueryLanes == 32 || QueryLanes == 16 || QueryLanes == 8);
  constexpr int edge_groups = 32 / QueryLanes;
  constexpr int queries_per_lane = 32 / QueryLanes;
  __shared__ float partial[warps_per_block * 32 * queries_per_lane];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int edge_group = lane / QueryLanes, query_lane = lane % QueryLanes;
  const uint32_t tiles = query_count / 32;
  const uint64_t warp_task = uint64_t(blockIdx.x) * warps_per_block + warp;
  const uint64_t total_tasks = Parallel ? uint64_t(vertex_count) * tiles : vertex_count;
  const bool valid = warp_task < total_tasks;
  const uint32_t vertex = valid ? (Parallel ? uint32_t(warp_task / tiles)
                                             : uint32_t(warp_task)) : 0;
  const uint32_t first_tile = Parallel && valid ? uint32_t(warp_task % tiles) * 32 : 0;
  const uint32_t tile_end = Parallel ? first_tile + 32 : query_count;
  const uint64_t begin = valid ? row[vertex] : 0, end = valid ? row[vertex + 1] : 0;
  for (uint32_t tile = first_tile; tile < tile_end; tile += 32) {
    const size_t storage_tile_base = Tiled ? size_t(tile >> 5) * (size_t(vertex_count) << 5) : 0;
    float best[queries_per_lane];
#pragma unroll
    for (int k = 0; k < queries_per_lane; ++k) {
      const uint32_t query = tile + k * QueryLanes + query_lane;
      const size_t index = Tiled
          ? storage_tile_base + (size_t(vertex) << 5) + k * QueryLanes + query_lane
          : size_t(vertex) * query_count + query;
      best[k] = valid ? old_values[index] : INFINITY;
    }
    if (valid) {
      for (uint64_t edge = begin + edge_group; edge < end; edge += edge_groups) {
        const uint32_t source = col[edge];
        const float edge_weight = weight[edge];
#pragma unroll
        for (int k = 0; k < queries_per_lane; ++k) {
          const uint32_t query = tile + k * QueryLanes + query_lane;
          const size_t index = Tiled
              ? storage_tile_base + (size_t(source) << 5) + k * QueryLanes + query_lane
              : size_t(source) * query_count + query;
          best[k] = fminf(best[k], old_values[index] + edge_weight);
        }
      }
    }
    if constexpr(QueryLanes == 32) {
      const size_t index = Tiled ? storage_tile_base + (size_t(vertex) << 5) + lane
                                 : size_t(vertex) * query_count + tile + lane;
      if (valid) new_values[index] = best[0];
    } else if constexpr(Shuffle) {
#pragma unroll
      for (int k = 0; k < queries_per_lane; ++k) {
        float reduced = best[k];
#pragma unroll
        for (int offset = 16; offset >= QueryLanes; offset >>= 1)
          reduced = fminf(reduced, __shfl_xor_sync(0xffffffffu, reduced, offset));
        if (valid && edge_group == 0)
          new_values[Tiled ? storage_tile_base + (size_t(vertex)<<5) + k*QueryLanes + query_lane
                           : size_t(vertex)*query_count + tile + k*QueryLanes + query_lane] = reduced;
      }
    } else {
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
          new_values[Tiled ? storage_tile_base + (size_t(vertex)<<5) + k*QueryLanes + query_lane
                           : size_t(vertex)*query_count + tile + k*QueryLanes + query_lane] = reduced;
        }
      }
      if constexpr(!Parallel) __syncthreads();
    }
  }
}

// One warp owns a vertex and all query tiles. Each 32-edge tile is cached in
// a warp-private shared-memory segment. This preserves the serial mapping and
// isolates row caching from tile-level parallelism.
template<int QueryLanes, bool Tiled>
__global__ void serial_smem_pull(const uint64_t* row, const uint32_t* col,
                                 const float* weight, const float* old_values,
                                 float* new_values, uint32_t vertex_count,
                                 uint32_t query_count) {
  static_assert(QueryLanes == 32 || QueryLanes == 16 || QueryLanes == 8);
  constexpr int queries_per_lane = 32 / QueryLanes;
  __shared__ uint32_t cached_source[warps_per_block * 32];
  __shared__ float cached_weight[warps_per_block * 32];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int edge_group = lane / QueryLanes, query_lane = lane % QueryLanes;
  const int cache_base = warp * 32;
  const uint64_t vertex = uint64_t(blockIdx.x) * warps_per_block + warp;
  const bool valid = vertex < vertex_count;
  const uint64_t begin = valid ? row[vertex] : 0, end = valid ? row[vertex + 1] : 0;
  for (uint32_t tile = 0; tile < query_count; tile += 32) {
    const size_t storage_tile_base = Tiled ? size_t(tile >> 5) * (size_t(vertex_count) << 5) : 0;
    float best[queries_per_lane];
#pragma unroll
    for (int k = 0; k < queries_per_lane; ++k) {
      const uint32_t query = tile + k * QueryLanes + query_lane;
      const size_t index = Tiled
          ? storage_tile_base + (size_t(vertex)<<5) + k*QueryLanes + query_lane
          : size_t(vertex)*query_count + query;
      best[k] = valid ? old_values[index] : INFINITY;
    }
    if (valid) {
      for (uint64_t base = begin; base < end; base += 32) {
        const uint64_t edge = base + lane;
        if (edge < end) {
          cached_source[cache_base + lane] = col[edge];
          cached_weight[cache_base + lane] = weight[edge];
        }
        __syncwarp();
        const int group_offset = edge_group * QueryLanes;
        int count = int(end - base) - group_offset;
        count = max(0, min(QueryLanes, count));
        for (int j = 0; j < count; ++j) {
          const uint32_t source = cached_source[cache_base + group_offset + j];
          const float edge_weight = cached_weight[cache_base + group_offset + j];
#pragma unroll
          for (int k = 0; k < queries_per_lane; ++k) {
            const uint32_t query = tile + k * QueryLanes + query_lane;
            const size_t index = Tiled
                ? storage_tile_base + (size_t(source)<<5) + k*QueryLanes + query_lane
                : size_t(source)*query_count + query;
            best[k] = fminf(best[k], old_values[index] + edge_weight);
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
        new_values[Tiled ? storage_tile_base + (size_t(vertex)<<5) + k*QueryLanes + query_lane
                         : size_t(vertex)*query_count + tile + k*QueryLanes + query_lane] = reduced;
    }
  }
}

// One block owns one vertex. Its D/32 query-tile warps cooperatively reuse one
// cached edge tile, so graph metadata is loaded once per vertex rather than
// once per query-tile warp.
template<int QueryCount, int QueryLanes, bool Tiled>
__global__ void parallel_vertex_smem_pull(
    const uint64_t* row, const uint32_t* col, const float* weight,
    const float* old_values, float* new_values, uint32_t vertex_count) {
  static_assert(QueryCount == 128 || QueryCount == 256);
  static_assert(QueryLanes == 32 || QueryLanes == 16 || QueryLanes == 8);
  constexpr int queries_per_lane = 32 / QueryLanes;
  __shared__ uint32_t cached_source[32];
  __shared__ float cached_weight[32];
  const uint32_t vertex = blockIdx.x;
  if (vertex >= vertex_count) return;
  const int lane = threadIdx.x % 32;
  const int query_tile = threadIdx.x / 32;
  const int edge_group = lane / QueryLanes, query_lane = lane % QueryLanes;
  const uint32_t tile = query_tile * 32;
  const size_t storage_tile_base = Tiled ? size_t(query_tile) * (size_t(vertex_count)<<5) : 0;
  float best[queries_per_lane];
#pragma unroll
  for (int k = 0; k < queries_per_lane; ++k) {
    const uint32_t query = tile + k * QueryLanes + query_lane;
    const size_t index = Tiled
        ? storage_tile_base + (size_t(vertex)<<5) + k*QueryLanes + query_lane
        : size_t(vertex)*QueryCount + query;
    best[k] = old_values[index];
  }
  const uint64_t begin = row[vertex], end = row[vertex + 1];
  for (uint64_t base = begin; base < end; base += 32) {
    if (threadIdx.x < 32) {
      const uint64_t edge = base + lane;
      if (edge < end) {
        cached_source[lane] = col[edge];
        cached_weight[lane] = weight[edge];
      }
    }
    __syncthreads();
    const int group_offset = edge_group * QueryLanes;
    int count = int(end - base) - group_offset;
    count = max(0, min(QueryLanes, count));
    for (int j = 0; j < count; ++j) {
      const uint32_t source = cached_source[group_offset + j];
      const float edge_weight = cached_weight[group_offset + j];
#pragma unroll
      for (int k = 0; k < queries_per_lane; ++k) {
        const uint32_t query = tile + k * QueryLanes + query_lane;
        const size_t index = Tiled
            ? storage_tile_base + (size_t(source)<<5) + k*QueryLanes + query_lane
            : size_t(source)*QueryCount + query;
        best[k] = fminf(best[k], old_values[index] + edge_weight);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int k = 0; k < queries_per_lane; ++k) {
    float reduced = best[k];
#pragma unroll
    for (int offset = 16; offset >= QueryLanes; offset >>= 1)
      reduced = fminf(reduced, __shfl_xor_sync(0xffffffffu, reduced, offset));
    if (edge_group == 0)
      new_values[Tiled ? storage_tile_base + (size_t(vertex)<<5) + k*QueryLanes + query_lane
                       : size_t(vertex)*QueryCount + tile + k*QueryLanes + query_lane] = reduced;
  }
}

// GE-SpMM-style feature coarsening: one warp owns one vertex, each lane keeps
// QueryCount/32 outputs live, and the sparse row is traversed only once.
// UseSmem additionally applies coalesced row caching to each 32-edge tile.
template<int QueryCount, int QueryLanes, bool UseSmem, bool Tiled>
__global__ void fused_serial_pull(
    const uint64_t* row, const uint32_t* col, const float* weight,
    const float* old_values, float* new_values, uint32_t vertex_count) {
  static_assert(QueryCount == 128 || QueryCount == 256);
  static_assert(QueryLanes == 32 || QueryLanes == 16 || QueryLanes == 8);
  constexpr int edge_groups = 32 / QueryLanes;
  constexpr int outputs_per_lane = QueryCount / QueryLanes;
  __shared__ uint32_t cached_source[warps_per_block * 32];
  __shared__ float cached_weight[warps_per_block * 32];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int edge_group = lane / QueryLanes, query_lane = lane % QueryLanes;
  const uint64_t vertex = uint64_t(blockIdx.x) * warps_per_block + warp;
  if (vertex >= vertex_count) return;
  const size_t tile_stride = size_t(vertex_count) << 5;
  size_t query_offset[outputs_per_lane];
#pragma unroll
  for (int k = 0; k < outputs_per_lane; ++k) {
    const int query = k * QueryLanes + query_lane;
    if constexpr(Tiled)
      query_offset[k] = size_t(query >> 5) * tile_stride + (query & 31);
    else
      query_offset[k] = query;
  }
  float best[outputs_per_lane];
#pragma unroll
  for (int k = 0; k < outputs_per_lane; ++k) {
    const int query = k * QueryLanes + query_lane;
    const size_t index = Tiled ? (size_t(vertex)<<5) + query_offset[k]
                               : size_t(vertex)*QueryCount + query;
    best[k] = old_values[index];
  }
  const uint64_t begin = row[vertex], end = row[vertex + 1];
  if constexpr(UseSmem) {
    const int cache_base = warp * 32;
    for (uint64_t base = begin; base < end; base += 32) {
      const uint64_t edge = base + lane;
      if (edge < end) {
        cached_source[cache_base + lane] = col[edge];
        cached_weight[cache_base + lane] = weight[edge];
      }
      __syncwarp();
      const int group_offset = QueryLanes == 32 ? 0 : edge_group * QueryLanes;
      int count = 0;
      if constexpr(QueryLanes == 32)
        count = min(32, int(end - base));
      else {
        count = int(end - base) - group_offset;
        count = max(0, min(QueryLanes, count));
      }
      for (int j = 0; j < count; ++j) {
        const uint32_t source = cached_source[cache_base + group_offset + j];
        const float edge_weight = cached_weight[cache_base + group_offset + j];
#pragma unroll
        for (int k = 0; k < outputs_per_lane; ++k) {
          const int query = k * QueryLanes + query_lane;
          const size_t index = Tiled ? (size_t(source)<<5) + query_offset[k]
                                     : size_t(source)*QueryCount + query;
          best[k] = fminf(best[k],
              old_values[index] + edge_weight);
        }
      }
      __syncwarp();
    }
  } else {
    for (uint64_t edge = begin + edge_group; edge < end; edge += edge_groups) {
      const uint32_t source = col[edge];
      const float edge_weight = weight[edge];
#pragma unroll
      for (int k = 0; k < outputs_per_lane; ++k) {
        const int query = k * QueryLanes + query_lane;
        const size_t index = Tiled ? (size_t(source)<<5) + query_offset[k]
                                   : size_t(source)*QueryCount + query;
        best[k] = fminf(best[k],
            old_values[index] + edge_weight);
      }
    }
  }
#pragma unroll
  for (int k = 0; k < outputs_per_lane; ++k) {
    if constexpr(QueryLanes == 32) {
      const size_t index = Tiled ? (size_t(vertex)<<5) + query_offset[k]
                                 : size_t(vertex)*QueryCount + k*32 + lane;
      new_values[index] = best[k];
    } else {
      float reduced = best[k];
#pragma unroll
      for (int offset = 16; offset >= QueryLanes; offset >>= 1)
        reduced = fminf(reduced, __shfl_xor_sync(0xffffffffu, reduced, offset));
      if (edge_group == 0)
        new_values[Tiled ? (size_t(vertex)<<5) + query_offset[k]
                         : size_t(vertex)*QueryCount + k*QueryLanes + query_lane] = reduced;
    }
  }
}

enum class Kind { Shared, Shuffle, Smem, FusedGlobal, FusedSmem };

template<int QueryCount, int QueryLanes, bool Parallel, Kind kind, bool Tiled>
void launch(const DeviceBuffer<uint64_t>& row, const DeviceBuffer<uint32_t>& col,
            const DeviceBuffer<float>& weight, const DeviceBuffer<float>& old_values,
            DeviceBuffer<float>& output, uint32_t vertices) {
  if constexpr(kind == Kind::FusedGlobal || kind == Kind::FusedSmem) {
    static_assert(!Parallel);
    fused_serial_pull<QueryCount,QueryLanes,kind == Kind::FusedSmem,Tiled>
        <<<(vertices + warps_per_block - 1) / warps_per_block,threads>>>(
            row.get(),col.get(),weight.get(),old_values.get(),output.get(),vertices);
  } else if constexpr(kind == Kind::Smem && Parallel) {
    parallel_vertex_smem_pull<QueryCount,QueryLanes,Tiled><<<vertices,QueryCount>>>(
        row.get(),col.get(),weight.get(),old_values.get(),output.get(),vertices);
  } else if constexpr(kind == Kind::Smem) {
    serial_smem_pull<QueryLanes,Tiled><<<(vertices + warps_per_block - 1) / warps_per_block,threads>>>(
        row.get(),col.get(),weight.get(),old_values.get(),output.get(),vertices,QueryCount);
  } else {
    uint64_t warps = vertices;
    if constexpr(Parallel) warps *= QueryCount / 32;
    const uint32_t blocks = uint32_t((warps + warps_per_block - 1) / warps_per_block);
    global_pull<QueryLanes,Parallel,kind == Kind::Shuffle,Tiled><<<blocks,threads>>>(
        row.get(),col.get(),weight.get(),old_values.get(),output.get(),vertices,QueryCount);
  }
  checked(cudaGetLastError());
}

__device__ uint64_t mix(uint64_t value) {
  value ^= value >> 30; value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27; value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

__global__ void fingerprint_kernel(const float* values, uint64_t cells,
                                   unsigned long long* sum,
                                   unsigned long long* xors) {
  uint64_t local_sum = 0, local_xor = 0;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < cells;
       i += uint64_t(blockDim.x) * gridDim.x) {
    const uint64_t token = mix((i << 32) ^ __float_as_uint(values[i]) ^
                               0x9e3779b97f4a7c15ULL);
    local_sum += token; local_xor ^= token;
  }
  atomicAdd(sum,local_sum); atomicXor(xors,local_xor);
}

std::array<uint64_t,2> fingerprint(const DeviceBuffer<float>& values,
                                   DeviceBuffer<unsigned long long>& digest) {
  checked(cudaMemset(digest.get(),0,2*sizeof(unsigned long long)));
  fingerprint_kernel<<<4096,threads>>>(values.get(),values.size(),digest.get(),digest.get()+1);
  checked(cudaGetLastError()); checked(cudaDeviceSynchronize());
  std::array<uint64_t,2> host{};
  checked(cudaMemcpy(host.data(),digest.get(),2*sizeof(uint64_t),cudaMemcpyDeviceToHost));
  return host;
}

double median(std::vector<float> values) {
  std::sort(values.begin(),values.end()); return values[values.size()/2];
}

template<class Baseline,class Candidate>
std::array<double,2> measure_pair(Baseline&& baseline,Candidate&& candidate,int repetitions) {
  cudaEvent_t start=nullptr,stop=nullptr;
  checked(cudaEventCreate(&start)); checked(cudaEventCreate(&stop));
  for(int i=0;i<3;++i){baseline();candidate();} checked(cudaDeviceSynchronize());
  std::array<std::vector<float>,2> samples;
  auto one=[&](int index,auto&& operation){
    checked(cudaEventRecord(start)); operation(); checked(cudaEventRecord(stop));
    checked(cudaEventSynchronize(stop)); float ms=0;
    checked(cudaEventElapsedTime(&ms,start,stop)); samples[index].push_back(ms);
  };
  for(int i=0;i<repetitions;++i) {
    if(i&1){one(1,candidate);one(0,baseline);} else {one(0,baseline);one(1,candidate);}
  }
  checked(cudaEventDestroy(start)); checked(cudaEventDestroy(stop));
  return {median(std::move(samples[0])),median(std::move(samples[1]))};
}

template<int QueryCount,int Q,bool CandidateParallel,Kind kind,bool BaselineParallel,bool Tiled>
void run_candidate(const char* name,const char* baseline_name,
                   const DeviceBuffer<uint64_t>& row,const DeviceBuffer<uint32_t>& col,
                   const DeviceBuffer<float>& weight,const DeviceBuffer<float>& old_values,
                   DeviceBuffer<float>& output,DeviceBuffer<unsigned long long>& digest,
                   uint32_t vertices,int repetitions,const std::array<uint64_t,2>& reference) {
  auto baseline=[&]{launch<QueryCount,32,BaselineParallel,Kind::Shared,Tiled>(row,col,weight,old_values,output,vertices);};
  auto candidate=[&]{launch<QueryCount,Q,CandidateParallel,kind,Tiled>(row,col,weight,old_values,output,vertices);};
  const auto times=measure_pair(baseline,candidate,repetitions);
  candidate(); checked(cudaDeviceSynchronize());
  if(fingerprint(output,digest)!=reference)throw std::runtime_error(std::string(name)+" fingerprint mismatch");
  std::cout<<"variant="<<name<<",baseline="<<baseline_name
           <<",baseline_ms="<<times[0]<<",candidate_ms="<<times[1]
           <<",speedup="<<times[0]/times[1]<<",fingerprint_match=1\n";
}

template<int QueryCount,bool Tiled>
void run_all(const graphweft::HostGraph& graph,int gpu,int repetitions,bool fused_only) {
  const uint64_t cells=uint64_t(graph.vertices)*QueryCount;
  std::cout<<std::fixed<<std::setprecision(6)<<"graph="<<graph.identity
           <<" V="<<graph.vertices<<" E="<<graph.edges()<<" D="<<QueryCount
           <<" layout="<<(Tiled?"tile32":"vertex-major")<<" gpu="<<gpu<<'\n';
  DeviceBuffer<uint64_t> row(graph.incoming_row.size());row.upload(graph.incoming_row);
  DeviceBuffer<uint32_t> col(graph.incoming_col.size());col.upload(graph.incoming_col);
  DeviceBuffer<float> weight(graph.incoming_weight.size());weight.upload(graph.incoming_weight);
  DeviceBuffer<float> old_values(cells),output(cells);
  DeviceBuffer<unsigned long long> digest(2);
  initialize_values<Tiled><<<4096,threads>>>(old_values.get(),cells,graph.vertices,QueryCount);checked(cudaGetLastError());
  launch<QueryCount,32,false,Kind::Shared,Tiled>(row,col,weight,old_values,output,graph.vertices);
  checked(cudaDeviceSynchronize()); const auto reference=fingerprint(output,digest);

  run_candidate<QueryCount,32,false,Kind::FusedGlobal,false,Tiled>("fused-serial-global-q32","serial-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);
  run_candidate<QueryCount,16,false,Kind::FusedGlobal,false,Tiled>("fused-serial-global-q16","serial-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);
  run_candidate<QueryCount,8,false,Kind::FusedGlobal,false,Tiled>("fused-serial-global-q8","serial-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);
  run_candidate<QueryCount,32,false,Kind::FusedSmem,false,Tiled>("fused-serial-smem-q32","serial-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);
  run_candidate<QueryCount,16,false,Kind::FusedSmem,false,Tiled>("fused-serial-smem-q16","serial-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);
  run_candidate<QueryCount,8,false,Kind::FusedSmem,false,Tiled>("fused-serial-smem-q8","serial-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);
  if(fused_only) {
    run_candidate<QueryCount,32,true,Kind::Shared,false,Tiled>("parallel-global-q32","serial-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);
    run_candidate<QueryCount,16,true,Kind::Smem,true,Tiled>("parallel-smem-shuffle-q16","parallel-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);
    return;
  }

  run_candidate<QueryCount,16,false,Kind::Shared,false,Tiled>("serial-shared-q16","serial-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);
  run_candidate<QueryCount,8,false,Kind::Shared,false,Tiled>("serial-shared-q8","serial-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);
  run_candidate<QueryCount,16,false,Kind::Shuffle,false,Tiled>("serial-shuffle-q16","serial-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);
  run_candidate<QueryCount,8,false,Kind::Shuffle,false,Tiled>("serial-shuffle-q8","serial-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);
  run_candidate<QueryCount,32,false,Kind::Smem,false,Tiled>("serial-smem-q32","serial-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);
  run_candidate<QueryCount,16,false,Kind::Smem,false,Tiled>("serial-smem-shuffle-q16","serial-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);
  run_candidate<QueryCount,8,false,Kind::Smem,false,Tiled>("serial-smem-shuffle-q8","serial-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);

  run_candidate<QueryCount,32,true,Kind::Shared,false,Tiled>("parallel-global-q32","serial-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);
  run_candidate<QueryCount,16,true,Kind::Shared,true,Tiled>("parallel-shared-q16","parallel-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);
  run_candidate<QueryCount,8,true,Kind::Shared,true,Tiled>("parallel-shared-q8","parallel-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);
  run_candidate<QueryCount,16,true,Kind::Shuffle,true,Tiled>("parallel-shuffle-q16","parallel-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);
  run_candidate<QueryCount,8,true,Kind::Shuffle,true,Tiled>("parallel-shuffle-q8","parallel-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);
  run_candidate<QueryCount,32,true,Kind::Smem,true,Tiled>("parallel-smem-q32","parallel-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);
  run_candidate<QueryCount,16,true,Kind::Smem,true,Tiled>("parallel-smem-shuffle-q16","parallel-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);
  run_candidate<QueryCount,8,true,Kind::Smem,true,Tiled>("parallel-smem-shuffle-q8","parallel-global-q32",row,col,weight,old_values,output,digest,graph.vertices,repetitions,reference);
}
}  // namespace

int main(int argc,char** argv) {
  try {
    if(argc<3 || argc>7)throw std::invalid_argument(
        "usage: graphweft_pull_wide_smem_shuffle GRAPH D [GPU=0] [REPETITIONS=7] [SUITE=all|fused] [LAYOUT=vm|tile32]");
    const uint32_t query_count=uint32_t(std::stoul(argv[2]));
    if(query_count!=128 && query_count!=256)throw std::invalid_argument("D must be 128 or 256");
    const int gpu=argc>=4?std::stoi(argv[3]):0;
    const int repetitions=argc>=5?std::stoi(argv[4]):7;
    const std::string suite=argc>=6?argv[5]:"all";
    if(suite!="all" && suite!="fused")throw std::invalid_argument("suite must be all or fused");
    const std::string layout=argc>=7?argv[6]:"vm";
    if(layout!="vm" && layout!="tile32")throw std::invalid_argument("layout must be vm or tile32");
    checked(cudaSetDevice(gpu));
    const auto graph=graphweft::HostGraph::load(argv[1],true,true);
    if(query_count==128) {
      if(layout=="tile32")run_all<128,true>(graph,gpu,repetitions,suite=="fused");
      else run_all<128,false>(graph,gpu,repetitions,suite=="fused");
    } else {
      if(layout=="tile32")run_all<256,true>(graph,gpu,repetitions,suite=="fused");
      else run_all<256,false>(graph,gpu,repetitions,suite=="fused");
    }
  } catch(const std::exception& error) {
    std::cerr<<"wide pull smem/shuffle: "<<error.what()<<'\n';return EXIT_FAILURE;
  }
}
