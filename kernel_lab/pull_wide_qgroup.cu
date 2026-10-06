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
    if (count_) checked(cudaMemcpy(data_, values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
  }
 private:
  T* data_ = nullptr;
  size_t count_ = 0;
};

__global__ void initialize_values(float* values, uint64_t cells, uint32_t query_count) {
  for (uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < cells; index += uint64_t(blockDim.x) * gridDim.x) {
    const uint64_t vertex = index / query_count;
    const uint32_t query = index % query_count;
    values[index] = float((vertex * 2654435761ULL + query * 97ULL) % 100003ULL);
  }
}

// One warp owns one vertex and loops over all 32-query tiles serially.
template<int QueryLanes>
__global__ void serial_tiles_pull(const uint64_t* row, const uint32_t* col,
                                  const float* weight, const float* old_values,
                                  float* new_values, uint32_t vertex_count,
                                  uint32_t query_count) {
  static_assert(32 % QueryLanes == 0);
  constexpr int edge_groups = 32 / QueryLanes;
  constexpr int queries_per_lane = 32 / QueryLanes;
  __shared__ float partial[warps_per_block * 32 * queries_per_lane];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int edge_group = lane / QueryLanes, query_lane = lane % QueryLanes;
  const uint64_t vertex = uint64_t(blockIdx.x) * warps_per_block + warp;
  const bool valid = vertex < vertex_count;
  const uint64_t begin = valid ? row[vertex] : 0;
  const uint64_t end = valid ? row[vertex + 1] : 0;
  for (uint32_t tile = 0; tile < query_count; tile += 32) {
    float best[queries_per_lane];
#pragma unroll
    for (int k = 0; k < queries_per_lane; ++k) {
      const uint32_t query = tile + k * QueryLanes + query_lane;
      best[k] = valid ? old_values[vertex * query_count + query] : INFINITY;
    }
    if (valid) for (uint64_t edge = begin + edge_group; edge < end; edge += edge_groups) {
      const uint32_t source = col[edge];
      const float edge_weight = weight[edge];
#pragma unroll
      for (int k = 0; k < queries_per_lane; ++k) {
        const uint32_t query = tile + k * QueryLanes + query_lane;
        best[k] = fminf(best[k],old_values[size_t(source) * query_count + query] + edge_weight);
      }
    }
    if constexpr(QueryLanes == 32) {
      if (valid) new_values[vertex * query_count + tile + lane] = best[0];
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
            reduced = fminf(reduced,partial[(warp * queries_per_lane + k) * 32 + group * QueryLanes + query_lane]);
          new_values[vertex * query_count + tile + k * QueryLanes + query_lane] = reduced;
        }
      }
      __syncthreads();
    }
  }
}

// A vertex owns query_count/32 independent warps. Each warp handles one query
// tile; QueryLanes only changes the edge/query split inside that tile.
template<int QueryLanes>
__global__ void parallel_tiles_pull(const uint64_t* row, const uint32_t* col,
                                    const float* weight, const float* old_values,
                                    float* new_values, uint32_t vertex_count,
                                    uint32_t query_count) {
  static_assert(32 % QueryLanes == 0);
  constexpr int edge_groups = 32 / QueryLanes;
  constexpr int queries_per_lane = 32 / QueryLanes;
  __shared__ float partial[warps_per_block * 32 * queries_per_lane];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int edge_group = lane / QueryLanes, query_lane = lane % QueryLanes;
  const uint32_t tiles = query_count / 32;
  const uint64_t task = uint64_t(blockIdx.x) * warps_per_block + warp;
  const uint64_t total = uint64_t(vertex_count) * tiles;
  const bool valid = task < total;
  const uint32_t vertex = valid ? uint32_t(task / tiles) : 0;
  const uint32_t tile = valid ? uint32_t(task % tiles) * 32 : 0;
  float best[queries_per_lane];
#pragma unroll
  for (int k = 0; k < queries_per_lane; ++k) {
    const uint32_t query = tile + k * QueryLanes + query_lane;
    best[k] = valid ? old_values[size_t(vertex) * query_count + query] : INFINITY;
  }
  if (valid) {
    const uint64_t begin = row[vertex], end = row[vertex + 1];
    for (uint64_t edge = begin + edge_group; edge < end; edge += edge_groups) {
      const uint32_t source = col[edge];
      const float edge_weight = weight[edge];
#pragma unroll
      for (int k = 0; k < queries_per_lane; ++k) {
        const uint32_t query = tile + k * QueryLanes + query_lane;
        best[k] = fminf(best[k],old_values[size_t(source) * query_count + query] + edge_weight);
      }
    }
  }
  if constexpr(QueryLanes == 32) {
    if (valid) new_values[size_t(vertex) * query_count + tile + lane] = best[0];
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
          reduced = fminf(reduced,partial[(warp * queries_per_lane + k) * 32 + group * QueryLanes + query_lane]);
        new_values[size_t(vertex) * query_count + tile + k * QueryLanes + query_lane] = reduced;
      }
    }
  }
}

template<int QueryLanes, bool Parallel>
void launch(const DeviceBuffer<uint64_t>& row,
            const DeviceBuffer<uint32_t>& col,
            const DeviceBuffer<float>& weight,
            const DeviceBuffer<float>& old_values,
            DeviceBuffer<float>& new_values,
            uint32_t vertices, uint32_t query_count) {
  uint64_t warps = vertices;
  if constexpr(Parallel) warps *= query_count / 32;
  const uint32_t blocks = uint32_t((warps + warps_per_block - 1) / warps_per_block);
  if constexpr(Parallel)
    parallel_tiles_pull<QueryLanes><<<blocks,threads>>>(row.get(),col.get(),weight.get(),old_values.get(),new_values.get(),vertices,query_count);
  else
    serial_tiles_pull<QueryLanes><<<blocks,threads>>>(row.get(),col.get(),weight.get(),old_values.get(),new_values.get(),vertices,query_count);
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
  for (uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < cells; index += uint64_t(blockDim.x) * gridDim.x) {
    const uint64_t bits = __float_as_uint(values[index]);
    const uint64_t token = mix((index << 32) ^ bits ^ 0x9e3779b97f4a7c15ULL);
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
  cudaEvent_t start=nullptr,stop=nullptr; checked(cudaEventCreate(&start)); checked(cudaEventCreate(&stop));
  for(int i=0;i<3;++i){baseline();candidate();} checked(cudaDeviceSynchronize());
  std::array<std::vector<float>,2> samples;
  auto one=[&](int index,auto&& operation){checked(cudaEventRecord(start));operation();checked(cudaEventRecord(stop));checked(cudaEventSynchronize(stop));float ms=0;checked(cudaEventElapsedTime(&ms,start,stop));samples[index].push_back(ms);};
  for(int i=0;i<repetitions;++i){if(i&1){one(1,candidate);one(0,baseline);}else{one(0,baseline);one(1,candidate);}}
  checked(cudaEventDestroy(start));checked(cudaEventDestroy(stop));
  return {median(std::move(samples[0])),median(std::move(samples[1]))};
}

template<int Q,bool Parallel>
void run_candidate(const std::string& name,
                   const DeviceBuffer<uint64_t>& row,const DeviceBuffer<uint32_t>& col,
                   const DeviceBuffer<float>& weight,const DeviceBuffer<float>& old_values,
                   DeviceBuffer<float>& output,DeviceBuffer<unsigned long long>& digest,
                   uint32_t vertices,uint32_t query_count,int repetitions,
                   const std::array<uint64_t,2>& reference) {
  auto baseline=[&]{launch<32,false>(row,col,weight,old_values,output,vertices,query_count);};
  auto candidate=[&]{launch<Q,Parallel>(row,col,weight,old_values,output,vertices,query_count);};
  const auto times=measure_pair(baseline,candidate,repetitions);
  candidate();checked(cudaDeviceSynchronize());const auto actual=fingerprint(output,digest);
  if(actual!=reference)throw std::runtime_error(name+" fingerprint mismatch");
  std::cout<<name<<",baseline_ms="<<times[0]<<",candidate_ms="<<times[1]
           <<",speedup="<<times[0]/times[1]<<",fingerprint_match=1\n";
}
} // namespace

int main(int argc,char** argv) {
  try {
    if(argc<3 || argc>5)throw std::invalid_argument("usage: graphweft_pull_wide_qgroup GRAPH D [GPU=0] [REPETITIONS=11]");
    const uint32_t query_count=uint32_t(std::stoul(argv[2]));
    if(query_count!=128 && query_count!=256)throw std::invalid_argument("D must be 128 or 256");
    const int gpu=argc>=4?std::stoi(argv[3]):0,repetitions=argc>=5?std::stoi(argv[4]):11;
    checked(cudaSetDevice(gpu));const auto graph=graphweft::HostGraph::load(argv[1],true,true);
    const uint64_t cells=uint64_t(graph.vertices)*query_count;
    std::cout<<std::fixed<<std::setprecision(6)<<"graph="<<graph.identity<<" V="<<graph.vertices
             <<" E="<<graph.edges()<<" D="<<query_count<<" gpu="<<gpu<<'\n';
    DeviceBuffer<uint64_t> row(graph.incoming_row.size());row.upload(graph.incoming_row);
    DeviceBuffer<uint32_t> col(graph.incoming_col.size());col.upload(graph.incoming_col);
    DeviceBuffer<float> weight(graph.incoming_weight.size());weight.upload(graph.incoming_weight);
    DeviceBuffer<float> old_values(cells),output(cells);DeviceBuffer<unsigned long long> digest(2);
    initialize_values<<<4096,threads>>>(old_values.get(),cells,query_count);checked(cudaGetLastError());
    launch<32,false>(row,col,weight,old_values,output,graph.vertices,query_count);checked(cudaDeviceSynchronize());
    const auto reference=fingerprint(output,digest);
    run_candidate<16,false>("serial-q16",row,col,weight,old_values,output,digest,graph.vertices,query_count,repetitions,reference);
    run_candidate<8,false>("serial-q8",row,col,weight,old_values,output,digest,graph.vertices,query_count,repetitions,reference);
    run_candidate<32,true>("parallel-q32",row,col,weight,old_values,output,digest,graph.vertices,query_count,repetitions,reference);
    run_candidate<16,true>("parallel-q16",row,col,weight,old_values,output,digest,graph.vertices,query_count,repetitions,reference);
    run_candidate<8,true>("parallel-q8",row,col,weight,old_values,output,digest,graph.vertices,query_count,repetitions,reference);
  } catch(const std::exception& error){std::cerr<<"wide pull qgroup: "<<error.what()<<'\n';return EXIT_FAILURE;}
}
