#include "graphweft/kernels.hpp"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace graphweft {
namespace {
void check(cudaError_t e) { if (e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }

/*
关于identity 和 reachable等的判断可能要更general一些
或者说每个算法要提供一个判断函数，直接调用注册函数即可

*/
__device__ float identity(Algorithm a) { return a==Algorithm::SSWP ? -INFINITY : INFINITY; }
__device__ bool reachable(float x, Algorithm a) { return a==Algorithm::SSWP ? x!=-INFINITY : x!=INFINITY; }
__device__ float relax(float x,float w,Algorithm a) { return a==Algorithm::BFS ? x+1.f : a==Algorithm::SSSP ? x+w : fminf(x,w); }
__device__ Algorithm slot_algorithm(const Context& c,uint32_t slot) {
  return c.slot_algorithms?c.slot_algorithms[slot]:c.algorithm;
}
__device__ Algorithm slot_algorithm(const FrontierContext& c,uint32_t slot) {
  return c.slot_algorithms?c.slot_algorithms[slot]:c.algorithm;
}

__device__ bool reduce_min_float(float* p,float x) {
  int* q=reinterpret_cast<int*>(p); int old=*q;
  while (__int_as_float(old)>x) {
    int prior=atomicCAS(q,old,__float_as_int(x));
    if (prior==old) return true;
    old=prior;
  }
  return false;
}
// SSWP capacities may be negative; integer bit order is not float order. CAS keeps max exact.
__device__ bool reduce_max_float(float* p,float x) {
  int* q=reinterpret_cast<int*>(p); int old=*q;
  while (__int_as_float(old)<x) {
    int prior=atomicCAS(q,old,__float_as_int(x));
    if (prior==old) return true;
    old=prior;
  }
  return false;
}
// Publish all newly improved queries for one (vertex, 64-query word) together.
// This keeps the common M<=64 path to one mask atomic per edge/target update,
// while retaining a vertex-level claim for uniqueness when M spans words.
__device__ __forceinline__ void mark_frontier_mask(const Context& c,uint32_t vertex,
                                                    uint32_t word_index,uint64_t bits) {
  if(!c.frontier_output.mask || !bits)return;
  auto* word=reinterpret_cast<unsigned long long*>(
    c.frontier_output.mask+size_t(vertex)*c.words+word_index);
  const uint64_t previous=atomicOr(word,static_cast<unsigned long long>(bits));
  uint64_t new_bits=bits&~previous;
  if(!new_bits)return;
  atomicAdd(reinterpret_cast<unsigned long long*>(c.frontier_output.pair_count),
            static_cast<unsigned long long>(__popcll(new_bits)));
  // Update-driven construction only needs to know whether a slot produced at
  // least one value, not how many vertices it produced. Reuse the slot-count
  // allocation as a compact active-query bitset and avoid one global atomic
  // for every new (vertex, slot) pair.
  atomicOr(reinterpret_cast<unsigned long long*>(c.frontier_output.slot_count)+word_index,
           static_cast<unsigned long long>(new_bits));
  // Only the first successful update of a word can possibly be the first
  // update of the vertex. For one word, previous==0 is the unique claim and
  // direct append does not need the auxiliary flag. With multiple words the
  // flag arbitrates the word winners.
  if(previous!=0)return;
  if(c.frontier_output.direct_append){
    const bool claimed=c.words==1 || atomicCAS(c.frontier_output.flags+vertex,0u,1u)==0u;
    if(claimed){
      const uint32_t position=atomicAdd(c.frontier_output.count,1u);
      c.frontier_output.list[position]=vertex;
    }
  } else {
    atomicExch(c.frontier_output.flags+vertex,1u);
  }
}
__device__ __forceinline__ void mark_frontier_legacy(const Context& c,uint32_t vertex,uint32_t slot) {
  if(!c.frontier_output.mask)return;
  const uint64_t bit=1ULL<<(slot%64);
  auto* word=reinterpret_cast<unsigned long long*>(
    c.frontier_output.mask+size_t(vertex)*c.words+slot/64);
  const uint64_t previous=atomicOr(word,static_cast<unsigned long long>(bit));
  if(previous&bit)return;
  atomicAdd(c.frontier_output.slot_count+slot,1u);
  atomicAdd(reinterpret_cast<unsigned long long*>(c.frontier_output.pair_count),1ULL);
  if(c.frontier_output.direct_append){
    if(atomicCAS(c.frontier_output.flags+vertex,0u,1u)==0u){
      const uint32_t position=atomicAdd(c.frontier_output.count,1u);
      c.frontier_output.list[position]=vertex;
    }
  } else atomicExch(c.frontier_output.flags+vertex,1u);
}
// Aggregate lanes which reach the same vertex/word at the same call site.
// Partitioned push/pull kernels assign query lanes to a shared target, so this
// turns their per-slot discoveries into the same 64-bit publication primitive.
__device__ __forceinline__ void mark_frontier(const Context& c,uint32_t vertex,uint32_t slot) {
  if(!c.frontier_output.mask)return;
  if(!c.frontier_output.mask64){mark_frontier_legacy(c,vertex,slot);return;}
  const uint32_t word_index=slot/64;
  const uint64_t bit=1ULL<<(slot%64);
  const unsigned active=__activemask();
  const unsigned long long key=(static_cast<unsigned long long>(vertex)<<32)|word_index;
  const unsigned peers=__match_any_sync(active,key);
  const int leader=__ffs(static_cast<int>(peers))-1;
  uint64_t combined=0;
  for(int source_lane=0;source_lane<32;++source_lane)
    if(peers&(1u<<source_lane))combined|=__shfl_sync(peers,bit,source_lane);
  if(int(threadIdx.x&31)==leader)mark_frontier_mask(c,vertex,word_index,combined);
}
__device__ uint32_t slot_for_cell(ValueView v,size_t i) {
  if(v.layout==Layout::VertexMajor)return uint32_t(i%v.slots);
  const size_t group_cells=size_t(v.vertices)*v.group_width;
  return uint32_t(i/group_cells)*v.group_width+uint32_t(i%v.group_width);
}
__global__ void init_kernel(ValueView v,Algorithm a,const Algorithm* algorithms,size_t size) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i<size) v.data[i]=identity(algorithms?algorithms[slot_for_cell(v,i)]:a);
}
__global__ void reset_slots_kernel(ValueView v,const uint8_t* reset,Algorithm a,
                                   const Algorithm* algorithms,size_t size) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<size){uint32_t s=slot_for_cell(v,i);if(reset[s])v.data[i]=identity(algorithms?algorithms[s]:a);}
}
__global__ void reset_slot_range_kernel(ValueView v,uint32_t first_slot,uint32_t slot_count,
                                        Algorithm a,const Algorithm* algorithms,size_t size) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=size)return;
  uint32_t local=uint32_t(i%slot_count),vertex=uint32_t(i/slot_count);
  uint32_t slot=first_slot+local;
  v.data[v.index(vertex,slot)]=identity(algorithms?algorithms[slot]:a);
}
__global__ void fill_value_range_kernel(float* data,float value,size_t size) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<size)data[i]=value;
}
__global__ void clear_kernel(uint64_t* m,size_t count) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(i<count)m[i]=0;
}
__global__ void activate_kernel(ValueView v,uint64_t* mask,const uint32_t* sources,const uint8_t* due,
                                uint32_t slots,uint32_t words,Algorithm algorithm,const Algorithm* algorithms) {
  uint32_t s=blockIdx.x*blockDim.x+threadIdx.x;
  if(s>=slots || !due[s])return;
  uint32_t source=sources[s];
  Algorithm a=algorithms?algorithms[s]:algorithm;
  v.data[v.index(source,s)]=a==Algorithm::SSWP ? INFINITY : 0.f;
  atomicOr(reinterpret_cast<unsigned long long*>(mask+size_t(source)*words+s/64),1ULL<<(s%64));
}
__global__ void clear_slot_mask_kernel(uint64_t* mask,uint32_t vertices,uint32_t words,
                                       const uint8_t* reset,uint32_t slots) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=size_t(vertices)*words)return;
  uint32_t word=uint32_t(i%words);uint64_t keep=~uint64_t{0};
  for(uint32_t b=0;b<64;++b){uint32_t s=word*64+b;if(s<slots&&reset[s])keep&=~(1ULL<<b);}
  mask[i]&=keep;
}

// A warp shares one frontier source. Each lane traverses different outgoing edges.
__global__ void push_kernel(Context c) {
  uint32_t warp=(blockIdx.x*blockDim.x+threadIdx.x)/32;
  uint32_t lane=threadIdx.x%32;
  uint32_t count=*c.frontier_count;
  for(uint32_t index=warp;index<count;index+=gridDim.x*(blockDim.x/32)) {
  uint32_t source=c.frontier[index];
  for(uint64_t e=c.graph.out_row[source]+lane;e<c.graph.out_row[source+1];e+=32) {
    uint32_t dest=c.graph.out_col[e]; float weight=c.graph.out_weight[e];
    for(uint32_t word=0;word<c.words;++word) {
      uint64_t bits=c.frontier_mask[size_t(source)*c.words+word];
      while(bits) {
        uint32_t bit=__ffsll(static_cast<long long>(bits))-1;
        uint32_t slot=word*64+bit; bits &= bits-1;
        if(slot>=c.slots || !c.live_slots[slot])continue;
        Algorithm algorithm=slot_algorithm(c,slot);
        float old=c.old_values.data[c.old_values.index(source,slot)];
        if(!reachable(old,algorithm))continue;
        if(algorithm==Algorithm::BFS && old>=16777216.f) { atomicExch(c.error_flag,1); continue; }
        float candidate=relax(old,weight,algorithm);
        float* target=c.new_values.data+c.new_values.index(dest,slot);
        bool improved=algorithm==Algorithm::SSWP?reduce_max_float(target,candidate):reduce_min_float(target,candidate);
        if(improved)mark_frontier_legacy(c,dest,slot);
      }
    }
  }
  }
}
__global__ void push_kernel_mask64(Context c) {
  uint32_t warp=(blockIdx.x*blockDim.x+threadIdx.x)/32;
  uint32_t lane=threadIdx.x%32;
  uint32_t count=*c.frontier_count;
  for(uint32_t index=warp;index<count;index+=gridDim.x*(blockDim.x/32)) {
    uint32_t source=c.frontier[index];
    for(uint64_t e=c.graph.out_row[source]+lane;e<c.graph.out_row[source+1];e+=32) {
      uint32_t dest=c.graph.out_col[e]; float weight=c.graph.out_weight[e];
      for(uint32_t word=0;word<c.words;++word) {
        uint64_t bits=c.frontier_mask[size_t(source)*c.words+word];
        uint64_t improved_bits=0;
        while(bits) {
          uint32_t bit=__ffsll(static_cast<long long>(bits))-1;
          uint32_t slot=word*64+bit; bits &= bits-1;
          if(slot>=c.slots || !c.live_slots[slot])continue;
          Algorithm algorithm=slot_algorithm(c,slot);
          float old=c.old_values.data[c.old_values.index(source,slot)];
          if(!reachable(old,algorithm))continue;
          if(algorithm==Algorithm::BFS && old>=16777216.f) { atomicExch(c.error_flag,1); continue; }
          float candidate=relax(old,weight,algorithm);
          float* target=c.new_values.data+c.new_values.index(dest,slot);
          bool improved=algorithm==Algorithm::SSWP?reduce_max_float(target,candidate):reduce_min_float(target,candidate);
          if(improved)improved_bits|=1ULL<<bit;
        }
        mark_frontier_mask(c,dest,word,improved_bits);
      }
    }
  }
}
// Each warp compacts the same 32-column query tile, independently of its
// edge/query grouping. Groups own edges; lanes within a group own query ranks.
template<int QueryLanes, int Warps, int Blocks>
__global__ void partition_push_kernel(Context c) {
  __shared__ uint32_t active_slots[Warps][32];
  constexpr int EdgeGroups=32/QueryLanes;
  const uint32_t warp=threadIdx.x/32, lane=threadIdx.x%32;
  const uint32_t group=lane/QueryLanes, query_lane=lane%QueryLanes;
  const unsigned group_mask=0xffffffffu >> (32-QueryLanes) << (group*QueryLanes);
  const uint64_t tasks=uint64_t(*c.frontier_count)*Blocks;
  for(uint64_t task=blockIdx.x;task<tasks;task+=gridDim.x) {
    const uint32_t source=c.frontier[task/Blocks];
    const uint32_t vertex_warp=uint32_t(task%Blocks)*Warps+warp;
    const uint64_t start=c.graph.out_row[source], end=c.graph.out_row[source+1];
    for(uint32_t tile=0;tile<c.slots;tile+=32) {
      const uint32_t slot=tile+lane;
      const bool enabled=slot<c.slots && c.live_slots[slot] &&
        ((c.frontier_mask[size_t(source)*c.words+slot/64]>>(slot%64))&1ULL);
      const unsigned bits=__ballot_sync(0xffffffffu,enabled);
      const uint32_t active=__popc(bits);
      if(enabled)active_slots[warp][__popc(bits&((1u<<lane)-1))]=slot;
      __syncwarp();
      if(active) {
        for(uint64_t edge=start+vertex_warp*EdgeGroups+group;edge<end;
            edge+=Blocks*Warps*EdgeGroups) {
          uint32_t dest=0;float weight=0;
          if(query_lane==0){dest=c.graph.out_col[edge];weight=c.graph.out_weight[edge];}
          dest=__shfl_sync(group_mask,dest,0,QueryLanes);
          weight=__shfl_sync(group_mask,weight,0,QueryLanes);
          for(uint32_t base=0;base<active;base+=QueryLanes) {
            if(base+query_lane>=active)continue;
            const uint32_t q=active_slots[warp][base+query_lane];
            const Algorithm algorithm=slot_algorithm(c,q);
            const float old=c.old_values.data[c.old_values.index(source,q)];
            if(!reachable(old,algorithm))continue;
            if(algorithm==Algorithm::BFS && old>=16777216.f){atomicExch(c.error_flag,1);continue;}
            const float candidate=relax(old,weight,algorithm);
            float* target=c.new_values.data+c.new_values.index(dest,q);
            bool improved=algorithm==Algorithm::SSWP?reduce_max_float(target,candidate):reduce_min_float(target,candidate);
            if(improved)mark_frontier(c,dest,q);
          }
        }
      }
      __syncwarp();
    }
  }
}
template<int Q,int W,int B>void launch_partition(const Context& c) {
  uint32_t n=c.frontier_size_hint==UINT32_MAX?c.graph.vertices:c.frontier_size_hint;
  if(!n)return;
  // The cap is divisible by four so parts of a vertex remain on separate blocks.
  uint32_t blocks=uint32_t(std::min<uint64_t>(65532,uint64_t(n)*B));
  partition_push_kernel<Q,W,B><<<blocks,W*32,0,c.stream>>>(c);
}
template<int Q>void launch_grain(int grain,const Context& c) {
  switch(grain){
    case 0:launch_partition<Q,1,1>(c);break;
    case 1:launch_partition<Q,2,1>(c);break;
    case 2:launch_partition<Q,4,1>(c);break;
    case 3:launch_partition<Q,4,2>(c);break;
    case 4:launch_partition<Q,4,4>(c);break;
  }
}
template<int QueryLanes,int Warps,int Blocks>
__device__ __forceinline__ void adaptive_push_tile(
    Context c,uint32_t source,uint32_t vertex_warp,uint32_t tile,
    uint32_t active,uint32_t active_slots[Warps][32]) {
  constexpr int EdgeGroups=32/QueryLanes;
  const uint32_t warp=threadIdx.x/32,lane=threadIdx.x%32;
  const uint32_t group=lane/QueryLanes,query_lane=lane%QueryLanes;
  const unsigned group_mask=(0xffffffffu>>(32-QueryLanes))<<(group*QueryLanes);
  const uint64_t start=c.graph.out_row[source],end=c.graph.out_row[source+1];
  for(uint64_t edge=start+vertex_warp*EdgeGroups+group;edge<end;
      edge+=Blocks*Warps*EdgeGroups) {
    uint32_t dest=0;float weight=0;
    if(query_lane==0){dest=c.graph.out_col[edge];weight=c.graph.out_weight[edge];}
    dest=__shfl_sync(group_mask,dest,0,QueryLanes);
    weight=__shfl_sync(group_mask,weight,0,QueryLanes);
    for(uint32_t base=0;base<active;base+=QueryLanes) {
      if(base+query_lane>=active)continue;
      const uint32_t q=active_slots[warp][base+query_lane];
      const Algorithm algorithm=slot_algorithm(c,q);
      const float old=c.old_values.data[c.old_values.index(source,q)];
      if(!reachable(old,algorithm))continue;
      if(algorithm==Algorithm::BFS && old>=16777216.f){atomicExch(c.error_flag,1);continue;}
      const float candidate=relax(old,weight,algorithm);
      float* target=c.new_values.data+c.new_values.index(dest,q);
      const bool improved=algorithm==Algorithm::SSWP?reduce_max_float(target,candidate):reduce_min_float(target,candidate);
      if(improved)mark_frontier(c,dest,q);
    }
  }
}
template<int Warps,int Blocks>
__global__ void adaptive_push_kernel(Context c,const uint32_t* vertices,uint32_t vertex_count) {
  __shared__ uint32_t active_slots[Warps][32];
  const uint32_t warp=threadIdx.x/32,lane=threadIdx.x%32;
  const uint64_t tasks=uint64_t(vertex_count)*Blocks;
  for(uint64_t task=blockIdx.x;task<tasks;task+=gridDim.x) {
    const uint32_t source=vertices[task/Blocks];
    const uint32_t vertex_warp=uint32_t(task%Blocks)*Warps+warp;
    for(uint32_t tile=0;tile<c.slots;tile+=32) {
      const uint32_t slot=tile+lane;
      const bool enabled=slot<c.slots && c.live_slots[slot] &&
        ((c.frontier_mask[size_t(source)*c.words+slot/64]>>(slot%64))&1ULL);
      const unsigned bits=__ballot_sync(0xffffffffu,enabled);
      const uint32_t active=__popc(bits);
      if(enabled)active_slots[warp][__popc(bits&((1u<<lane)-1))]=slot;
      __syncwarp();
      if(!active){__syncwarp();continue;}
      if(active<=1)adaptive_push_tile<1,Warps,Blocks>(c,source,vertex_warp,tile,active,active_slots);
      else if(active<=2)adaptive_push_tile<2,Warps,Blocks>(c,source,vertex_warp,tile,active,active_slots);
      else if(active<=4)adaptive_push_tile<4,Warps,Blocks>(c,source,vertex_warp,tile,active,active_slots);
      else if(active<=8)adaptive_push_tile<8,Warps,Blocks>(c,source,vertex_warp,tile,active,active_slots);
      else if(active<=16)adaptive_push_tile<16,Warps,Blocks>(c,source,vertex_warp,tile,active,active_slots);
      else adaptive_push_tile<32,Warps,Blocks>(c,source,vertex_warp,tile,active,active_slots);
      __syncwarp();
    }
  }
}
template<int Warps,int Blocks>
void launch_adaptive_bucket(const Context& c,const uint32_t* vertices,uint32_t count) {
  if(!count)return;
  const uint32_t blocks=uint32_t(std::min<uint64_t>(65532,uint64_t(count)*Blocks));
  adaptive_push_kernel<Warps,Blocks><<<blocks,Warps*32,0,c.stream>>>(c,vertices,count);
}
__global__ void pull_kernel(Context c,uint32_t tiles) {
  uint64_t index=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  uint64_t total=uint64_t(c.graph.vertices)*tiles*32;
  if(index>=total)return;
  uint32_t slot=index%32+(index/32%tiles)*32;
  uint32_t vertex=index/(uint64_t(tiles)*32);
  if(slot>=c.slots || !c.live_slots[slot])return;
  Algorithm algorithm=slot_algorithm(c,slot);
  float best=c.old_values.data[c.old_values.index(vertex,slot)];
  for(uint64_t e=c.graph.in_row[vertex];e<c.graph.in_row[vertex+1];++e) {
    uint32_t source=c.graph.in_col[e];
    float old=c.old_values.data[c.old_values.index(source,slot)];
    if(!reachable(old,algorithm))continue;
    if(algorithm==Algorithm::BFS && old>=16777216.f) { atomicExch(c.error_flag,1); continue; }
    float candidate=relax(old,c.graph.in_weight[e],algorithm);
    best=algorithm==Algorithm::SSWP ? fmaxf(best,candidate) : fminf(best,candidate);
  }
  c.new_values.data[c.new_values.index(vertex,slot)]=best;
  const bool improved=algorithm==Algorithm::SSWP?best>c.old_values.data[c.old_values.index(vertex,slot)]:
    best<c.old_values.data[c.old_values.index(vertex,slot)];
  if(improved)mark_frontier(c,vertex,slot);
}
// A block owns one target/part.  Each group of QueryLanes cooperates on an
// incoming edge; the other groups and warps traverse disjoint incoming edges.
// QueryLanes is lanes per edge (q8 means four edge groups per warp). With
// Q32 vertex-major float values, eight adjacent lanes fill one 32-byte sector;
// narrower groups may revisit that sector as k advances through queries.
template<int QueryLanes,int Warps,int Blocks,bool Check>
__global__ void partition_pull_kernel(Context c) {
  constexpr int EdgeGroups=32/QueryLanes;
  constexpr int Groups=Warps*EdgeGroups;
  constexpr int QueriesPerLane=32/QueryLanes;
  __shared__ float partial[QueriesPerLane][Warps*32];
  const uint32_t lane=threadIdx.x%32,warp=threadIdx.x/32;
  const uint32_t edge_group=lane/QueryLanes,query_lane=lane%QueryLanes;
  const uint32_t part=blockIdx.x%Blocks;
  const uint32_t first=blockIdx.x/Blocks,step=gridDim.x/Blocks;
  for(uint32_t target=first;target<c.graph.vertices;target+=step) {
    const uint64_t begin=c.graph.in_row[target],end=c.graph.in_row[target+1];
    if(begin==end)continue;
    for(uint32_t tile=0;tile<c.slots;tile+=32) {
      float best[QueriesPerLane];
#pragma unroll
      for(int k=0;k<QueriesPerLane;++k){
        uint32_t slot=tile+k*QueryLanes+query_lane;
        best[k]=identity(slot<c.slots?slot_algorithm(c,slot):c.algorithm);
      }
      for(uint64_t edge=begin+part*Groups+warp*EdgeGroups+edge_group;
          edge<end;edge+=Blocks*Groups) {
        const uint32_t source=c.graph.in_col[edge];
        const float weight=c.graph.in_weight[edge];
        uint64_t active=~uint64_t{0};
        if constexpr(Check) {
          active=c.frontier_mask[size_t(source)*c.words+tile/64];
          if(!(active>>(tile%64)&0xffffffffULL))continue;
        }
#pragma unroll
        for(int k=0;k<QueriesPerLane;++k) {
          const uint32_t slot=tile+k*QueryLanes+query_lane;
          if(slot>=c.slots || !c.live_slots[slot])continue;
          const Algorithm algorithm=slot_algorithm(c,slot);
          if constexpr(Check) if(!((active>>(slot%64))&1ULL))continue;
          const float old=c.old_values.data[c.old_values.index(source,slot)];
          if(!reachable(old,algorithm))continue;
          if(algorithm==Algorithm::BFS && old>=16777216.f){atomicExch(c.error_flag,1);continue;}
          const float candidate=relax(old,weight,algorithm);
          best[k]=algorithm==Algorithm::SSWP?fmaxf(best[k],candidate):fminf(best[k],candidate);
        }
      }
#pragma unroll
      for(int k=0;k<QueriesPerLane;++k)partial[k][threadIdx.x]=best[k];
      __syncthreads();
      if(threadIdx.x<32) {
        const uint32_t slot=tile+threadIdx.x;
        const uint32_t k=threadIdx.x/QueryLanes;
        const uint32_t query_lane_out=threadIdx.x%QueryLanes;
        if(slot<c.slots && c.live_slots[slot]) {
          const Algorithm algorithm=slot_algorithm(c,slot);
          float reduced=identity(algorithm);
          for(int group=0;group<Groups;++group)
            reduced=algorithm==Algorithm::SSWP?
              fmaxf(reduced,partial[k][group*QueryLanes+query_lane_out]):
              fminf(reduced,partial[k][group*QueryLanes+query_lane_out]);
          const size_t position=c.new_values.index(target,slot);
          const float previous=c.old_values.data[c.old_values.index(target,slot)];
          if(algorithm==Algorithm::SSWP) {
            if(reduced>previous) {
              if constexpr(Blocks==1){c.new_values.data[position]=reduced;mark_frontier(c,target,slot);}
              else if(reduce_max_float(c.new_values.data+position,reduced))mark_frontier(c,target,slot);
            }
          } else if(reduced<previous) {
            if constexpr(Blocks==1){c.new_values.data[position]=reduced;mark_frontier(c,target,slot);}
            else if(reduce_min_float(c.new_values.data+position,reduced))mark_frontier(c,target,slot);
          }
        }
      }
      __syncthreads();
    }
  }
}

template<int Q,int W,int B,bool Check>void launch_pull_partition(const Context& c) {
  if(!c.graph.vertices)return;
  const uint32_t blocks=uint32_t(std::min<uint64_t>(65532,uint64_t(c.graph.vertices)*B));
  partition_pull_kernel<Q,W,B,Check><<<blocks,W*32,0,c.stream>>>(c);
}
template<int Q,bool Check>void launch_pull_grain(int grain,const Context& c) {
  switch(grain){
    case 0:launch_pull_partition<Q,1,1,Check>(c);break;
    case 1:launch_pull_partition<Q,2,1,Check>(c);break;
    case 2:launch_pull_partition<Q,4,1,Check>(c);break;
    case 3:launch_pull_partition<Q,4,2,Check>(c);break;
    case 4:launch_pull_partition<Q,4,4,Check>(c);break;
  }
}
template<bool Check>void launch_pull_group(int group,int grain,const Context& c) {
  switch(group){
    case 0:launch_pull_grain<1,Check>(grain,c);break;
    case 1:launch_pull_grain<2,Check>(grain,c);break;
    case 2:launch_pull_grain<4,Check>(grain,c);break;
    case 3:launch_pull_grain<8,Check>(grain,c);break;
    case 4:launch_pull_grain<16,Check>(grain,c);break;
    case 5:launch_pull_grain<32,Check>(grain,c);break;
  }
}

// G=8 experiment kernel. A block owns one target vertex. The four warps own
// exactly the four edge streams that q8_w1 formerly placed inside one warp:
// warp w visits begin+w, begin+w+4, ... . Every warp covers all 32 queries in
// a tile, then warp 0 reduces the four partial results for each query.
__global__ void grouped_g8_edge4_warp4_pull_kernel(Context c) {
  __shared__ float partial[4][32];
  const uint32_t warp=threadIdx.x/32;
  const uint32_t lane=threadIdx.x%32;
  const size_t group_stride=size_t(c.old_values.vertices)*8;
  const uint32_t first=blockIdx.x;
  const uint32_t step=gridDim.x;
  for(uint32_t target=first;target<c.graph.vertices;target+=step) {
    const uint64_t begin=c.graph.in_row[target],end=c.graph.in_row[target+1];
    if(begin==end)continue;
    for(uint32_t tile=0;tile<c.slots;tile+=32) {
      float best=identity(c.algorithm);
      const uint32_t slot=tile+lane;
      const bool enabled=slot<c.slots && c.live_slots[slot];
      const Algorithm algorithm=slot<c.slots?slot_algorithm(c,slot):c.algorithm;
      best=identity(algorithm);
      for(uint64_t edge=begin+warp;edge<end;edge+=4) {
        uint32_t source=0;
        float weight=0;
        if(lane==0) {
          source=c.graph.in_col[edge];
          weight=c.graph.in_weight[edge];
        }
        source=__shfl_sync(0xffffffffu,source,0);
        weight=__shfl_sync(0xffffffffu,weight,0);
        if(!enabled)continue;
        const size_t source_base=size_t(tile>>3)*group_stride+size_t(source)*8;
        const size_t position=source_base+size_t(lane>>3)*group_stride+(lane&7);
        const float old=c.old_values.data[position];
        if(!reachable(old,algorithm))continue;
        if(algorithm==Algorithm::BFS && old>=16777216.f) {
          atomicExch(c.error_flag,1);
          continue;
        }
        const float candidate=relax(old,weight,algorithm);
        best=algorithm==Algorithm::SSWP?fmaxf(best,candidate):fminf(best,candidate);
      }
      partial[warp][lane]=best;
      __syncthreads();
      if(warp==0 && enabled) {
        float reduced=partial[0][lane];
#pragma unroll
        for(int w=1;w<4;++w)
          reduced=algorithm==Algorithm::SSWP?fmaxf(reduced,partial[w][lane]):fminf(reduced,partial[w][lane]);
        const size_t target_base=size_t(tile>>3)*group_stride+size_t(target)*8;
        const size_t position=target_base+size_t(lane>>3)*group_stride+(lane&7);
        const float previous=c.old_values.data[position];
        if(algorithm==Algorithm::SSWP) {
          if(reduced>previous){c.new_values.data[position]=reduced;mark_frontier(c,target,slot);}
        } else if(reduced<previous){c.new_values.data[position]=reduced;mark_frontier(c,target,slot);}
      }
      __syncthreads();
    }
  }
}

void launch_grouped_g8_edge4_warp4_pull(const Context& c) {
  if(c.old_values.layout!=Layout::Grouped || c.new_values.layout!=Layout::Grouped ||
     c.old_values.group_width!=8 || c.new_values.group_width!=8)
    throw std::invalid_argument("grouped G8 edge4-warp4 Pull requires --layout=grouped --group_width=8");
  if(!c.graph.vertices)return;
  const uint32_t blocks=uint32_t(std::min<uint64_t>(65532,c.graph.vertices));
  grouped_g8_edge4_warp4_pull_kernel<<<blocks,4*32,0,c.stream>>>(c);
}
__global__ void compare_kernel(FrontierContext c) {
  uint32_t v=blockIdx.x*blockDim.x+threadIdx.x;
  if(v>=c.old_values.vertices)return;
  uint32_t active=0;
  for(uint32_t w=0;w<c.words;++w) {
    uint64_t bits=0;
    for(uint32_t b=0;b<64;++b) {
      uint32_t s=w*64+b;
      if(s>=c.slots)break;
      if(!c.live_slots[s])continue;
      Algorithm algorithm=slot_algorithm(c,s);
      float old=c.old_values.data[c.old_values.index(v,s)];
      float next=c.new_values.data[c.new_values.index(v,s)];
      bool improved=algorithm==Algorithm::SSWP ? next>old : next<old;
      if(improved) { bits|=1ULL<<b; ++active; atomicAdd(c.slot_count+s,1u); }
    }
    c.mask[size_t(v)*c.words+w]=bits;
  }
  c.flags[v]=active!=0;
  if(active)atomicAdd(reinterpret_cast<unsigned long long*>(c.pair_count),static_cast<unsigned long long>(active));
}
__global__ void flags_from_mask(FrontierContext c) {
  uint32_t v=blockIdx.x*blockDim.x+threadIdx.x;
  if(v>=c.old_values.vertices)return;
  uint32_t active=0;
  for(uint32_t w=0;w<c.words;++w)active+=__popcll(c.mask[size_t(v)*c.words+w]);
  c.flags[v]=active!=0;
  if(active)atomicAdd(reinterpret_cast<unsigned long long*>(c.pair_count),static_cast<unsigned long long>(active));
}
__global__ void edge_pairs_kernel(GraphView g,const uint64_t* mask,const uint32_t* list,const uint32_t* count,
                                  uint32_t words,uint64_t* result) {
  uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=*count)return;
  uint32_t v=list[i], active=0;
  for(uint32_t w=0;w<words;++w)active+=__popcll(mask[size_t(v)*words+w]);
  atomicAdd(reinterpret_cast<unsigned long long*>(result),static_cast<unsigned long long>((g.out_row[v+1]-g.out_row[v])*active));
}
__global__ void classify_edge_pairs_kernel(GraphView g,const uint64_t* mask,const uint32_t* list,
                                            const uint32_t* count,uint32_t words,uint64_t* result,
                                            uint8_t* categories,uint32_t* category_counts) {
  uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=*count)return;
  uint32_t v=list[i],active=0;
  for(uint32_t w=0;w<words;++w)active+=__popcll(mask[size_t(v)*words+w]);
  const uint64_t degree=g.out_row[v+1]-g.out_row[v];
  const uint64_t load=active && degree>UINT64_MAX/active?UINT64_MAX:degree*active;
  atomicAdd(reinterpret_cast<unsigned long long*>(result),
            static_cast<unsigned long long>(load));
  const int category=adaptive_push_bucket(load);
  categories[i]=category<0?uint8_t(255):uint8_t(category);
  if(category>=0)atomicAdd(category_counts+category,1u);
}
__global__ void scatter_adaptive_kernel(const uint32_t* list,const uint32_t* count,
                                        const uint8_t* categories,uint32_t* buckets,
                                        uint32_t* cursors,uint32_t o0,uint32_t o1,
                                        uint32_t o2,uint32_t o3) {
  uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=*count)return;
  const uint32_t category=categories[i];
  if(category>=adaptive_push_bucket_count)return;
  const uint32_t offsets[]={o0,o1,o2,o3};
  const uint32_t offset=category?offsets[category-1]:0;
  buckets[offset+atomicAdd(cursors+category,1u)]=list[i];
}
__global__ void unordered_kernel(const uint32_t* flags,uint32_t* list,uint32_t* count,uint32_t n) {
  uint32_t v=blockIdx.x*blockDim.x+threadIdx.x;
  uint32_t lane=threadIdx.x&31;
  bool selected=v<n && flags[v]!=0;
  unsigned bits=__ballot_sync(0xffffffff,selected);
  uint32_t base=0;
  if(lane==0 && bits)base=atomicAdd(count,__popc(bits));
  base=__shfl_sync(0xffffffff,base,0);
  if(selected)list[base+__popc(bits&((1u<<lane)-1))]=v;
}
}
void initialize_values(ValueView v,Algorithm a,cudaStream_t stream) {
  initialize_values(v,a,nullptr,stream);
}
void initialize_values(ValueView v,Algorithm a,const Algorithm* algorithms,cudaStream_t stream) {
  size_t count=size_t(v.vertices)*v.slots;
  init_kernel<<<(count+255)/256,256,0,stream>>>(v,a,algorithms,count); check(cudaGetLastError());
}
void reset_slots(ValueView v,const uint8_t* reset,Algorithm a,const Algorithm* algorithms,cudaStream_t stream) {
  size_t count=size_t(v.vertices)*v.slots;
  reset_slots_kernel<<<(count+255)/256,256,0,stream>>>(v,reset,a,algorithms,count);check(cudaGetLastError());
}
void reset_slot_range(ValueView v,uint32_t first_slot,uint32_t slot_count,Algorithm a,
                      const Algorithm* algorithms,cudaStream_t stream) {
  if(!slot_count || first_slot+slot_count>v.slots)throw std::invalid_argument("invalid reset slot range");
  size_t count=size_t(v.vertices)*slot_count;
  // A complete grouped partition is one contiguous allocation.  Refill always
  // replaces such aligned, same-algorithm groups, so avoid the general
  // per-element slot/vertex division and the algorithm lookup here.
  if(v.layout==Layout::Grouped && first_slot%v.group_width==0 && slot_count==v.group_width && !algorithms){
    const size_t offset=size_t(first_slot/v.group_width)*v.vertices*v.group_width;
    const float value=a==Algorithm::SSWP?-std::numeric_limits<float>::infinity():
                                           std::numeric_limits<float>::infinity();
    fill_value_range_kernel<<<(count+255)/256,256,0,stream>>>(v.data+offset,value,count);
    check(cudaGetLastError());return;
  }
  reset_slot_range_kernel<<<(count+255)/256,256,0,stream>>>(
    v,first_slot,slot_count,a,algorithms,count);check(cudaGetLastError());
}
void clear_mask(uint64_t* mask,uint32_t vertices,uint32_t words,cudaStream_t stream) {
  size_t count=size_t(vertices)*words;
  clear_kernel<<<(count+255)/256,256,0,stream>>>(mask,count); check(cudaGetLastError());
}
void activate(ValueView v,uint64_t* mask,const uint32_t* sources,const uint8_t* due,
              uint32_t slots,uint32_t words,Algorithm algorithm,cudaStream_t stream) {
  activate(v,mask,sources,due,slots,words,algorithm,nullptr,stream);
}
void activate(ValueView v,uint64_t* mask,const uint32_t* sources,const uint8_t* due,
              uint32_t slots,uint32_t words,Algorithm algorithm,const Algorithm* algorithms,cudaStream_t stream) {
  activate_kernel<<<(slots+255)/256,256,0,stream>>>(v,mask,sources,due,slots,words,algorithm,algorithms); check(cudaGetLastError());
}
void clear_slot_mask(uint64_t* mask,uint32_t vertices,uint32_t words,const uint8_t* reset,
                     uint32_t slots,cudaStream_t stream) {
  size_t count=size_t(vertices)*words;
  clear_slot_mask_kernel<<<(count+255)/256,256,0,stream>>>(mask,vertices,words,reset,slots);check(cudaGetLastError());
}
void shared_push(const Context& c) {
  // Launch for every possible frontier vertex because count is held on device.
  uint32_t warps=4; uint32_t blocks=std::min(65535u,(c.graph.vertices+warps-1)/warps);
  if(c.frontier_output.mask && c.frontier_output.mask64)
    push_kernel_mask64<<<blocks,warps*32,0,c.stream>>>(c);
  else push_kernel<<<blocks,warps*32,0,c.stream>>>(c);
  check(cudaGetLastError());
}
void dense_pull(const Context& c) {
  uint32_t tiles=(c.slots+31)/32;
  uint64_t count=uint64_t(c.graph.vertices)*tiles*32;
  pull_kernel<<<(count+255)/256,256,0,c.stream>>>(c,tiles); check(cudaGetLastError());
}
void adaptive_push(const Context& c,const uint32_t* buckets,
                   const uint32_t counts[adaptive_push_bucket_count]) {
  uint32_t offset=0;
  launch_adaptive_bucket<1,1>(c,buckets+offset,counts[0]);offset+=counts[0];
  launch_adaptive_bucket<2,1>(c,buckets+offset,counts[1]);offset+=counts[1];
  launch_adaptive_bucket<4,1>(c,buckets+offset,counts[2]);offset+=counts[2];
  launch_adaptive_bucket<4,2>(c,buckets+offset,counts[3]);offset+=counts[3];
  launch_adaptive_bucket<4,4>(c,buckets+offset,counts[4]);
  check(cudaGetLastError());
}
KernelId push_partition_id(int index) {
  if(index<0 || index>=push_partition_count)throw std::invalid_argument("invalid Push partition index");
  return static_cast<KernelId>(int(KernelId::PushPartitionBase)+index);
}
PushPartition push_partition(int index) {
  push_partition_id(index);
  const uint32_t warps[]={1,2,4,4,4},blocks[]={1,1,1,2,4};
  return {1u<<(index/5),warps[index%5],blocks[index%5]};
}
KernelId pull_partition_id(int index,bool check) {
  if(index<0 || index>=pull_partition_count)throw std::invalid_argument("invalid Pull partition index");
  return static_cast<KernelId>(int(check?KernelId::PullCheckBase:KernelId::PullCheckFreeBase)+index);
}
PushPartition pull_partition(int index) { return push_partition(index); }
void launch(KernelId id,const Context& c) {
  if(id==KernelId::SharedPush){shared_push(c);return;}
  if(id==KernelId::DensePull){dense_pull(c);return;}
  if(id==KernelId::AdaptivePush)throw std::invalid_argument("adaptive Push requires prepared buckets");
  if(id==KernelId::GroupedG8Edge4Warp4Pull){
    launch_grouped_g8_edge4_warp4_pull(c);check(cudaGetLastError());return;
  }
  int pull_index=int(id)-int(KernelId::PullCheckFreeBase);
  if(pull_index>=0 && pull_index<pull_partition_count){
    launch_pull_group<false>(pull_index/5,pull_index%5,c);check(cudaGetLastError());return;
  }
  pull_index=int(id)-int(KernelId::PullCheckBase);
  if(pull_index>=0 && pull_index<pull_partition_count){
    launch_pull_group<true>(pull_index/5,pull_index%5,c);check(cudaGetLastError());return;
  }
  int index=int(id)-int(KernelId::PushPartitionBase);
  push_partition_id(index);
  switch(index/5){
    case 0:launch_grain<1>(index%5,c);break;
    case 1:launch_grain<2>(index%5,c);break;
    case 2:launch_grain<4>(index%5,c);break;
    case 3:launch_grain<8>(index%5,c);break;
    case 4:launch_grain<16>(index%5,c);break;
    case 5:launch_grain<32>(index%5,c);break;
  }
  check(cudaGetLastError());
}
size_t stable_temp_bytes(uint32_t n) {
  size_t bytes=0; auto input=cub::CountingInputIterator<uint32_t>(0);
  cub::DeviceSelect::Flagged(nullptr,bytes,input,static_cast<const uint32_t*>(nullptr),
                             static_cast<uint32_t*>(nullptr),static_cast<uint32_t*>(nullptr),n);
  return bytes;
}
void build_frontier(const FrontierContext& c,cudaEvent_t compare_start,cudaEvent_t compare_end) {
  check(cudaMemsetAsync(c.pair_count,0,sizeof(uint64_t),c.stream));
  check(cudaMemsetAsync(c.count,0,sizeof(uint32_t),c.stream));
  uint32_t n=c.old_values.vertices;
  check(cudaMemsetAsync(c.slot_count,0,size_t(c.slots)*sizeof(uint32_t),c.stream));
  if(compare_start)check(cudaEventRecord(compare_start,c.stream));
  compare_kernel<<<(n+255)/256,256,0,c.stream>>>(c); check(cudaGetLastError());
  if(compare_end)check(cudaEventRecord(compare_end,c.stream));
  if(c.mode==FrontierMode::Unordered) {
    unordered_kernel<<<(n+255)/256,256,0,c.stream>>>(c.flags,c.list,c.count,n); check(cudaGetLastError());
  } else {
    auto input=cub::CountingInputIterator<uint32_t>(0); size_t bytes=c.stable_temp_bytes;
    check(cub::DeviceSelect::Flagged(c.stable_temp,bytes,input,c.flags,c.list,c.count,n,c.stream));
  }
}
void prepare_fused_frontier(const FrontierContext& c) {
  check(cudaMemsetAsync(c.mask,0,size_t(c.old_values.vertices)*c.words*sizeof(uint64_t),c.stream));
  check(cudaMemsetAsync(c.flags,0,size_t(c.old_values.vertices)*sizeof(uint32_t),c.stream));
  check(cudaMemsetAsync(c.pair_count,0,sizeof(uint64_t),c.stream));
  check(cudaMemsetAsync(c.count,0,sizeof(uint32_t),c.stream));
  check(cudaMemsetAsync(c.slot_count,0,std::max(size_t(c.slots)*sizeof(uint32_t),
                                               size_t(c.words)*sizeof(uint64_t)),c.stream));
}
void finish_fused_frontier(const FrontierContext& c) {
  const uint32_t n=c.old_values.vertices;
  if(c.mode==FrontierMode::Unordered) {
    unordered_kernel<<<(n+255)/256,256,0,c.stream>>>(c.flags,c.list,c.count,n);check(cudaGetLastError());
  } else {
    auto input=cub::CountingInputIterator<uint32_t>(0);size_t bytes=c.stable_temp_bytes;
    check(cub::DeviceSelect::Flagged(c.stable_temp,bytes,input,c.flags,c.list,c.count,n,c.stream));
  }
}
void finish_direct_frontier(const FrontierContext&) {
  // The relaxation kernel has already produced the unique vertex list and
  // count. Kernel completion is the publication barrier for mask/list data.

}
void rebuild_frontier(const FrontierContext& c) {
  check(cudaMemsetAsync(c.pair_count,0,sizeof(uint64_t),c.stream));
  check(cudaMemsetAsync(c.count,0,sizeof(uint32_t),c.stream));
  uint32_t n=c.old_values.vertices;
  flags_from_mask<<<(n+255)/256,256,0,c.stream>>>(c); check(cudaGetLastError());
  if(c.mode==FrontierMode::Unordered) {
    unordered_kernel<<<(n+255)/256,256,0,c.stream>>>(c.flags,c.list,c.count,n); check(cudaGetLastError());
  } else {
    auto input=cub::CountingInputIterator<uint32_t>(0); size_t bytes=c.stable_temp_bytes;
    check(cub::DeviceSelect::Flagged(c.stable_temp,bytes,input,c.flags,c.list,c.count,n,c.stream));
  }
}
void count_edge_pairs(GraphView g,const uint64_t* mask,const uint32_t* list,const uint32_t* count,
                      uint32_t words,uint64_t* output,cudaStream_t stream) {
  check(cudaMemsetAsync(output,0,sizeof(uint64_t),stream));
  edge_pairs_kernel<<<(g.vertices+255)/256,256,0,stream>>>(g,mask,list,count,words,output);
  check(cudaGetLastError());
}
void classify_edge_pairs(GraphView g,const uint64_t* mask,const uint32_t* list,
                         const uint32_t* count,uint32_t words,uint64_t* output,
                         uint8_t* categories,uint32_t* category_counts,
                         cudaStream_t stream) {
  check(cudaMemsetAsync(output,0,sizeof(uint64_t),stream));
  check(cudaMemsetAsync(category_counts,0,adaptive_push_bucket_count*sizeof(uint32_t),stream));
  classify_edge_pairs_kernel<<<(g.vertices+255)/256,256,0,stream>>>(
    g,mask,list,count,words,output,categories,category_counts);
  check(cudaGetLastError());
}
void scatter_adaptive_buckets(const uint32_t* list,const uint32_t* count,
                              const uint8_t* categories,uint32_t* buckets,
                              uint32_t* cursors,uint32_t maximum_count,
                              const uint32_t counts[adaptive_push_bucket_count],
                              cudaStream_t stream) {
  check(cudaMemsetAsync(cursors,0,adaptive_push_bucket_count*sizeof(uint32_t),stream));
  const uint32_t total=counts[0]+counts[1]+counts[2]+counts[3]+counts[4];
  if(!total)return;
  scatter_adaptive_kernel<<<(maximum_count+255)/256,256,0,stream>>>(
    list,count,categories,buckets,cursors,counts[0],counts[0]+counts[1],
    counts[0]+counts[1]+counts[2],counts[0]+counts[1]+counts[2]+counts[3]);
  check(cudaGetLastError());
}

__device__ __forceinline__ uint64_t fingerprint_mix(uint64_t value) {
  value^=value>>30;value*=0xbf58476d1ce4e5b9ULL;
  value^=value>>27;value*=0x94d049bb133111ebULL;
  return value^(value>>31);
}
__global__ void fingerprint_kernel(ValueView values,uint32_t first_slot,uint64_t* sums,uint64_t* xors) {
  const uint32_t local_slot=blockIdx.y,slot=first_slot+local_slot;
  uint64_t sum=0,xor_value=0;
  for(uint32_t vertex=blockIdx.x*blockDim.x+threadIdx.x;vertex<values.vertices;
      vertex+=gridDim.x*blockDim.x){
    uint32_t bits=__float_as_uint(values.data[values.index(vertex,slot)]);
    uint64_t mixed=fingerprint_mix((uint64_t(vertex)<<32)^bits^0x9e3779b97f4a7c15ULL);
    sum+=mixed;xor_value^=mixed;
  }
  __shared__ uint64_t shared_sum[256],shared_xor[256];
  shared_sum[threadIdx.x]=sum;shared_xor[threadIdx.x]=xor_value;__syncthreads();
  for(uint32_t stride=blockDim.x/2;stride;stride>>=1){
    if(threadIdx.x<stride){shared_sum[threadIdx.x]+=shared_sum[threadIdx.x+stride];
      shared_xor[threadIdx.x]^=shared_xor[threadIdx.x+stride];}
    __syncthreads();
  }
  if(threadIdx.x==0){atomicAdd(reinterpret_cast<unsigned long long*>(sums+local_slot),shared_sum[0]);
    atomicXor(reinterpret_cast<unsigned long long*>(xors+local_slot),shared_xor[0]);}
}
void fingerprint_values(ValueView values,uint32_t first_slot,uint32_t slot_count,
                        uint64_t* sums,uint64_t* xors,cudaStream_t stream) {
  if(!slot_count)return;
  check(cudaMemsetAsync(sums,0,size_t(slot_count)*sizeof(uint64_t),stream));
  check(cudaMemsetAsync(xors,0,size_t(slot_count)*sizeof(uint64_t),stream));
  uint32_t blocks=std::max(1u,std::min(4096u,(values.vertices+255)/256));
  fingerprint_kernel<<<dim3(blocks,slot_count),256,0,stream>>>(values,first_slot,sums,xors);
  check(cudaGetLastError());
}
}
