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

__device__ void reduce_min_float(float* p,float x) {
  int* q=reinterpret_cast<int*>(p); int old=*q;
  while (__int_as_float(old)>x) {
    int prior=atomicCAS(q,old,__float_as_int(x));
    if (prior==old) break;
    old=prior;
  }
}
// SSWP capacities may be negative; integer bit order is not float order. CAS keeps max exact.
__device__ void reduce_max_float(float* p,float x) {
  int* q=reinterpret_cast<int*>(p); int old=*q;
  while (__int_as_float(old)<x) {
    int prior=atomicCAS(q,old,__float_as_int(x));
    if (prior==old) break;
    old=prior;
  }
}
__global__ void init_kernel(ValueView v,Algorithm a,size_t size) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i<size) v.data[i]=identity(a);
}
__global__ void clear_kernel(uint64_t* m,size_t count) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(i<count)m[i]=0;
}
__global__ void activate_kernel(ValueView v,uint64_t* mask,const uint32_t* sources,const uint8_t* due,
                                uint32_t slots,uint32_t words,Algorithm algorithm) {
  uint32_t s=blockIdx.x*blockDim.x+threadIdx.x;
  if(s>=slots || !due[s])return;
  uint32_t source=sources[s];
  v.data[v.index(source,s)]=algorithm==Algorithm::SSWP ? INFINITY : 0.f;
  atomicOr(reinterpret_cast<unsigned long long*>(mask+size_t(source)*words+s/64),1ULL<<(s%64));
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
        float old=c.old_values.data[c.old_values.index(source,slot)];
        if(!reachable(old,c.algorithm))continue;
        if(c.algorithm==Algorithm::BFS && old>=16777216.f) { atomicExch(c.error_flag,1); continue; }
        float candidate=relax(old,weight,c.algorithm);
        float* target=c.new_values.data+c.new_values.index(dest,slot);
        if(c.algorithm==Algorithm::SSWP)reduce_max_float(target,candidate);
        else reduce_min_float(target,candidate);
      }
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
            const float old=c.old_values.data[c.old_values.index(source,q)];
            if(!reachable(old,c.algorithm))continue;
            if(c.algorithm==Algorithm::BFS && old>=16777216.f){atomicExch(c.error_flag,1);continue;}
            const float candidate=relax(old,weight,c.algorithm);
            float* target=c.new_values.data+c.new_values.index(dest,q);
            if(c.algorithm==Algorithm::SSWP)reduce_max_float(target,candidate);
            else reduce_min_float(target,candidate);
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
__global__ void pull_kernel(Context c,uint32_t tiles) {
  uint64_t index=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  uint64_t total=uint64_t(c.graph.vertices)*tiles*32;
  if(index>=total)return;
  uint32_t slot=index%32+(index/32%tiles)*32;
  uint32_t vertex=index/(uint64_t(tiles)*32);
  if(slot>=c.slots || !c.live_slots[slot])return;
  float best=c.old_values.data[c.old_values.index(vertex,slot)];
  for(uint64_t e=c.graph.in_row[vertex];e<c.graph.in_row[vertex+1];++e) {
    uint32_t source=c.graph.in_col[e];
    float old=c.old_values.data[c.old_values.index(source,slot)];
    if(!reachable(old,c.algorithm))continue;
    if(c.algorithm==Algorithm::BFS && old>=16777216.f) { atomicExch(c.error_flag,1); continue; }
    float candidate=relax(old,c.graph.in_weight[e],c.algorithm);
    best=c.algorithm==Algorithm::SSWP ? fmaxf(best,candidate) : fminf(best,candidate);
  }
  c.new_values.data[c.new_values.index(vertex,slot)]=best;
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
      for(int k=0;k<QueriesPerLane;++k)best[k]=identity(c.algorithm);
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
          if constexpr(Check) if(!((active>>(slot%64))&1ULL))continue;
          const float old=c.old_values.data[c.old_values.index(source,slot)];
          if(!reachable(old,c.algorithm))continue;
          if(c.algorithm==Algorithm::BFS && old>=16777216.f){atomicExch(c.error_flag,1);continue;}
          const float candidate=relax(old,weight,c.algorithm);
          best[k]=c.algorithm==Algorithm::SSWP?fmaxf(best[k],candidate):fminf(best[k],candidate);
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
          float reduced=identity(c.algorithm);
          for(int group=0;group<Groups;++group)
            reduced=c.algorithm==Algorithm::SSWP?
              fmaxf(reduced,partial[k][group*QueryLanes+query_lane_out]):
              fminf(reduced,partial[k][group*QueryLanes+query_lane_out]);
          const size_t position=c.new_values.index(target,slot);
          const float previous=c.old_values.data[c.old_values.index(target,slot)];
          if(c.algorithm==Algorithm::SSWP) {
            if(reduced>previous) {
              if constexpr(Blocks==1)c.new_values.data[position]=reduced;
              else reduce_max_float(c.new_values.data+position,reduced);
            }
          } else if(reduced<previous) {
            if constexpr(Blocks==1)c.new_values.data[position]=reduced;
            else reduce_min_float(c.new_values.data+position,reduced);
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
        if(!reachable(old,c.algorithm))continue;
        if(c.algorithm==Algorithm::BFS && old>=16777216.f) {
          atomicExch(c.error_flag,1);
          continue;
        }
        const float candidate=relax(old,weight,c.algorithm);
        best=c.algorithm==Algorithm::SSWP?fmaxf(best,candidate):fminf(best,candidate);
      }
      partial[warp][lane]=best;
      __syncthreads();
      if(warp==0 && enabled) {
        float reduced=partial[0][lane];
#pragma unroll
        for(int w=1;w<4;++w)
          reduced=c.algorithm==Algorithm::SSWP?fmaxf(reduced,partial[w][lane]):fminf(reduced,partial[w][lane]);
        const size_t target_base=size_t(tile>>3)*group_stride+size_t(target)*8;
        const size_t position=target_base+size_t(lane>>3)*group_stride+(lane&7);
        const float previous=c.old_values.data[position];
        if(c.algorithm==Algorithm::SSWP) {
          if(reduced>previous)c.new_values.data[position]=reduced;
        } else if(reduced<previous)c.new_values.data[position]=reduced;
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
      float old=c.old_values.data[c.old_values.index(v,s)];
      float next=c.new_values.data[c.new_values.index(v,s)];
      bool improved=c.algorithm==Algorithm::SSWP ? next>old : next<old;
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
  size_t count=size_t(v.vertices)*v.slots;
  init_kernel<<<(count+255)/256,256,0,stream>>>(v,a,count); check(cudaGetLastError());
}
void clear_mask(uint64_t* mask,uint32_t vertices,uint32_t words,cudaStream_t stream) {
  size_t count=size_t(vertices)*words;
  clear_kernel<<<(count+255)/256,256,0,stream>>>(mask,count); check(cudaGetLastError());
}
void activate(ValueView v,uint64_t* mask,const uint32_t* sources,const uint8_t* due,
              uint32_t slots,uint32_t words,Algorithm algorithm,cudaStream_t stream) {
  activate_kernel<<<(slots+255)/256,256,0,stream>>>(v,mask,sources,due,slots,words,algorithm); check(cudaGetLastError());
}
void shared_push(const Context& c) {
  // Launch for every possible frontier vertex because count is held on device.
  uint32_t warps=4; uint32_t blocks=std::min(65535u,(c.graph.vertices+warps-1)/warps);
  push_kernel<<<blocks,warps*32,0,c.stream>>>(c); check(cudaGetLastError());
}
void dense_pull(const Context& c) {
  uint32_t tiles=(c.slots+31)/32;
  uint64_t count=uint64_t(c.graph.vertices)*tiles*32;
  pull_kernel<<<(count+255)/256,256,0,c.stream>>>(c,tiles); check(cudaGetLastError());
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
}
