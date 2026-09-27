#pragma once
#include "graphweft/engine.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace graphweft::lab {
inline void checked(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
inline std::string candidate_name(int k){
  if(k==0)return "legacy_push";
  auto p=push_partition(k-1);
  return "q"+std::to_string(p.group_size)+(p.blocks_per_vertex==1?"_w"+std::to_string(p.warps_per_block):"_b"+std::to_string(p.blocks_per_vertex));
}
inline KernelId candidate_id(int k){return k?push_partition_id(k-1):KernelId::SharedPush;}
inline double quantile(std::vector<double> a,double q){
  std::sort(a.begin(),a.end());double p=q*(a.size()-1);size_t lo=size_t(p),hi=std::min(lo+1,a.size()-1);
  return a[lo]+(p-lo)*(a[hi]-a[lo]);
}
// The reference and old snapshot are host-only. No third V*Q device array is allocated.
class PartitionProbe {
  const HostGraph& graph;
  std::ofstream raw,summary,features,histogram;
  cudaEvent_t start=nullptr,stop=nullptr;
  float* chunk=nullptr;
  static constexpr size_t chunk_cells=1<<20;
  std::vector<float> reference,old_snapshot;
  std::vector<uint64_t> mask_snapshot;
  std::vector<uint32_t> list_snapshot;
  std::vector<uint8_t> live_snapshot;
  int expected_error=0;
  std::mt19937 rng{0x4757};
  void restore(const Context& c,size_t cells){
    checked(cudaMemcpyAsync(c.new_values.data,c.old_values.data,cells*sizeof(float),cudaMemcpyDeviceToDevice,c.stream));
    checked(cudaMemsetAsync(c.error_flag,0,sizeof(int),c.stream));
  }
  void exact(const void* device,const void* host,size_t bytes,const std::string& label){
    for(size_t offset=0;offset<bytes;offset+=chunk_cells*sizeof(float)){
      size_t n=std::min(chunk_cells*sizeof(float),bytes-offset);
      checked(cudaMemcpy(chunk,static_cast<const char*>(device)+offset,n,cudaMemcpyDeviceToHost));
      if(std::memcmp(chunk,static_cast<const char*>(host)+offset,n)!=0)
        throw std::runtime_error(label+" byte mismatch in chunk at "+std::to_string(offset));
    }
  }
  void verify_candidate(const Context& c,size_t cells,int k){
    restore(c,cells);launch(candidate_id(k),c);checked(cudaStreamSynchronize(c.stream));
    int error=0;checked(cudaMemcpy(&error,c.error_flag,sizeof(error),cudaMemcpyDeviceToHost));
    if(error!=expected_error)throw std::runtime_error(candidate_name(k)+" precision flag mismatch");
    exact(c.new_values.data,reference.data(),cells*sizeof(float),candidate_name(k));
  }
public:
  bool measure=true,reverse_check=false;
  uint64_t checked_rounds=0,checked_candidates=0;
  explicit PartitionProbe(const HostGraph& g,const std::string& output=""):graph(g){
    checked(cudaEventCreate(&start));checked(cudaEventCreate(&stop));
    checked(cudaMallocHost(reinterpret_cast<void**>(&chunk),chunk_cells*sizeof(float)));
    if(!output.empty()){
      std::filesystem::create_directories(output);
      raw.open(output+"/samples.csv");summary.open(output+"/rounds.csv");features.open(output+"/features.csv");histogram.open(output+"/histogram.csv");
      if(!raw || !summary || !features || !histogram)throw std::runtime_error("cannot open study output");
      raw<<"batch,round,candidate,group_size,warps_per_block,blocks_per_vertex,sample,order,kernel_ms\n";
      summary<<"batch,round,candidate,median_ms,iqr_over_median,samples,speedup_vs_legacy,exact_match\n";
      features<<"batch,round,U,Q_live,P,E_union,E_pair,max_degree,max_work,query_runs,probe_wall_ms\n";
      histogram<<"batch,round,degree_log2,active_queries,vertices,edges,edge_pairs\n";
      raw<<std::setprecision(10);summary<<std::setprecision(10);features<<std::setprecision(10);
    }
  }
  ~PartitionProbe(){if(chunk)cudaFreeHost(chunk);if(start)cudaEventDestroy(start);if(stop)cudaEventDestroy(stop);}
  void operator()(const Context& original,uint64_t batch,uint32_t round){
    auto wall_start=std::chrono::steady_clock::now();
    Context c=original;
    uint32_t count=0;checked(cudaMemcpy(&count,c.frontier_count,sizeof(count),cudaMemcpyDeviceToHost));
    c.frontier_size_hint=count;
    size_t cells=size_t(c.old_values.vertices)*c.old_values.slots;
    old_snapshot.resize(cells);reference.resize(cells);mask_snapshot.resize(size_t(c.graph.vertices)*c.words);
    list_snapshot.resize(count);live_snapshot.resize(c.slots);
    checked(cudaMemcpy(old_snapshot.data(),c.old_values.data,cells*sizeof(float),cudaMemcpyDeviceToHost));
    checked(cudaMemcpy(mask_snapshot.data(),c.frontier_mask,mask_snapshot.size()*sizeof(uint64_t),cudaMemcpyDeviceToHost));
    if(count)checked(cudaMemcpy(list_snapshot.data(),c.frontier,count*sizeof(uint32_t),cudaMemcpyDeviceToHost));
    checked(cudaMemcpy(live_snapshot.data(),c.live_slots,c.slots,cudaMemcpyDeviceToHost));
    uint64_t pairs=0,edges=0,active_pairs=0,max_degree=0,max_work=0,runs=0;
    uint32_t live=std::count(live_snapshot.begin(),live_snapshot.end(),uint8_t{1});
    std::map<std::pair<int,int>,std::array<uint64_t,3>> bins;
    for(uint32_t v:list_snapshot){
      uint64_t d=graph.row[v+1]-graph.row[v];uint32_t a=0;bool previous=false;
      for(uint32_t slot=0;slot<c.slots;++slot){
        bool enabled=live_snapshot[slot] && ((mask_snapshot[size_t(v)*c.words+slot/64]>>(slot%64))&1ULL);
        a+=enabled;runs+=enabled && !previous;previous=enabled;
      }
      uint64_t work=d*a;active_pairs+=a;edges+=d;pairs+=work;
      max_degree=std::max(max_degree,d);max_work=std::max(max_work,work);
      int degree_bin=d?64-__builtin_clzll(d):0;
      auto& b=bins[{degree_bin,int(a)}];++b[0];b[1]+=d;b[2]+=work;
    }
    restore(c,cells);launch(KernelId::SharedPush,c);checked(cudaStreamSynchronize(c.stream));
    checked(cudaMemcpy(&expected_error,c.error_flag,sizeof(int),cudaMemcpyDeviceToHost));
    checked(cudaMemcpy(reference.data(),c.new_values.data,cells*sizeof(float),cudaMemcpyDeviceToHost));
    for(int k=1;k<=push_partition_count;++k){verify_candidate(c,cells,k);++checked_candidates;}
    if(reverse_check && count){
      auto reversed=list_snapshot;std::reverse(reversed.begin(),reversed.end());
      checked(cudaMemcpy(const_cast<uint32_t*>(c.frontier),reversed.data(),count*sizeof(uint32_t),cudaMemcpyHostToDevice));
      for(int k=1;k<=push_partition_count;++k)verify_candidate(c,cells,k);
      checked(cudaMemcpy(const_cast<uint32_t*>(c.frontier),list_snapshot.data(),count*sizeof(uint32_t),cudaMemcpyHostToDevice));
    }
    std::array<std::vector<double>,push_partition_count+1> times;
    std::array<std::vector<int>,push_partition_count+1> sample_order;int next_order=0;
    if(measure){
      auto one=[&](int k,bool record){
        restore(c,cells);checked(cudaEventRecord(start,c.stream));launch(candidate_id(k),c);
        checked(cudaEventRecord(stop,c.stream));checked(cudaEventSynchronize(stop));float ms=0;
        checked(cudaEventElapsedTime(&ms,start,stop));
        if(record){times[k].push_back(ms);sample_order[k].push_back(next_order++);}
      };
      for(int k=0;k<=push_partition_count;++k)one(k,false);
      auto repetitions=[&](int n){
        std::vector<int> order(push_partition_count+1);std::iota(order.begin(),order.end(),0);
        for(int i=0;i<n;++i){std::shuffle(order.begin(),order.end(),rng);for(int k:order)one(k,true);}
      };
      repetitions(10);
      auto ratio=[&](int k){return (quantile(times[k],.75)-quantile(times[k],.25))/quantile(times[k],.5);};
      bool noisy=false;for(int k=0;k<=push_partition_count;++k)noisy|=ratio(k)>.1;
      if(noisy)repetitions(20);
      double baseline=quantile(times[0],.5);
      for(int k=0;k<=push_partition_count;++k){
        PushPartition p=k?push_partition(k-1):PushPartition{0,0,0};
        if(raw.is_open())for(size_t i=0;i<times[k].size();++i)
          raw<<batch<<','<<round<<','<<candidate_name(k)<<','<<p.group_size<<','<<p.warps_per_block<<','<<p.blocks_per_vertex<<','<<i<<','<<sample_order[k][i]<<','<<times[k][i]<<'\n';
        if(summary.is_open())summary<<batch<<','<<round<<','<<candidate_name(k)<<','<<quantile(times[k],.5)<<','<<ratio(k)<<','<<times[k].size()<<','<<baseline/quantile(times[k],.5)<<",1\n";
      }
    }
    exact(c.old_values.data,old_snapshot.data(),cells*sizeof(float),"old state");
    exact(c.frontier_mask,mask_snapshot.data(),mask_snapshot.size()*sizeof(uint64_t),"current mask");
    exact(c.frontier,list_snapshot.data(),list_snapshot.size()*sizeof(uint32_t),"current frontier");
    exact(c.live_slots,live_snapshot.data(),live_snapshot.size(),"live slots");
    uint32_t after_count=0;checked(cudaMemcpy(&after_count,c.frontier_count,4,cudaMemcpyDeviceToHost));
    if(after_count!=count)throw std::runtime_error("frontier count modified");
    ++checked_rounds;
    double wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-wall_start).count();
    if(features.is_open())features<<batch<<','<<round<<','<<count<<','<<live<<','<<active_pairs<<','<<edges<<','<<pairs<<','<<max_degree<<','<<max_work<<','<<runs<<','<<wall_ms<<'\n';
    if(histogram.is_open())for(auto& item:bins)histogram<<batch<<','<<round<<','<<item.first.first<<','<<item.first.second<<','<<item.second[0]<<','<<item.second[1]<<','<<item.second[2]<<'\n';
    if(raw.is_open()){raw.flush();summary.flush();features.flush();histogram.flush();}
    if(measure)std::cout<<"round="<<round<<" U="<<count<<" E_pair="<<pairs<<" checked=30 samples="<<times[0].size()<<" probe_s="<<wall_ms/1000<<std::endl;
  }
};
}
