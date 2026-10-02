#include "graphweft/checkpoint.hpp"
#include "graphweft/kernels.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using namespace graphweft;
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Buffer{
 T* p=nullptr;explicit Buffer(size_t n){if(n)check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
 ~Buffer(){if(p)cudaFree(p);}Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
 void upload(const std::vector<T>& v){if(!v.empty())check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
};
double quantile(std::vector<double> v,double at){
 std::sort(v.begin(),v.end());double pos=at*(v.size()-1);size_t lo=size_t(pos),hi=std::min(lo+1,v.size()-1);
 return v[lo]+(v[hi]-v[lo])*(pos-lo);
}
struct Sample{int candidate,index;double ms;};
}
int main(int argc,char** argv){
 try{
  if(argc<4 || argc>6)throw std::invalid_argument("usage: graphweft_kernel_lab GRAPH CHECKPOINT SAMPLES.csv [directed=0|1] [cold=0|1]");
  auto graph=HostGraph::load(argv[1],argc>=5 && std::string(argv[4])=="1");
  auto cp=load_checkpoint(argv[2]);
  bool cold=argc==6 && std::string(argv[5])=="1";
  if(cp.graph_identity!=graph.identity || cp.vertices!=graph.vertices)throw std::runtime_error("checkpoint graph identity mismatch");
  Buffer<uint64_t> row(graph.row.size()),in_row(graph.directed?graph.incoming_row.size():0),mask(cp.frontier_mask.size());
  Buffer<uint32_t> col(graph.col.size()),in_col(graph.directed?graph.incoming_col.size():0),list(cp.frontier_count),count(1);
  Buffer<float> weight(graph.weight.size()),in_weight(graph.directed?graph.incoming_weight.size():0),old(cp.old_values.size()),next(cp.old_values.size());
  Buffer<uint8_t> live(cp.live_slots.size());Buffer<Algorithm> algorithms(cp.slot_algorithms.size());Buffer<int> error(1);
  Buffer<unsigned char> eviction(cold?64ULL*1024*1024:0);
  row.upload(graph.row);col.upload(graph.col);weight.upload(graph.weight);
  if(graph.directed){in_row.upload(graph.incoming_row);in_col.upload(graph.incoming_col);in_weight.upload(graph.incoming_weight);}
  mask.upload(cp.frontier_mask);list.upload(cp.frontier);live.upload(cp.live_slots);algorithms.upload(cp.slot_algorithms);
  check(cudaMemcpy(count.p,&cp.frontier_count,4,cudaMemcpyHostToDevice));
  GraphView gv{graph.vertices,graph.edges(),row.p,col.p,weight.p,
    graph.directed?in_row.p:row.p,graph.directed?in_col.p:col.p,graph.directed?in_weight.p:weight.p};
  auto value_view=[&](float* data){return ValueView{data,cp.vertices,cp.physical_slots,cp.group_width,cp.layout};};
  auto context=[&](){return Context{gv,value_view(old.p),value_view(next.p),mask.p,list.p,count.p,live.p,
                                cp.slots,cp.words,cp.algorithm,nullptr,error.p,UINT32_MAX,algorithms.p};};
  auto restore=[&](){
    old.upload(cp.old_values);check(cudaMemcpy(next.p,old.p,cp.old_values.size()*4,cudaMemcpyDeviceToDevice));
    check(cudaMemset(error.p,0,4));
  };
  std::vector<std::vector<float>> outputs(2);
  for(int k=0;k<2;++k){
    restore();launch(k?KernelId::DensePull:KernelId::SharedPush,context());check(cudaDeviceSynchronize());
    int flag=0;check(cudaMemcpy(&flag,error.p,4,cudaMemcpyDeviceToHost));if(flag)throw std::runtime_error("BFS precision boundary");
    outputs[k].resize(cp.old_values.size());check(cudaMemcpy(outputs[k].data(),next.p,cp.old_values.size()*4,cudaMemcpyDeviceToHost));
  }
  if(outputs[0]!=outputs[1])throw std::runtime_error("push/pull output mismatch on checkpoint");
  auto shuffled=cp.frontier;std::reverse(shuffled.begin(),shuffled.end());
  list.upload(shuffled);restore();launch(KernelId::SharedPush,context());check(cudaDeviceSynchronize());
  std::vector<float> shuffled_output(cp.old_values.size());
  check(cudaMemcpy(shuffled_output.data(),next.p,shuffled_output.size()*4,cudaMemcpyDeviceToHost));
  if(shuffled_output!=outputs[0])throw std::runtime_error("frontier order changed push output");
  list.upload(cp.frontier);
  std::vector<float> verify_old(cp.old_values.size());check(cudaMemcpy(verify_old.data(),old.p,verify_old.size()*4,cudaMemcpyDeviceToHost));
  if(verify_old!=cp.old_values)throw std::runtime_error("kernel modified old value buffer");
  std::vector<uint64_t> verify_mask(cp.frontier_mask.size());
  check(cudaMemcpy(verify_mask.data(),mask.p,verify_mask.size()*8,cudaMemcpyDeviceToHost));
  if(verify_mask!=cp.frontier_mask)throw std::runtime_error("kernel modified frontier mask");
  std::vector<uint32_t> verify_list(cp.frontier_count);
  if(cp.frontier_count)check(cudaMemcpy(verify_list.data(),list.p,verify_list.size()*4,cudaMemcpyDeviceToHost));
  if(verify_list!=cp.frontier)throw std::runtime_error("kernel modified frontier list");
  std::vector<Sample> samples;std::vector<double> timings[2];
  cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
  auto one=[&](int k,bool record){
    restore();
    if(cold){check(cudaMemset(eviction.p,0x5a,64ULL*1024*1024));check(cudaDeviceSynchronize());}
    check(cudaEventRecord(begin));
    launch(k?KernelId::DensePull:KernelId::SharedPush,context());
    check(cudaEventRecord(end));check(cudaEventSynchronize(end));
    float ms=0;check(cudaEventElapsedTime(&ms,begin,end));
    if(record){samples.push_back({k,int(timings[k].size()),ms});timings[k].push_back(ms);}
  };
  one(0,false);one(1,false);
  std::mt19937 rng(0x4757);
  auto batch=[&](int repetitions){
    std::vector<int> order;for(int i=0;i<repetitions;++i){order.push_back(0);order.push_back(1);}
    std::shuffle(order.begin(),order.end(),rng);
    for(int k:order)one(k,true);
  };
  batch(10);
  auto ratio=[&](int k){return (quantile(timings[k],.75)-quantile(timings[k],.25))/quantile(timings[k],.5);};
  if(ratio(0)>.1 || ratio(1)>.1)batch(20);
  check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
  std::ofstream csv(argv[3]);if(!csv)throw std::runtime_error("cannot create samples CSV");
  csv<<"candidate,sample_index,kernel_ms,cache_mode\n";
  for(const auto& s:samples)csv<<(s.candidate?"dense_pull":"shared_push")<<','<<s.index<<','<<s.ms<<','<<(cold?"cold":"normal")<<'\n';
  std::cout<<"graph="<<graph.identity<<" round="<<cp.round<<" V="<<cp.vertices<<" Q="<<cp.slots
           <<" frontier="<<cp.frontier_count<<" cache_mode="<<(cold?"cold":"normal")<<" samples_per_candidate="<<timings[0].size()<<" exact_match=1\n";
  for(int k=0;k<2;++k)std::cout<<(k?"dense_pull":"shared_push")<<" median_ms="<<quantile(timings[k],.5)
                             <<" iqr_over_median="<<ratio(k)<<'\n';
  return 0;
 }catch(const std::exception& e){std::cerr<<"kernel_lab: "<<e.what()<<'\n';return 1;}
}
