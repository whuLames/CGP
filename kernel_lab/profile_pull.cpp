#include "graphweft/checkpoint.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace graphweft;
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Buffer {
  T* p=nullptr;explicit Buffer(size_t n){if(n)check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
  ~Buffer(){if(p)cudaFree(p);}
  void upload(const std::vector<T>& data){if(!data.empty())check(cudaMemcpy(p,data.data(),data.size()*sizeof(T),cudaMemcpyHostToDevice));}
};
KernelId candidate(const std::string& name) {
  if(name=="dense_pull")return KernelId::DensePull;
  if(name.rfind("pull-dense-",0)==0 || name.rfind("pull-vm-",0)==0)
    return parse_dense_pull_token(name);
  for(int k=0;k<pull_partition_count;++k) {
    auto p=pull_partition(k);
    std::string shape="q"+std::to_string(p.group_size)+
      (p.blocks_per_vertex==1?"_w"+std::to_string(p.warps_per_block):"_b"+std::to_string(p.blocks_per_vertex));
    if(name=="check_free_"+shape)return pull_partition_id(k,false);
    if(name=="check_"+shape)return pull_partition_id(k,true);
  }
  throw std::invalid_argument("unknown candidate");
}
}
int main(int argc,char** argv) {
  try {
    if(argc!=5)throw std::invalid_argument("usage: profile_pull GRAPH CHECKPOINT CANDIDATE GPU");
    check(cudaSetDevice(std::stoi(argv[4])));
    auto graph=HostGraph::load(argv[1],true,true);auto cp=load_checkpoint(argv[2]);
    if(cp.graph_identity!=graph.identity || cp.vertices!=graph.vertices)
      throw std::runtime_error("checkpoint graph identity mismatch");
    Buffer<uint64_t> row(graph.row.size()),in_row(graph.incoming_row.size()),mask(cp.frontier_mask.size());
    Buffer<uint32_t> col(graph.col.size()),in_col(graph.incoming_col.size()),list(cp.frontier_count),count(1);
    Buffer<float> weight(graph.weight.size()),in_weight(graph.incoming_weight.size()),old(cp.old_values.size()),next(cp.old_values.size());
    Buffer<uint8_t> live(cp.live_slots.size());Buffer<Algorithm> algorithms(cp.slot_algorithms.size());Buffer<int> error(1);
    row.upload(graph.row);col.upload(graph.col);weight.upload(graph.weight);
    in_row.upload(graph.incoming_row);in_col.upload(graph.incoming_col);in_weight.upload(graph.incoming_weight);
    mask.upload(cp.frontier_mask);list.upload(cp.frontier);live.upload(cp.live_slots);algorithms.upload(cp.slot_algorithms);old.upload(cp.old_values);
    check(cudaMemcpy(count.p,&cp.frontier_count,4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(next.p,old.p,cp.old_values.size()*4,cudaMemcpyDeviceToDevice));check(cudaMemset(error.p,0,4));
    GraphView gv{graph.vertices,graph.edges(),row.p,col.p,weight.p,in_row.p,in_col.p,in_weight.p};
    Context context{gv,{old.p,cp.vertices,cp.physical_slots,cp.group_width,cp.layout},
      {next.p,cp.vertices,cp.physical_slots,cp.group_width,cp.layout},mask.p,list.p,count.p,live.p,
      cp.slots,cp.words,cp.algorithm,nullptr,error.p,cp.frontier_count,algorithms.p};
    launch(candidate(argv[3]),context);check(cudaDeviceSynchronize());
    int flag=0;check(cudaMemcpy(&flag,error.p,4,cudaMemcpyDeviceToHost));
    if(flag)throw std::runtime_error("precision flag set");
    std::cout<<"profiled candidate="<<argv[3]<<" round="<<cp.round<<" U="<<cp.frontier_count
             <<" Q="<<cp.slots<<" graph="<<graph.identity<<std::endl;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
