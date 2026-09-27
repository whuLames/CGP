#include "graphweft/kernels.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace graphweft;
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct D{T* p=nullptr;D(size_t n){check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}~D(){cudaFree(p);}};
}
int main(){
  constexpr uint32_t V=7,Q=65,W=2;
  std::vector<float> old(V*Q,INFINITY),next=old;
  next[1*Q+0]=1;next[4*Q+64]=2;next[5*Q+63]=3;
  std::vector<uint8_t> live(Q,1);
  D<float> a(old.size()),b(next.size());D<uint64_t> mask(V*W),pairs(1);
  D<uint32_t> flags(V),list(V),count(1),slots(Q);
  check(cudaMemcpy(a.p,old.data(),old.size()*4,cudaMemcpyHostToDevice));
  check(cudaMemcpy(b.p,next.data(),next.size()*4,cudaMemcpyHostToDevice));
  D<uint8_t> d_live(Q);check(cudaMemcpy(d_live.p,live.data(),Q,cudaMemcpyHostToDevice));
  size_t temp_bytes=stable_temp_bytes(V);D<unsigned char> temp(temp_bytes);
  std::vector<uint64_t> baseline_mask;
  for(auto mode:{FrontierMode::Unordered,FrontierMode::Stable}){
    FrontierContext c{{a.p,V,Q,8,Layout::VertexMajor},{b.p,V,Q,8,Layout::VertexMajor},
      mask.p,flags.p,list.p,count.p,pairs.p,slots.p,d_live.p,Q,W,Algorithm::BFS,mode,temp.p,temp_bytes,nullptr};
    build_frontier(c);check(cudaDeviceSynchronize());
    uint32_t n=0;check(cudaMemcpy(&n,count.p,4,cudaMemcpyDeviceToHost));
    if(n!=3)throw std::runtime_error("wrong frontier count");
    std::vector<uint32_t> output(n);check(cudaMemcpy(output.data(),list.p,n*4,cudaMemcpyDeviceToHost));
    if(mode==FrontierMode::Unordered)std::sort(output.begin(),output.end());
    if(output!=std::vector<uint32_t>({1,4,5}))throw std::runtime_error("wrong frontier ordering or set");
    std::vector<uint64_t> bits(V*W);check(cudaMemcpy(bits.data(),mask.p,bits.size()*8,cudaMemcpyDeviceToHost));
    if(bits[1*W]!=1 || bits[4*W+1]!=1 || bits[5*W]!=(1ULL<<63))throw std::runtime_error("cross-word mask wrong");
    if(baseline_mask.empty())baseline_mask=bits;else if(bits!=baseline_mask)throw std::runtime_error("compression changed mask");
  }
  // A frontier distance at 2^24 must fail before float32 addition can round it away.
  D<uint64_t> row(3),in_row(3),edge_mask(2);D<uint32_t> col(1),in_col(1),one_list(1),one_count(1);
  D<float> weight(1),boundary_old(2),boundary_new(2);D<uint8_t> one_live(1);D<int> error(1);
  std::vector<uint64_t> rows{0,1,1},incoming_rows{0,0,1},masks{1,0};std::vector<float> vals{16777216.f,INFINITY};
  uint32_t destination=1,zero=0,one=1;float edge_weight=1.f;uint8_t live_one=1;
  check(cudaMemcpy(row.p,rows.data(),24,cudaMemcpyHostToDevice));
  check(cudaMemcpy(in_row.p,incoming_rows.data(),24,cudaMemcpyHostToDevice));
  check(cudaMemcpy(edge_mask.p,masks.data(),16,cudaMemcpyHostToDevice));
  check(cudaMemcpy(col.p,&destination,4,cudaMemcpyHostToDevice));
  check(cudaMemcpy(in_col.p,&zero,4,cudaMemcpyHostToDevice));
  check(cudaMemcpy(weight.p,&edge_weight,4,cudaMemcpyHostToDevice));
  check(cudaMemcpy(boundary_old.p,vals.data(),8,cudaMemcpyHostToDevice));
  check(cudaMemcpy(boundary_new.p,vals.data(),8,cudaMemcpyHostToDevice));
  check(cudaMemcpy(one_list.p,&zero,4,cudaMemcpyHostToDevice));
  check(cudaMemcpy(one_count.p,&one,4,cudaMemcpyHostToDevice));
  check(cudaMemcpy(one_live.p,&live_one,1,cudaMemcpyHostToDevice));
  GraphView g{2,1,row.p,col.p,weight.p,in_row.p,in_col.p,weight.p};
  Context boundary{g,{boundary_old.p,2,1,8,Layout::VertexMajor},{boundary_new.p,2,1,8,Layout::VertexMajor},
    edge_mask.p,one_list.p,one_count.p,one_live.p,1,1,Algorithm::BFS,nullptr,error.p};
  std::vector<KernelId> kernels{KernelId::SharedPush,KernelId::DensePull};
  for(int i=0;i<push_partition_count;++i)kernels.push_back(push_partition_id(i));
  for(int i=0;i<pull_partition_count;++i){
    kernels.push_back(pull_partition_id(i,false));
    kernels.push_back(pull_partition_id(i,true));
  }
  for(auto kernel:kernels){
    check(cudaMemset(error.p,0,4));launch(kernel,boundary);check(cudaDeviceSynchronize());
    int flag=0;check(cudaMemcpy(&flag,error.p,4,cudaMemcpyDeviceToHost));
    if(flag!=1)throw std::runtime_error("BFS precision boundary was not rejected");
  }
  check(cudaMemcpy(one_count.p,&zero,4,cudaMemcpyHostToDevice));
  for(int i=0;i<push_partition_count;++i){
    check(cudaMemset(error.p,0,4));launch(push_partition_id(i),boundary);check(cudaDeviceSynchronize());
    int flag=0;check(cudaMemcpy(&flag,error.p,4,cudaMemcpyDeviceToHost));
    if(flag)throw std::runtime_error("empty frontier executed relaxation");
  }
  std::cout<<"frontier validation passed\n";
}
