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
  // Classification uses degree * active queries and preserves all five exact
  // threshold boundaries while skipping zero-load vertices.
  {
    const std::vector<uint64_t> degrees{0,1,256,257,1024,1025,4096,4097,16384,16385};
    std::vector<uint64_t> rows(degrees.size()+1),bits(degrees.size(),1);
    for(size_t i=0;i<degrees.size();++i)rows[i+1]=rows[i]+degrees[i];
    std::vector<uint32_t> vertices(degrees.size());
    for(uint32_t i=0;i<vertices.size();++i)vertices[i]=i;
    uint32_t vertex_count=uint32_t(vertices.size());
    D<uint64_t> d_rows(rows.size()),d_bits(bits.size()),edge_pairs(1);
    D<uint32_t> d_vertices(vertices.size()),d_count(1),category_counts(adaptive_push_bucket_count),
      bucket_vertices(vertices.size()),cursors(adaptive_push_bucket_count);
    D<uint8_t> categories(vertices.size());
    check(cudaMemcpy(d_rows.p,rows.data(),rows.size()*8,cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_bits.p,bits.data(),bits.size()*8,cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_vertices.p,vertices.data(),vertices.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_count.p,&vertex_count,4,cudaMemcpyHostToDevice));
    GraphView classification_graph{vertex_count,rows.back(),d_rows.p,nullptr,nullptr,d_rows.p,nullptr,nullptr};
    classify_edge_pairs(classification_graph,d_bits.p,d_vertices.p,d_count.p,1,edge_pairs.p,
                        categories.p,category_counts.p,nullptr);
    uint32_t host_counts[adaptive_push_bucket_count]{};
    check(cudaMemcpy(host_counts,category_counts.p,sizeof(host_counts),cudaMemcpyDeviceToHost));
    const uint32_t expected_counts[]={2,2,2,2,1};
    if(!std::equal(host_counts,host_counts+adaptive_push_bucket_count,expected_counts))
      throw std::runtime_error("adaptive classification boundary mismatch");
    scatter_adaptive_buckets(d_vertices.p,d_count.p,categories.p,bucket_vertices.p,cursors.p,
                             vertex_count,host_counts,nullptr);
    check(cudaDeviceSynchronize());
    std::vector<uint32_t> scattered(9);check(cudaMemcpy(scattered.data(),bucket_vertices.p,9*4,cudaMemcpyDeviceToHost));
    for(size_t offset=0;offset<8;offset+=2)std::sort(scattered.begin()+offset,scattered.begin()+offset+2);
    const std::vector<uint32_t> expected_vertices{1,2,3,4,5,6,7,8,9};
    if(scattered!=expected_vertices)throw std::runtime_error("adaptive bucket scatter mismatch");
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
  // Thousands of duplicate relaxations must create exactly one frontier pair.
  constexpr uint32_t duplicate_edges=4096;
  D<uint64_t> duplicate_row(3),duplicate_mask(2),duplicate_next_mask(2),duplicate_pair(1);
  D<uint32_t> duplicate_col(duplicate_edges),duplicate_current_list(1),duplicate_next_list(2),
    duplicate_current_count(1),duplicate_next_count(1),duplicate_flags(2),duplicate_slot_count(2);
  D<float> duplicate_weight(duplicate_edges),duplicate_old(2),duplicate_new(2);
  D<uint8_t> duplicate_live(1);D<int> duplicate_error(1);
  std::vector<uint64_t> duplicate_rows{0,duplicate_edges,duplicate_edges},duplicate_current_bits{1,0};
  std::vector<uint32_t> duplicate_columns(duplicate_edges,1);std::vector<float> duplicate_weights(duplicate_edges,1.f);
  std::vector<float> duplicate_values{0.f,INFINITY};
  check(cudaMemcpy(duplicate_row.p,duplicate_rows.data(),24,cudaMemcpyHostToDevice));
  check(cudaMemcpy(duplicate_mask.p,duplicate_current_bits.data(),16,cudaMemcpyHostToDevice));
  check(cudaMemcpy(duplicate_col.p,duplicate_columns.data(),duplicate_edges*4,cudaMemcpyHostToDevice));
  check(cudaMemcpy(duplicate_weight.p,duplicate_weights.data(),duplicate_edges*4,cudaMemcpyHostToDevice));
  check(cudaMemcpy(duplicate_old.p,duplicate_values.data(),8,cudaMemcpyHostToDevice));
  check(cudaMemcpy(duplicate_new.p,duplicate_values.data(),8,cudaMemcpyHostToDevice));
  check(cudaMemcpy(duplicate_current_list.p,&zero,4,cudaMemcpyHostToDevice));
  check(cudaMemcpy(duplicate_current_count.p,&one,4,cudaMemcpyHostToDevice));
  check(cudaMemcpy(duplicate_live.p,&live_one,1,cudaMemcpyHostToDevice));
  GraphView duplicate_graph{2,duplicate_edges,duplicate_row.p,duplicate_col.p,duplicate_weight.p,
    duplicate_row.p,duplicate_col.p,duplicate_weight.p};
  FrontierContext duplicate_frontier{{duplicate_old.p,2,1,8,Layout::VertexMajor},
    {duplicate_new.p,2,1,8,Layout::VertexMajor},duplicate_next_mask.p,duplicate_flags.p,
    duplicate_next_list.p,duplicate_next_count.p,duplicate_pair.p,duplicate_slot_count.p,
    duplicate_live.p,1,1,Algorithm::BFS,FrontierMode::Unordered,nullptr,0,nullptr};
  for(bool direct:{false,true}){
    check(cudaMemcpy(duplicate_new.p,duplicate_values.data(),8,cudaMemcpyHostToDevice));
    prepare_fused_frontier(duplicate_frontier);
    Context duplicate_context{duplicate_graph,{duplicate_old.p,2,1,8,Layout::VertexMajor},
      {duplicate_new.p,2,1,8,Layout::VertexMajor},duplicate_mask.p,duplicate_current_list.p,
      duplicate_current_count.p,duplicate_live.p,1,1,Algorithm::BFS,nullptr,duplicate_error.p,UINT32_MAX,nullptr,
      {duplicate_next_mask.p,duplicate_flags.p,duplicate_next_list.p,duplicate_next_count.p,
       duplicate_pair.p,duplicate_slot_count.p,direct,true}};
    launch(KernelId::SharedPush,duplicate_context);
    if(direct)finish_direct_frontier(duplicate_frontier);else finish_fused_frontier(duplicate_frontier);
    check(cudaDeviceSynchronize());
    uint64_t pair_count=0,next_bits[2]{};uint32_t slot_count=0,next_count=0,next_vertex=0;float improved_value=0;
    check(cudaMemcpy(&pair_count,duplicate_pair.p,8,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(next_bits,duplicate_next_mask.p,16,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&slot_count,duplicate_slot_count.p,4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&next_count,duplicate_next_count.p,4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&next_vertex,duplicate_next_list.p,4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&improved_value,duplicate_new.p+1,4,cudaMemcpyDeviceToHost));
    if(pair_count!=1 || slot_count!=1 || next_count!=1 || next_vertex!=1 || next_bits[1]!=1 || improved_value!=1.f)
      throw std::runtime_error(direct?"direct duplicate relaxation accounting mismatch":
        "fused duplicate relaxation accounting mismatch");
  }
  std::cout<<"frontier validation passed\n";
}
