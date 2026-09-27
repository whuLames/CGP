#include "graphweft/checkpoint.hpp"
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
using namespace graphweft;
int main(int argc,char** argv) {
  try {
    if(argc!=6)throw std::invalid_argument("usage: capture_pull_rounds GRAPH QUERIES OUTPUT ROUND,ROUND,... GPU");
    if(cudaSetDevice(std::stoi(argv[5]))!=cudaSuccess)throw std::runtime_error("cudaSetDevice failed");
    auto graph=HostGraph::load(argv[1],true,true);
    auto queries=load_queries(argv[2],graph.vertices,graph.identity);
    std::set<uint32_t> requested;std::istringstream input(argv[4]);std::string item;
    while(std::getline(input,item,','))requested.insert(uint32_t(std::stoul(item)));
    std::filesystem::create_directories(argv[3]);
    Options options;options.algorithm=Algorithm::SSSP;options.capacity=32;
    options.selector=Options::Selector::Push;
    run(graph,queries,options,{}, {},[&](const Context& c,uint64_t batch,uint32_t round){
      if(batch || !requested.count(round))return;
      Checkpoint cp;cp.graph_identity=graph.identity;cp.algorithm=c.algorithm;
      cp.layout=c.old_values.layout;cp.vertices=graph.vertices;cp.slots=c.slots;
      cp.physical_slots=c.old_values.slots;cp.group_width=c.old_values.group_width;
      cp.words=c.words;cp.round=round;
      if(cudaMemcpy(&cp.frontier_count,c.frontier_count,4,cudaMemcpyDeviceToHost)!=cudaSuccess)
        throw std::runtime_error("frontier count copy failed");
      cp.old_values.resize(size_t(graph.vertices)*cp.physical_slots);
      cp.frontier_mask.resize(size_t(graph.vertices)*cp.words);
      cp.frontier.resize(cp.frontier_count);cp.live_slots.resize(c.slots);
      if(cudaMemcpy(cp.old_values.data(),c.old_values.data,cp.old_values.size()*4,cudaMemcpyDeviceToHost)!=cudaSuccess ||
         cudaMemcpy(cp.frontier_mask.data(),c.frontier_mask,cp.frontier_mask.size()*8,cudaMemcpyDeviceToHost)!=cudaSuccess ||
         cudaMemcpy(cp.live_slots.data(),c.live_slots,cp.live_slots.size(),cudaMemcpyDeviceToHost)!=cudaSuccess ||
         (cp.frontier_count && cudaMemcpy(cp.frontier.data(),c.frontier,cp.frontier.size()*4,cudaMemcpyDeviceToHost)!=cudaSuccess))
        throw std::runtime_error("checkpoint input copy failed");
      auto path=std::filesystem::path(argv[3])/("round-"+std::to_string(round)+".bin");
      save_checkpoint(cp,path.string());requested.erase(round);
      std::cout<<"saved "<<path<<" U="<<cp.frontier_count<<std::endl;
    });
    if(!requested.empty())throw std::runtime_error("some requested rounds were not reached");
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
