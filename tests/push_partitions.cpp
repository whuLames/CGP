#include "../kernel_lab/partition_probe.hpp"
#include <spdlog/spdlog.h>
using namespace graphweft;
int main(int argc,char**){
 try{
  spdlog::set_level(spdlog::level::warn);
  std::vector<uint32_t> src,dst;std::vector<float> weights;
  // Divergent degrees, many incoming updates, duplicates, self loops, unreachable vertex.
  for(uint32_t i=0;i<4097;++i){src.push_back(0);dst.push_back(1+i%37);weights.push_back(float(i%9));}
  for(uint32_t i=1;i<39;++i){src.push_back(i);dst.push_back((i*7)%39);weights.push_back(float(i%5));}
  auto g=HostGraph::from_edges(40,src,dst,weights,true);
  uint64_t rounds=0,candidates=0;
  std::vector<uint32_t> qs=argc>1?std::vector<uint32_t>{65}:std::vector<uint32_t>{1,8,31,32,33,63,64,65,127,128,129,256};
  for(auto algorithm:{Algorithm::BFS,Algorithm::SSSP,Algorithm::SSWP})
   for(auto layout:{Layout::VertexMajor,Layout::Grouped})for(auto q:qs){
    Options o;o.algorithm=algorithm;o.layout=layout;o.capacity=q;o.selector=Options::Selector::Push;
    o.frontier=q%2?FrontierMode::Stable:FrontierMode::Unordered;o.use_offsets=true;
    std::vector<Query> queries;for(uint32_t i=0;i<q+1;++i)queries.push_back({i,i%5==4?39:i%4,0,i%3});
    lab::PartitionProbe probe(g);probe.measure=false;probe.reverse_check=true;
    run(g,queries,o,{}, {},std::ref(probe));rounds+=probe.checked_rounds;candidates+=probe.checked_candidates;
   }
  auto negative=HostGraph::from_edges(4,{0,1,1,2,2},{1,0,2,1,2},{-2,-2,0,0,-4},false);
  for(auto a:{Algorithm::BFS,Algorithm::SSWP}){
    Options o;o.algorithm=a;o.capacity=33;o.selector=Options::Selector::Push;
    lab::PartitionProbe p(negative);p.measure=false;p.reverse_check=true;
    run(negative,{{0,0},{1,3}},o,{}, {},std::ref(p));rounds+=p.checked_rounds;candidates+=p.checked_candidates;
  }
  Options production;production.algorithm=Algorithm::SSSP;production.capacity=65;
  production.frontier=FrontierMode::Stable;production.selector=Options::Selector::Push;
  std::vector<Query> prod_queries{{0,0},{1,1},{2,39}};
  std::vector<RoundSnapshot> baseline;
  run(g,prod_queries,production,{},[&](const RoundSnapshot& state){baseline.push_back(state);});
  production.selector=Options::Selector::Replay;
  production.frontier_build=FrontierBuildMode::Fused;
  for(int k=0;k<push_partition_count;++k){
    production.replay={push_partition_id(k)};size_t round=0;
    auto stats=run(g,prod_queries,production,{},[&](const RoundSnapshot& state){
      if(round>=baseline.size() || state.values!=baseline[round].values || state.mask!=baseline[round].mask || state.frontier!=baseline[round].frontier)
        throw std::runtime_error("production partition trajectory mismatch");
      ++round;
    });
    if(round!=baseline.size() || stats.pull_rounds || stats.push_rounds!=stats.rounds)
      throw std::runtime_error("production partition round accounting");
  }
  std::cout<<"partition validation passed rounds="<<rounds<<" candidate_rounds="<<candidates<<" reversed_frontier=1 production_variants=30\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
