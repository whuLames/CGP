#include "graphweft/engine.hpp"
#include "graphweft/kernels.hpp"
#include <spdlog/spdlog.h>
#include <iostream>
#include <stdexcept>
#include <tuple>
using namespace graphweft;
int main() {
  try {
    spdlog::set_level(spdlog::level::warn);
    std::vector<uint32_t> src,dst;std::vector<float> weight;
    for(uint32_t i=0;i<1025;++i){src.push_back(i%9);dst.push_back(9+i%7);weight.push_back(float(i%11));}
    for(uint32_t i=0;i<18;++i){src.push_back(i);dst.push_back(i);weight.push_back(0);}
    for(uint32_t i=9;i<17;++i){src.push_back(i);dst.push_back((i+1)%17);weight.push_back(float(i%3));}
    auto graph=HostGraph::from_edges(19,src,dst,weight,true);
    uint64_t trajectories=0,rounds=0;
    for(auto algorithm:{Algorithm::BFS,Algorithm::SSSP,Algorithm::SSWP})
      for(auto layout:{Layout::VertexMajor,Layout::Grouped})
        for(uint32_t q:{1u,31u,32u,33u,63u,64u,65u,127u,128u,129u}) {
          Options options;options.algorithm=algorithm;options.layout=layout;
          options.capacity=q;options.selector=Options::Selector::Push;
          options.frontier=FrontierMode::Stable;options.use_offsets=true;
          std::vector<Query> queries;
          for(uint32_t i=0;i<q;++i)queries.push_back({i,i%10,0,i%3});
          std::vector<RoundSnapshot> reference;
          run(graph,queries,options,{},[&](const RoundSnapshot& s){reference.push_back(s);});
          options.selector=Options::Selector::Replay;
          options.frontier_build=FrontierBuildMode::Fused;
          for(int check=0;check<2;++check)for(int k=0;k<pull_partition_count;++k) {
            options.replay={pull_partition_id(k,check!=0)};size_t index=0;
            auto stats=run(graph,queries,options,{},[&](const RoundSnapshot& s){
              if(index>=reference.size() || s.values!=reference[index].values ||
                 s.mask!=reference[index].mask || s.frontier!=reference[index].frontier)
                throw std::runtime_error("Pull partition trajectory mismatch");
              ++index;
            });
            if(index!=reference.size() || stats.pull_rounds!=stats.rounds)
              throw std::runtime_error("Pull partition round accounting mismatch");
            ++trajectories;rounds+=stats.rounds;
          }
        }
    auto negative=HostGraph::from_edges(5,{0,0,1,1,2,3,3},{1,2,2,3,3,3,4},
                                        {-2,-7,-4,-3,-1,-6,-5},true);
    Options negative_options;negative_options.algorithm=Algorithm::SSWP;
    negative_options.capacity=65;negative_options.selector=Options::Selector::Push;
    negative_options.frontier=FrontierMode::Stable;
    std::vector<Query> negative_queries{{0,0},{1,1},{2,4}};
    std::vector<RoundSnapshot> negative_reference;
    run(negative,negative_queries,negative_options,{},[&](const RoundSnapshot& s){negative_reference.push_back(s);});
    negative_options.selector=Options::Selector::Replay;
    negative_options.frontier_build=FrontierBuildMode::Fused;
    for(int check=0;check<2;++check)for(int k=0;k<pull_partition_count;++k){
      negative_options.replay={pull_partition_id(k,check!=0)};size_t index=0;
      run(negative,negative_queries,negative_options,{},[&](const RoundSnapshot& s){
        if(index>=negative_reference.size() || s.values!=negative_reference[index].values)
          throw std::runtime_error("negative capacity Pull reduction mismatch");
        ++index;
      });
      if(index!=negative_reference.size())throw std::runtime_error("negative capacity round count mismatch");
      ++trajectories;rounds+=index;
    }
    // Exact experiment matrix on the adversarial SSSP graph above: Q=32/64/96,
    // grouped widths 8/16/32, both requested mappings, and early completion.
    auto partition_id=[](uint32_t group){
      for(int k=0;k<pull_partition_count;++k){auto p=pull_partition(k);
        if(p.group_size==group && p.warps_per_block==1 && p.blocks_per_vertex==1)
          return pull_partition_id(k,false);
      }
      throw std::runtime_error("required Pull partition is unavailable");
    };
    struct LogicalRound { std::vector<float> values;std::vector<uint64_t> mask;std::vector<uint32_t> frontier; };
    uint64_t matrix_configurations=0;
    for(uint32_t q:{32u,64u,96u}){
      std::vector<Query> queries;for(uint32_t s=0;s<q;++s)queries.push_back({s,s%19});
      std::vector<LogicalRound> reference_rounds;std::vector<QueryResult> reference_results;
      Options baseline;baseline.algorithm=Algorithm::SSSP;baseline.capacity=q;baseline.layout=Layout::VertexMajor;
      baseline.selector=Options::Selector::Replay;baseline.replay={partition_id(8)};baseline.frontier=FrontierMode::Stable;
      auto baseline_stats=run(graph,queries,baseline,[&](const QueryResult& r){reference_results.push_back(r);},
        [&](const RoundSnapshot& s){reference_rounds.push_back({s.values,s.mask,s.frontier});});
      std::vector<std::tuple<Layout,uint32_t,KernelId>> matrix{
        {Layout::VertexMajor,8,partition_id(32)},
        {Layout::Grouped,8,partition_id(8)},
        {Layout::Grouped,8,KernelId::GroupedG8Edge4Warp4Pull},
        {Layout::Grouped,16,partition_id(8)}};
      if(q!=32)matrix.push_back({Layout::Grouped,32,partition_id(32)});
      for(auto [layout,width,kernel]:matrix){
        Options candidate=baseline;candidate.layout=layout;candidate.group_width=width;candidate.replay={kernel};
        candidate.frontier_build=FrontierBuildMode::Fused;
        size_t round_index=0,result_index=0;
        auto stats=run(graph,queries,candidate,[&](const QueryResult& r){
          if(result_index>=reference_results.size() || r.id!=reference_results[result_index].id ||
             r.completion_local_round!=reference_results[result_index].completion_local_round ||
             r.values!=reference_results[result_index].values)
            throw std::runtime_error("experiment-matrix final result mismatch");
          ++result_index;
        },[&](const RoundSnapshot& s){
          if(round_index>=reference_rounds.size())throw std::runtime_error("experiment-matrix extra round");
          std::vector<float> logical(size_t(graph.vertices)*q);
          ValueView view{const_cast<float*>(s.values.data()),graph.vertices,s.physical_slots,width,layout};
          for(uint32_t v=0;v<graph.vertices;++v)for(uint32_t slot=0;slot<q;++slot)
            logical[size_t(v)*q+slot]=s.values[view.index(v,slot)];
          const auto& expected=reference_rounds[round_index++];
          if(logical!=expected.values || s.mask!=expected.mask || s.frontier!=expected.frontier)
            throw std::runtime_error("experiment-matrix round trajectory mismatch");
        });
        if(stats.rounds!=baseline_stats.rounds || round_index!=reference_rounds.size() ||
           result_index!=reference_results.size())throw std::runtime_error("experiment-matrix round/result count mismatch");
        ++matrix_configurations;
      }
    }
    std::cout<<"pull partition validation passed trajectories="<<trajectories
             <<" rounds="<<rounds<<" experiment_matrix_configurations="<<matrix_configurations<<"\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
