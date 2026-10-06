#include "graphweft/engine.hpp"
#include "graphweft/kernels.hpp"
#include <spdlog/spdlog.h>
#include <iostream>
#include <algorithm>
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
    // Every new VM Pull specialization must preserve the complete synchronous
    // trajectory, including frontier publication and completion rounds.
    uint64_t vm_trajectories=0;
    for(auto algorithm:{Algorithm::BFS,Algorithm::SSSP,Algorithm::SSWP})
      for(uint32_t q:{32u,64u,96u,128u,160u,192u,224u,256u}){
        Options baseline;baseline.algorithm=algorithm;baseline.layout=Layout::Grouped;
        baseline.group_width=32;baseline.capacity=q;baseline.selector=Options::Selector::Push;
        baseline.frontier=FrontierMode::Stable;baseline.frontier_build=FrontierBuildMode::Fused;
        std::vector<Query> queries;for(uint32_t s=0;s<q;++s)queries.push_back({s,s%10,0,s%3});
        std::vector<RoundSnapshot> expected;auto baseline_stats=run(
          graph,queries,baseline,{},[&](const RoundSnapshot& s){expected.push_back(s);});
        for(KernelId kernel:dense_pull_candidates){
          Options candidate=baseline;candidate.selector=Options::Selector::Replay;candidate.replay={kernel};
          size_t index=0;auto stats=run(graph,queries,candidate,{},[&](const RoundSnapshot& s){
            if(index>=expected.size() || s.values!=expected[index].values ||
               s.mask!=expected[index].mask || s.frontier!=expected[index].frontier)
              throw std::runtime_error("VM Pull trajectory mismatch");
            ++index;
          });
          if(index!=expected.size() || stats.rounds!=baseline_stats.rounds ||
             stats.pull_rounds!=stats.rounds || stats.completions.size()!=baseline_stats.completions.size())
            throw std::runtime_error("VM Pull completion/round accounting mismatch");
          ++vm_trajectories;
        }
      }
    // Mixed algorithms use the per-slot fallback and must retain the same
    // round-by-round values and frontier as Push for every dense candidate.
    {
      constexpr uint32_t q=64;
      Options baseline;baseline.capacity=q;baseline.selector=Options::Selector::Push;
      baseline.frontier=FrontierMode::Stable;baseline.frontier_build=FrontierBuildMode::Fused;
      std::vector<Query> queries;
      for(uint32_t s=0;s<q;++s){Query query{s,s%10,0,s%3};query.algorithm=int(s%3);queries.push_back(query);}
      std::vector<RoundSnapshot> expected;
      auto baseline_stats=run(graph,queries,baseline,{},[&](const RoundSnapshot& snapshot){expected.push_back(snapshot);});
      for(KernelId kernel:dense_pull_candidates){
        Options candidate=baseline;candidate.selector=Options::Selector::Replay;candidate.replay={kernel};
        size_t index=0;auto stats=run(graph,queries,candidate,{},[&](const RoundSnapshot& snapshot){
          if(index>=expected.size() || snapshot.values!=expected[index].values ||
             snapshot.mask!=expected[index].mask || snapshot.frontier!=expected[index].frontier)
            throw std::runtime_error("mixed dense Pull trajectory mismatch");
          ++index;
        });
        if(index!=expected.size() || stats.rounds!=baseline_stats.rounds)
          throw std::runtime_error("mixed dense Pull round count mismatch");
        ++vm_trajectories;
      }
    }
    // Exercise every legal frontier epilogue configuration for every dense
    // candidate. Unordered Direct compares sets and explicitly rejects
    // duplicate vertices; Stable configurations retain exact list order.
    {
      constexpr uint32_t q=64;
      std::vector<Query> queries;for(uint32_t s=0;s<q;++s)queries.push_back({s,s%10});
      struct Configuration { FrontierBuildMode build;FrontierMode mode;bool mask64; };
      const Configuration configurations[]={
        {FrontierBuildMode::Scan,FrontierMode::Stable,true},
        {FrontierBuildMode::Fused,FrontierMode::Stable,false},
        {FrontierBuildMode::Fused,FrontierMode::Stable,true},
        {FrontierBuildMode::Direct,FrontierMode::Unordered,false},
        {FrontierBuildMode::Direct,FrontierMode::Unordered,true}};
      Options reference_options;reference_options.capacity=q;reference_options.selector=Options::Selector::Push;
      reference_options.frontier=FrontierMode::Stable;reference_options.frontier_build=FrontierBuildMode::Scan;
      std::vector<RoundSnapshot> reference;
      run(graph,queries,reference_options,{},[&](const RoundSnapshot& snapshot){reference.push_back(snapshot);});
      for(const auto& configuration:configurations)for(KernelId kernel:dense_pull_candidates){
        Options candidate=reference_options;candidate.selector=Options::Selector::Replay;candidate.replay={kernel};
        candidate.frontier_build=configuration.build;candidate.frontier=configuration.mode;
        candidate.frontier_mask64=configuration.mask64;size_t index=0;
        run(graph,queries,candidate,{},[&](const RoundSnapshot& snapshot){
          if(index>=reference.size() || snapshot.values!=reference[index].values || snapshot.mask!=reference[index].mask)
            throw std::runtime_error("dense frontier epilogue trajectory mismatch");
          auto actual=snapshot.frontier,expected=reference[index].frontier;
          if(configuration.mode==FrontierMode::Unordered){
            std::sort(actual.begin(),actual.end());std::sort(expected.begin(),expected.end());
            if(std::adjacent_find(actual.begin(),actual.end())!=actual.end())
              throw std::runtime_error("dense Direct frontier contains a duplicate vertex");
          }
          if(actual!=expected)throw std::runtime_error("dense frontier set/order mismatch");
          ++index;
        });
        if(index!=reference.size())throw std::runtime_error("dense frontier epilogue round count mismatch");
        ++vm_trajectories;
      }
    }
    // Unsupported physical capacities must preserve explicit Replay priority
    // semantically while falling back to the general DensePull kernel.
    {
      constexpr uint32_t q=33;std::vector<Query> queries;
      for(uint32_t s=0;s<q;++s)queries.push_back({s,s%10});
      Options baseline;baseline.capacity=q;baseline.selector=Options::Selector::Push;baseline.frontier=FrontierMode::Stable;
      std::vector<RoundSnapshot> expected;run(graph,queries,baseline,{},[&](const RoundSnapshot& s){expected.push_back(s);});
      baseline.selector=Options::Selector::Replay;baseline.replay={dense_pull_candidates[0]};size_t index=0;
      run(graph,queries,baseline,{},[&](const RoundSnapshot& s){
        if(index>=expected.size() || s.values!=expected[index].values || s.mask!=expected[index].mask || s.frontier!=expected[index].frontier)
          throw std::runtime_error("unsupported dense capacity fallback mismatch");
        ++index;
      });
      if(index!=expected.size())throw std::runtime_error("unsupported dense capacity round count mismatch");
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
          ValueView view{const_cast<float*>(s.values.data()),graph.vertices,s.physical_slots,width,s.layout};
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
             <<" rounds="<<rounds<<" vm_trajectories="<<vm_trajectories
             <<" experiment_matrix_configurations="<<matrix_configurations<<"\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
