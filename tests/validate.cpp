#include "graphweft/engine.hpp"
#include "graphweft/scheduler.hpp"
#include "graphweft/iteration_model.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>
using namespace graphweft;
namespace {
std::vector<float> reference(const HostGraph& g,Algorithm a,uint32_t source){
  float missing=a==Algorithm::SSWP?-INFINITY:INFINITY;
  std::vector<float> old(g.vertices,missing),next;old[source]=a==Algorithm::SSWP?INFINITY:0.f;
  for(uint32_t round=0;round<g.vertices*2+2;++round){
    next=old;
    for(uint32_t u=0;u<g.vertices;++u)for(uint64_t e=g.row[u];e<g.row[u+1];++e){
      if(old[u]==missing)continue;
      float candidate=a==Algorithm::BFS?old[u]+1.f:a==Algorithm::SSSP?old[u]+g.weight[e]:std::min(old[u],g.weight[e]);
      auto& value=next[g.col[e]];
      value=a==Algorithm::SSWP?std::max(value,candidate):std::min(value,candidate);
    }
    if(next==old)return old;
    old.swap(next);
  }
  throw std::runtime_error("CPU reference did not converge");
}
void verify(const HostGraph& g,Algorithm a,Layout layout,FrontierMode mode,uint32_t q,uint32_t n,
            Options::Selector selector,bool offsets,FrontierBuildMode build=FrontierBuildMode::Scan,
            Options::PushMapping mapping=Options::PushMapping::Shared,uint32_t group_width=8,
            bool mask64=true){
  Options o;o.algorithm=a;o.layout=layout;o.frontier=mode;o.capacity=q;o.group_width=group_width;
  o.frontier_build=build;
  o.push_mapping=mapping;o.frontier_mask64=mask64;
  o.copy_results_to_cpu=true;o.selector=selector;o.replay={KernelId::SharedPush,KernelId::DensePull};
  o.use_offsets=offsets;
  std::vector<Query> queries;queries.reserve(n);
  for(uint32_t i=0;i<n;++i)queries.push_back({1000+i,i%g.vertices,double(i%5),offsets?i%4:0});
  std::map<uint64_t,QueryResult> results;
  auto stats=run(g,queries,o,[&](const QueryResult& r){results.emplace(r.id,r);});
  if(results.size()!=n || stats.batches!=(n+q-1)/q)throw std::runtime_error("batch/result count mismatch");
  for(const auto& query:queries){
    const auto& result=results.at(query.id);
    auto expected=reference(g,a,query.source);
    if(result.values!=expected){
      std::cerr<<"Mismatch id="<<query.id<<" algorithm="<<int(a)<<" Q="<<q<<" N="<<n
               <<" selector="<<int(selector)<<"\n";
      for(uint32_t v=0;v<g.vertices;++v)std::cerr<<v<<": "<<result.values[v]<<" expected "<<expected[v]<<"\n";
      throw std::runtime_error("GPU/CPU mismatch");
    }
    if(result.completion_local_round==0)throw std::runtime_error("missing completion round");
  }
}
}
int main(){
  {
    if(dense_pull_candidate_count!=21)throw std::runtime_error("dense Pull candidate count");
    for(KernelId id:dense_pull_candidates){
      const char* token=dense_pull_token(id);
      if(!token || parse_dense_pull_token(token)!=id || !is_dense_pull_kernel(id) || !is_pull_kernel(id))
        throw std::runtime_error("dense Pull token round trip");
    }
    bool invalid_dense_token=false;
    try{(void)parse_dense_pull_token("pull-dense-not-a-kernel");}
    catch(const std::invalid_argument&){invalid_dense_token=true;}
    if(!invalid_dense_token)throw std::runtime_error("invalid dense Pull token accepted");
    Options o;o.sort_by_score=true;
    std::vector<Query> input{{10,0,2},{11,0,3},{12,0,3},{13,0,1}};
    auto ordered=BatchPlanner::plan(input,o);
    if(ordered[0].id!=11 || ordered[1].id!=12 || ordered[2].id!=10 || ordered[3].id!=13)
      throw std::runtime_error("stable imported score order");
    o.predictor=Options::Predictor::ImportedKey;
    input[0].feature_key=5;input[1].feature_key=2;input[2].feature_key=2;input[3].feature_key=7;
    ordered=BatchPlanner::plan(input,o);
    if(ordered[0].id!=11 || ordered[1].id!=12 || ordered[2].id!=10 || ordered[3].id!=13)
      throw std::runtime_error("stable feature key order");
    o.selector=Options::Selector::Push;o.push_mapping=Options::PushMapping::Static;
    o.push_query_lanes=8;o.push_grain=2;
    if(ConfiguredSelector(o).choose({},0)!=push_partition_id(17))throw std::runtime_error("static Push mapping");
    o.push_mapping=Options::PushMapping::Degree;
    if(ConfiguredSelector(o).choose({1024,64,.1,true,4},0)!=push_partition_id(2))
      throw std::runtime_error("degree Push mapping");
    o.push_mapping=Options::PushMapping::Density;
    if(ConfiguredSelector(o).choose({1,1,.2,true,1},0)!=push_partition_id(22))
      throw std::runtime_error("density Push mapping");
    o.push_mapping=Options::PushMapping::Adaptive;
    if(ConfiguredSelector(o).choose({},0)!=KernelId::AdaptivePush)
      throw std::runtime_error("adaptive Push mapping");
    o.push_mapping=Options::PushMapping::Iteration;
    RoundFeatures iteration_features{100000,32,.1,true,1000,100,10000,500000};
    auto iteration=predict_iteration_push(iteration_features);
    if(iteration.candidate<0 || iteration.candidate>=push_partition_count ||
       ConfiguredSelector(o).choose(iteration_features,0)!=push_partition_id(iteration.candidate))
      throw std::runtime_error("iteration Push mapping");
    if(predict_iteration_push({0,32,.1,true,100,10,100,1000}).candidate!=0)
      throw std::runtime_error("iteration zero-edge fallback");
    std::array<double,30> tied{};tied.fill(1);tied[9]=0;tied[10]=0;
    if(iteration_argmin(tied)!=9)throw std::runtime_error("iteration deterministic tie break");
    auto clipped=predict_iteration_push({UINT64_MAX,UINT64_MAX,1e100,true,1,UINT32_MAX,1,UINT64_MAX});
    for(double x:clipped.clipped)if(!std::isfinite(x))throw std::runtime_error("iteration feature clipping");
    ConfiguredSelector hybrid(o);o.selector=Options::Selector::Threshold;o.pull_threshold=.05;
    if(hybrid.choose(iteration_features,0)!=KernelId::DensePull || hybrid.used_iteration_model())
      throw std::runtime_error("iteration model ran on Pull round");
    if(default_pull_kernel(32,100,5000)!=KernelId::VmFusedSerialSmemQ32 ||
       default_pull_kernel(64,100,1900)!=KernelId::VmFusedSerialSmemQ32 ||
       default_pull_kernel(64,100,2000)!=KernelId::VmParallelSmemShuffleQ16 ||
       default_pull_kernel(33,100,5000)!=KernelId::DensePull ||
       default_pull_kernel(288,100,5000)!=KernelId::DensePull)
      throw std::runtime_error("default VM Pull selection boundary");
    const uint64_t loads[]={0,1,256,257,1024,1025,4096,4097,16384,16385};
    const int expected[]={-1,0,0,1,1,2,2,3,3,4};
    for(int i=0;i<10;++i)if(adaptive_push_bucket(loads[i])!=expected[i])
      throw std::runtime_error("adaptive Push load boundary");
  }
  auto undirected=HostGraph::from_edges(6,{0,1,1,2,2,3,3,1,4,5},{1,0,2,1,3,2,3,1,5,4},
    {2,2,0,0,3,3,4,4,-2,-2},false);
  auto directed=HostGraph::from_edges(6,{0,1,0,2,2,3,4},{1,2,2,3,3,3,5},{2,0,8,3,4,4,1},true);
  if(undirected.directed || undirected.incoming_row.size() || directed.incoming_row.empty())throw std::runtime_error("graph view storage");
  {
    const auto path=std::filesystem::temp_directory_path()/"graphweft_round_oracle_validate.csv";
    Options o;o.algorithm=Algorithm::SSSP;o.capacity=32;o.group_width=32;o.layout=Layout::Grouped;
    o.selector=Options::Selector::Threshold;o.push_mapping=Options::PushMapping::Iteration;
    o.frontier_build=FrontierBuildMode::Fused;o.frontier_mask64=true;
    o.round_oracle_profile=true;o.round_oracle_output_path=path.string();
    std::vector<Query> queries;for(uint32_t i=0;i<32;++i)queries.push_back({9000+i,i%directed.vertices});
    auto stats=run(directed,queries,o);
    std::ifstream input(path);std::string line;uint64_t rows=0;
    while(std::getline(input,line))if(rows || line.rfind("batch,round,",0)!=0)++rows;
    std::filesystem::remove(path);
    if(rows!=stats.rounds*uint64_t(push_partition_count+2+dense_pull_candidate_count))
      throw std::runtime_error("round oracle row count mismatch");
    o.round_oracle_paired=true;
    auto paired=run(directed,queries,o);
    std::ifstream paired_input(path);rows=0;
    while(std::getline(paired_input,line))if(rows || line.rfind("batch,round,",0)!=0)++rows;
    paired_input.close();std::filesystem::remove(path);
    if(rows!=paired.rounds*2)throw std::runtime_error("paired round oracle row count mismatch");
  }
  {
    bool rejected=false;
    try{
      Options o;o.push_mapping=Options::PushMapping::Adaptive;o.layout=Layout::VertexMajor;
      run(directed,{{0,0}},o);
    }catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("adaptive Push accepted a non-grouped layout");
  }
  {
    bool rejected=false;
    try{
      Options o;o.push_mapping=Options::PushMapping::Iteration;o.layout=Layout::Grouped;o.group_width=8;
      run(directed,{{0,0}},o);
    }catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("iteration Push accepted non-G32 layout");
  }
  {
    bool rejected=false;
    try{
      Options o;o.frontier=FrontierMode::Stable;o.frontier_build=FrontierBuildMode::Direct;
      run(directed,{{0,0}},o);
    }catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("direct frontier silently accepted stable ordering");
  }
  {
    bool rejected=false;
    try{
      Options o;o.layout=Layout::Grouped;o.capacity=32;o.group_width=32;o.group_refill=true;
      o.interference_bridge_refill=true;
      run(directed,{{0,0}},o);
    }catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("bridge refill accepted missing prediction keys");
  }
  for(const auto& g:{undirected,directed})for(auto a:{Algorithm::BFS,Algorithm::SSSP,Algorithm::SSWP}){
    if(a==Algorithm::SSSP && !g.directed)continue; // negative edge rejection has a separate assertion.
    for(auto layout:{Layout::VertexMajor,Layout::Grouped})for(auto mode:{FrontierMode::Unordered,FrontierMode::Stable}){
      std::vector<FrontierBuildMode> builds{FrontierBuildMode::Scan,FrontierBuildMode::Fused};
      if(mode==FrontierMode::Unordered)builds.push_back(FrontierBuildMode::Direct);
      for(auto build:builds){
        for(auto selector:{Options::Selector::Push,Options::Selector::Pull,Options::Selector::Replay,Options::Selector::Threshold})
          verify(g,a,layout,mode,65,7,selector,true,build);
        for(uint32_t q:{1u,8u,32u,63u,64u,65u,127u,128u,129u,256u})
          verify(g,a,layout,mode,q,q+1,Options::Selector::Threshold,false,build);
      }
    }
  }
  for(auto build:{FrontierBuildMode::Scan,FrontierBuildMode::Fused,FrontierBuildMode::Direct})for(uint32_t n:{512u,1024u})
    verify(directed,Algorithm::BFS,Layout::Grouped,FrontierMode::Unordered,65,n,Options::Selector::Threshold,true,build);
  for(auto a:{Algorithm::BFS,Algorithm::SSSP,Algorithm::SSWP})
    for(uint32_t q:{1u,31u,32u,33u,63u,64u,65u})
      for(bool mask64:{false,true})
        for(auto mode:{FrontierMode::Stable,FrontierMode::Unordered}){
          verify(directed,a,Layout::Grouped,mode,q,q+1,Options::Selector::Push,true,
                 FrontierBuildMode::Scan,Options::PushMapping::Adaptive,32,mask64);
          verify(directed,a,Layout::Grouped,mode,q,q+1,Options::Selector::Push,true,
                 FrontierBuildMode::Fused,Options::PushMapping::Adaptive,32,mask64);
          if(mode==FrontierMode::Unordered)
            verify(directed,a,Layout::Grouped,mode,q,q+1,Options::Selector::Push,true,
                   FrontierBuildMode::Direct,Options::PushMapping::Adaptive,32,mask64);
        }
  for(auto a:{Algorithm::BFS,Algorithm::SSSP,Algorithm::SSWP})
    for(uint32_t q:{1u,31u,32u,33u,63u,64u,65u})
      verify(directed,a,Layout::Grouped,FrontierMode::Unordered,q,q+1,Options::Selector::Push,true,
             FrontierBuildMode::Direct,Options::PushMapping::Iteration,32,true);
  {
    Options o;o.algorithm=Algorithm::SSSP;o.capacity=4;o.use_offsets=true;o.phase_offsets=true;
    o.landmarks=4;o.copy_results_to_cpu=true;
    std::vector<Query> queries{{100,0},{101,1},{102,2},{103,4},{104,0}};
    std::map<uint64_t,QueryResult> results;
    auto stats=run(directed,queries,o,[&](const QueryResult& result){results.emplace(result.id,result);});
    if(stats.batches!=2 || results.size()!=queries.size())throw std::runtime_error("phase planning batch mismatch");
    for(const auto& query:queries)if(results.at(query.id).values!=reference(directed,Algorithm::SSSP,query.source))
      throw std::runtime_error("phase planning changed result");
  }
  for(auto predictor:{Options::Predictor::CoreDistance,Options::Predictor::WeightedBoundary}){
    Options o;o.algorithm=predictor==Options::Predictor::CoreDistance?Algorithm::BFS:Algorithm::SSSP;
    o.predictor=predictor;o.sort_by_score=true;o.capacity=4;o.copy_results_to_cpu=true;
    std::vector<Query> qs{{0,0},{1,1},{2,2},{3,4},{4,0}};
    std::map<uint64_t,QueryResult> results;
    run(directed,qs,o,[&](const QueryResult& r){results.emplace(r.id,r);});
    if(results.size()!=qs.size())throw std::runtime_error("predicted grouping result count");
    for(const auto& q:qs)if(results.at(q.id).values!=reference(directed,o.algorithm,q.source))
      throw std::runtime_error("predicted grouping changed result");
  }
  // Heterogeneous group refill: tail groups, simultaneous reclamation and a
  // physical slot carrying different algorithms over its lifetime.
  for(auto build:{FrontierBuildMode::Scan,FrontierBuildMode::Fused,FrontierBuildMode::Direct}){
    Options o;o.capacity=4;o.group_width=2;o.layout=Layout::Grouped;o.group_refill=true;
    o.copy_results_to_cpu=true;o.selector=Options::Selector::Push;o.frontier_build=build;
    std::vector<Query> qs;
    for(uint32_t i=0;i<5;++i){Query q{200+i,i%directed.vertices};q.algorithm=int(Algorithm::BFS);qs.push_back(q);}
    for(uint32_t i=0;i<5;++i){Query q{300+i,(i+1)%directed.vertices};q.algorithm=int(Algorithm::SSSP);qs.push_back(q);}
    for(uint32_t i=0;i<3;++i){Query q{400+i,(i+2)%directed.vertices};q.algorithm=int(Algorithm::SSWP);qs.push_back(q);}
    std::map<uint64_t,QueryResult> results;
    auto stats=run(directed,qs,o,[&](const QueryResult& r){results.emplace(r.id,r);});
    if(results.size()!=qs.size() || stats.completions.size()!=qs.size() || !stats.group_refills)
      throw std::runtime_error("heterogeneous refill accounting mismatch");
    for(const auto& r:stats.completions){
      if(r.activation_ms<0 || r.completion_ms<r.activation_ms || r.waiting_ms!=r.activation_ms ||
         r.service_ms<0 || r.submit_to_completion_ms!=r.completion_ms ||
         std::abs((r.waiting_ms+r.service_ms)-r.submit_to_completion_ms)>1e-6)
        throw std::runtime_error("completion wall-clock accounting mismatch");
    }
    for(const auto& q:qs){auto a=Algorithm(q.algorithm);
      if(results.at(q.id).values!=reference(directed,a,q.source))throw std::runtime_error("heterogeneous refill result mismatch");}
    std::map<uint32_t,std::vector<Algorithm>> history;
    for(const auto& r:stats.completions)history[r.slot].push_back(r.algorithm);
    bool reused_across_algorithms=false;
    for(const auto& [_,algorithms]:history)for(size_t i=1;i<algorithms.size();++i)
      reused_across_algorithms|=algorithms[i]!=algorithms[i-1];
    if(!reused_across_algorithms)throw std::runtime_error("physical slots were not reused across algorithms");
  }
  // Exercise each coalesced VM group-reset specialization, including a final
  // partially filled logical group admitted through refill.
  for(uint32_t width:{8u,16u,32u}){
    Options o;o.algorithm=Algorithm::BFS;o.capacity=2*width;o.group_width=width;
    o.layout=Layout::Grouped;o.group_refill=true;o.copy_results_to_cpu=true;
    o.selector=Options::Selector::Push;o.frontier_build=FrontierBuildMode::Direct;
    std::vector<Query> qs;for(uint32_t i=0;i<2*width+3;++i)qs.push_back({7000+i,i%directed.vertices});
    std::map<uint64_t,QueryResult> results;
    auto stats=run(directed,qs,o,[&](const QueryResult& r){results.emplace(r.id,r);});
    if(results.size()!=qs.size() || !stats.group_refills)
      throw std::runtime_error("VM group reset/refill accounting mismatch");
    for(const auto& query:qs)if(results.at(query.id).values!=reference(directed,Algorithm::BFS,query.source))
      throw std::runtime_error("VM group reset changed result");
  }
  // Interference-aware refill may leave completed physical groups vacant, but
  // must eventually admit every waiting group without changing query results.
  for(auto selector:{Options::Selector::Push,Options::Selector::Threshold}){
    Options o;o.algorithm=Algorithm::SSSP;o.capacity=4;o.group_width=2;o.layout=Layout::Grouped;
    o.group_refill=true;o.interference_aware_refill=true;o.refill_max_active_groups=1;
    o.copy_results_to_cpu=true;o.selector=selector;o.frontier_build=FrontierBuildMode::Direct;
    std::vector<Query> qs;for(uint32_t i=0;i<9;++i)qs.push_back({500+i,i%directed.vertices});
    std::map<uint64_t,QueryResult> results;
    auto stats=run(directed,qs,o,[&](const QueryResult& r){results.emplace(r.id,r);});
    if(results.size()!=qs.size() || stats.completions.size()!=qs.size() || !stats.refill_admitted_groups)
      throw std::runtime_error("interference-aware refill accounting mismatch");
    for(const auto& q:qs)if(results.at(q.id).values!=reference(directed,Algorithm::SSSP,q.source))
      throw std::runtime_error("interference-aware refill changed result");
  }
  for(auto selector:{Options::Selector::Push,Options::Selector::Threshold}){
    Options o;o.algorithm=Algorithm::SSSP;o.capacity=4;o.group_width=2;o.layout=Layout::Grouped;
    o.group_refill=true;o.interference_bridge_refill=true;o.copy_results_to_cpu=true;
    o.predictor=Options::Predictor::ImportedKey;
    o.selector=selector;o.frontier_build=FrontierBuildMode::Direct;
    std::vector<Query> qs;for(uint32_t i=0;i<11;++i){Query q{600+i,i%directed.vertices};
      q.feature_key=10-i;qs.push_back(q);}
    std::map<uint64_t,QueryResult> results;
    auto stats=run(directed,qs,o,[&](const QueryResult& r){results.emplace(r.id,r);});
    if(results.size()!=qs.size() || stats.completions.size()!=qs.size() || !stats.refill_admitted_groups)
      throw std::runtime_error("interference bridge refill accounting mismatch");
    for(const auto& q:qs)if(results.at(q.id).values!=reference(directed,Algorithm::SSSP,q.source))
      throw std::runtime_error("interference bridge refill changed result");
  }
  // Compare every synchronous state, multiword mask and frontier set with a CPU recurrence.
  for(auto a:{Algorithm::BFS,Algorithm::SSSP,Algorithm::SSWP})for(auto layout:{Layout::VertexMajor,Layout::Grouped})
    for(auto build:{FrontierBuildMode::Scan,FrontierBuildMode::Fused,FrontierBuildMode::Direct})
    for(auto selector:{Options::Selector::Push,Options::Selector::Pull,Options::Selector::Replay}){
      Options o;o.algorithm=a;o.layout=layout;o.capacity=65;o.selector=selector;
      o.frontier_build=build;
      o.replay={KernelId::SharedPush,KernelId::DensePull};
      std::vector<Query> qs;for(uint32_t i=0;i<65;++i)qs.push_back({i,i%directed.vertices});
      float missing=a==Algorithm::SSWP?-INFINITY:INFINITY;
      std::vector<std::vector<float>> previous(65,std::vector<float>(directed.vertices,missing));
      for(uint32_t s=0;s<65;++s)previous[s][qs[s].source]=a==Algorithm::SSWP?INFINITY:0.f;
      uint32_t snapshots=0;
      run(directed,qs,o,{},[&](const RoundSnapshot& snap){
        ++snapshots;std::vector<uint32_t> expected_frontier;
        std::vector<uint64_t> expected_mask(directed.vertices*2,0);
        for(uint32_t slot=0;slot<65;++slot){
          auto next=previous[slot];
          for(uint32_t u=0;u<directed.vertices;++u)for(uint64_t e=directed.row[u];e<directed.row[u+1];++e){
            if(previous[slot][u]==missing)continue;
            float candidate=a==Algorithm::BFS?previous[slot][u]+1.f:a==Algorithm::SSSP?
              previous[slot][u]+directed.weight[e]:std::min(previous[slot][u],directed.weight[e]);
            auto& target=next[directed.col[e]];
            target=a==Algorithm::SSWP?std::max(target,candidate):std::min(target,candidate);
          }
          for(uint32_t v=0;v<directed.vertices;++v){
            if(snap.layout!=Layout::VertexMajor)
              throw std::runtime_error("production snapshot is not vertex-major");
            size_t index=size_t(v)*snap.physical_slots+slot;
            if(snap.values[index]!=next[v])throw std::runtime_error("round value mismatch");
            bool improved=a==Algorithm::SSWP?next[v]>previous[slot][v]:next[v]<previous[slot][v];
            if(improved)expected_mask[size_t(v)*2+slot/64]|=1ULL<<(slot%64);
          }
          previous[slot]=std::move(next);
        }
        if(expected_mask!=snap.mask)throw std::runtime_error("round mask mismatch");
        for(uint32_t v=0;v<directed.vertices;++v)
          if(expected_mask[size_t(v)*2] || expected_mask[size_t(v)*2+1])expected_frontier.push_back(v);
        auto actual=snap.frontier;std::sort(actual.begin(),actual.end());
        if(actual!=expected_frontier)throw std::runtime_error("round frontier mismatch");
      });
      if(!snapshots)throw std::runtime_error("missing round trace");
    }
  bool rejected=false;
  try{HostGraph::from_edges(2,{0},{1},{1},false);}catch(const std::invalid_argument&){rejected=true;}
  if(!rejected)throw std::runtime_error("asymmetric graph accepted");
  rejected=false;
  try{Options o;o.algorithm=Algorithm::SSSP;run(undirected,{{0,0}},o);}catch(const std::invalid_argument&){rejected=true;}
  if(!rejected)throw std::runtime_error("negative weight accepted");
  std::cout<<"validation passed\n";
}
