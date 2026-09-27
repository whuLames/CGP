#include "graphweft/engine.hpp"
#include "graphweft/scheduler.hpp"
#include <algorithm>
#include <cmath>
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
            Options::Selector selector,bool offsets){
  Options o;o.algorithm=a;o.layout=layout;o.frontier=mode;o.capacity=q;o.group_width=8;
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
  }
  auto undirected=HostGraph::from_edges(6,{0,1,1,2,2,3,3,1,4,5},{1,0,2,1,3,2,3,1,5,4},
    {2,2,0,0,3,3,4,4,-2,-2},false);
  auto directed=HostGraph::from_edges(6,{0,1,0,2,2,3,4},{1,2,2,3,3,3,5},{2,0,8,3,4,4,1},true);
  if(undirected.directed || undirected.incoming_row.size() || directed.incoming_row.empty())throw std::runtime_error("graph view storage");
  for(const auto& g:{undirected,directed})for(auto a:{Algorithm::BFS,Algorithm::SSSP,Algorithm::SSWP}){
    if(a==Algorithm::SSSP && !g.directed)continue; // negative edge rejection has a separate assertion.
    for(auto layout:{Layout::VertexMajor,Layout::Grouped})for(auto mode:{FrontierMode::Unordered,FrontierMode::Stable}){
      for(auto selector:{Options::Selector::Push,Options::Selector::Pull,Options::Selector::Replay,Options::Selector::Threshold})
        verify(g,a,layout,mode,65,7,selector,true);
      for(uint32_t q:{1u,8u,32u,63u,64u,65u,127u,128u,129u,256u})
        verify(g,a,layout,mode,q,q+1,Options::Selector::Threshold,false);
    }
  }
  for(uint32_t n:{512u,1024u})verify(directed,Algorithm::BFS,Layout::Grouped,FrontierMode::Unordered,65,n,Options::Selector::Threshold,true);
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
  // Compare every synchronous state, multiword mask and frontier set with a CPU recurrence.
  for(auto a:{Algorithm::BFS,Algorithm::SSSP,Algorithm::SSWP})for(auto layout:{Layout::VertexMajor,Layout::Grouped})
    for(auto selector:{Options::Selector::Push,Options::Selector::Pull,Options::Selector::Replay}){
      Options o;o.algorithm=a;o.layout=layout;o.capacity=65;o.selector=selector;
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
            size_t index=layout==Layout::VertexMajor?size_t(v)*snap.physical_slots+slot:
              (size_t(slot/8)*directed.vertices+v)*8+slot%8;
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
