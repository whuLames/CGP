#include "graphweft/scheduler.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <tuple>
namespace graphweft {
namespace {
uint64_t splitmix64(uint64_t x){
  x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
  x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);
}
const std::vector<uint64_t>& in_row(const HostGraph& g){return g.directed?g.incoming_row:g.row;}
const std::vector<uint32_t>& in_col(const HostGraph& g){return g.directed?g.incoming_col:g.col;}
const std::vector<float>& in_weight(const HostGraph& g){return g.directed?g.incoming_weight:g.weight;}
std::vector<uint32_t> reverse_bfs(const HostGraph& g,const std::vector<uint32_t>& hubs,uint32_t count){
  std::vector<uint32_t> distance(g.vertices,UINT32_MAX),queue;
  queue.reserve(g.vertices);
  for(uint32_t i=0;i<count;++i)if(distance[hubs[i]]==UINT32_MAX){distance[hubs[i]]=0;queue.push_back(hubs[i]);}
  const auto& row=in_row(g);const auto& col=in_col(g);
  for(size_t head=0;head<queue.size();++head){
    uint32_t v=queue[head];
    for(uint64_t e=row[v];e<row[v+1];++e){uint32_t pred=col[e];
      if(distance[pred]==UINT32_MAX){distance[pred]=distance[v]+1;queue.push_back(pred);}
    }
  }
  return distance;
}
struct Path {float distance;uint16_t hops;};
std::vector<Path> reverse_dijkstra(const HostGraph& g,uint32_t landmark){
  std::vector<Path> result(g.vertices,{INFINITY,UINT16_MAX});
  using Item=std::tuple<float,uint16_t,uint32_t>;
  std::priority_queue<Item,std::vector<Item>,std::greater<Item>> queue;
  result[landmark]={0,0};queue.emplace(0.f,0,landmark);
  const auto& row=in_row(g);const auto& col=in_col(g);const auto& weight=in_weight(g);
  while(!queue.empty()){
    auto [d,h,v]=queue.top();queue.pop();
    if(d!=result[v].distance || h!=result[v].hops)continue;
    for(uint64_t e=row[v];e<row[v+1];++e){
      float candidate=d+weight[e];uint16_t hops=h==UINT16_MAX-1?UINT16_MAX:uint16_t(h+1);
      if(!std::isfinite(candidate))continue;
      auto& target=result[col[e]];
      if(candidate<target.distance || (candidate==target.distance && hops<target.hops)){
        target={candidate,hops};queue.emplace(candidate,hops,col[e]);
      }
    }
  }
  return result;
}
}
void CoreDistanceProvider::predict(const HostGraph& g,std::vector<Query>& queries)const{
  if(!g.vertices)return;
  for(auto& query:queries)query.feature_key=0;
  const uint32_t tiers[]={1,16,256,4096};
  std::vector<uint64_t> indegree(g.vertices,0);
  for(uint32_t dst:g.col)++indegree[dst];
  std::vector<uint32_t> hubs(g.vertices);std::iota(hubs.begin(),hubs.end(),0);
  uint32_t count=std::min<uint32_t>(4096,g.vertices);
  std::partial_sort(hubs.begin(),hubs.begin()+count,hubs.end(),[&](uint32_t a,uint32_t b){
    uint64_t da=g.row[a+1]-g.row[a]+indegree[a],db=g.row[b+1]-g.row[b]+indegree[b];
    return da!=db?da>db:a<b;
  });
  hubs.resize(count);
  for(uint32_t tier:tiers){
    auto distance=reverse_bfs(g,hubs,std::min<uint32_t>(tier,count));
    for(auto& query:queries){
      uint32_t d=distance[query.source];
      // Encode unreachable as -1 and finite distances as their natural order.
      uint64_t part=d==UINT32_MAX?0:uint64_t(std::min<uint32_t>(d+1,65535));
      query.feature_key=(query.feature_key<<16)|part;
    }
  }
}
void WeightedBoundaryProvider::predict(const HostGraph& g,std::vector<Query>& queries)const{
  if(!g.vertices)return;
  std::vector<uint32_t> landmarks(g.vertices);std::iota(landmarks.begin(),landmarks.end(),0);
  uint32_t count=std::min<uint32_t>(256,g.vertices);
  std::partial_sort(landmarks.begin(),landmarks.begin()+count,landmarks.end(),[&](uint32_t a,uint32_t b){
    uint64_t da=g.row[a+1]-g.row[a],db=g.row[b+1]-g.row[b];
    if(da!=db)return da<db;
    uint64_t ah=splitmix64(a^0x4c454e475448ULL),bh=splitmix64(b^0x4c454e475448ULL);
    return ah!=bh?ah<bh:a<b;
  });
  landmarks.resize(count);
  std::vector<uint16_t> max_hops(g.vertices,0),reachable(g.vertices,0);
  std::vector<uint32_t> sum_hops(g.vertices,0);
  for(uint32_t landmark:landmarks){
    auto paths=reverse_dijkstra(g,landmark);
    for(uint32_t v=0;v<g.vertices;++v)if(std::isfinite(paths[v].distance)){
      max_hops[v]=std::max(max_hops[v],paths[v].hops);
      sum_hops[v]+=paths[v].hops;++reachable[v];
    }
  }
  for(auto& query:queries){
    uint32_t v=query.source;
    uint32_t mean=reachable[v]?std::min<uint32_t>(65535,uint64_t(sum_hops[v])*256/reachable[v]):0;
    query.feature_key=uint64_t(mean)+128ULL*max_hops[v];
    if(query.feature_key>65535)throw std::overflow_error("weighted-boundary score exceeds uint16");
  }
}
std::vector<Query> BatchPlanner::plan(std::vector<Query> queries,const Options& options){
  if(options.sort_by_score){
    if(options.predictor==Options::Predictor::Imported)
      std::stable_sort(queries.begin(),queries.end(),[](const Query& a,const Query& b){return a.score>b.score;});
    else
      std::stable_sort(queries.begin(),queries.end(),[](const Query& a,const Query& b){return a.feature_key<b.feature_key;});
  }
  return queries;
}
KernelId ConfiguredSelector::choose(const RoundFeatures& features,uint64_t round)const{
  switch(options_.selector){
    case Options::Selector::Push:return KernelId::SharedPush;
    case Options::Selector::Pull:return KernelId::DensePull;
    case Options::Selector::Replay:return options_.replay[round%options_.replay.size()];
    case Options::Selector::Threshold:
      if(!features.active_queries || !features.density_valid)return KernelId::SharedPush;
      return features.density>=options_.pull_threshold?KernelId::DensePull:KernelId::SharedPush;
  }
  return KernelId::SharedPush;
}
}
