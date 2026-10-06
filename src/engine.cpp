#include "graphweft/engine.hpp"
#include "graphweft/checkpoint.hpp"
#include "graphweft/scheduler.hpp"
#include "graphweft/iteration_model.hpp"
#include "../third_party/puercgp/online_offset_evaluator.hxx"
#include <filesystem>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <memory>
#include <sstream>
#include <unordered_set>
#include <stdexcept>
namespace graphweft {
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point a) { return std::chrono::duration<double,std::milli>(Clock::now()-a).count(); }
void check(cudaError_t e,const char* action) { if(e!=cudaSuccess)throw std::runtime_error(std::string(action)+": "+cudaGetErrorString(e)); }
uint64_t mul(uint64_t a,uint64_t b) { if(a && b>UINT64_MAX/a)throw std::overflow_error("allocation size overflow");return a*b; }
uint64_t add(uint64_t a,uint64_t b) { if(b>UINT64_MAX-a)throw std::overflow_error("allocation size overflow");return a+b; }
struct KernelDescription {
  const char* family;
  uint32_t group_size=0,warps_per_block=0,blocks_per_vertex=0;
  bool check=false;
};
KernelDescription describe(KernelId id) {
  if(id==KernelId::SharedPush)return {"shared_push"};
  if(id==KernelId::DensePull)return {"dense_pull"};
  if(id==KernelId::AdaptivePush)return {"adaptive_push"};
  if(id==KernelId::GroupedG8Edge4Warp4Pull)return {"grouped_g8_edge4_warp4_pull",8,4,1,false};
  int raw=int(id),base=int(KernelId::PullCheckFreeBase);
  bool check=false;
  if(raw>=int(KernelId::PullCheckBase) && raw<int(KernelId::PullCheckBase)+pull_partition_count){
    base=int(KernelId::PullCheckBase);check=true;
  }
  if(raw>=base && raw<base+pull_partition_count){auto p=pull_partition(raw-base);return {"partition_pull",p.group_size,p.warps_per_block,p.blocks_per_vertex,check};}
  base=int(KernelId::PushPartitionBase);
  if(raw>=base && raw<base+push_partition_count){auto p=push_partition(raw-base);return {"partition_push",p.group_size,p.warps_per_block,p.blocks_per_vertex,false};}
  return {"unknown"};
}
template<class T> class Device {
 public:
  Device()=default;
  explicit Device(size_t n,const char* label):label_(label) { if(n)check(cudaMalloc(reinterpret_cast<void**>(&ptr_),mul(n,sizeof(T))),label); }
  ~Device(){ if(ptr_)cudaFree(ptr_); }
  Device(const Device&)=delete; Device& operator=(const Device&)=delete;
  T* get()const{return ptr_;}
  void upload(const std::vector<T>& v) { if(!v.empty())check(cudaMemcpy(ptr_,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice),label_); }
 private:T* ptr_=nullptr;const char* label_="device buffer";
};
class Event {
 public:
  explicit Event(bool enabled){if(enabled)check(cudaEventCreate(&event_),"cudaEventCreate");}
  ~Event(){if(event_)cudaEventDestroy(event_);}
  Event(const Event&)=delete;Event& operator=(const Event&)=delete;
  cudaEvent_t get()const{return event_;}
 private:cudaEvent_t event_=nullptr;
};
struct DeviceGraph {
  Device<uint64_t> row, in_row; Device<uint32_t> col,in_col; Device<float> weight,in_weight;
  GraphView view;
  explicit DeviceGraph(const HostGraph& g):row(g.row.size(),"out row"),col(g.col.size(),"out col"),
      weight(g.weight.size(),"out weight"),in_row(g.directed?g.incoming_row.size():0,"in row"),
      in_col(g.directed?g.incoming_col.size():0,"in col"),in_weight(g.directed?g.incoming_weight.size():0,"in weight") {
    row.upload(g.row);col.upload(g.col);weight.upload(g.weight);
    if(g.directed){in_row.upload(g.incoming_row);in_col.upload(g.incoming_col);in_weight.upload(g.incoming_weight);}
    view={g.vertices,g.edges(),row.get(),col.get(),weight.get(),
          g.directed?in_row.get():row.get(),g.directed?in_col.get():col.get(),g.directed?in_weight.get():weight.get()};
  }
};
uint32_t physical_slots(const Options& o,uint32_t q) {
  if(o.layout==Layout::VertexMajor)return q;
  if(!o.group_width)throw std::invalid_argument("group width must be positive");
  uint64_t p=(uint64_t(q)+o.group_width-1)/o.group_width*o.group_width;
  if(p>UINT32_MAX)throw std::overflow_error("padded slot count overflow");return uint32_t(p);
}
void validate(const HostGraph& g,const Options& o,const std::vector<Query>& queries) {
  if(o.frontier_build==FrontierBuildMode::Direct && o.frontier==FrontierMode::Stable)
    throw std::invalid_argument("direct frontier construction requires unordered frontier order");
  if(!g.vertices || g.row.size()!=uint64_t(g.vertices)+1 || g.row.back()!=g.edges() || g.weight.size()!=g.edges())throw std::invalid_argument("invalid graph");
  if(o.sort_by_score && o.predictor==Options::Predictor::CoreDistance && o.algorithm!=Algorithm::BFS)
    throw std::invalid_argument("core-distance predictor requires BFS");
  if(o.sort_by_score && o.predictor==Options::Predictor::WeightedBoundary && o.algorithm!=Algorithm::SSSP)
    throw std::invalid_argument("weighted-boundary predictor requires SSSP");
  if(o.phase_offsets && (o.landmarks==0 || o.max_offset>1024))throw std::invalid_argument("invalid phase evaluator options");
  if(o.capacity==0 || o.group_width==0 || o.memory_fraction<=0 || o.memory_fraction>1 || o.pull_threshold<0 || o.pull_threshold>1)throw std::invalid_argument("invalid options");
  if(o.push_grain>4 || (o.push_query_lanes!=1&&o.push_query_lanes!=2&&o.push_query_lanes!=4&&
      o.push_query_lanes!=8&&o.push_query_lanes!=16&&o.push_query_lanes!=32))throw std::invalid_argument("invalid Push mapping");
  if((o.push_mapping==Options::PushMapping::Adaptive || o.push_mapping==Options::PushMapping::Iteration) &&
     (o.layout!=Layout::Grouped || o.group_width!=32))
    throw std::invalid_argument("adaptive/iteration Push mapping requires --layout=grouped --group_width=32");
  if(o.eager_sssp_refill && !o.group_refill)
    throw std::invalid_argument("eager SSSP refill requires group refill");
  if(o.interference_aware_refill && !o.group_refill)
    throw std::invalid_argument("interference-aware refill requires group refill");
  if(o.interference_bridge_refill && !o.group_refill)
    throw std::invalid_argument("interference bridge refill requires group refill");
  if(o.interference_aware_refill && o.eager_sssp_refill)
    throw std::invalid_argument("interference-aware and eager SSSP refill are mutually exclusive");
  if(o.interference_bridge_refill && (o.eager_sssp_refill || o.interference_aware_refill))
    throw std::invalid_argument("interference bridge refill is mutually exclusive with other experimental refill policies");
  if(o.interference_bridge_refill && o.predictor!=Options::Predictor::ImportedKey)
    throw std::invalid_argument("interference bridge refill requires imported length-prediction keys");
  if(!o.refill_max_active_groups)
    throw std::invalid_argument("refill active-group budget must be positive");
  if(o.group_iteration_mapping && (!o.group_refill || o.push_mapping!=Options::PushMapping::Iteration))
    throw std::invalid_argument("group Iteration mapping requires group refill and iteration Push mapping");
  if((o.group_refill||o.same_algorithm_groups) && (o.capacity%o.group_width)!=0)throw std::invalid_argument("group scheduling requires capacity divisible by group width");
  if(o.group_refill && o.use_offsets)throw std::invalid_argument("group refill uses zero offsets");
  std::unordered_set<uint64_t> ids;
  for(const auto& q:queries){if(q.source>=g.vertices)throw std::invalid_argument("query source out of range");
    if(!ids.insert(q.id).second)throw std::invalid_argument("duplicate query ID");
    if(!std::isfinite(q.score))throw std::invalid_argument("nonfinite query score");
    if(q.algorithm<-1 || q.algorithm>int(Algorithm::SSWP))throw std::invalid_argument("invalid query algorithm");}
  bool has_sssp=o.algorithm==Algorithm::SSSP;
  for(const auto& q:queries)has_sssp|=q.algorithm==int(Algorithm::SSSP);
  if(has_sssp)for(float w:g.weight)if(w<0 || !std::isfinite(w))throw std::invalid_argument("SSSP requires nonnegative finite weights");
  if(o.selector==Options::Selector::Replay && o.replay.empty())throw std::invalid_argument("empty replay selector");
}
}
uint64_t AllocationPlan::total() const { uint64_t sum=0;for(auto [_,v]:bytes)sum=add(sum,v);return sum; }
AllocationPlan allocation_plan(const HostGraph& g,const Options& o,uint32_t q) {
  if(!q)throw std::invalid_argument("capacity must be positive");
  uint64_t v=g.vertices,e=g.edges(),words=(uint64_t(q)+63)/64,p=physical_slots(o,q);
  AllocationPlan a;
  a.bytes["topology_out_row"]=mul(v+1,8);
  a.bytes["topology_out_col"]=mul(e,4);
  a.bytes["topology_out_weight"]=mul(e,4);
  a.bytes["topology_in_row"]=g.directed?mul(v+1,8):0;
  a.bytes["topology_in_col"]=g.directed?mul(e,4):0;
  a.bytes["topology_in_weight"]=g.directed?mul(e,4):0;
  a.bytes["value_double"]=mul(mul(v,p),8);
  a.bytes["mask_double"]=mul(mul(v,words),16);
  a.bytes["frontier_double"]=mul(v,8);
  a.bytes["frontier_flags"]=mul(v,4);
  // Scan mode stores one uint32 count per slot; update-driven modes reuse the
  // same allocation as one uint64 active-query mask per 64 slots.
  a.bytes["slot_counts"]=std::max(mul(q,4),mul(words,8));
  a.bytes["slot_sources"]=mul(q,4);
  a.bytes["slot_live_due"]=mul(q,2);
  a.bytes["slot_reset"]=mul(q,1);
  // Grouped value initialization visits padded cells, so its per-cell slot
  // lookup must also be valid for padded physical slots.
  a.bytes["slot_algorithms"]=mul(p,sizeof(Algorithm));
  a.bytes["group_metadata"]=mul((q+o.group_width-1)/o.group_width,32);
  a.bytes["group_iteration_features"]=o.group_iteration_mapping?
    mul((q+o.group_width-1)/o.group_width,sizeof(uint32_t)+2*sizeof(uint64_t)):0;
  a.bytes["scalars"]=2*4+3*8+4;
  a.bytes["stable_temp"]=o.frontier==FrontierMode::Stable?stable_temp_bytes(g.vertices):0;
  a.bytes["adaptive_categories"]=o.push_mapping==Options::PushMapping::Adaptive?mul(v,1):0;
  a.bytes["adaptive_buckets"]=o.push_mapping==Options::PushMapping::Adaptive?mul(v,4):0;
  a.bytes["adaptive_counters"]=o.push_mapping==Options::PushMapping::Adaptive?
    adaptive_push_bucket_count*2*sizeof(uint32_t):0;
  return a;
}
uint32_t max_capacity(const HostGraph& g,const Options& o,uint64_t allowed,uint32_t limit) {
  if(!limit)return 0;uint32_t lo=0,hi=limit;
  while(lo<hi){uint32_t mid=lo+(hi-lo+1)/2;if(allocation_plan(g,o,mid).total()<=allowed)lo=mid;else hi=mid-1;}return lo;
}
std::vector<Query> load_queries(const std::string& path,uint32_t vertices,const std::string& graph_identity,bool require_identity,uint32_t expected_capacity) {
  std::ifstream f(path);if(!f)throw std::runtime_error("cannot open queries: "+path);
  std::vector<Query> out;std::string line;bool identity_seen=false;
  while(std::getline(f,line)){
    if(line.empty())continue;
    if(line[0]=='#'){
      constexpr const char* capacity_prefix="# capacity=";
      if(line.rfind(capacity_prefix,0)==0 && expected_capacity &&
         std::stoul(line.substr(11))!=expected_capacity)
        throw std::runtime_error("frozen plan capacity mismatch");
      constexpr const char* prefix="# graph_identity=";
      if(line.rfind(prefix,0)==0){
        identity_seen=true;
        if(!graph_identity.empty() && line.substr(17)!=graph_identity)throw std::runtime_error("query graph identity mismatch");
      }
      continue;
    }
    std::replace(line.begin(),line.end(),',',' ');
    std::istringstream in(line);Query q{};
    if(!(in>>q.id>>q.source))throw std::runtime_error("invalid query line");
    if(!(in>>q.score)){q.score=0;in.clear();}
    if(!(in>>q.offset)){q.offset=0;in.clear();}
    if(!(in>>q.feature_key)){q.feature_key=0;in.clear();}
    if(!(in>>q.algorithm)){q.algorithm=-1;in.clear();}
    if(!(in>>q.reference_rounds)){q.reference_rounds=0;in.clear();}
    if(q.source>=vertices)throw std::runtime_error("query source out of range");
    out.push_back(q);
  }
  if(require_identity && !identity_seen)throw std::runtime_error("frozen schedule requires # graph_identity=<hash> header");
  return out;
}
RunStats run(const HostGraph& g,std::vector<Query> queries,const Options& o,const ResultCallback& callback,const RoundCallback& round_callback,const DeviceRoundProbe& probe,const FingerprintCallback& fingerprint_callback) {
  auto task_start=Clock::now(); validate(g,o,queries);RunStats stats;
  if(queries.empty())return stats;
  size_t free_bytes=0,total_bytes=0;check(cudaMemGetInfo(&free_bytes,&total_bytes),"cudaMemGetInfo");
  uint64_t allowed=std::min<uint64_t>(free_bytes,uint64_t(double(total_bytes)*o.memory_fraction));
  auto required=allocation_plan(g,o,o.capacity);
  if(required.total()>allowed){auto max=max_capacity(g,o,allowed,uint32_t(std::min<size_t>(queries.size(),UINT32_MAX)));
    throw std::runtime_error("Q exceeds GPU budget; requested="+std::to_string(o.capacity)+" bytes="+std::to_string(required.total())+" allowed="+std::to_string(allowed)+" suggested_Q="+std::to_string(max));}
  auto plan_start=Clock::now();
  auto prediction_start=Clock::now();
  if(o.sort_by_score){
    if(o.predictor==Options::Predictor::CoreDistance){CoreDistanceProvider provider;provider.predict(g,queries);}
    else if(o.predictor==Options::Predictor::WeightedBoundary){WeightedBoundaryProvider provider;provider.predict(g,queries);}
    else {FrozenPredictionProvider provider;provider.predict(g,queries);}
  }
  stats.prediction_ms=elapsed(prediction_start);
  queries=(o.group_refill||o.same_algorithm_groups)?BatchPlanner::plan_groups(std::move(queries),o):BatchPlanner::plan(std::move(queries),o);
  std::unique_ptr<puercgp::scheduling::landmark_phase_index> phase_index;
  if(o.use_offsets && o.phase_offsets){
    auto phase_start=Clock::now();
    auto index=puercgp::scheduling::landmark_phase_index::build(g.row,g.col,
      int(std::min<uint32_t>(o.landmarks,g.vertices)));
    phase_index=std::make_unique<puercgp::scheduling::landmark_phase_index>(std::move(index));
    stats.prediction_ms+=elapsed(phase_start);
  }
  stats.planning_ms=elapsed(plan_start)-stats.prediction_ms;
  DeviceGraph dg(g);
  auto task_wall_start=Clock::now();
  uint32_t q=o.capacity,words=(q+63)/64,p=physical_slots(o,q);size_t cells=mul(g.vertices,p);
  Device<float> value0(cells,"value0"),value1(cells,"value1");
  Device<uint64_t> mask0(mul(g.vertices,words),"mask0"),mask1(mul(g.vertices,words),"mask1");
  Device<uint32_t> list0(g.vertices,"frontier0"),list1(g.vertices,"frontier1"),flags(g.vertices,"flags");
  const uint32_t slot_storage=std::max<uint32_t>(q,words*2);
  Device<uint32_t> count0(1,"count0"),count1(1,"count1"),slot_counts(slot_storage,"slot_counts"),sources(q,"sources");
  Device<uint8_t> live(q,"live"),due(q,"due");
  Device<uint8_t> reset(q,"reset");
  Device<Algorithm> slot_algorithms(p,"slot algorithms");
  Device<uint64_t> pair0(1,"pair0"),pair1(1,"pair1"),edge_pairs(1,"edge_pairs");
  const uint32_t group_count=q/o.group_width;
  Device<uint32_t> group_frontier_vertices(o.group_iteration_mapping?group_count:0,"group frontier vertices");
  Device<uint64_t> group_vertex_pairs(o.group_iteration_mapping?group_count:0,"group vertex pairs");
  Device<uint64_t> group_edge_pairs(o.group_iteration_mapping?group_count:0,"group edge pairs");
  Device<uint8_t> adaptive_categories(o.push_mapping==Options::PushMapping::Adaptive?g.vertices:0,"adaptive categories");
  Device<uint32_t> adaptive_buckets(o.push_mapping==Options::PushMapping::Adaptive?g.vertices:0,"adaptive buckets");
  Device<uint32_t> adaptive_counts(o.push_mapping==Options::PushMapping::Adaptive?adaptive_push_bucket_count:0,"adaptive counts");
  Device<uint32_t> adaptive_cursors(o.push_mapping==Options::PushMapping::Adaptive?adaptive_push_bucket_count:0,"adaptive cursors");
  Device<int> error(1,"error");
  Device<uint64_t> fingerprint_sum(q,"fingerprint sum"),fingerprint_xor(q,"fingerprint xor");
  Device<unsigned char> temp(o.frontier==FrontierMode::Stable?stable_temp_bytes(g.vertices):0,"stable_temp");
  const bool profile_kernel=o.profile_kernel || !o.round_metrics_path.empty();
  Event compare_start(o.profile_compare),compare_end(o.profile_compare);
  Event kernel_start(profile_kernel),kernel_end(profile_kernel);
  const bool update_driven_frontier=o.frontier_build!=FrontierBuildMode::Scan;
  Event fused_prepare_start(update_driven_frontier);
  Event fused_prepare_end(update_driven_frontier);
  float* old=value0.get();float* next=value1.get();uint64_t* current_mask=mask0.get();uint64_t* next_mask=mask1.get();
  uint32_t* current_list=list0.get();uint32_t* next_list=list1.get();uint32_t* current_count=count0.get();uint32_t* next_count=count1.get();
  uint64_t* current_pair=pair0.get();uint64_t* next_pair=pair1.get();
  cudaStream_t stream=nullptr;
  std::vector<float> host_values;
  std::vector<uint8_t> host_live(q),host_due(q),host_reset(q);
  std::vector<Algorithm> host_algorithms(p,o.algorithm);
  std::vector<size_t> slot_query(q,SIZE_MAX);
  std::vector<uint32_t> host_sources(q),host_slot_counts(q),start_round(q),completion(q);
  std::vector<double> activation_wall_ms(q,0.0);
  std::vector<uint64_t> host_active_words(words);
  std::vector<uint32_t> host_group_frontier_vertices(group_count);
  std::vector<uint64_t> host_group_vertex_pairs(group_count),host_group_edge_pairs(group_count);
  std::vector<uint8_t> group_launch_live(q);
  ConfiguredSelector selector(o);
  std::ofstream round_metrics;
  if(!o.round_metrics_path.empty()){
    auto parent=std::filesystem::path(o.round_metrics_path).parent_path();
    if(!parent.empty())std::filesystem::create_directories(parent);
    round_metrics.open(o.round_metrics_path,std::ios::trunc);
    if(!round_metrics)throw std::runtime_error("cannot write round metrics: "+o.round_metrics_path);
    round_metrics<<"batch,round,live_queries,kernel_id,kernel_family,group_size,warps_per_block,blocks_per_vertex,check,kernel_gpu_ms,adaptive_preparation_ms,adaptive_w1_vertices,adaptive_w2_vertices,adaptive_w4_vertices,adaptive_b2_vertices,adaptive_b4_vertices,frontier_vertices,vertex_pairs,edge_pairs,density,mean_active_queries,mean_degree,mean_edge_pairs,selector_ms,predicted_log_cost,predicted_relative_cost,iteration_model_version,group_mapping_launches,group_mapping_divergent,group_mappings\n";
  }
  bool checkpoint_saved=false;
  auto execution_start=Clock::now();
  uint32_t submission_round_base=0;
  for(size_t begin=0;begin<queries.size();begin+=o.group_refill?queries.size():q){
    uint32_t used=o.group_refill?q:uint32_t(std::min<size_t>(q,queries.size()-begin));++stats.batches;
    if(phase_index){
      auto prediction_start=Clock::now();
      std::vector<int> batch_sources;batch_sources.reserve(used);
      for(uint32_t s=0;s<used;++s)batch_sources.push_back(int(queries[begin+s].source));
      puercgp::scheduling::online_offset_evaluator evaluator(*phase_index,int(o.max_offset));
      auto offsets=evaluator.evaluate(batch_sources);
      for(uint32_t s=0;s<used;++s)queries[begin+s].offset=uint32_t(offsets.offsets[s]);
      stats.prediction_ms+=elapsed(prediction_start);
    }
    std::fill(host_live.begin(),host_live.end(),0);std::fill(host_due.begin(),host_due.end(),0);
    std::fill(host_sources.begin(),host_sources.end(),0);std::fill(start_round.begin(),start_round.end(),UINT32_MAX);
    std::fill(completion.begin(),completion.end(),0);std::fill(slot_query.begin(),slot_query.end(),SIZE_MAX);
    std::fill(activation_wall_ms.begin(),activation_wall_ms.end(),0.0);
    size_t next_query=begin;uint32_t pending=0,global_round=0,last_refill_round=0;
    uint32_t bridge_base=UINT32_MAX;bool bridge_attempted=false;
    auto fill_group=[&](uint32_t base){
      if(next_query>=queries.size())return uint32_t(0);
      Algorithm a=queries[next_query].algorithm<0?o.algorithm:Algorithm(queries[next_query].algorithm);
      uint32_t filled=0;
      while(filled<o.group_width && next_query<queries.size()){
        Algorithm next_a=queries[next_query].algorithm<0?o.algorithm:Algorithm(queries[next_query].algorithm);
        if(next_a!=a)break;
        uint32_t s=base+filled;slot_query[s]=next_query;host_sources[s]=queries[next_query].source;
        host_algorithms[s]=a;start_round[s]=UINT32_MAX;completion[s]=0;++filled;++next_query;++pending;
      }
      return filled;
    };
    if(o.group_refill){for(uint32_t base=0;base<q;base+=o.group_width)fill_group(base);}
    else for(uint32_t s=0;s<used;++s){slot_query[s]=begin+s;host_sources[s]=queries[begin+s].source;
      host_algorithms[s]=queries[begin+s].algorithm<0?o.algorithm:Algorithm(queries[begin+s].algorithm);++next_query;}
    if(!o.group_refill)pending=used;
    sources.upload(host_sources);slot_algorithms.upload(host_algorithms);
    auto t=Clock::now();
    initialize_values({old,g.vertices,p,o.group_width,o.layout},o.algorithm,slot_algorithms.get(),stream);
    clear_mask(current_mask,g.vertices,words,stream);clear_mask(next_mask,g.vertices,words,stream);
    check(cudaMemsetAsync(current_count,0,4,stream),"reset count");
    check(cudaStreamSynchronize(stream),"initialize batch");stats.initialization_ms+=elapsed(t);
    while(pending || (o.group_refill&&next_query<queries.size()) || std::any_of(host_live.begin(),host_live.begin()+used,[](uint8_t x){return x!=0;})){
      auto round_start=Clock::now();
      bool activated=false;
      for(uint32_t s=0;s<used;++s){
        if(slot_query[s]!=SIZE_MAX && start_round[s]==UINT32_MAX && (!o.use_offsets || queries[slot_query[s]].offset<=global_round)){
          host_due[s]=1;host_live[s]=1;start_round[s]=global_round;--pending;activated=true;
        } else host_due[s]=0;
      }
      if(activated){
        live.upload(host_live);due.upload(host_due);
        t=Clock::now();
        activate({old,g.vertices,p,o.group_width,o.layout},current_mask,sources.get(),due.get(),q,words,o.algorithm,slot_algorithms.get(),stream);
        FrontierContext fc{{old,g.vertices,p,o.group_width,o.layout},{next,g.vertices,p,o.group_width,o.layout},current_mask,
          flags.get(),current_list,current_count,current_pair,slot_counts.get(),live.get(),q,words,o.algorithm,o.frontier,
          temp.get(),o.frontier==FrontierMode::Stable?stable_temp_bytes(g.vertices):0,stream,slot_algorithms.get()};
        rebuild_frontier(fc);check(cudaStreamSynchronize(stream),"activate frontier");stats.frontier_ms+=elapsed(t);
        // Query latency starts when execution begins, after one-time graph and
        // device setup.  Those costs are not queueing delay and can vary by
        // seconds between otherwise identical large-graph processes.
        const double activation_timestamp_ms=elapsed(execution_start);
        for(uint32_t s=0;s<used;++s)if(host_due[s])activation_wall_ms[s]=activation_timestamp_ms;
      }
      uint32_t active=0;for(uint32_t s=0;s<used;++s)active+=host_live[s];
      if(!active){
        if(pending){uint32_t next_start=UINT32_MAX;for(uint32_t s=0;s<used;++s)if(slot_query[s]!=SIZE_MAX&&start_round[s]==UINT32_MAX)next_start=std::min(next_start,queries[slot_query[s]].offset);
          global_round=std::max(global_round+1,next_start);continue;}
        break;
      }
      if(!checkpoint_saved && !o.checkpoint_path.empty() && global_round==o.checkpoint_round){
        Checkpoint cp;cp.graph_identity=g.identity;cp.algorithm=o.algorithm;cp.layout=o.layout;
        cp.vertices=g.vertices;cp.slots=q;cp.physical_slots=p;cp.group_width=o.group_width;
        cp.words=words;cp.round=global_round;
        check(cudaMemcpy(&cp.frontier_count,current_count,4,cudaMemcpyDeviceToHost),"checkpoint frontier count");
        cp.old_values.resize(cells);cp.frontier_mask.resize(size_t(g.vertices)*words);
        cp.frontier.resize(cp.frontier_count);cp.live_slots=host_live;
        cp.slot_algorithms.assign(host_algorithms.begin(),host_algorithms.begin()+q);
        check(cudaMemcpy(cp.old_values.data(),old,cells*4,cudaMemcpyDeviceToHost),"checkpoint values");
        check(cudaMemcpy(cp.frontier_mask.data(),current_mask,cp.frontier_mask.size()*8,cudaMemcpyDeviceToHost),"checkpoint mask");
        if(cp.frontier_count)check(cudaMemcpy(cp.frontier.data(),current_list,cp.frontier_count*4,cudaMemcpyDeviceToHost),"checkpoint list");
        auto parent=std::filesystem::path(o.checkpoint_path).parent_path();
        if(!parent.empty())std::filesystem::create_directories(parent);
        save_checkpoint(cp,o.checkpoint_path);checkpoint_saved=true;
      }
      uint64_t pairs=0,vertex_pairs=0;
      uint32_t host_adaptive_counts[adaptive_push_bucket_count]={0,0,0,0,0};
      double round_adaptive_preparation_ms=0;
      const bool adaptive_features=o.push_mapping==Options::PushMapping::Adaptive &&
        o.selector!=Options::Selector::Pull && o.selector!=Options::Selector::Replay;
      const bool iteration_features=o.push_mapping==Options::PushMapping::Iteration &&
        o.selector!=Options::Selector::Pull && o.selector!=Options::Selector::Replay;
      uint32_t frontier_vertices=UINT32_MAX;
      t=Clock::now();
      if(adaptive_features)
        classify_edge_pairs(dg.view,current_mask,current_list,current_count,words,edge_pairs.get(),
                            adaptive_categories.get(),adaptive_counts.get(),stream);
      else count_edge_pairs(dg.view,current_mask,current_list,current_count,words,edge_pairs.get(),stream);
      if(o.group_iteration_mapping){
        check(cudaMemsetAsync(group_frontier_vertices.get(),0,group_count*sizeof(uint32_t),stream),"reset group frontier features");
        check(cudaMemsetAsync(group_vertex_pairs.get(),0,group_count*sizeof(uint64_t),stream),"reset group vertex features");
        check(cudaMemsetAsync(group_edge_pairs.get(),0,group_count*sizeof(uint64_t),stream),"reset group edge features");
        count_group_features(dg.view,current_mask,current_list,current_count,words,group_count,
                             group_frontier_vertices.get(),group_vertex_pairs.get(),group_edge_pairs.get(),stream);
        check(cudaMemcpyAsync(host_group_frontier_vertices.data(),group_frontier_vertices.get(),
                              group_count*sizeof(uint32_t),cudaMemcpyDeviceToHost,stream),"group frontier features");
        check(cudaMemcpyAsync(host_group_vertex_pairs.data(),group_vertex_pairs.get(),
                              group_count*sizeof(uint64_t),cudaMemcpyDeviceToHost,stream),"group vertex features");
        check(cudaMemcpyAsync(host_group_edge_pairs.data(),group_edge_pairs.get(),
                              group_count*sizeof(uint64_t),cudaMemcpyDeviceToHost,stream),"group edge features");
      }
      check(cudaMemcpyAsync(&pairs,edge_pairs.get(),8,cudaMemcpyDeviceToHost,stream),"edge pairs");
      check(cudaMemcpyAsync(&vertex_pairs,current_pair,8,cudaMemcpyDeviceToHost,stream),"vertex pairs");
      if(iteration_features)check(cudaMemcpyAsync(&frontier_vertices,current_count,4,cudaMemcpyDeviceToHost,stream),"iteration frontier count");
      if(adaptive_features)check(cudaMemcpyAsync(host_adaptive_counts,adaptive_counts.get(),sizeof(host_adaptive_counts),cudaMemcpyDeviceToHost,stream),"adaptive counts");
      check(cudaStreamSynchronize(stream),"features");
      const double feature_elapsed=elapsed(t);stats.feature_ms+=feature_elapsed;
      if(adaptive_features){stats.adaptive_preparation_ms+=feature_elapsed;round_adaptive_preparation_ms+=feature_elapsed;}
      double rho=g.edges()?double(pairs)/(double(g.edges())*active):0;
      t=Clock::now();
      KernelId chosen=selector.choose({pairs,active,rho,g.edges()!=0,vertex_pairs,
        iteration_features?frontier_vertices:0,g.vertices,g.edges()},stats.rounds);
      if(chosen==KernelId::DensePull ||
         chosen==KernelId::GroupedG8Edge4Warp4Pull ||
         (int(chosen)>=int(KernelId::PullCheckFreeBase) && int(chosen)<int(KernelId::PullCheckFreeBase)+pull_partition_count) ||
         (int(chosen)>=int(KernelId::PullCheckBase) && int(chosen)<int(KernelId::PullCheckBase)+pull_partition_count))
        ++stats.pull_rounds;
      else ++stats.push_rounds;
      std::vector<int> round_group_mappings;
      if(o.group_iteration_mapping && int(chosen)>=int(KernelId::PushPartitionBase) &&
         int(chosen)<int(KernelId::PushPartitionBase)+push_partition_count){
        round_group_mappings.assign(group_count,-1);
        for(uint32_t group=0;group<group_count;++group){
          uint32_t group_active=0;
          for(uint32_t s=group*o.group_width;s<(group+1)*o.group_width;++s)group_active+=host_live[s];
          if(!group_active)continue;
          const double group_density=g.edges()?double(host_group_edge_pairs[group])/
            (double(g.edges())*group_active):0;
          auto prediction=predict_iteration_push({host_group_edge_pairs[group],group_active,group_density,
            g.edges()!=0,host_group_vertex_pairs[group],host_group_frontier_vertices[group],g.vertices,g.edges()});
          round_group_mappings[group]=prediction.candidate;
        }
      }
      const double round_selector_ms=elapsed(t);stats.selector_ms+=round_selector_ms;
      if(chosen==KernelId::AdaptivePush){
        auto feature_start=Clock::now();
        scatter_adaptive_buckets(current_list,current_count,adaptive_categories.get(),adaptive_buckets.get(),
                                 adaptive_cursors.get(),g.vertices,host_adaptive_counts,stream);
        check(cudaStreamSynchronize(stream),"adaptive bucket scatter");
        const double scatter_ms=elapsed(feature_start);
        stats.feature_ms+=scatter_ms;stats.adaptive_preparation_ms+=scatter_ms;
        round_adaptive_preparation_ms+=scatter_ms;
      }
      uint32_t launch_frontier_size=UINT32_MAX;
      if(int(chosen)>=int(KernelId::PushPartitionBase) &&
         int(chosen)<int(KernelId::PushPartitionBase)+push_partition_count){
        if(iteration_features)launch_frontier_size=frontier_vertices;
        else {
          auto feature_start=Clock::now();
          check(cudaMemcpyAsync(&launch_frontier_size,current_count,sizeof(uint32_t),cudaMemcpyDeviceToHost,stream),"partition frontier size");
          check(cudaStreamSynchronize(stream),"partition launch features");
          stats.feature_ms+=elapsed(feature_start);
          frontier_vertices=launch_frontier_size;
        }
      }
      Context context{dg.view,{old,g.vertices,p,o.group_width,o.layout},{next,g.vertices,p,o.group_width,o.layout},
                      current_mask,current_list,current_count,live.get(),q,words,o.algorithm,stream,error.get(),launch_frontier_size,
                      slot_algorithms.get()};
      if(probe)probe(context,stats.batches-1,global_round);
      t=Clock::now();check(cudaMemcpyAsync(next,old,cells*sizeof(float),cudaMemcpyDeviceToDevice,stream),"value copy");
      check(cudaMemsetAsync(error.get(),0,4,stream),"error reset");
      check(cudaStreamSynchronize(stream),"copy");stats.copy_ms+=elapsed(t);
      FrontierContext fc{{old,g.vertices,p,o.group_width,o.layout},{next,g.vertices,p,o.group_width,o.layout},next_mask,
        flags.get(),next_list,next_count,next_pair,slot_counts.get(),live.get(),q,words,o.algorithm,o.frontier,
        temp.get(),o.frontier==FrontierMode::Stable?stable_temp_bytes(g.vertices):0,stream,slot_algorithms.get()};
      if(update_driven_frontier){
        check(cudaEventRecord(fused_prepare_start.get(),stream),"fused frontier prepare start");
        prepare_fused_frontier(fc);
        check(cudaEventRecord(fused_prepare_end.get(),stream),"fused frontier prepare end");
        context.frontier_output={next_mask,flags.get(),next_list,next_count,next_pair,slot_counts.get(),
          o.frontier_build==FrontierBuildMode::Direct,o.frontier_mask64};
      }
      t=Clock::now();
      if(profile_kernel)check(cudaEventRecord(kernel_start.get(),stream),"kernel event start");
      uint32_t round_group_launches=0;
      bool round_group_divergent=false;
      if(!round_group_mappings.empty()){
        std::vector<int> distinct;
        for(int candidate:round_group_mappings)if(candidate>=0 &&
            std::find(distinct.begin(),distinct.end(),candidate)==distinct.end())distinct.push_back(candidate);
        round_group_launches=uint32_t(distinct.size());round_group_divergent=distinct.size()>1;
        ++stats.group_mapping_rounds;stats.group_mapping_launches+=round_group_launches;
        stats.group_mapping_divergent_rounds+=round_group_divergent;
        for(int candidate:distinct){
          std::fill(group_launch_live.begin(),group_launch_live.end(),0);
          for(uint32_t group=0;group<group_count;++group)if(round_group_mappings[group]==candidate)
            std::copy(host_live.begin()+group*o.group_width,host_live.begin()+(group+1)*o.group_width,
                      group_launch_live.begin()+group*o.group_width);
          check(cudaMemcpyAsync(live.get(),group_launch_live.data(),q*sizeof(uint8_t),cudaMemcpyHostToDevice,stream),
                "group mapping live mask");
          launch(push_partition_id(candidate),context);
        }
        check(cudaMemcpyAsync(live.get(),host_live.data(),q*sizeof(uint8_t),cudaMemcpyHostToDevice,stream),
              "restore live mask");
      }else if(chosen==KernelId::AdaptivePush)adaptive_push(context,adaptive_buckets.get(),host_adaptive_counts);
      else launch(chosen,context);
      if(profile_kernel)check(cudaEventRecord(kernel_end.get(),stream),"kernel event end");
      check(cudaStreamSynchronize(stream),"kernel");stats.kernel_ms+=elapsed(t);
      if(update_driven_frontier){
        float ms=0;check(cudaEventElapsedTime(&ms,fused_prepare_start.get(),fused_prepare_end.get()),"fused frontier prepare elapsed");
        stats.frontier_ms+=ms;
      }
      float kernel_gpu_ms=0;
      if(profile_kernel){
        check(cudaEventElapsedTime(&kernel_gpu_ms,kernel_start.get(),kernel_end.get()),"kernel elapsed");
        stats.kernel_gpu_ms+=kernel_gpu_ms;
      }
      if(round_metrics.is_open()){
        auto d=describe(chosen);
        round_metrics<<(stats.batches-1)<<','<<global_round<<','<<active<<','<<int(chosen)<<','<<d.family<<','
          <<d.group_size<<','<<d.warps_per_block<<','<<d.blocks_per_vertex<<','<<int(d.check)<<','<<kernel_gpu_ms<<','
          <<round_adaptive_preparation_ms<<','<<host_adaptive_counts[0]<<','<<host_adaptive_counts[1]<<','
          <<host_adaptive_counts[2]<<','<<host_adaptive_counts[3]<<','<<host_adaptive_counts[4]<<','
          <<(frontier_vertices==UINT32_MAX?0:frontier_vertices)<<','<<vertex_pairs<<','<<pairs<<','<<rho<<','
          <<(frontier_vertices!=UINT32_MAX&&frontier_vertices?double(vertex_pairs)/frontier_vertices:0)<<','
          <<(vertex_pairs?double(pairs)/vertex_pairs:0)<<','
          <<(frontier_vertices!=UINT32_MAX&&frontier_vertices?double(pairs)/frontier_vertices:0)<<','
          <<round_selector_ms<<',';
        if(selector.used_iteration_model())round_metrics<<selector.iteration_prediction().log_cost<<','
          <<selector.iteration_prediction().relative_cost<<',';
        else round_metrics<<"nan,nan,";
        if(o.push_mapping==Options::PushMapping::Iteration)round_metrics<<iteration_model_version();
        round_metrics<<','<<round_group_launches<<','<<int(round_group_divergent)<<',';
        for(size_t i=0;i<round_group_mappings.size();++i){if(i)round_metrics<<'|';round_metrics<<round_group_mappings[i];}
        round_metrics<<'\n';
        if(!round_metrics)throw std::runtime_error("round metrics write failed: "+o.round_metrics_path);
      }
      int err=0;check(cudaMemcpy(&err,error.get(),4,cudaMemcpyDeviceToHost),"precision flag");
      if(err)throw std::runtime_error("BFS distance reached float32 exact-integer boundary 2^24");
      t=Clock::now();
      if(o.frontier_build==FrontierBuildMode::Scan)
        build_frontier(fc,compare_start.get(),compare_end.get());
      else if(o.frontier_build==FrontierBuildMode::Fused)finish_fused_frontier(fc);
      else finish_direct_frontier(fc);
      if(update_driven_frontier && o.frontier_mask64)
        check(cudaMemcpyAsync(host_active_words.data(),slot_counts.get(),size_t(words)*8,cudaMemcpyDeviceToHost,stream),"active query words");
      else
        check(cudaMemcpyAsync(host_slot_counts.data(),slot_counts.get(),size_t(q)*4,cudaMemcpyDeviceToHost,stream),"slot counts");
      check(cudaStreamSynchronize(stream),"frontier");stats.frontier_ms+=elapsed(t);
      const double completion_timestamp_ms=elapsed(execution_start);
      if(o.profile_compare && o.frontier_build==FrontierBuildMode::Scan){
        float ms=0;check(cudaEventElapsedTime(&ms,compare_start.get(),compare_end.get()),"compare elapsed");stats.compare_ms+=ms;
      }
      for(uint32_t s=0;s<used;++s)if(host_live[s] &&
          ((update_driven_frontier&&o.frontier_mask64)?
            ((host_active_words[s/64]>>(s%64)&1ULL)==0):host_slot_counts[s]==0)){
        host_live[s]=0;completion[s]=global_round-start_round[s]+1;
        const auto& query=queries[slot_query[s]];
        const double activation_timestamp_ms=activation_wall_ms[s];
        const double service_ms=std::max(0.0,completion_timestamp_ms-activation_timestamp_ms);
        stats.completions.push_back({query.id,query.source,host_algorithms[s],s,s/o.group_width,
          submission_round_base+start_round[s],submission_round_base+global_round+1,completion[s],
          activation_timestamp_ms,completion_timestamp_ms,activation_timestamp_ms,service_ms,
          completion_timestamp_ms});
      }
      stats.active_slot_rounds+=active;stats.capacity_slot_rounds+=q;
      live.upload(host_live);
      std::swap(old,next);std::swap(current_mask,next_mask);std::swap(current_list,next_list);
      std::swap(current_count,next_count);std::swap(current_pair,next_pair);
      if(round_callback){
        RoundSnapshot snap{};snap.batch_index=stats.batches-1;snap.global_round=global_round;
        snap.used_slots=used;snap.physical_slots=p;snap.words=words;snap.layout=o.layout;
        for(uint32_t s=0;s<used;++s)if(slot_query[s]!=SIZE_MAX)snap.query_ids.push_back(queries[slot_query[s]].id);
        snap.values.resize(cells);snap.mask.resize(size_t(g.vertices)*words);
        uint32_t count=0;
        check(cudaMemcpy(snap.values.data(),old,cells*4,cudaMemcpyDeviceToHost),"round values");
        check(cudaMemcpy(snap.mask.data(),current_mask,snap.mask.size()*8,cudaMemcpyDeviceToHost),"round mask");
        check(cudaMemcpy(&count,current_count,4,cudaMemcpyDeviceToHost),"round frontier count");
        snap.frontier.resize(count);
        if(count)check(cudaMemcpy(snap.frontier.data(),current_list,count*4,cudaMemcpyDeviceToHost),"round frontier");
        round_callback(snap);
      }
      // Count only waiting within a not-yet-complete group.  A completed
      // group waiting for a whole-batch barrier is intentionally excluded.
      for(uint32_t base=0;base<used;base+=o.group_width){
        bool any=false,done=true;
        uint64_t finished=0;
        for(uint32_t s=base;s<std::min<uint32_t>(used,base+o.group_width);++s)if(slot_query[s]!=SIZE_MAX){
          any=true;if(host_live[s]||!completion[s])done=false;else ++finished;
        }
        if(any&&!done)stats.completed_slot_rounds+=finished;
      }
      if(o.group_refill){
        std::vector<uint32_t> reclaimed;
        uint32_t occupied_groups=0;
        bool resident_algorithm[3]={false,false,false};
        for(uint32_t base=0;base<q;base+=o.group_width){
          bool any=false,done=true;
          Algorithm group_algorithm=o.algorithm;
          for(uint32_t s=base;s<base+o.group_width;++s)if(slot_query[s]!=SIZE_MAX){
            any=true;group_algorithm=host_algorithms[s];if(host_live[s] || !completion[s])done=false;
          }
          if(any){++occupied_groups;resident_algorithm[int(group_algorithm)]=true;
            if(done)reclaimed.push_back(base);}
        }
        // Different algorithms can have very different per-round costs and
        // convergence lengths.  Refilling only the early-finishing algorithm
        // desynchronizes the resident groups and made short queries contend
        // with a long query for many extra rounds.  Keep heterogeneous groups
        // in the same replacement wave; homogeneous groups still refill as
        // soon as an individual group completes.
        const uint32_t resident_algorithms=uint32_t(resident_algorithm[0])+uint32_t(resident_algorithm[1])+
          uint32_t(resident_algorithm[2]);
        // SSSP is also wave-scheduled: advancing one SSSP group ahead of its
        // peers changes the resident frontier mix enough to cost more kernel
        // time than refill can recover on the long-tail workloads.  BFS keeps
        // eager refill, where the shorter rounds do benefit from replenishment.
        const bool sssp_wave=!o.eager_sssp_refill && !o.interference_aware_refill &&
          !o.interference_bridge_refill && resident_algorithms==1 &&
          resident_algorithm[int(Algorithm::SSSP)];
        if((resident_algorithms>1 || sssp_wave) && reclaimed.size()<occupied_groups)
          reclaimed.clear();
        if(!reclaimed.empty()){
          auto recycle_start=Clock::now();
          if(o.copy_results_to_cpu){
            auto transfer_start=Clock::now();host_values.resize(cells);
            check(cudaMemcpy(host_values.data(),old,cells*sizeof(float),cudaMemcpyDeviceToHost),"reclaimed result transfer");
            ValueView view{host_values.data(),g.vertices,p,o.group_width,o.layout};
            for(uint32_t base:reclaimed)for(uint32_t s=base;s<base+o.group_width;++s)if(slot_query[s]!=SIZE_MAX){
              const auto& query=queries[slot_query[s]];
              QueryResult result{query.id,query.source,completion[s],std::vector<float>(g.vertices)};
              for(uint32_t v=0;v<g.vertices;++v)result.values[v]=host_values[view.index(v,s)];
              if(callback)callback(result);
            }
            stats.transfer_ms+=elapsed(transfer_start);
          }
          if(fingerprint_callback){
            std::vector<uint64_t> sums(o.group_width),xors(o.group_width);
            for(uint32_t base:reclaimed){
              fingerprint_values({old,g.vertices,p,o.group_width,o.layout},base,o.group_width,
                                 fingerprint_sum.get(),fingerprint_xor.get(),stream);
              check(cudaMemcpyAsync(sums.data(),fingerprint_sum.get(),o.group_width*sizeof(uint64_t),cudaMemcpyDeviceToHost,stream),"fingerprint sums");
              check(cudaMemcpyAsync(xors.data(),fingerprint_xor.get(),o.group_width*sizeof(uint64_t),cudaMemcpyDeviceToHost,stream),"fingerprint xors");
              check(cudaStreamSynchronize(stream),"fingerprint reclaimed group");
              for(uint32_t local=0;local<o.group_width;++local){uint32_t s=base+local;
                if(slot_query[s]!=SIZE_MAX){const auto& query=queries[slot_query[s]];
                  fingerprint_callback({query.id,query.source,completion[s],g.vertices,sums[local],xors[local]});}}
            }
          }
          std::fill(host_reset.begin(),host_reset.end(),0);
          const uint32_t resident_groups=occupied_groups-uint32_t(reclaimed.size());
          const bool round_was_pull=chosen==KernelId::DensePull ||
            chosen==KernelId::GroupedG8Edge4Warp4Pull ||
            (int(chosen)>=int(KernelId::PullCheckFreeBase) &&
             int(chosen)<int(KernelId::PullCheckFreeBase)+pull_partition_count) ||
            (int(chosen)>=int(KernelId::PullCheckBase) &&
             int(chosen)<int(KernelId::PullCheckBase)+pull_partition_count);
          const bool bridge_survives=bridge_base!=UINT32_MAX &&
            std::find(reclaimed.begin(),reclaimed.end(),bridge_base)==reclaimed.end();
          const bool old_cohort_completed=std::any_of(reclaimed.begin(),reclaimed.end(),
            [&](uint32_t base){return base!=bridge_base;});
          for(uint32_t base:reclaimed){
            for(uint32_t s=base;s<base+o.group_width;++s){host_reset[s]=1;host_live[s]=host_due[s]=0;
              slot_query[s]=SIZE_MAX;start_round[s]=UINT32_MAX;completion[s]=0;activation_wall_ms[s]=0;
              host_sources[s]=0;host_algorithms[s]=o.algorithm;}
          }
          if(bridge_base!=UINT32_MAX && !bridge_survives)bridge_base=UINT32_MAX;
          std::vector<uint32_t> available_bases;
          for(uint32_t base=0;base<q;base+=o.group_width)if(slot_query[base]==SIZE_MAX)
            available_bases.push_back(base);
          const uint32_t waiting_groups=uint32_t((queries.size()-next_query+o.group_width-1)/o.group_width);
          const uint32_t eligible_groups=std::min<uint32_t>(uint32_t(available_bases.size()),waiting_groups);
          uint64_t resident_prediction_key=0,candidate_prediction_key=0;
          for(uint32_t s=0;s<q;++s)if(host_live[s] && slot_query[s]!=SIZE_MAX)
            resident_prediction_key=std::max(resident_prediction_key,queries[slot_query[s]].feature_key);
          for(size_t i=next_query;i<std::min(queries.size(),next_query+o.group_width);++i)
            candidate_prediction_key=std::max(candidate_prediction_key,queries[i].feature_key);
          // Imported weighted-boundary keys are ordered with predicted-long
          // groups first.  Admit a bridge only if it is at least as long as
          // the lone resident group; short groups do not preempt a long tail.
          const bool bridge_compatible=candidate_prediction_key<=resident_prediction_key;
          uint32_t admission_budget=std::min<uint32_t>(uint32_t(reclaimed.size()),eligible_groups);
          bool opening_bridge=false;
          if(o.interference_aware_refill && resident_groups){
            if(round_was_pull)admission_budget=0;
            else if(resident_groups>=o.refill_max_active_groups)admission_budget=0;
            else admission_budget=std::min<uint32_t>(admission_budget,
              o.refill_max_active_groups-resident_groups);
          }else if(o.interference_bridge_refill){
            admission_budget=0;
            if(!resident_groups){
              admission_budget=eligible_groups;bridge_attempted=false;bridge_base=UINT32_MAX;
            }else if(bridge_survives && old_cohort_completed){
              // The old cohort drained while its one bridge group is still
              // alive.  Fill every vacancy to recover normal batch width.
              admission_budget=eligible_groups;bridge_attempted=false;bridge_base=UINT32_MAX;
            }else if(!bridge_attempted && resident_groups==1 && !round_was_pull &&
                     bridge_compatible && eligible_groups){
              admission_budget=1;opening_bridge=true;
            }
          }
          bool refilled=false;
          std::vector<uint32_t> refilled_bases;
          uint32_t admitted=0;
          for(uint32_t base:available_bases){
            if(admitted>=admission_budget)break;
            for(uint32_t s=base;s<base+o.group_width;++s)host_reset[s]=1;
            if(fill_group(base)){++admitted;++stats.group_refills;
              ++stats.refill_admitted_groups;last_refill_round=global_round+1;refilled=true;
              refilled_bases.push_back(base);}
          }
          if(opening_bridge && admitted){bridge_base=refilled_bases.front();bridge_attempted=true;}
          if((o.interference_aware_refill || o.interference_bridge_refill) && admitted<eligible_groups){
            const uint64_t deferred=eligible_groups-admitted;
            stats.refill_deferred_groups+=deferred;
            if(round_was_pull && resident_groups)stats.refill_deferred_pull_groups+=deferred;
            else if(o.interference_aware_refill && resident_groups>=o.refill_max_active_groups)
              stats.refill_deferred_capacity_groups+=deferred;
            else if(o.interference_bridge_refill && resident_groups==1 && !bridge_attempted &&
                    !bridge_compatible)stats.refill_deferred_incompatible_groups+=deferred;
          }
          live.upload(host_live);sources.upload(host_sources);slot_algorithms.upload(host_algorithms);
          // Only `old` survives into the next iteration: the normal round copy
          // overwrites every cell of `next`, and the next frontier publication
          // overwrites next_mask.  Reset only groups that receive new queries;
          // retired groups are never read again.  A refilled group is activated
          // at the top of the next loop, whose rebuild also subsumes the reclaim
          // rebuild.
          const bool live_after_reclaim=std::any_of(host_live.begin(),host_live.begin()+used,
                                                    [](uint8_t x){return x!=0;});
          for(uint32_t base:refilled_bases)
            reset_slot_range({old,g.vertices,p,o.group_width,o.layout},base,o.group_width,
                             host_algorithms[base],nullptr,stream);
          if(refilled || live_after_reclaim){
            reset.upload(host_reset);
            clear_slot_mask(current_mask,g.vertices,words,reset.get(),q,stream);
          }
          if(!refilled && live_after_reclaim){
            FrontierContext refill_fc{{old,g.vertices,p,o.group_width,o.layout},{next,g.vertices,p,o.group_width,o.layout},current_mask,
              flags.get(),current_list,current_count,current_pair,slot_counts.get(),live.get(),q,words,o.algorithm,o.frontier,
              temp.get(),o.frontier==FrontierMode::Stable?stable_temp_bytes(g.vertices):0,stream,slot_algorithms.get()};
            rebuild_frontier(refill_fc);
          }
          check(cudaStreamSynchronize(stream),"reclaim groups");
          stats.recycle_ms+=elapsed(recycle_start);
        }
      }
      const bool partition_pull=(int(chosen)>=int(KernelId::PullCheckFreeBase) &&
          int(chosen)<int(KernelId::PullCheckFreeBase)+pull_partition_count) ||
          (int(chosen)>=int(KernelId::PullCheckBase) &&
          int(chosen)<int(KernelId::PullCheckBase)+pull_partition_count) ||
          chosen==KernelId::GroupedG8Edge4Warp4Pull;
      spdlog::debug("scheduler round={} active={} edge_pairs={} density={} kernel={} kernel_id={}",global_round,active,pairs,rho,
                    chosen==KernelId::DensePull?"dense_pull":chosen==KernelId::SharedPush?"shared_push":partition_pull?"partition_pull":"partition_push",int(chosen));
      stats.round_ms+=elapsed(round_start);
      ++global_round;++stats.rounds;
    }
    if(o.copy_results_to_cpu && !o.group_refill){
      t=Clock::now();host_values.resize(cells);
      check(cudaMemcpy(host_values.data(),old,cells*sizeof(float),cudaMemcpyDeviceToHost),"result transfer");
      for(uint32_t s=0;s<used;++s){
        QueryResult result{queries[begin+s].id,host_sources[s],completion[s],std::vector<float>(g.vertices)};
        ValueView view{host_values.data(),g.vertices,p,o.group_width,o.layout};
        for(uint32_t v=0;v<g.vertices;++v)result.values[v]=host_values[view.index(v,s)];
        if(callback)callback(result);
      }
      stats.transfer_ms+=elapsed(t);
    }
    if(fingerprint_callback && !o.group_refill){
      std::vector<uint64_t> sums(used),xors(used);
      fingerprint_values({old,g.vertices,p,o.group_width,o.layout},0,used,
                         fingerprint_sum.get(),fingerprint_xor.get(),stream);
      check(cudaMemcpyAsync(sums.data(),fingerprint_sum.get(),used*sizeof(uint64_t),cudaMemcpyDeviceToHost,stream),"fingerprint sums");
      check(cudaMemcpyAsync(xors.data(),fingerprint_xor.get(),used*sizeof(uint64_t),cudaMemcpyDeviceToHost,stream),"fingerprint xors");
      check(cudaStreamSynchronize(stream),"fingerprint batch");
      for(uint32_t s=0;s<used;++s){const auto& query=queries[begin+s];
        fingerprint_callback({query.id,query.source,completion[s],g.vertices,sums[s],xors[s]});}
    }
    spdlog::info("scheduler batch={} queries={} rounds={}",stats.batches,used,global_round);
    if(o.group_refill)stats.final_drain_rounds=global_round-last_refill_round;
    submission_round_base+=global_round;
  }
  if(!o.checkpoint_path.empty() && !checkpoint_saved)
    throw std::runtime_error("checkpoint round was not reached: "+std::to_string(o.checkpoint_round));
  stats.execution_ms=elapsed(execution_start);
  stats.task_wall_ms=elapsed(task_wall_start);
  stats.workload_ms=stats.planning_ms+stats.task_wall_ms;
  if(!o.plan_output_path.empty()){
    auto parent=std::filesystem::path(o.plan_output_path).parent_path();
    if(!parent.empty())std::filesystem::create_directories(parent);
    std::ofstream out(o.plan_output_path);if(!out)throw std::runtime_error("cannot write plan output");
    out<<"# graph_identity="<<g.identity<<'\n';
    out<<"# capacity="<<o.capacity<<'\n';
    out<<"# id,source,score,offset,feature_key,algorithm,reference_rounds,batch,slot\n";
    for(size_t i=0;i<queries.size();++i){const auto& query=queries[i];
      out<<query.id<<','<<query.source<<','<<query.score<<','<<query.offset<<','<<query.feature_key<<','<<query.algorithm<<','
         <<query.reference_rounds<<','<<i/o.capacity<<','<<i%o.capacity<<'\n';
    }
    if(!out)throw std::runtime_error("plan output write failed");
  }
  if(!o.completion_output_path.empty()){
    auto parent=std::filesystem::path(o.completion_output_path).parent_path();
    if(!parent.empty())std::filesystem::create_directories(parent);
    std::ofstream out(o.completion_output_path);if(!out)throw std::runtime_error("cannot write completion metadata");
    out<<std::setprecision(17);
    out<<"query_id,source,algorithm,slot,group,activation_round,completion_round,waiting_rounds,service_rounds,submit_to_completion_rounds,activation_ms,completion_ms,waiting_ms,service_ms,submit_to_completion_ms\n";
    for(const auto& r:stats.completions)out<<r.id<<','<<r.source<<','<<int(r.algorithm)<<','<<r.slot<<','<<r.group<<','
      <<r.activation_round<<','<<r.completion_round<<','<<r.activation_round<<','<<r.service_rounds<<','<<r.completion_round<<','
      <<r.activation_ms<<','<<r.completion_ms<<','<<r.waiting_ms<<','<<r.service_ms<<','<<r.submit_to_completion_ms<<'\n';
  }
  if(!o.schedule_events_path.empty()){
    auto parent=std::filesystem::path(o.schedule_events_path).parent_path();
    if(!parent.empty())std::filesystem::create_directories(parent);
    std::ofstream out(o.schedule_events_path);if(!out)throw std::runtime_error("cannot write schedule events");
    out<<"event,round,query_id,slot,group,algorithm\n";
    for(const auto& r:stats.completions){
      out<<"activate,"<<r.activation_round<<','<<r.id<<','<<r.slot<<','<<r.group<<','<<int(r.algorithm)<<'\n';
      out<<"complete,"<<r.completion_round<<','<<r.id<<','<<r.slot<<','<<r.group<<','<<int(r.algorithm)<<'\n';
    }
  }
  stats.total_ms=elapsed(task_start);return stats;
}
}
