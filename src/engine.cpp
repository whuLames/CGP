#include "graphweft/engine.hpp"
#include "graphweft/checkpoint.hpp"
#include "graphweft/scheduler.hpp"
#include "../third_party/puercgp/online_offset_evaluator.hxx"
#include <filesystem>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
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
  if(!g.vertices || g.row.size()!=uint64_t(g.vertices)+1 || g.row.back()!=g.edges() || g.weight.size()!=g.edges())throw std::invalid_argument("invalid graph");
  if(o.sort_by_score && o.predictor==Options::Predictor::CoreDistance && o.algorithm!=Algorithm::BFS)
    throw std::invalid_argument("core-distance predictor requires BFS");
  if(o.sort_by_score && o.predictor==Options::Predictor::WeightedBoundary && o.algorithm!=Algorithm::SSSP)
    throw std::invalid_argument("weighted-boundary predictor requires SSSP");
  if(o.phase_offsets && (o.landmarks==0 || o.max_offset>1024))throw std::invalid_argument("invalid phase evaluator options");
  if(o.capacity==0 || o.group_width==0 || o.memory_fraction<=0 || o.memory_fraction>1 || o.pull_threshold<0 || o.pull_threshold>1)throw std::invalid_argument("invalid options");
  std::unordered_set<uint64_t> ids;
  for(const auto& q:queries){if(q.source>=g.vertices)throw std::invalid_argument("query source out of range");
    if(!ids.insert(q.id).second)throw std::invalid_argument("duplicate query ID");
    if(!std::isfinite(q.score))throw std::invalid_argument("nonfinite query score");}
  if(o.algorithm==Algorithm::SSSP)for(float w:g.weight)if(w<0 || !std::isfinite(w))throw std::invalid_argument("SSSP requires nonnegative finite weights");
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
  a.bytes["slot_counts"]=mul(q,4);
  a.bytes["slot_sources"]=mul(q,4);
  a.bytes["slot_live_due"]=mul(q,2);
  a.bytes["scalars"]=2*4+3*8+4;
  a.bytes["stable_temp"]=o.frontier==FrontierMode::Stable?stable_temp_bytes(g.vertices):0;
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
    if(q.source>=vertices)throw std::runtime_error("query source out of range");
    out.push_back(q);
  }
  if(require_identity && !identity_seen)throw std::runtime_error("frozen schedule requires # graph_identity=<hash> header");
  return out;
}
RunStats run(const HostGraph& g,std::vector<Query> queries,const Options& o,const ResultCallback& callback,const RoundCallback& round_callback,const DeviceRoundProbe& probe) {
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
  queries=BatchPlanner::plan(std::move(queries),o);
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
  Device<uint32_t> count0(1,"count0"),count1(1,"count1"),slot_counts(q,"slot_counts"),sources(q,"sources");
  Device<uint8_t> live(q,"live"),due(q,"due");
  Device<uint64_t> pair0(1,"pair0"),pair1(1,"pair1"),edge_pairs(1,"edge_pairs");
  Device<int> error(1,"error");
  Device<unsigned char> temp(o.frontier==FrontierMode::Stable?stable_temp_bytes(g.vertices):0,"stable_temp");
  const bool profile_kernel=o.profile_kernel || !o.round_metrics_path.empty();
  Event compare_start(o.profile_compare),compare_end(o.profile_compare);
  Event kernel_start(profile_kernel),kernel_end(profile_kernel);
  float* old=value0.get();float* next=value1.get();uint64_t* current_mask=mask0.get();uint64_t* next_mask=mask1.get();
  uint32_t* current_list=list0.get();uint32_t* next_list=list1.get();uint32_t* current_count=count0.get();uint32_t* next_count=count1.get();
  uint64_t* current_pair=pair0.get();uint64_t* next_pair=pair1.get();
  cudaStream_t stream=nullptr;
  std::vector<float> host_values;
  std::vector<uint8_t> host_live(q),host_due(q);
  std::vector<uint32_t> host_sources(q),host_slot_counts(q),start_round(q),completion(q);
  ConfiguredSelector selector(o);
  std::ofstream round_metrics;
  if(!o.round_metrics_path.empty()){
    auto parent=std::filesystem::path(o.round_metrics_path).parent_path();
    if(!parent.empty())std::filesystem::create_directories(parent);
    round_metrics.open(o.round_metrics_path,std::ios::trunc);
    if(!round_metrics)throw std::runtime_error("cannot write round metrics: "+o.round_metrics_path);
    round_metrics<<"batch,round,live_queries,kernel_id,kernel_family,group_size,warps_per_block,blocks_per_vertex,check,kernel_gpu_ms\n";
  }
  bool checkpoint_saved=false;
  auto execution_start=Clock::now();
  for(size_t begin=0;begin<queries.size();begin+=q){
    uint32_t used=uint32_t(std::min<size_t>(q,queries.size()-begin));++stats.batches;
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
    std::fill(completion.begin(),completion.end(),0);
    uint32_t pending=used,global_round=0;
    for(uint32_t s=0;s<used;++s)host_sources[s]=queries[begin+s].source;
    sources.upload(host_sources);
    auto t=Clock::now();
    initialize_values({old,g.vertices,p,o.group_width,o.layout},o.algorithm,stream);
    clear_mask(current_mask,g.vertices,words,stream);clear_mask(next_mask,g.vertices,words,stream);
    check(cudaMemsetAsync(current_count,0,4,stream),"reset count");
    check(cudaStreamSynchronize(stream),"initialize batch");stats.initialization_ms+=elapsed(t);
    while(pending || std::any_of(host_live.begin(),host_live.begin()+used,[](uint8_t x){return x!=0;})){
      auto round_start=Clock::now();
      bool activated=false;
      for(uint32_t s=0;s<used;++s){
        if(start_round[s]==UINT32_MAX && (!o.use_offsets || queries[begin+s].offset<=global_round)){
          host_due[s]=1;host_live[s]=1;start_round[s]=global_round;--pending;activated=true;
        } else host_due[s]=0;
      }
      if(activated){
        live.upload(host_live);due.upload(host_due);
        t=Clock::now();
        activate({old,g.vertices,p,o.group_width,o.layout},current_mask,sources.get(),due.get(),q,words,o.algorithm,stream);
        FrontierContext fc{{old,g.vertices,p,o.group_width,o.layout},{next,g.vertices,p,o.group_width,o.layout},current_mask,
          flags.get(),current_list,current_count,current_pair,slot_counts.get(),live.get(),q,words,o.algorithm,o.frontier,
          temp.get(),o.frontier==FrontierMode::Stable?stable_temp_bytes(g.vertices):0,stream};
        rebuild_frontier(fc);check(cudaStreamSynchronize(stream),"activate frontier");stats.frontier_ms+=elapsed(t);
      }
      uint32_t active=0;for(uint32_t s=0;s<used;++s)active+=host_live[s];
      if(!active){
        if(pending){uint32_t next_start=UINT32_MAX;for(uint32_t s=0;s<used;++s)if(start_round[s]==UINT32_MAX)next_start=std::min(next_start,queries[begin+s].offset);
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
        check(cudaMemcpy(cp.old_values.data(),old,cells*4,cudaMemcpyDeviceToHost),"checkpoint values");
        check(cudaMemcpy(cp.frontier_mask.data(),current_mask,cp.frontier_mask.size()*8,cudaMemcpyDeviceToHost),"checkpoint mask");
        if(cp.frontier_count)check(cudaMemcpy(cp.frontier.data(),current_list,cp.frontier_count*4,cudaMemcpyDeviceToHost),"checkpoint list");
        auto parent=std::filesystem::path(o.checkpoint_path).parent_path();
        if(!parent.empty())std::filesystem::create_directories(parent);
        save_checkpoint(cp,o.checkpoint_path);checkpoint_saved=true;
      }
      uint64_t pairs=0;
      t=Clock::now();count_edge_pairs(dg.view,current_mask,current_list,current_count,words,edge_pairs.get(),stream);
      check(cudaMemcpyAsync(&pairs,edge_pairs.get(),8,cudaMemcpyDeviceToHost,stream),"edge pairs");
      check(cudaStreamSynchronize(stream),"features");stats.feature_ms+=elapsed(t);
      double rho=g.edges()?double(pairs)/(double(g.edges())*active):0;
      t=Clock::now();
      KernelId chosen=selector.choose({pairs,active,rho,g.edges()!=0},stats.rounds);
      if(chosen==KernelId::DensePull ||
         chosen==KernelId::GroupedG8Edge4Warp4Pull ||
         (int(chosen)>=int(KernelId::PullCheckFreeBase) && int(chosen)<int(KernelId::PullCheckFreeBase)+pull_partition_count) ||
         (int(chosen)>=int(KernelId::PullCheckBase) && int(chosen)<int(KernelId::PullCheckBase)+pull_partition_count))
        ++stats.pull_rounds;
      else ++stats.push_rounds;
      stats.selector_ms+=elapsed(t);
      uint32_t launch_frontier_size=UINT32_MAX;
      if(int(chosen)>=int(KernelId::PushPartitionBase) &&
         int(chosen)<int(KernelId::PushPartitionBase)+push_partition_count){
        auto feature_start=Clock::now();
        check(cudaMemcpyAsync(&launch_frontier_size,current_count,sizeof(uint32_t),cudaMemcpyDeviceToHost,stream),"partition frontier size");
        check(cudaStreamSynchronize(stream),"partition launch features");
        stats.feature_ms+=elapsed(feature_start);
      }
      Context context{dg.view,{old,g.vertices,p,o.group_width,o.layout},{next,g.vertices,p,o.group_width,o.layout},
                      current_mask,current_list,current_count,live.get(),q,words,o.algorithm,stream,error.get(),launch_frontier_size};
      if(probe)probe(context,stats.batches-1,global_round);
      t=Clock::now();check(cudaMemcpyAsync(next,old,cells*sizeof(float),cudaMemcpyDeviceToDevice,stream),"value copy");
      check(cudaMemsetAsync(error.get(),0,4,stream),"error reset");
      check(cudaStreamSynchronize(stream),"copy");stats.copy_ms+=elapsed(t);
      t=Clock::now();
      if(profile_kernel)check(cudaEventRecord(kernel_start.get(),stream),"kernel event start");
      launch(chosen,context);
      if(profile_kernel)check(cudaEventRecord(kernel_end.get(),stream),"kernel event end");
      check(cudaStreamSynchronize(stream),"kernel");stats.kernel_ms+=elapsed(t);
      float kernel_gpu_ms=0;
      if(profile_kernel){
        check(cudaEventElapsedTime(&kernel_gpu_ms,kernel_start.get(),kernel_end.get()),"kernel elapsed");
        stats.kernel_gpu_ms+=kernel_gpu_ms;
      }
      if(round_metrics.is_open()){
        auto d=describe(chosen);
        round_metrics<<(stats.batches-1)<<','<<global_round<<','<<active<<','<<int(chosen)<<','<<d.family<<','
          <<d.group_size<<','<<d.warps_per_block<<','<<d.blocks_per_vertex<<','<<int(d.check)<<','<<kernel_gpu_ms<<'\n';
        if(!round_metrics)throw std::runtime_error("round metrics write failed: "+o.round_metrics_path);
      }
      int err=0;check(cudaMemcpy(&err,error.get(),4,cudaMemcpyDeviceToHost),"precision flag");
      if(err)throw std::runtime_error("BFS distance reached float32 exact-integer boundary 2^24");
      t=Clock::now();
      FrontierContext fc{{old,g.vertices,p,o.group_width,o.layout},{next,g.vertices,p,o.group_width,o.layout},next_mask,
        flags.get(),next_list,next_count,next_pair,slot_counts.get(),live.get(),q,words,o.algorithm,o.frontier,
        temp.get(),o.frontier==FrontierMode::Stable?stable_temp_bytes(g.vertices):0,stream};
      build_frontier(fc,compare_start.get(),compare_end.get());
      check(cudaMemcpyAsync(host_slot_counts.data(),slot_counts.get(),q*4,cudaMemcpyDeviceToHost,stream),"slot counts");
      check(cudaStreamSynchronize(stream),"frontier");stats.frontier_ms+=elapsed(t);
      if(o.profile_compare){float ms=0;check(cudaEventElapsedTime(&ms,compare_start.get(),compare_end.get()),"compare elapsed");stats.compare_ms+=ms;}
      for(uint32_t s=0;s<used;++s)if(host_live[s] && host_slot_counts[s]==0){host_live[s]=0;completion[s]=global_round-start_round[s]+1;}
      live.upload(host_live);
      std::swap(old,next);std::swap(current_mask,next_mask);std::swap(current_list,next_list);
      std::swap(current_count,next_count);std::swap(current_pair,next_pair);
      if(round_callback){
        RoundSnapshot snap{};snap.batch_index=stats.batches-1;snap.global_round=global_round;
        snap.used_slots=used;snap.physical_slots=p;snap.words=words;snap.layout=o.layout;
        for(uint32_t s=0;s<used;++s)snap.query_ids.push_back(queries[begin+s].id);
        snap.values.resize(cells);snap.mask.resize(size_t(g.vertices)*words);
        uint32_t count=0;
        check(cudaMemcpy(snap.values.data(),old,cells*4,cudaMemcpyDeviceToHost),"round values");
        check(cudaMemcpy(snap.mask.data(),current_mask,snap.mask.size()*8,cudaMemcpyDeviceToHost),"round mask");
        check(cudaMemcpy(&count,current_count,4,cudaMemcpyDeviceToHost),"round frontier count");
        snap.frontier.resize(count);
        if(count)check(cudaMemcpy(snap.frontier.data(),current_list,count*4,cudaMemcpyDeviceToHost),"round frontier");
        round_callback(snap);
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
    if(o.copy_results_to_cpu){
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
    spdlog::info("scheduler batch={} queries={} rounds={}",stats.batches,used,global_round);
  }
  if(!o.checkpoint_path.empty() && !checkpoint_saved)
    throw std::runtime_error("checkpoint round was not reached: "+std::to_string(o.checkpoint_round));
  stats.execution_ms=elapsed(execution_start);
  stats.task_wall_ms=elapsed(task_wall_start);
  if(!o.plan_output_path.empty()){
    auto parent=std::filesystem::path(o.plan_output_path).parent_path();
    if(!parent.empty())std::filesystem::create_directories(parent);
    std::ofstream out(o.plan_output_path);if(!out)throw std::runtime_error("cannot write plan output");
    out<<"# graph_identity="<<g.identity<<'\n';
    out<<"# capacity="<<o.capacity<<'\n';
    out<<"# id,source,score,offset,feature_key,batch,slot\n";
    for(size_t i=0;i<queries.size();++i){const auto& query=queries[i];
      out<<query.id<<','<<query.source<<','<<query.score<<','<<query.offset<<','<<query.feature_key<<','<<i/o.capacity<<','<<i%o.capacity<<'\n';
    }
    if(!out)throw std::runtime_error("plan output write failed");
  }
  stats.total_ms=elapsed(task_start);return stats;
}
}
