#include "graphweft/engine.hpp"
#include "graphweft/scheduler.hpp"
#include "graphweft/iteration_model.hpp"
#include <gflags/gflags.h>
#include <spdlog/spdlog.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <cstring>
DEFINE_string(graph,"","Edge list or puercgp CSR directory");
DEFINE_bool(directed,false,"Use directed graph and incoming CSC");
DEFINE_bool(legacy_int_weights,false,"Read legacy CSR weights as int32 before conversion to float32");
DEFINE_string(algorithm,"bfs","bfs, sssp, sswp");
DEFINE_string(queries,"","Query CSV: id,source,score,offset");
DEFINE_int32(n,32,"Generated number of queries when queries is empty");
DEFINE_int32(q,32,"Concurrent query capacity");
DEFINE_bool(auto_q,false,"Choose largest capacity within budget, bounded by N");
DEFINE_string(layout,"vertex","vertex or grouped");
DEFINE_int32(group_width,8,"Grouped layout width");
DEFINE_bool(group_refill,false,"Reclaim and refill a same-algorithm group as soon as it completes");
DEFINE_bool(eager_sssp_refill,false,"Experimental: refill completed SSSP groups without waiting for a synchronized wave");
DEFINE_bool(interference_aware_refill,false,"Selectively refill only under a Push round and a bounded active-group budget");
DEFINE_int32(refill_max_active_groups,2,"Maximum simultaneously resident groups after interference-aware refill");
DEFINE_bool(interference_bridge_refill,false,"Give one next-cohort group a Push-only head start, then restore full cohorts");
DEFINE_bool(group_iteration_mapping,false,"Experimental: select Iteration Push mapping per G and merge equal mappings");
DEFINE_bool(same_algorithm_groups,false,"Form same-algorithm groups even with batch-barrier reclamation");
DEFINE_bool(oracle_order,false,"Diagnostic only: sort within each algorithm by imported reference rounds");
DEFINE_string(planner,"fifo","fifo or length");
DEFINE_string(predictor,"import","import, core_distance or weighted_boundary");
DEFINE_bool(offsets,false,"Enable delayed query starts");
DEFINE_string(offset_source,"import","import or phase");
DEFINE_int32(landmarks,16,"Phase index landmark count");
DEFINE_int32(max_offset,16,"Phase evaluator maximum start offset");
DEFINE_string(selector,"threshold","threshold, push, pull, replay");
DEFINE_string(replay,"","Comma separated push,pull choices");
DEFINE_string(pull_kernel,"auto","auto or a pull-dense-* kernel token");
DEFINE_double(pull_threshold,.2,"Dense pull threshold");
DEFINE_string(push_mapping,"shared","shared, static, degree, density, adaptive or iteration Push mapping");
DEFINE_int32(push_query_lanes,8,"Static Push query lanes per edge: 1/2/4/8/16/32");
DEFINE_int32(push_grain,0,"Push grain: 0/1/2=1/2/4 warps, 3/4=2/4 blocks");
DEFINE_string(frontier,"unordered","unordered or stable");
DEFINE_string(frontier_build,"fused","scan, fused or direct frontier construction");
DEFINE_bool(frontier_mask64,true,"Aggregate update-driven frontier publication in 64-query words");
DEFINE_bool(copy_results_to_cpu,false,"Return per query results");
DEFINE_bool(profile_compare,false,"Measure compare_kernel GPU time with CUDA events");
DEFINE_bool(profile_kernel,false,"Measure production graph kernels with CUDA events");
DEFINE_string(round_metrics,"","Write per-round kernel timing CSV (also enables --profile_kernel)");
DEFINE_string(output,"","CSV result output path");
DEFINE_string(binary_output,"","Write one query as raw float32 values for exact comparison");
DEFINE_uint64(binary_query_id,0,"Logical query ID written by --binary_output");
DEFINE_string(checkpoint,"","Save first active round input for kernel_lab");
DEFINE_uint32(checkpoint_round,0,"Global round saved by --checkpoint");
DEFINE_string(result_hashes,"","Write per-query final-value FNV-1a hashes for validation");
DEFINE_string(result_fingerprints,"","Write GPU-computed per-query sum/xor fingerprints for validation");
DEFINE_string(plan_output,"","Write graph-bound frozen plan CSV");
DEFINE_bool(plan_only,false,"Build/freeze predictor keys without allocating or executing GPU state");
DEFINE_string(completion_output,"","Write per-query completion metadata without copying result vectors");
DEFINE_string(schedule_events,"","Write activation/completion scheduling events");
DEFINE_double(memory_fraction,.8,"Fraction of total GPU memory usable");
DEFINE_int32(device,0,"CUDA device number");
DEFINE_string(log_level,"info","trace, debug, info, warn, error");
namespace gw=graphweft;
int main(int argc,char** argv){
  try{
    gflags::ParseCommandLineFlags(&argc,&argv,true);
    spdlog::set_level(spdlog::level::from_str(FLAGS_log_level));
    if(FLAGS_graph.empty())throw std::invalid_argument("--graph is required");
    auto g=gw::HostGraph::load(FLAGS_graph,FLAGS_directed,FLAGS_legacy_int_weights);
    gw::Options o;
    if(FLAGS_algorithm=="bfs")o.algorithm=gw::Algorithm::BFS;
    else if(FLAGS_algorithm=="sssp")o.algorithm=gw::Algorithm::SSSP;
    else if(FLAGS_algorithm=="sswp")o.algorithm=gw::Algorithm::SSWP;
    else throw std::invalid_argument("invalid algorithm");
    if(FLAGS_layout=="grouped")o.layout=gw::Layout::Grouped;
    else if(FLAGS_layout!="vertex")throw std::invalid_argument("invalid layout");
    if(FLAGS_frontier=="stable")o.frontier=gw::FrontierMode::Stable;
    else if(FLAGS_frontier!="unordered")throw std::invalid_argument("invalid frontier");
    if(FLAGS_frontier_build=="fused")o.frontier_build=gw::FrontierBuildMode::Fused;
    else if(FLAGS_frontier_build=="direct")o.frontier_build=gw::FrontierBuildMode::Direct;
    else if(FLAGS_frontier_build!="scan")throw std::invalid_argument("invalid frontier build mode");
    o.frontier_mask64=FLAGS_frontier_mask64;
    if(FLAGS_predictor=="core_distance")o.predictor=gw::Options::Predictor::CoreDistance;
    else if(FLAGS_predictor=="weighted_boundary")o.predictor=gw::Options::Predictor::WeightedBoundary;
    else if(FLAGS_predictor=="import_key")o.predictor=gw::Options::Predictor::ImportedKey;
    else if(FLAGS_predictor!="import")throw std::invalid_argument("invalid predictor");
    if(FLAGS_planner=="length")o.sort_by_score=true;
    else if(FLAGS_planner!="fifo")throw std::invalid_argument("invalid planner");
    if(FLAGS_group_width<=0 || FLAGS_q<=0)throw std::invalid_argument("Q and group width must be positive");
    o.group_width=FLAGS_group_width;o.capacity=FLAGS_q;o.use_offsets=FLAGS_offsets;
    o.group_refill=FLAGS_group_refill;o.same_algorithm_groups=FLAGS_same_algorithm_groups||FLAGS_group_refill;
    o.eager_sssp_refill=FLAGS_eager_sssp_refill;
    o.interference_aware_refill=FLAGS_interference_aware_refill;
    o.interference_bridge_refill=FLAGS_interference_bridge_refill;
    if(FLAGS_refill_max_active_groups<=0)throw std::invalid_argument("--refill_max_active_groups must be positive");
    o.refill_max_active_groups=uint32_t(FLAGS_refill_max_active_groups);
    o.group_iteration_mapping=FLAGS_group_iteration_mapping;
    o.oracle_order=FLAGS_oracle_order;
    if(FLAGS_offset_source=="phase")o.phase_offsets=true;
    else if(FLAGS_offset_source!="import")throw std::invalid_argument("invalid offset source");
    if(FLAGS_landmarks<=0 || FLAGS_max_offset<0)throw std::invalid_argument("invalid phase index options");
    o.landmarks=FLAGS_landmarks;o.max_offset=FLAGS_max_offset;
    o.pull_threshold=FLAGS_pull_threshold;o.memory_fraction=FLAGS_memory_fraction;
    if(FLAGS_pull_kernel!="auto")o.pull_kernel=gw::parse_dense_pull_token(FLAGS_pull_kernel);
    if(FLAGS_push_mapping=="static")o.push_mapping=gw::Options::PushMapping::Static;
    else if(FLAGS_push_mapping=="degree")o.push_mapping=gw::Options::PushMapping::Degree;
    else if(FLAGS_push_mapping=="density")o.push_mapping=gw::Options::PushMapping::Density;
    else if(FLAGS_push_mapping=="adaptive")o.push_mapping=gw::Options::PushMapping::Adaptive;
    else if(FLAGS_push_mapping=="iteration")o.push_mapping=gw::Options::PushMapping::Iteration;
    else if(FLAGS_push_mapping!="shared")throw std::invalid_argument("invalid Push mapping");
    if((o.push_mapping==gw::Options::PushMapping::Adaptive || o.push_mapping==gw::Options::PushMapping::Iteration) &&
       (o.layout!=gw::Layout::Grouped || o.group_width!=32))
      throw std::invalid_argument(FLAGS_push_mapping+" Push mapping requires --layout=grouped --group_width=32");
    if(o.eager_sssp_refill && !o.group_refill)
      throw std::invalid_argument("--eager_sssp_refill requires --group_refill");
    if(o.interference_aware_refill && !o.group_refill)
      throw std::invalid_argument("--interference_aware_refill requires --group_refill");
    if(o.interference_bridge_refill && !o.group_refill)
      throw std::invalid_argument("--interference_bridge_refill requires --group_refill");
    if(o.interference_aware_refill && o.eager_sssp_refill)
      throw std::invalid_argument("--interference_aware_refill and --eager_sssp_refill are mutually exclusive");
    if(o.interference_bridge_refill && (o.eager_sssp_refill || o.interference_aware_refill))
      throw std::invalid_argument("--interference_bridge_refill is mutually exclusive with other experimental refill policies");
    if(o.interference_bridge_refill && o.predictor!=gw::Options::Predictor::ImportedKey)
      throw std::invalid_argument("--interference_bridge_refill requires --predictor=import_key");
    if(o.group_iteration_mapping && (!o.group_refill || o.push_mapping!=gw::Options::PushMapping::Iteration))
      throw std::invalid_argument("--group_iteration_mapping requires --group_refill --push_mapping=iteration");
    o.push_query_lanes=FLAGS_push_query_lanes;o.push_grain=FLAGS_push_grain;
    o.copy_results_to_cpu=FLAGS_copy_results_to_cpu || !FLAGS_binary_output.empty();o.profile_compare=FLAGS_profile_compare;
    o.profile_kernel=FLAGS_profile_kernel;o.round_metrics_path=FLAGS_round_metrics;
    o.checkpoint_path=FLAGS_checkpoint;o.checkpoint_round=FLAGS_checkpoint_round;o.plan_output_path=FLAGS_plan_output;
    o.completion_output_path=FLAGS_completion_output;o.schedule_events_path=FLAGS_schedule_events;
    if(FLAGS_selector=="push")o.selector=gw::Options::Selector::Push;
    else if(FLAGS_selector=="pull")o.selector=gw::Options::Selector::Pull;
    else if(FLAGS_selector=="replay"){
      o.selector=gw::Options::Selector::Replay;std::string token;
      std::istringstream in(FLAGS_replay);
      while(std::getline(in,token,',')){
        if(token=="push")o.replay.push_back(gw::KernelId::SharedPush);
        else if(token=="pull")o.replay.push_back(gw::KernelId::DensePull);
        else if(token.rfind("pull-dense-",0)==0 || token.rfind("pull-vm-",0)==0)
          o.replay.push_back(gw::parse_dense_pull_token(token));
        else if(token=="pull-grouped-g8-edge4-warp4")
          o.replay.push_back(gw::KernelId::GroupedG8Edge4Warp4Pull);
        else {
          bool found=false;
          for(int k=0;k<gw::pull_partition_count && !found;++k){
            auto p=gw::pull_partition(k);
            std::string shape="q"+std::to_string(p.group_size)+
              (p.blocks_per_vertex==1?"_w"+std::to_string(p.warps_per_block):"_b"+std::to_string(p.blocks_per_vertex));
            if(token=="pull-check-free-"+shape || token=="pull-check-"+shape){
              o.replay.push_back(gw::pull_partition_id(k,token.rfind("pull-check-"+shape,0)==0));
              found=true;
            }
          }
          if(!found)throw std::invalid_argument("invalid replay token");
        }
      }
    } else if(FLAGS_selector!="threshold")throw std::invalid_argument("invalid selector");
    if(cudaSetDevice(FLAGS_device)!=cudaSuccess)throw std::runtime_error("cudaSetDevice failed");
    if(o.push_mapping==gw::Options::PushMapping::Iteration){
      cudaDeviceProp property{};
      if(cudaGetDeviceProperties(&property,FLAGS_device)!=cudaSuccess)throw std::runtime_error("cudaGetDeviceProperties failed");
      if(std::string(property.name).find("V100")==std::string::npos)
        spdlog::warn("iteration Push model is calibrated only for NVIDIA Tesla V100; detected {}",property.name);
      if(!gw::iteration_model_is_calibrated())
        spdlog::warn("iteration Push is using the deterministic bootstrap model {}; run the seed-45 calibration before performance use",gw::iteration_model_version());
    }
    std::vector<gw::Query> queries;
    if(!FLAGS_queries.empty())queries=gw::load_queries(FLAGS_queries,g.vertices,g.identity,(o.sort_by_score && (o.predictor==gw::Options::Predictor::Imported || o.predictor==gw::Options::Predictor::ImportedKey)) || (o.use_offsets && !o.phase_offsets),FLAGS_auto_q?0:uint32_t(FLAGS_q));
    else{
      if(FLAGS_n<0)throw std::invalid_argument("negative N");
      for(int i=0;i<FLAGS_n;++i)queries.push_back({uint64_t(i),uint32_t(i%g.vertices),0,0});
    }
    if(((o.sort_by_score && (o.predictor==gw::Options::Predictor::Imported || o.predictor==gw::Options::Predictor::ImportedKey)) || (o.use_offsets && !o.phase_offsets)) && FLAGS_queries.empty())
      throw std::invalid_argument("length/offset scheduling requires frozen query import");
    if(o.phase_offsets && o.algorithm==gw::Algorithm::SSWP)
      throw std::invalid_argument("SSWP requires imported offsets");
    if(FLAGS_plan_only){
      if(FLAGS_plan_output.empty())throw std::invalid_argument("--plan_only requires --plan_output");
      if(!o.sort_by_score)throw std::invalid_argument("--plan_only requires --planner=length");
      auto started=std::chrono::steady_clock::now();
      if(o.predictor==gw::Options::Predictor::CoreDistance){gw::CoreDistanceProvider provider;provider.predict(g,queries);}
      else if(o.predictor==gw::Options::Predictor::WeightedBoundary){gw::WeightedBoundaryProvider provider;provider.predict(g,queries);}
      else {gw::FrozenPredictionProvider provider;provider.predict(g,queries);}
      // Validate the eventual ordering work, but freeze features in the
      // caller's original submission order.  Imported execution performs the
      // sort online; pre-sorting this file would silently turn FIFO controls
      // into length-sorted workloads.
      auto planned=gw::BatchPlanner::plan(queries,o);
      (void)planned;
      auto parent=std::filesystem::path(FLAGS_plan_output).parent_path();
      if(!parent.empty())std::filesystem::create_directories(parent);
      std::ofstream out(FLAGS_plan_output);if(!out)throw std::runtime_error("cannot write plan output");
      out<<"# graph_identity="<<g.identity<<'\n'<<"# capacity="<<o.capacity<<'\n';
      out<<"# id,source,score,offset,feature_key,algorithm,reference_rounds,input_batch,input_slot\n";
      for(size_t i=0;i<queries.size();++i){const auto& query=queries[i];
        out<<query.id<<','<<query.source<<','<<query.score<<','<<query.offset<<','<<query.feature_key<<','
           <<query.algorithm<<','<<query.reference_rounds<<','<<i/o.capacity<<','<<i%o.capacity<<'\n';
      }
      auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
      std::cout<<"graph="<<g.identity<<" plan_only=1 queries="<<queries.size()<<" prediction_and_planning_ms="<<ms<<'\n';
      return 0;
    }
    if(FLAGS_auto_q && !queries.empty()){
      size_t free_b=0,total_b=0;cudaMemGetInfo(&free_b,&total_b);
      uint64_t allowed=std::min<uint64_t>(free_b,uint64_t(total_b*o.memory_fraction));
      o.capacity=gw::max_capacity(g,o,allowed,uint32_t(queries.size()));
      if(!o.capacity)throw std::runtime_error("no capacity fits memory budget");
    }
    size_t free_b=0,total_b=0;cudaMemGetInfo(&free_b,&total_b);
    uint64_t allowed=std::min<uint64_t>(free_b,uint64_t(total_b*o.memory_fraction));
    auto plan=gw::allocation_plan(g,o,o.capacity);
    std::cout<<"graph="<<g.identity<<" V="<<g.vertices<<" E="<<g.edges()<<" directed="<<g.directed
             <<" gpu_total="<<total_b<<" gpu_free="<<free_b<<" budget="<<allowed<<" Q="<<o.capacity
             <<" max_Q="<<gw::max_capacity(g,o,allowed,uint32_t(std::max<size_t>(queries.size(),o.capacity)))
             <<" frontier_build="<<(o.frontier_build==gw::FrontierBuildMode::Scan?"scan":
                 o.frontier_build==gw::FrontierBuildMode::Fused?"fused":"direct")<<'\n';
    for(auto [name,bytes]:plan.bytes)std::cout<<"memory."<<name<<'='<<bytes<<'\n';
    std::cout<<"memory.total="<<plan.total()<<'\n';
    std::ofstream result;
    if(o.copy_results_to_cpu && !FLAGS_output.empty()){
      auto parent=std::filesystem::path(FLAGS_output).parent_path();
      if(!parent.empty())std::filesystem::create_directories(parent);
      result.open(FLAGS_output);if(!result)throw std::runtime_error("cannot open output");
      result<<"query_id,source,vertex,value,completion_local_round\n";
    }
    std::ofstream hashes;
    if(!FLAGS_result_hashes.empty()){
      auto parent=std::filesystem::path(FLAGS_result_hashes).parent_path();
      if(!parent.empty())std::filesystem::create_directories(parent);
      hashes.open(FLAGS_result_hashes);if(!hashes)throw std::runtime_error("cannot open result hashes");
      hashes<<"query_id,source,completion_local_round,vertices,fnv1a64\n";
      o.copy_results_to_cpu=true;
    }
    std::ofstream fingerprints;
    if(!FLAGS_result_fingerprints.empty()){
      auto parent=std::filesystem::path(FLAGS_result_fingerprints).parent_path();
      if(!parent.empty())std::filesystem::create_directories(parent);
      fingerprints.open(FLAGS_result_fingerprints);if(!fingerprints)throw std::runtime_error("cannot open result fingerprints");
      fingerprints<<"query_id,source,completion_local_round,vertices,sum64,xor64\n";
    }
    bool binary_written=false;
    auto callback=[&](const gw::QueryResult& r){
      if(!FLAGS_binary_output.empty() && r.id==FLAGS_binary_query_id){
        auto parent=std::filesystem::path(FLAGS_binary_output).parent_path();
        if(!parent.empty())std::filesystem::create_directories(parent);
        std::ofstream binary(FLAGS_binary_output,std::ios::binary|std::ios::trunc);
        if(!binary)throw std::runtime_error("cannot create binary result");
        binary.write(reinterpret_cast<const char*>(r.values.data()),r.values.size()*sizeof(float));
        if(!binary)throw std::runtime_error("binary result write failed");
        binary_written=true;
      }
      if(hashes){
        uint64_t hash=14695981039346656037ULL;
        for(float value:r.values){uint32_t bits;std::memcpy(&bits,&value,sizeof(bits));
          for(int byte=0;byte<4;++byte){hash^=(bits>>(byte*8))&255U;hash*=1099511628211ULL;}}
        hashes<<r.id<<','<<r.source<<','<<r.completion_local_round<<','<<r.values.size()<<','<<std::hex<<hash<<std::dec<<'\n';
      }
      if(!result)return;
      for(uint32_t v=0;v<r.values.size();++v)result<<r.id<<','<<r.source<<','<<v<<','<<r.values[v]<<','<<r.completion_local_round<<'\n';
    };
    auto fingerprint_callback=[&](const gw::QueryFingerprint& r){
      if(fingerprints)fingerprints<<r.id<<','<<r.source<<','<<r.completion_local_round<<','<<r.vertices<<','
        <<std::hex<<r.sum<<','<<r.xor_value<<std::dec<<'\n';
    };
    auto stats=gw::run(g,queries,o,callback,{}, {},fingerprint_callback);
    if(!FLAGS_binary_output.empty() && !binary_written)throw std::runtime_error("binary query ID not found");
    std::cout<<"batches="<<stats.batches<<" rounds="<<stats.rounds<<" push_rounds="<<stats.push_rounds<<" pull_rounds="<<stats.pull_rounds<<" execution_ms="<<stats.execution_ms<<" task_wall_ms="<<stats.task_wall_ms<<" workload_ms="<<stats.workload_ms
             <<" throughput_qps="<<(stats.workload_ms?queries.size()*1000.0/stats.workload_ms:0.0)<<" total_ms="<<stats.total_ms
             <<" planning_ms="<<stats.planning_ms<<" prediction_ms="<<stats.prediction_ms<<" initialization_ms="<<stats.initialization_ms<<" recycle_ms="<<stats.recycle_ms
             <<" copy_ms="<<stats.copy_ms<<" kernel_ms="<<stats.kernel_ms<<" kernel_gpu_ms="<<stats.kernel_gpu_ms
             <<" frontier_ms="<<stats.frontier_ms<<" compare_ms="<<stats.compare_ms<<" feature_ms="<<stats.feature_ms<<" adaptive_preparation_ms="<<stats.adaptive_preparation_ms<<" selector_ms="<<stats.selector_ms<<" round_ms="<<stats.round_ms
             <<" transfer_ms="<<stats.transfer_ms<<" group_refills="<<stats.group_refills
             <<" refill_admitted_groups="<<stats.refill_admitted_groups
             <<" refill_deferred_groups="<<stats.refill_deferred_groups
             <<" refill_deferred_pull_groups="<<stats.refill_deferred_pull_groups
             <<" refill_deferred_capacity_groups="<<stats.refill_deferred_capacity_groups
             <<" refill_deferred_incompatible_groups="<<stats.refill_deferred_incompatible_groups
             <<" group_mapping_rounds="<<stats.group_mapping_rounds
             <<" group_mapping_divergent_rounds="<<stats.group_mapping_divergent_rounds
             <<" group_mapping_launches="<<stats.group_mapping_launches
             <<" completed_slot_rounds="<<stats.completed_slot_rounds<<" final_drain_rounds="<<stats.final_drain_rounds<<" active_slot_ratio="
             <<(stats.capacity_slot_rounds?double(stats.active_slot_rounds)/stats.capacity_slot_rounds:0.0)<<'\n';
    return 0;
  }catch(const std::exception& e){std::cerr<<"GraphWeft: "<<e.what()<<'\n';return 1;}
}
