#include "graphweft/engine.hpp"
#include <gflags/gflags.h>
#include <spdlog/spdlog.h>
#include <cuda_runtime.h>
#include <algorithm>
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
DEFINE_string(planner,"fifo","fifo or length");
DEFINE_string(predictor,"import","import, core_distance or weighted_boundary");
DEFINE_bool(offsets,false,"Enable delayed query starts");
DEFINE_string(offset_source,"import","import or phase");
DEFINE_int32(landmarks,16,"Phase index landmark count");
DEFINE_int32(max_offset,16,"Phase evaluator maximum start offset");
DEFINE_string(selector,"threshold","threshold, push, pull, replay");
DEFINE_string(replay,"","Comma separated push,pull choices");
DEFINE_double(pull_threshold,.2,"Dense pull threshold");
DEFINE_string(frontier,"unordered","unordered or stable");
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
DEFINE_string(plan_output,"","Write graph-bound frozen plan CSV");
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
    if(FLAGS_predictor=="core_distance")o.predictor=gw::Options::Predictor::CoreDistance;
    else if(FLAGS_predictor=="weighted_boundary")o.predictor=gw::Options::Predictor::WeightedBoundary;
    else if(FLAGS_predictor=="import_key")o.predictor=gw::Options::Predictor::ImportedKey;
    else if(FLAGS_predictor!="import")throw std::invalid_argument("invalid predictor");
    if(FLAGS_planner=="length")o.sort_by_score=true;
    else if(FLAGS_planner!="fifo")throw std::invalid_argument("invalid planner");
    if(FLAGS_group_width<=0 || FLAGS_q<=0)throw std::invalid_argument("Q and group width must be positive");
    o.group_width=FLAGS_group_width;o.capacity=FLAGS_q;o.use_offsets=FLAGS_offsets;
    if(FLAGS_offset_source=="phase")o.phase_offsets=true;
    else if(FLAGS_offset_source!="import")throw std::invalid_argument("invalid offset source");
    if(FLAGS_landmarks<=0 || FLAGS_max_offset<0)throw std::invalid_argument("invalid phase index options");
    o.landmarks=FLAGS_landmarks;o.max_offset=FLAGS_max_offset;
    o.pull_threshold=FLAGS_pull_threshold;o.memory_fraction=FLAGS_memory_fraction;
    o.copy_results_to_cpu=FLAGS_copy_results_to_cpu || !FLAGS_binary_output.empty();o.profile_compare=FLAGS_profile_compare;
    o.profile_kernel=FLAGS_profile_kernel;o.round_metrics_path=FLAGS_round_metrics;
    o.checkpoint_path=FLAGS_checkpoint;o.checkpoint_round=FLAGS_checkpoint_round;o.plan_output_path=FLAGS_plan_output;
    if(FLAGS_selector=="push")o.selector=gw::Options::Selector::Push;
    else if(FLAGS_selector=="pull")o.selector=gw::Options::Selector::Pull;
    else if(FLAGS_selector=="replay"){
      o.selector=gw::Options::Selector::Replay;std::string token;
      std::istringstream in(FLAGS_replay);
      while(std::getline(in,token,',')){
        if(token=="push")o.replay.push_back(gw::KernelId::SharedPush);
        else if(token=="pull")o.replay.push_back(gw::KernelId::DensePull);
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
             <<" max_Q="<<gw::max_capacity(g,o,allowed,uint32_t(std::max<size_t>(queries.size(),o.capacity)))<<'\n';
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
    auto stats=gw::run(g,queries,o,callback);
    if(!FLAGS_binary_output.empty() && !binary_written)throw std::runtime_error("binary query ID not found");
    std::cout<<"batches="<<stats.batches<<" rounds="<<stats.rounds<<" push_rounds="<<stats.push_rounds<<" pull_rounds="<<stats.pull_rounds<<" execution_ms="<<stats.execution_ms<<" task_wall_ms="<<stats.task_wall_ms<<" total_ms="<<stats.total_ms
             <<" planning_ms="<<stats.planning_ms<<" prediction_ms="<<stats.prediction_ms<<" initialization_ms="<<stats.initialization_ms
             <<" copy_ms="<<stats.copy_ms<<" kernel_ms="<<stats.kernel_ms<<" kernel_gpu_ms="<<stats.kernel_gpu_ms
             <<" frontier_ms="<<stats.frontier_ms<<" compare_ms="<<stats.compare_ms<<" feature_ms="<<stats.feature_ms<<" selector_ms="<<stats.selector_ms<<" round_ms="<<stats.round_ms
             <<" transfer_ms="<<stats.transfer_ms<<'\n';
    return 0;
  }catch(const std::exception& e){std::cerr<<"GraphWeft: "<<e.what()<<'\n';return 1;}
}
