#include "pull_probe.hpp"
#include <gflags/gflags.h>
#include <spdlog/spdlog.h>
DEFINE_string(graph,"","CSR graph directory");
DEFINE_string(queries,"","Frozen query CSV");
DEFINE_string(output,"","Study output directory");
DEFINE_bool(legacy_int_weights,true,"Original int32 CSR weights");
DEFINE_bool(directed,true,"Build directed graph");
DEFINE_uint32(q,32,"Query capacity");
int main(int argc,char** argv){
 try{
  gflags::ParseCommandLineFlags(&argc,&argv,true);
  if(FLAGS_graph.empty() || FLAGS_queries.empty() || FLAGS_output.empty())
    throw std::invalid_argument("--graph, --queries, --output required");
  spdlog::set_level(spdlog::level::warn);
  auto graph=graphweft::HostGraph::load(FLAGS_graph,FLAGS_directed,FLAGS_legacy_int_weights);
  auto queries=graphweft::load_queries(FLAGS_queries,graph.vertices,graph.identity);
  graphweft::Options options;options.algorithm=graphweft::Algorithm::SSSP;options.capacity=FLAGS_q;
  options.selector=graphweft::Options::Selector::Push;
  graphweft::lab::PullProbe probe(graph,FLAGS_output);
  std::cout<<"graph_identity="<<graph.identity<<" V="<<graph.vertices<<" E="<<graph.edges()
           <<" N="<<queries.size()<<" Q="<<FLAGS_q<<std::endl;
  auto stats=graphweft::run(graph,queries,options,{}, {},std::ref(probe));
  std::ofstream done(FLAGS_output+"/completed.txt");
  done<<"rounds="<<stats.rounds<<" checked_candidates="<<probe.checked_candidates<<" exact_match=1\n";
  done<<"production_kernel_ms="<<stats.kernel_ms<<" production_frontier_ms="<<stats.frontier_ms
      <<" production_copy_ms="<<stats.copy_ms<<" diagnostic_wall_ms="<<stats.task_wall_ms<<'\n';
  std::cout<<"complete rounds="<<stats.rounds<<" checked_candidates="<<probe.checked_candidates<<std::endl;
 }catch(const std::exception& e){std::cerr<<"pull study: "<<e.what()<<std::endl;return 1;}
}
