#include "graphweft/iteration_model.hpp"
#include "graphweft/iteration_model_coefficients.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>
namespace graphweft {
IterationPrediction predict_iteration_push(const RoundFeatures& f) {
  IterationPrediction out;
  if(!f.edge_pairs || !f.frontier_vertices)return out;
  const double frontier=f.frontier_vertices,vertices=f.vertex_pairs;
  std::array<double,iteration_base_feature_count> raw={
    std::log1p(frontier),std::log1p(double(f.active_queries)),
    std::log1p(vertices/frontier),
    std::log1p(vertices?double(f.edge_pairs)/vertices:0.0),
    std::log1p(double(f.edge_pairs)/frontier),
    std::log1p(f.graph_vertices?double(f.graph_edges)/f.graph_vertices:0.0),f.density
  },z{};
  for(size_t i=0;i<raw.size();++i){
    out.clipped[i]=std::clamp(raw[i],iteration_model_data::minimum[i],iteration_model_data::maximum[i]);
    z[i]=(out.clipped[i]-iteration_model_data::mean[i])/iteration_model_data::scale[i];
  }
  std::array<double,iteration_expanded_feature_count> x{};x[0]=1;
  for(size_t i=0;i<z.size();++i)x[1+i]=z[i];
  size_t at=1+z.size();
  for(size_t i=0;i<z.size();++i)for(size_t j=i;j<z.size();++j)x[at++]=z[i]*z[j];
  std::array<double,30> scores{};
  for(int candidate=0;candidate<push_partition_count;++candidate){
    for(size_t i=0;i<x.size();++i)scores[candidate]+=iteration_model_data::coefficients[candidate][i]*x[i];
  }
  out.candidate=iteration_argmin(scores);out.log_cost=scores[out.candidate];
  out.relative_cost=std::exp(std::clamp(out.log_cost,-50.0,50.0));
  return out;
}
const char* iteration_model_version(){return iteration_model_data::version;}
bool iteration_model_is_calibrated(){return std::string_view(iteration_model_data::version).find("bootstrap")==std::string_view::npos;}
}
