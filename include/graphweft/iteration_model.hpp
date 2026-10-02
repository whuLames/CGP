#pragma once
#include "graphweft/engine.hpp"
#include <array>
#include <cstddef>
#include <limits>
namespace graphweft {
constexpr size_t iteration_base_feature_count = 7;
constexpr size_t iteration_expanded_feature_count = 36;
struct IterationPrediction {
  int candidate = 0;
  double log_cost = 0;
  double relative_cost = 1;
  std::array<double,iteration_base_feature_count> clipped{};
};
inline int iteration_argmin(const std::array<double,30>& scores) {
  int best=0;
  for(int i=1;i<30;++i)if(scores[i]<scores[best])best=i;
  return best;
}
IterationPrediction predict_iteration_push(const RoundFeatures&);
const char* iteration_model_version();
bool iteration_model_is_calibrated();
}
