#pragma once
#include "graphweft/engine.hpp"
#include "graphweft/iteration_model.hpp"
#include <vector>
namespace graphweft {
// A provider populates the distinct score and start-offset fields of Query.
class PredictionProvider {
 public:
  virtual ~PredictionProvider() = default;
  virtual void predict(const HostGraph&, std::vector<Query>&) const = 0;
};
class FrozenPredictionProvider final : public PredictionProvider {
 public:
  void predict(const HostGraph&, std::vector<Query>&) const override {} // imported Query fields are immutable here
};
class CoreDistanceProvider final : public PredictionProvider {
 public:
  void predict(const HostGraph&, std::vector<Query>&) const override;
};
class WeightedBoundaryProvider final : public PredictionProvider {
 public:
  void predict(const HostGraph&, std::vector<Query>&) const override;
};
class BatchPlanner {
 public:
  static std::vector<Query> plan(std::vector<Query> queries,const Options& options);
  static std::vector<Query> plan_groups(std::vector<Query> queries,const Options& options);
};
class OnlineSelector {
 public:
  virtual ~OnlineSelector() = default;
  virtual KernelId choose(const RoundFeatures&,uint64_t global_round) const = 0;
};
class ConfiguredSelector final : public OnlineSelector {
 public:
  explicit ConfiguredSelector(const Options& options):options_(options){}
  KernelId choose(const RoundFeatures&,uint64_t global_round) const override;
  bool used_iteration_model() const { return used_iteration_model_; }
  const IterationPrediction& iteration_prediction() const { return iteration_prediction_; }
 private:
  const Options& options_;
  mutable bool used_iteration_model_ = false;
  mutable IterationPrediction iteration_prediction_{};
};
}
