#pragma once
#include "graphweft/engine.hpp"
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
 private:
  const Options& options_;
};
}
