#pragma once

#include <chrono>
#include <stdexcept>
#include <vector>

#include <puercgp/scheduling/online_offset_evaluator.hxx>
#include <puercgp/scheduling/slot_start_schedule.hxx>

namespace puercgp {
namespace scheduling {

struct online_execution_batch {
  std::vector<int> query_ids;
  std::vector<int> sources;
  slot_start_schedule start_schedule;
  std::vector<int> predicted_lengths;
  std::uint64_t alignment_score = 0;
  double offset_evaluator_ms = 0.0;
};

struct online_execution_plan {
  std::vector<online_execution_batch> batches;
  double grouping_ms = 0.0;
  double offset_evaluator_ms = 0.0;
  std::uint64_t grouping_alignment_score = 0;

  double evaluator_ms() const {
    return grouping_ms + offset_evaluator_ms;
  }
};

inline online_execution_plan make_online_execution_plan(
    const online_offset_evaluator& evaluator,
    const std::vector<int>& sources,
    int batch_size,
    int max_swaps = 16) {
  if (sources.empty() || batch_size <= 0 || batch_size > 64) {
    throw std::invalid_argument("invalid online runner configuration");
  }

  auto grouping = evaluator.plan_batches(sources, batch_size, max_swaps);
  online_execution_plan result;
  result.grouping_ms = grouping.evaluator_ms;
  result.grouping_alignment_score = grouping.alignment_score;
  result.batches.reserve(grouping.query_ids.size());

  for (auto& query_ids : grouping.query_ids) {
    std::vector<int> batch_sources;
    batch_sources.reserve(query_ids.size());
    for (int query_id : query_ids) {
      if (query_id < 0 || query_id >= static_cast<int>(sources.size())) {
        throw std::runtime_error("online evaluator returned an invalid query id");
      }
      batch_sources.push_back(sources[query_id]);
    }

    auto offset_plan = evaluator.evaluate(batch_sources);
    result.offset_evaluator_ms += offset_plan.evaluator_ms;
    result.batches.push_back(
        {std::move(query_ids), std::move(batch_sources),
         slot_start_schedule(std::move(offset_plan.offsets)),
         std::move(offset_plan.predicted_lengths),
         offset_plan.alignment_score, offset_plan.evaluator_ms});
  }
  return result;
}

inline online_execution_plan make_online_batching_execution_plan(
    const online_offset_evaluator& evaluator,
    const std::vector<int>& sources,
    int batch_size,
    int max_swaps = 16) {
  if (sources.empty() || batch_size <= 0 || batch_size > 64) {
    throw std::invalid_argument("invalid online batching configuration");
  }
  auto grouping = evaluator.plan_batches(sources, batch_size, max_swaps);
  online_execution_plan result;
  result.grouping_ms = grouping.evaluator_ms;
  result.grouping_alignment_score = grouping.alignment_score;
  result.batches.reserve(grouping.query_ids.size());
  for (auto& query_ids : grouping.query_ids) {
    std::vector<int> batch_sources;
    batch_sources.reserve(query_ids.size());
    for (int query_id : query_ids) batch_sources.push_back(sources[query_id]);
    std::vector<int> zero_offsets(batch_sources.size(), 0);
    result.batches.push_back(
        {std::move(query_ids), std::move(batch_sources),
         slot_start_schedule(std::move(zero_offsets)), {}, 0, 0.0});
  }
  return result;
}

inline online_execution_plan make_online_offset_execution_plan(
    const online_offset_evaluator& evaluator,
    const std::vector<online_execution_batch>& batches) {
  if (batches.empty()) {
    throw std::invalid_argument("cannot offset an empty batch plan");
  }
  online_execution_plan result;
  result.batches.reserve(batches.size());
  for (const auto& batch : batches) {
    auto offset_plan = evaluator.evaluate(batch.sources);
    result.offset_evaluator_ms += offset_plan.evaluator_ms;
    result.batches.push_back(
        {batch.query_ids, batch.sources,
         slot_start_schedule(std::move(offset_plan.offsets)),
         std::move(offset_plan.predicted_lengths),
         offset_plan.alignment_score, offset_plan.evaluator_ms});
  }
  return result;
}

inline std::vector<online_execution_batch> make_sequential_execution_plan(
    const std::vector<int>& sources,
    int batch_size) {
  if (sources.empty() || batch_size <= 0 || batch_size > 64) {
    throw std::invalid_argument("invalid sequential runner configuration");
  }
  std::vector<online_execution_batch> batches;
  for (int first = 0; first < static_cast<int>(sources.size());
       first += batch_size) {
    const int last = std::min(first + batch_size,
                              static_cast<int>(sources.size()));
    std::vector<int> query_ids;
    std::vector<int> batch_sources;
    std::vector<int> offsets;
    query_ids.reserve(last - first);
    batch_sources.reserve(last - first);
    offsets.assign(last - first, 0);
    for (int query = first; query < last; ++query) {
      query_ids.push_back(query);
      batch_sources.push_back(sources[query]);
    }
    batches.push_back({std::move(query_ids), std::move(batch_sources),
                       slot_start_schedule(std::move(offsets)), {}, 0, 0.0});
  }
  return batches;
}

}  // namespace scheduling
}  // namespace puercgp
