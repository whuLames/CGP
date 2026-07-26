#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace puercgp {
namespace scheduling {

class landmark_phase_index {
 public:
  static constexpr std::uint16_t unreachable =
      std::numeric_limits<std::uint16_t>::max();

  template <typename row_container_t, typename column_container_t>
  static landmark_phase_index build(const row_container_t& row_offsets,
                                    const column_container_t& column_indices,
                                    int landmark_count) {
    if (row_offsets.size() < 2 || landmark_count <= 0 ||
        landmark_count > static_cast<int>(row_offsets.size() - 1)) {
      throw std::invalid_argument("invalid graph or landmark count");
    }
    landmark_phase_index index;
    index.vertex_count_ = static_cast<int>(row_offsets.size() - 1);
    index.landmark_count_ = landmark_count;
    index.phase_landmark_count_ = std::min(4, landmark_count);
    index.landmarks_.reserve(landmark_count);
    index.landmark_weights_.reserve(landmark_count);
    index.distances_.reserve(
        static_cast<std::size_t>(index.vertex_count_) * landmark_count);

    int first = 0;
    int best_degree = -1;
    for (int vertex = 0; vertex < index.vertex_count_; ++vertex) {
      int degree = static_cast<int>(row_offsets[vertex + 1] -
                                    row_offsets[vertex]);
      if (degree > best_degree) {
        best_degree = degree;
        first = vertex;
      }
    }

    std::vector<std::uint16_t> nearest(index.vertex_count_, unreachable);
    std::vector<char> selected(index.vertex_count_, 0);
    int landmark = first;
    for (int i = 0; i < index.phase_landmark_count_; ++i) {
      auto distances = bfs_distances(row_offsets, column_indices, landmark);
      index.landmarks_.push_back(landmark);
      index.landmark_weights_.push_back(0);
      selected[landmark] = 1;
      index.distances_.insert(index.distances_.end(), distances.begin(),
                              distances.end());
      for (int vertex = 0; vertex < index.vertex_count_; ++vertex) {
        nearest[vertex] = std::min(nearest[vertex], distances[vertex]);
      }

      int next = -1;
      std::uint16_t farthest = 0;
      for (int vertex = 0; vertex < index.vertex_count_; ++vertex) {
        if (selected[vertex]) continue;
        if (nearest[vertex] == unreachable) {
          next = vertex;
          break;
        }
        if (next < 0 || nearest[vertex] > farthest) {
          farthest = nearest[vertex];
          next = vertex;
        }
      }
      if (next >= 0) landmark = next;
    }

    const std::uint64_t edge_count =
        static_cast<std::uint64_t>(column_indices.size());
    std::uint64_t random_state = 0x9E3779B97F4A7C15ULL;
    int failed_samples = 0;
    while (static_cast<int>(index.landmarks_.size()) < landmark_count) {
      random_state = random_state * 2862933555777941757ULL + 3037000493ULL;
      int vertex = 0;
      if (edge_count == 0) {
        vertex = static_cast<int>(random_state % index.vertex_count_);
      } else {
        auto edge = static_cast<typename row_container_t::value_type>(
            random_state % edge_count);
        vertex = static_cast<int>(
            std::upper_bound(row_offsets.begin(), row_offsets.end(), edge) -
            row_offsets.begin() - 1);
      }
      if (vertex < 0 || vertex >= index.vertex_count_ || selected[vertex]) {
        if (++failed_samples >= 64) {
          vertex = static_cast<int>(random_state % index.vertex_count_);
          while (selected[vertex]) {
            vertex = (vertex + 1) % index.vertex_count_;
          }
          failed_samples = 0;
        } else {
          continue;
        }
      }
      if (selected[vertex])
        continue;
      failed_samples = 0;
      auto distances = bfs_distances(row_offsets, column_indices, vertex);
      index.landmarks_.push_back(vertex);
      index.landmark_weights_.push_back(1);
      index.distances_.insert(index.distances_.end(), distances.begin(),
                              distances.end());
      selected[vertex] = 1;
    }
    return index;
  }

  static landmark_phase_index load(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open phase index: " + path.string());
    header file_header{};
    input.read(reinterpret_cast<char*>(&file_header), sizeof(file_header));
    if (!input || file_header.magic != kMagic || file_header.version != 2 ||
        file_header.vertex_count <= 0 || file_header.landmark_count <= 0) {
      throw std::runtime_error("invalid phase index: " + path.string());
    }
    landmark_phase_index index;
    index.vertex_count_ = file_header.vertex_count;
    index.landmark_count_ = file_header.landmark_count;
    index.phase_landmark_count_ = file_header.phase_landmark_count;
    index.landmarks_.resize(index.landmark_count_);
    index.landmark_weights_.resize(index.landmark_count_);
    index.distances_.resize(static_cast<std::size_t>(index.vertex_count_) *
                            index.landmark_count_);
    input.read(reinterpret_cast<char*>(index.landmarks_.data()),
               index.landmarks_.size() * sizeof(int));
    input.read(reinterpret_cast<char*>(index.landmark_weights_.data()),
               index.landmark_weights_.size() * sizeof(std::uint64_t));
    input.read(reinterpret_cast<char*>(index.distances_.data()),
               index.distances_.size() * sizeof(std::uint16_t));
    if (!input) throw std::runtime_error("truncated phase index: " + path.string());
    return index;
  }

  void save(const std::filesystem::path& path) const {
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create phase index: " + path.string());
    header file_header{kMagic, 2, vertex_count_, landmark_count_,
                       phase_landmark_count_};
    output.write(reinterpret_cast<const char*>(&file_header),
                 sizeof(file_header));
    output.write(reinterpret_cast<const char*>(landmarks_.data()),
                 landmarks_.size() * sizeof(int));
    output.write(reinterpret_cast<const char*>(landmark_weights_.data()),
                 landmark_weights_.size() * sizeof(std::uint64_t));
    output.write(reinterpret_cast<const char*>(distances_.data()),
                 distances_.size() * sizeof(std::uint16_t));
    if (!output) throw std::runtime_error("failed to write phase index: " + path.string());
  }

  int vertex_count() const { return vertex_count_; }
  int landmark_count() const { return landmark_count_; }
  int phase_landmark_count() const { return phase_landmark_count_; }
  const std::vector<int>& landmarks() const { return landmarks_; }

  std::uint64_t landmark_weight(int landmark_id) const {
    return landmark_weights_[landmark_id];
  }

  std::uint16_t distance(int landmark_id, int vertex) const {
    if (landmark_id < 0 || landmark_id >= landmark_count_ || vertex < 0 ||
        vertex >= vertex_count_) {
      return unreachable;
    }
    return distances_[static_cast<std::size_t>(landmark_id) * vertex_count_ +
                      vertex];
  }

  int estimate_phase_length(int source) const {
    int estimate = 1;
    for (int landmark = 0; landmark < phase_landmark_count_; ++landmark) {
      auto value = distance(landmark, source);
      if (value != unreachable) estimate = std::max(estimate, int(value) + 1);
    }
    return estimate;
  }

  std::uint64_t estimate_alignment_affinity(int first, int second,
                                             int max_offset) const {
    if (max_offset < 0) return 0;
    const int width = 2 * max_offset + 1;
    std::vector<std::uint64_t> scores(width, 0);
    for (int landmark = phase_landmark_count_; landmark < landmark_count_;
         ++landmark) {
      auto first_distance = distance(landmark, first);
      auto second_distance = distance(landmark, second);
      if (first_distance == unreachable || second_distance == unreachable)
        continue;
      int delta = int(second_distance) - int(first_distance);
      if (std::abs(delta) <= max_offset) {
        scores[delta + max_offset] += landmark_weights_[landmark];
      }
    }
    return *std::max_element(scores.begin(), scores.end());
  }

 private:
  struct header {
    std::uint64_t magic;
    int version;
    int vertex_count;
    int landmark_count;
    int phase_landmark_count;
  };

  static constexpr std::uint64_t kMagic = 0x505545524F464653ULL;

  template <typename row_container_t, typename column_container_t>
  static std::vector<std::uint16_t> bfs_distances(
      const row_container_t& row_offsets,
      const column_container_t& column_indices, int source) {
    const int vertex_count = static_cast<int>(row_offsets.size() - 1);
    std::vector<std::uint16_t> distances(vertex_count, unreachable);
    std::vector<int> queue;
    queue.reserve(vertex_count);
    distances[source] = 0;
    queue.push_back(source);
    for (std::size_t head = 0; head < queue.size(); ++head) {
      int vertex = queue[head];
      std::uint16_t next_distance =
          distances[vertex] == unreachable - 1
              ? distances[vertex]
              : static_cast<std::uint16_t>(distances[vertex] + 1);
      for (auto edge = row_offsets[vertex]; edge < row_offsets[vertex + 1];
           ++edge) {
        int neighbor = column_indices[edge];
        if (distances[neighbor] != unreachable) continue;
        distances[neighbor] = next_distance;
        queue.push_back(neighbor);
      }
    }
    return distances;
  }

  int vertex_count_ = 0;
  int landmark_count_ = 0;
  int phase_landmark_count_ = 0;
  std::vector<int> landmarks_;
  std::vector<std::uint64_t> landmark_weights_;
  std::vector<std::uint16_t> distances_;
};

struct online_offset_plan {
  std::vector<int> predicted_lengths;
  std::vector<int> offsets;
  double evaluator_ms = 0.0;
  std::uint64_t alignment_score = 0;
};

struct online_batch_plan {
  std::vector<std::vector<int>> query_ids;
  double evaluator_ms = 0.0;
  std::uint64_t alignment_score = 0;
};

class online_offset_evaluator {
 public:
  explicit online_offset_evaluator(const landmark_phase_index& index,
                                   int max_offset = 16)
      : index_(index), max_offset_(max_offset) {
    if (max_offset < 0) {
      throw std::invalid_argument("max offset must be non-negative");
    }
  }

  online_batch_plan plan_batches(const std::vector<int>& sources,
                                 int batch_size,
                                 int max_swaps = 16) const {
    if (sources.empty() || batch_size <= 0 || max_swaps < 0) {
      throw std::invalid_argument("invalid online batch configuration");
    }
    auto start = std::chrono::steady_clock::now();
    const int query_count = static_cast<int>(sources.size());
    std::vector<std::vector<std::uint64_t>> affinity(
        query_count, std::vector<std::uint64_t>(query_count, 0));
    for (int query = 0; query < query_count; ++query) {
      for (int other = query + 1; other < query_count; ++other) {
        affinity[query][other] = affinity[other][query] =
            index_.estimate_alignment_affinity(sources[query], sources[other],
                                               max_offset_);
      }
    }

    online_batch_plan plan;
    std::vector<char> assigned(query_count, 0);
    int remaining = query_count;
    while (remaining > 0) {
      int seed = -1;
      std::uint64_t best_total = 0;
      for (int query = 0; query < query_count; ++query) {
        if (assigned[query]) continue;
        std::uint64_t total = 0;
        for (int other = 0; other < query_count; ++other) {
          if (!assigned[other]) total += affinity[query][other];
        }
        if (seed < 0 || total > best_total) {
          seed = query;
          best_total = total;
        }
      }

      std::vector<int> batch{seed};
      assigned[seed] = 1;
      --remaining;
      int target_size = std::min(batch_size, remaining + 1);
      while (static_cast<int>(batch.size()) < target_size) {
        int best = -1;
        std::uint64_t best_score = 0;
        for (int query = 0; query < query_count; ++query) {
          if (assigned[query]) continue;
          std::uint64_t score = 0;
          for (int member : batch) score += affinity[query][member];
          if (best < 0 || score > best_score) {
            best = query;
            best_score = score;
          }
        }
        batch.push_back(best);
        assigned[best] = 1;
        --remaining;
      }
      plan.query_ids.push_back(std::move(batch));
    }

    for (int pass = 0; pass < max_swaps; ++pass) {
      std::int64_t best_gain = 0;
      int best_first_batch = -1;
      int best_first_slot = -1;
      int best_second_batch = -1;
      int best_second_slot = -1;
      for (int first_batch = 0;
           first_batch < static_cast<int>(plan.query_ids.size());
           ++first_batch) {
        for (int second_batch = first_batch + 1;
             second_batch < static_cast<int>(plan.query_ids.size());
             ++second_batch) {
          for (int first_slot = 0;
               first_slot < static_cast<int>(plan.query_ids[first_batch].size());
               ++first_slot) {
            int first = plan.query_ids[first_batch][first_slot];
            for (int second_slot = 0;
                 second_slot <
                     static_cast<int>(plan.query_ids[second_batch].size());
                 ++second_slot) {
              int second = plan.query_ids[second_batch][second_slot];
              std::int64_t gain = 0;
              for (int member : plan.query_ids[first_batch]) {
                if (member == first) continue;
                gain += static_cast<std::int64_t>(affinity[second][member]) -
                        static_cast<std::int64_t>(affinity[first][member]);
              }
              for (int member : plan.query_ids[second_batch]) {
                if (member == second) continue;
                gain += static_cast<std::int64_t>(affinity[first][member]) -
                        static_cast<std::int64_t>(affinity[second][member]);
              }
              if (gain > best_gain) {
                best_gain = gain;
                best_first_batch = first_batch;
                best_first_slot = first_slot;
                best_second_batch = second_batch;
                best_second_slot = second_slot;
              }
            }
          }
        }
      }
      if (best_gain <= 0) break;
      std::swap(plan.query_ids[best_first_batch][best_first_slot],
                plan.query_ids[best_second_batch][best_second_slot]);
    }

    for (const auto& batch : plan.query_ids) {
      for (std::size_t first = 0; first < batch.size(); ++first) {
        for (std::size_t second = first + 1; second < batch.size(); ++second) {
          plan.alignment_score += affinity[batch[first]][batch[second]];
        }
      }
    }
    plan.evaluator_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count();
    return plan;
  }

  online_offset_plan evaluate(const std::vector<int>& sources) const {
    if (sources.empty()) {
      throw std::invalid_argument("cannot evaluate an empty query batch");
    }
    auto start = std::chrono::steady_clock::now();
    online_offset_plan plan;
    plan.predicted_lengths.reserve(sources.size());
    int longest = 0;
    for (int source : sources) {
      int length = index_.estimate_phase_length(source);
      plan.predicted_lengths.push_back(length);
      longest = std::max(longest, length);
    }
    plan.offsets.reserve(sources.size());
    for (int length : plan.predicted_lengths) {
      plan.offsets.push_back(std::min(max_offset_, longest - length));
    }
    int minimum = *std::min_element(plan.offsets.begin(), plan.offsets.end());
    for (int& offset : plan.offsets) offset -= minimum;

    const int query_count = static_cast<int>(sources.size());
    const int width = 2 * max_offset_ + 1;
    std::vector<std::uint64_t> pair_scores(
        static_cast<std::size_t>(query_count) * query_count * width, 0);
    auto score_index = [&](int first, int second, int delta) {
      return (static_cast<std::size_t>(first) * query_count + second) * width +
             delta + max_offset_;
    };
    for (int first = 0; first < query_count; ++first) {
      for (int second = first + 1; second < query_count; ++second) {
        for (int landmark = 0; landmark < index_.landmark_count(); ++landmark) {
          auto first_distance = index_.distance(landmark, sources[first]);
          auto second_distance = index_.distance(landmark, sources[second]);
          if (first_distance == landmark_phase_index::unreachable ||
              second_distance == landmark_phase_index::unreachable)
            continue;
          int delta = int(second_distance) - int(first_distance);
          if (std::abs(delta) > max_offset_) continue;
          std::uint64_t weight = index_.landmark_weight(landmark);
          pair_scores[score_index(first, second, delta)] += weight;
          pair_scores[score_index(second, first, -delta)] += weight;
        }
      }
    }

    auto total_score = [&](const std::vector<int>& offsets) {
      std::uint64_t score = 0;
      for (int first = 0; first < query_count; ++first) {
        for (int second = first + 1; second < query_count; ++second) {
          int delta = offsets[first] - offsets[second];
          if (std::abs(delta) <= max_offset_) {
            score += pair_scores[score_index(first, second, delta)];
          }
        }
      }
      return score;
    };

    std::vector<std::vector<int>> starts(2,
                                         std::vector<int>(query_count, 0));
    starts[1] = plan.offsets;
    std::mt19937 random(42);
    for (int start_id = 0; start_id < 6; ++start_id) {
      std::vector<int> random_start(query_count);
      for (int& offset : random_start) {
        offset = static_cast<int>(random() % (max_offset_ + 1));
      }
      starts.push_back(std::move(random_start));
    }
    std::vector<int> best = plan.offsets;
    std::uint64_t best_score = total_score(best);
    for (auto offsets : starts) {
      for (int pass = 0; pass < 8; ++pass) {
        bool changed = false;
        for (int query = 0; query < query_count; ++query) {
          int best_offset = offsets[query];
          std::uint64_t best_local = 0;
          int completion_offset = plan.offsets[query];
          for (int candidate = 0; candidate <= max_offset_; ++candidate) {
            std::uint64_t local = 0;
            for (int other = 0; other < query_count; ++other) {
              if (query == other) continue;
              int delta = candidate - offsets[other];
              if (std::abs(delta) <= max_offset_) {
                local += pair_scores[score_index(query, other, delta)];
              }
            }
            if (local > best_local ||
                (local == best_local &&
                 std::abs(candidate - completion_offset) <
                     std::abs(best_offset - completion_offset))) {
              best_local = local;
              best_offset = candidate;
            }
          }
          changed |= best_offset != offsets[query];
          offsets[query] = best_offset;
        }
        if (!changed) break;
      }
      int min_offset = *std::min_element(offsets.begin(), offsets.end());
      for (int& offset : offsets) offset -= min_offset;
      std::uint64_t score = total_score(offsets);
      if (score > best_score) {
        best_score = score;
        best = std::move(offsets);
      }
    }
    plan.offsets = std::move(best);
    plan.alignment_score = best_score;
    auto stop = std::chrono::steady_clock::now();
    plan.evaluator_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    return plan;
  }

 private:
  const landmark_phase_index& index_;
  int max_offset_ = 16;
};

}  // namespace scheduling
}  // namespace puercgp
