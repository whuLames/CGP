#pragma once

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace puercgp {

template <typename query_t> class algorithm_query_batch {
public:
  using query_type = query_t;
  using container_type = std::vector<query_t>;
  using const_iterator = typename container_type::const_iterator;

  algorithm_query_batch() = default;

  algorithm_query_batch(std::initializer_list<query_t> queries)
      : queries_(queries) {}

  explicit algorithm_query_batch(container_type queries)
      : queries_(std::move(queries)) {}

  std::size_t size() const noexcept { return queries_.size(); }
  bool empty() const noexcept { return queries_.empty(); }
  const container_type &queries() const noexcept { return queries_; }
  const query_t &operator[](std::size_t index) const {
    return queries_.at(index);
  }
  const_iterator begin() const noexcept { return queries_.begin(); }
  const_iterator end() const noexcept { return queries_.end(); }

  void validate(std::size_t max_queries) const {
    if (queries_.empty()) {
      throw std::invalid_argument(
          "algorithm_query_batch must contain at least one query");
    }
    if (queries_.size() > 64) {
      throw std::invalid_argument(
          "algorithm_query_batch supports at most 64 queries");
    }
    if (max_queries != 0 && queries_.size() > max_queries) {
      throw std::invalid_argument("algorithm_query_batch exceeds max_queries");
    }
  }

private:
  container_type queries_;
};

} // namespace puercgp
