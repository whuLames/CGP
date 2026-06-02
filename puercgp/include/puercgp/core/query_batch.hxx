#pragma once

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace puercgp {

template <typename vertex_t>
class query_batch {
  static_assert(std::is_integral<vertex_t>::value,
                "query_batch vertex_t must be an integral type");

 public:
  using vertex_type = vertex_t;
  using container_type = std::vector<vertex_t>;
  using const_iterator = typename container_type::const_iterator;

  query_batch() = default;

  query_batch(std::initializer_list<vertex_t> sources) : sources_(sources) {}

  explicit query_batch(container_type sources) : sources_(std::move(sources)) {}

  std::size_t size() const noexcept { return sources_.size(); }

  bool empty() const noexcept { return sources_.empty(); }

  const vertex_t* data() const noexcept { return sources_.data(); }

  const container_type& sources() const noexcept { return sources_; }

  const vertex_t& operator[](std::size_t index) const {
    return sources_.at(index);
  }

  const_iterator begin() const noexcept { return sources_.begin(); }

  const_iterator end() const noexcept { return sources_.end(); }

  void validate(std::size_t max_queries) const {
    if (sources_.empty()) {
      throw std::invalid_argument("query_batch must contain at least one source");
    }
    if (max_queries != 0 && sources_.size() > max_queries) {
      throw std::invalid_argument("query_batch exceeds max_queries");
    }
    for (auto source : sources_) {
      if (source < 0) {
        throw std::invalid_argument("query_batch sources must be non-negative");
      }
    }
  }

 private:
  container_type sources_;
};

}  // namespace puercgp
