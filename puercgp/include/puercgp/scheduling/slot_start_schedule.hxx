#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <puercgp/core/types.hxx>

namespace puercgp {
namespace scheduling {

class slot_start_schedule {
 public:
  slot_start_schedule() = default;

  explicit slot_start_schedule(std::vector<int> offsets)
      : offsets_(std::move(offsets)) {
    if (offsets_.empty() || offsets_.size() > 64) {
      throw std::invalid_argument(
          "slot start schedule requires between 1 and 64 offsets");
    }
    if (*std::min_element(offsets_.begin(), offsets_.end()) < 0) {
      throw std::invalid_argument("slot start offsets must be non-negative");
    }
  }

  void validate(int query_count) const {
    if (query_count != static_cast<int>(offsets_.size())) {
      throw std::invalid_argument(
          "slot start schedule size must match the query count");
    }
  }

  int max_offset() const {
    return *std::max_element(offsets_.begin(), offsets_.end());
  }

  query_mask_t activation_mask(int global_step) const {
    query_mask_t result = 0;
    for (int slot = 0; slot < static_cast<int>(offsets_.size()); ++slot) {
      if (offsets_[slot] == global_step) {
        result |= query_mask_t{1} << slot;
      }
    }
    return result;
  }

  const std::vector<int>& offsets() const noexcept { return offsets_; }

 private:
  std::vector<int> offsets_;
};

}  // namespace scheduling
}  // namespace puercgp
