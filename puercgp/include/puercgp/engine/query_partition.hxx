#pragma once

#include <puercgp/core/types.hxx>

namespace puercgp {

inline query_mask_t query_slots_mask(int query_count) {
  return query_count >= 64 ? ~query_mask_t{0}
                           : ((query_mask_t{1} << query_count) - 1);
}

struct query_partition_t {
  query_mask_t active_slots = ~query_mask_t{0};
  traversal_mode_t mode = traversal_mode_t::hybrid;

  static query_partition_t all_slots(int query_count,
                                     traversal_mode_t mode_value) {
    return {query_slots_mask(query_count), mode_value};
  }
};

}  // namespace puercgp
