#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include <puercgp/algorithms/hybrid.hxx>
#include <puercgp/core/types.hxx>
#include <puercgp/kernels/replenish/slot_io_kernels.hxx>

namespace puercgp {

class slot_io_manager {
 public:
  static void snapshot_slot(int slot,
                            algorithms::algo_kind_t kind,
                            int query_count,
                            std::size_t vertex_count,
                            const algorithms::unified_value_t* values,
                            const query_mask_t* visited_mask,
                            algorithms::unified_value_t* final_row,
                            cudaStream_t stream) {
    hybrid_detail::launch_snapshot_slot_values(
        slot, kind, query_count, vertex_count, values, visited_mask, final_row,
        stream);
  }

  static void snapshot_slots(
      const int* converged_slots,
      const int* final_query_ids,
      const algorithms::algo_kind_t* converged_kinds,
      int num_converged,
      int query_count,
      std::size_t vertex_count,
      const algorithms::unified_value_t* values,
      const query_mask_t* visited_mask,
      algorithms::unified_value_t* final_buffer,
      std::size_t final_row_stride,
      cudaStream_t stream) {
    hybrid_detail::launch_snapshot_multi_slot(
        converged_slots, final_query_ids, num_converged, converged_kinds,
        query_count, vertex_count, values, visited_mask, final_buffer,
        final_row_stride, stream);
  }

  static void reset_reused_slots(
      const int* value_reset_slots,
      int num_value_reset_slots,
      query_mask_t visited_clear_bits,
      int query_count,
      std::size_t vertex_count,
      algorithms::unified_value_t* values,
      query_mask_t* visited_mask,
      cudaStream_t stream) {
    hybrid_detail::launch_reset_reused_slots(
        value_reset_slots, num_value_reset_slots, visited_clear_bits,
        query_count, vertex_count, values, visited_mask, stream);
  }

  static void reinit_slots(const int* slots,
                           const algorithms::algo_kind_t* kinds,
                           const int* sources,
                           const algorithms::unified_value_t* source_values,
                           int num_slots,
                           int query_count,
                           algorithms::unified_value_t* values,
                           query_mask_t* visited_mask,
                           query_mask_t* frontier_mask,
                           int* frontier_vertices,
                           unsigned long long* unique_count,
                           int* slot_start_levels,
                           int start_level,
                           cudaStream_t stream) {
    hybrid_detail::launch_set_sources_multi(
        slots, kinds, sources, source_values, num_slots, query_count, values,
        visited_mask, frontier_mask, frontier_vertices, unique_count,
        slot_start_levels, start_level, stream);
  }

};

}  // namespace puercgp
