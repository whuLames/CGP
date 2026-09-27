#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace graphweft {
struct HostGraph {
  uint32_t vertices = 0;
  bool directed = false;
  std::vector<uint64_t> row;
  std::vector<uint32_t> col;
  std::vector<float> weight;
  std::vector<uint64_t> incoming_row;
  std::vector<uint32_t> incoming_col;
  std::vector<float> incoming_weight;
  std::string identity;
  static HostGraph from_edges(uint32_t vertices, const std::vector<uint32_t>& source,
                              const std::vector<uint32_t>& destination,
                              const std::vector<float>& weights, bool directed);
  static HostGraph load(const std::string& path, bool directed, bool legacy_int_weights = false);
  uint64_t edges() const { return col.size(); }
};
struct GraphView {
  uint32_t vertices;
  uint64_t edges;
  const uint64_t* out_row;
  const uint32_t* out_col;
  const float* out_weight;
  const uint64_t* in_row;
  const uint32_t* in_col;
  const float* in_weight;
};
}
