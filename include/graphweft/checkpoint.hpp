#pragma once
#include "graphweft/engine.hpp"
#include <array>
#include <cstdint>
#include <string>
#include <vector>
namespace graphweft {
struct Checkpoint {
  std::string graph_identity;
  Algorithm algorithm;
  Layout layout;
  uint32_t vertices, slots, physical_slots, group_width, words, frontier_count, round;
  std::vector<float> old_values;
  std::vector<uint64_t> frontier_mask;
  std::vector<uint32_t> frontier;
  std::vector<uint8_t> live_slots;
};
void save_checkpoint(const Checkpoint&, const std::string& path);
Checkpoint load_checkpoint(const std::string& path);
}
