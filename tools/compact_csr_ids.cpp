#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
template <typename T>
std::vector<T> read_array(const fs::path& path) {
  const auto bytes = fs::file_size(path);
  if (bytes % sizeof(T)) throw std::runtime_error("misaligned file: " + path.string());
  std::vector<T> values(bytes / sizeof(T));
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char*>(values.data()), std::streamsize(bytes));
  if (!input) throw std::runtime_error("failed to read: " + path.string());
  return values;
}

void write_value(std::ofstream& output, int32_t value, const char* label) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!output) throw std::runtime_error(std::string("failed to write ") + label);
}
}

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::invalid_argument("usage: graphweft_compact_csr_ids INPUT_CSR OUTPUT_CSR");
    const fs::path input = fs::absolute(argv[1]);
    const fs::path output = fs::absolute(argv[2]);
    if (fs::exists(output)) throw std::invalid_argument("output already exists: " + output.string());
    const auto row_path = input / "csr_vlist.bin";
    const auto edge_path = input / "csr_elist.bin";
    const auto weight_path = input / "csr_weightlist.bin";
    if (!fs::is_regular_file(row_path) || !fs::is_regular_file(edge_path) || !fs::is_regular_file(weight_path))
      throw std::runtime_error("input CSR is incomplete");

    const auto rows = read_array<int32_t>(row_path);
    if (rows.size() < 2 || rows.front() != 0) throw std::runtime_error("invalid CSR row offsets");
    const uint64_t old_vertices = rows.size() - 1;
    const uint64_t edges = fs::file_size(edge_path) / sizeof(int32_t);
    if (fs::file_size(edge_path) % sizeof(int32_t) || fs::file_size(weight_path) != edges * sizeof(int32_t) ||
        rows.back() < 0 || uint64_t(rows.back()) != edges)
      throw std::runtime_error("CSR edge/weight/offset sizes disagree");
    for (size_t i = 1; i < rows.size(); ++i)
      if (rows[i] < rows[i - 1]) throw std::runtime_error("CSR row offsets are not monotonic");

    std::vector<uint32_t> remap(old_vertices, std::numeric_limits<uint32_t>::max());
    std::vector<uint32_t> external_ids;
    external_ids.reserve(old_vertices);
    for (uint64_t vertex = 0; vertex < old_vertices; ++vertex) {
      if (rows[vertex] == rows[vertex + 1]) continue;
      if (external_ids.size() >= uint64_t(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("compacted vertex count exceeds int32 CSR limit");
      remap[vertex] = uint32_t(external_ids.size());
      external_ids.push_back(uint32_t(vertex));
    }
    if (external_ids.empty()) throw std::runtime_error("CSR has no non-isolated vertices");

    fs::create_directories(output);
    std::ifstream edge_input(edge_path, std::ios::binary), weight_input(weight_path, std::ios::binary);
    std::ofstream row_output(output / "csr_vlist.bin", std::ios::binary),
      edge_output(output / "csr_elist.bin", std::ios::binary),
      weight_output(output / "csr_weightlist.bin", std::ios::binary),
      map_output(output / "external_vertex_ids.bin", std::ios::binary);
    if (!edge_input || !weight_input || !row_output || !edge_output || !weight_output || !map_output)
      throw std::runtime_error("failed to open compacted CSR files");

    int32_t compact_offset = 0;
    write_value(row_output, compact_offset, "row offset");
    constexpr size_t block_entries = 1 << 20;
    std::vector<int32_t> destinations(block_entries), weights(block_entries);
    for (uint64_t old_vertex = 0; old_vertex < old_vertices; ++old_vertex) {
      uint64_t remaining = uint64_t(rows[old_vertex + 1] - rows[old_vertex]);
      while (remaining) {
        const size_t count = size_t(std::min<uint64_t>(remaining, block_entries));
        edge_input.read(reinterpret_cast<char*>(destinations.data()), std::streamsize(count * sizeof(int32_t)));
        weight_input.read(reinterpret_cast<char*>(weights.data()), std::streamsize(count * sizeof(int32_t)));
        if (!edge_input || !weight_input) throw std::runtime_error("failed while streaming source CSR");
        for (size_t i = 0; i < count; ++i) {
          const int32_t destination = destinations[i];
          if (destination < 0 || uint64_t(destination) >= old_vertices ||
              remap[size_t(destination)] == std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("edge references an invalid or isolated destination");
          destinations[i] = int32_t(remap[size_t(destination)]);
        }
        edge_output.write(reinterpret_cast<const char*>(destinations.data()), std::streamsize(count * sizeof(int32_t)));
        weight_output.write(reinterpret_cast<const char*>(weights.data()), std::streamsize(count * sizeof(int32_t)));
        if (!edge_output || !weight_output) throw std::runtime_error("failed while writing compacted CSR");
        remaining -= count;
      }
      if (remap[old_vertex] != std::numeric_limits<uint32_t>::max()) {
        compact_offset = rows[old_vertex + 1];
        write_value(row_output, compact_offset, "row offset");
      }
    }
    map_output.write(reinterpret_cast<const char*>(external_ids.data()),
                     std::streamsize(external_ids.size() * sizeof(uint32_t)));
    if (!map_output) throw std::runtime_error("failed to write external vertex mapping");

    std::ofstream manifest(output / "conversion_manifest.json");
    manifest << "{\n  \"schema\": 1,\n  \"construction\": \"compact_existing_symmetric_csr\",\n"
      << "  \"source\": \"" << input.string() << "\",\n  \"old_vertices\": " << old_vertices
      << ",\n  \"vertices\": " << external_ids.size() << ",\n  \"symmetric_edges\": " << edges
      << ",\n  \"external_ids_compacted\": true,\n  \"external_vertex_map\": \"external_vertex_ids.bin\"\n}\n";
    if (!manifest) throw std::runtime_error("failed to write conversion manifest");
    std::cout << "old_vertices=" << old_vertices << " vertices=" << external_ids.size()
              << " symmetric_edges=" << edges << " dropped_isolated_rows="
              << old_vertices - external_ids.size() << '\n';
  } catch (const std::exception& error) {
    std::cerr << "compact_csr_ids: " << error.what() << '\n';
    return 1;
  }
}
