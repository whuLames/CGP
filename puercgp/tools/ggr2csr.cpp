#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct header_t {
  std::uint64_t version;
  std::uint64_t edge_type_size;
  std::uint64_t vertices;
  std::uint64_t edges;
};

void copy_int32_values(std::ifstream &input, const fs::path &output_path,
                       std::uint64_t count) {
  std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create " + output_path.string());
  }
  constexpr std::size_t chunk_values = 1U << 20;
  std::vector<std::int32_t> buffer(chunk_values);
  while (count != 0) {
    const auto current = static_cast<std::size_t>(
        std::min<std::uint64_t>(count, buffer.size()));
    input.read(reinterpret_cast<char *>(buffer.data()),
               static_cast<std::streamsize>(current * sizeof(std::int32_t)));
    if (!input) {
      throw std::runtime_error("truncated GR edge data");
    }
    output.write(reinterpret_cast<const char *>(buffer.data()),
                 static_cast<std::streamsize>(current * sizeof(std::int32_t)));
    if (!output) {
      throw std::runtime_error("failed writing " + output_path.string());
    }
    count -= current;
  }
}

void write_unit_weights(const fs::path &output_path, std::uint64_t count) {
  std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create " + output_path.string());
  }
  constexpr std::size_t chunk_values = 1U << 20;
  std::vector<std::int32_t> buffer(chunk_values, 1);
  while (count != 0) {
    const auto current = static_cast<std::size_t>(
        std::min<std::uint64_t>(count, buffer.size()));
    output.write(reinterpret_cast<const char *>(buffer.data()),
                 static_cast<std::streamsize>(current * sizeof(std::int32_t)));
    if (!output) {
      throw std::runtime_error("failed writing " + output_path.string());
    }
    count -= current;
  }
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3 && argc != 4) {
      throw std::invalid_argument(
          "Usage: ggr2csr <input.gr> <output-dir> [--unit-weight]");
    }
    const fs::path input_path = argv[1];
    const fs::path output_dir = argv[2];
    const bool unit_weight = argc == 4 && std::string(argv[3]) == "--unit-weight";
    if (argc == 4 && !unit_weight) {
      throw std::invalid_argument("unknown option: " + std::string(argv[3]));
    }
    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("cannot open " + input_path.string());
    }

    header_t header{};
    input.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!input || header.version != 1 || header.edge_type_size == 0 ||
        header.vertices >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        header.edges >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("unsupported weighted 32-bit GR graph");
    }

    std::vector<std::uint64_t> row_ends(
        static_cast<std::size_t>(header.vertices));
    input.read(reinterpret_cast<char *>(row_ends.data()),
               static_cast<std::streamsize>(row_ends.size() *
                                            sizeof(std::uint64_t)));
    if (!input || (!row_ends.empty() && row_ends.back() != header.edges)) {
      throw std::runtime_error("invalid GR row offsets");
    }

    std::vector<std::int32_t> row_offsets(row_ends.size() + 1);
    for (std::size_t vertex = 0; vertex < row_ends.size(); ++vertex) {
      if (row_ends[vertex] >
              static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
          (vertex != 0 && row_ends[vertex] < row_ends[vertex - 1])) {
        throw std::runtime_error("GR row offset exceeds 32-bit CSR range");
      }
      row_offsets[vertex + 1] = static_cast<std::int32_t>(row_ends[vertex]);
    }

    fs::create_directories(output_dir);
    {
      std::ofstream rows(output_dir / "csr_vlist.bin",
                         std::ios::binary | std::ios::trunc);
      rows.write(reinterpret_cast<const char *>(row_offsets.data()),
                 static_cast<std::streamsize>(row_offsets.size() *
                                              sizeof(std::int32_t)));
      if (!rows) {
        throw std::runtime_error("failed writing CSR row offsets");
      }
    }
    copy_int32_values(input, output_dir / "csr_elist.bin", header.edges);
    if ((header.edges & 1ULL) != 0) {
      input.seekg(sizeof(std::int32_t), std::ios::cur);
    }
    if (unit_weight) {
      write_unit_weights(output_dir / "csr_weightlist.bin", header.edges);
    } else {
      copy_int32_values(input, output_dir / "csr_weightlist.bin", header.edges);
    }

    std::cout << "vertices=" << header.vertices << " edges=" << header.edges
              << " output=" << output_dir
              << " unit_weight=" << (unit_weight ? 1 : 0) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
