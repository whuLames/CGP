#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

namespace fs = std::filesystem;

enum class algorithm_t { bfs, sssp, sswp };

bool is_minimum(algorithm_t algorithm) {
  return algorithm != algorithm_t::sswp;
}

const char *algorithm_name(algorithm_t algorithm) {
  if (algorithm == algorithm_t::bfs) return "bfs";
  return algorithm == algorithm_t::sssp ? "sssp" : "sswp";
}

struct options_t {
  fs::path input_dir;
  fs::path output_dir;
  int maximum_degree = 4;
  int threads = 0;
  algorithm_t algorithm = algorithm_t::sssp;
};

struct csr_t {
  int vertices = 0;
  std::vector<int> row_offsets;
  std::vector<int> columns;
  std::vector<int> weights;
};

struct edge_t {
  int neighbor;
  int weight;
};

double elapsed_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

options_t parse_options(int argc, char **argv) {
  if (argc < 3) {
    throw std::invalid_argument(
        "Usage: add_low_degree_shortcuts <input-dir> <output-dir> "
        "[--max-degree=4] [--threads=0] [--algorithm=bfs|sssp|sswp]");
  }
  options_t options;
  options.input_dir = argv[1];
  options.output_dir = argv[2];
  for (int i = 3; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument.rfind("--max-degree=", 0) == 0) {
      options.maximum_degree =
          std::stoi(argument.substr(std::string("--max-degree=").size()));
    } else if (argument.rfind("--threads=", 0) == 0) {
      options.threads =
          std::stoi(argument.substr(std::string("--threads=").size()));
    } else if (argument == "--algorithm=bfs") {
      options.algorithm = algorithm_t::bfs;
    } else if (argument == "--algorithm=sssp") {
      options.algorithm = algorithm_t::sssp;
    } else if (argument == "--algorithm=sswp") {
      options.algorithm = algorithm_t::sswp;
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  if (options.maximum_degree < 1 || options.threads < 0) {
    throw std::invalid_argument("invalid shortcut options");
  }
  return options;
}

template <typename value_t>
std::vector<value_t> read_binary_vector(const fs::path &path) {
  if (!fs::is_regular_file(path)) {
    throw std::runtime_error("missing input file: " + path.string());
  }
  const auto bytes = fs::file_size(path);
  if (bytes % sizeof(value_t) != 0) {
    throw std::runtime_error("invalid binary file size: " + path.string());
  }
  std::vector<value_t> values(bytes / sizeof(value_t));
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char *>(values.data()),
             static_cast<std::streamsize>(bytes));
  if (!input) {
    throw std::runtime_error("failed to read: " + path.string());
  }
  return values;
}

template <typename value_t>
void write_binary_vector(const fs::path &path,
                         const std::vector<value_t> &values) {
  const fs::path temporary(path.string() + ".tmp");
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(value_t)));
  output.close();
  if (!output) {
    throw std::runtime_error("failed to write: " + path.string());
  }
  fs::rename(temporary, path);
}

csr_t load_csr(const fs::path &directory) {
  const auto start = std::chrono::steady_clock::now();
  csr_t graph;
  graph.row_offsets = read_binary_vector<int>(directory / "csr_vlist.bin");
  graph.columns = read_binary_vector<int>(directory / "csr_elist.bin");
  graph.weights =
      read_binary_vector<int>(directory / "csr_weightlist.bin");
  if (graph.row_offsets.size() < 2 ||
      graph.row_offsets.size() - 1 >
          static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      graph.columns.size() != graph.weights.size() ||
      graph.columns.size() >
          static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("invalid 32-bit weighted CSR");
  }
  graph.vertices = static_cast<int>(graph.row_offsets.size() - 1);
  if (graph.row_offsets.front() != 0 ||
      graph.row_offsets.back() != static_cast<int>(graph.columns.size())) {
    throw std::runtime_error("inconsistent CSR offsets");
  }
  for (int weight : graph.weights) {
    if (weight <= 0 || weight > std::numeric_limits<int>::max() / 2) {
      throw std::runtime_error("shortcut weights require positive int sums");
    }
  }
  std::cout << "load_ms=" << elapsed_ms(start)
            << " vertices=" << graph.vertices
            << " edges=" << graph.columns.size() << '\n';
  return graph;
}

int degree(const csr_t &graph, int vertex) {
  return graph.row_offsets[static_cast<std::size_t>(vertex) + 1] -
         graph.row_offsets[static_cast<std::size_t>(vertex)];
}

csr_t add_shortcuts(const csr_t &graph, int maximum_degree,
                    algorithm_t algorithm) {
  const auto start = std::chrono::steady_clock::now();
  std::vector<std::uint64_t> expanded_offsets(
      static_cast<std::size_t>(graph.vertices) + 1);

#pragma omp parallel for schedule(dynamic, 2048)
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    std::uint64_t count = static_cast<std::uint64_t>(degree(graph, vertex));
    for (int edge = graph.row_offsets[static_cast<std::size_t>(vertex)];
         edge < graph.row_offsets[static_cast<std::size_t>(vertex) + 1];
         ++edge) {
      const int center = graph.columns[static_cast<std::size_t>(edge)];
      const int center_degree = degree(graph, center);
      if (center_degree >= 2 && center_degree <= maximum_degree) {
        for (int second =
                 graph.row_offsets[static_cast<std::size_t>(center)];
             second <
             graph.row_offsets[static_cast<std::size_t>(center) + 1];
             ++second) {
          count += graph.columns[static_cast<std::size_t>(second)] != vertex;
        }
      }
    }
    expanded_offsets[static_cast<std::size_t>(vertex) + 1] = count;
  }
  for (std::size_t i = 1; i < expanded_offsets.size(); ++i) {
    expanded_offsets[i] += expanded_offsets[i - 1];
  }
  if (expanded_offsets.back() >
      static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("expanded graph exceeds 32-bit CSR limit");
  }

  std::vector<edge_t> expanded(
      static_cast<std::size_t>(expanded_offsets.back()));
#pragma omp parallel for schedule(dynamic, 512)
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    std::size_t position =
        static_cast<std::size_t>(expanded_offsets[vertex]);
    const int begin = graph.row_offsets[static_cast<std::size_t>(vertex)];
    const int end = graph.row_offsets[static_cast<std::size_t>(vertex) + 1];
    for (int edge = begin; edge < end; ++edge) {
      expanded[position++] = {graph.columns[static_cast<std::size_t>(edge)],
                              graph.weights[static_cast<std::size_t>(edge)]};
    }
    for (int edge = begin; edge < end; ++edge) {
      const int center = graph.columns[static_cast<std::size_t>(edge)];
      const int center_degree = degree(graph, center);
      if (center_degree < 2 || center_degree > maximum_degree) {
        continue;
      }
      const int first_weight = graph.weights[static_cast<std::size_t>(edge)];
      for (int second = graph.row_offsets[static_cast<std::size_t>(center)];
           second <
           graph.row_offsets[static_cast<std::size_t>(center) + 1];
           ++second) {
        const int neighbor = graph.columns[static_cast<std::size_t>(second)];
        if (neighbor == vertex) {
          continue;
        }
        const int second_weight =
            graph.weights[static_cast<std::size_t>(second)];
        expanded[position++] = {
            neighbor, is_minimum(algorithm)
                          ? first_weight + second_weight
                          : std::min(first_weight, second_weight)};
      }
    }
  }

  std::vector<int> unique_counts(static_cast<std::size_t>(graph.vertices));
#pragma omp parallel for schedule(dynamic, 512)
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    const auto begin = expanded.begin() +
                       static_cast<std::ptrdiff_t>(expanded_offsets[vertex]);
    const auto end =
        expanded.begin() +
        static_cast<std::ptrdiff_t>(expanded_offsets[vertex + 1]);
    std::sort(begin, end, [algorithm](const edge_t &left,
                                     const edge_t &right) {
      if (left.neighbor != right.neighbor) {
        return left.neighbor < right.neighbor;
      }
      return is_minimum(algorithm) ? left.weight < right.weight
                                   : left.weight > right.weight;
    });
    auto output = begin;
    for (auto input = begin; input != end; ++input) {
      if (output == begin || (output - 1)->neighbor != input->neighbor) {
        *output++ = *input;
      }
    }
    unique_counts[static_cast<std::size_t>(vertex)] =
        static_cast<int>(output - begin);
  }

  csr_t output;
  output.vertices = graph.vertices;
  output.row_offsets.resize(static_cast<std::size_t>(graph.vertices) + 1);
  std::uint64_t unique_edges = 0;
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    output.row_offsets[static_cast<std::size_t>(vertex)] =
        static_cast<int>(unique_edges);
    unique_edges +=
        static_cast<std::uint64_t>(unique_counts[static_cast<std::size_t>(vertex)]);
    if (unique_edges > static_cast<std::uint64_t>(
                           std::numeric_limits<int>::max())) {
      throw std::runtime_error("deduplicated graph exceeds 32-bit CSR limit");
    }
  }
  output.row_offsets.back() = static_cast<int>(unique_edges);
  output.columns.resize(static_cast<std::size_t>(unique_edges));
  output.weights.resize(static_cast<std::size_t>(unique_edges));

#pragma omp parallel for schedule(dynamic, 1024)
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    const std::size_t input =
        static_cast<std::size_t>(expanded_offsets[vertex]);
    const std::size_t output_begin =
        static_cast<std::size_t>(output.row_offsets[vertex]);
    const int count = unique_counts[static_cast<std::size_t>(vertex)];
    for (int i = 0; i < count; ++i) {
      output.columns[output_begin + static_cast<std::size_t>(i)] =
          expanded[input + static_cast<std::size_t>(i)].neighbor;
      output.weights[output_begin + static_cast<std::size_t>(i)] =
          expanded[input + static_cast<std::size_t>(i)].weight;
    }
  }
  std::cout << "shortcut_ms=" << elapsed_ms(start)
            << " expanded_edges=" << expanded_offsets.back()
            << " output_edges=" << unique_edges
            << " edge_delta="
            << static_cast<std::int64_t>(unique_edges) -
                   static_cast<std::int64_t>(graph.columns.size())
            << '\n';
  return output;
}

void write_csr(const fs::path &directory, const csr_t &graph) {
  const auto start = std::chrono::steady_clock::now();
  fs::create_directories(directory);
  write_binary_vector(directory / "csr_vlist.bin", graph.row_offsets);
  write_binary_vector(directory / "csr_elist.bin", graph.columns);
  write_binary_vector(directory / "csr_weightlist.bin", graph.weights);
  std::cout << "write_ms=" << elapsed_ms(start) << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = parse_options(argc, argv);
#ifdef _OPENMP
    if (options.threads > 0) {
      omp_set_num_threads(options.threads);
    }
#endif
    const auto total_start = std::chrono::steady_clock::now();
    const auto graph = load_csr(options.input_dir);
    const auto output =
        add_shortcuts(graph, options.maximum_degree, options.algorithm);
    write_csr(options.output_dir, output);
    std::cout << "max_degree=" << options.maximum_degree
              << " algorithm="
              << algorithm_name(options.algorithm)
              << " total_ms=" << elapsed_ms(total_start) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
