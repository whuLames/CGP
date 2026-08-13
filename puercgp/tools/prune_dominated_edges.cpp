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
  fs::path protected_base_dir;
  int maximum_intersection_degree = 256;
  int threads = 0;
  algorithm_t algorithm = algorithm_t::sssp;
};

struct csr_t {
  int vertices = 0;
  std::vector<int> row_offsets;
  std::vector<int> columns;
  std::vector<int> weights;
};

double elapsed_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

options_t parse_options(int argc, char **argv) {
  if (argc < 3) {
    throw std::invalid_argument(
        "Usage: prune_dominated_edges <input-dir> <output-dir> "
        "[--max-intersection-degree=256] [--protect-shortcuts-from=dir] "
        "[--threads=0] [--algorithm=bfs|sssp|sswp]");
  }
  options_t options;
  options.input_dir = argv[1];
  options.output_dir = argv[2];
  for (int i = 3; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument.rfind("--max-intersection-degree=", 0) == 0) {
      options.maximum_intersection_degree = std::stoi(argument.substr(
          std::string("--max-intersection-degree=").size()));
    } else if (argument.rfind("--protect-shortcuts-from=", 0) == 0) {
      options.protected_base_dir = argument.substr(
          std::string("--protect-shortcuts-from=").size());
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
  if (options.maximum_intersection_degree < 2 || options.threads < 0) {
    throw std::invalid_argument("invalid pruning options");
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
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    const int begin = graph.row_offsets[static_cast<std::size_t>(vertex)];
    const int end = graph.row_offsets[static_cast<std::size_t>(vertex) + 1];
    for (int edge = begin; edge < end; ++edge) {
      if (graph.weights[static_cast<std::size_t>(edge)] <= 0) {
        throw std::runtime_error("pruning requires positive edge weights");
      }
      if (edge != begin &&
          graph.columns[static_cast<std::size_t>(edge - 1)] >=
              graph.columns[static_cast<std::size_t>(edge)]) {
        throw std::runtime_error(
            "pruning requires sorted, duplicate-free CSR rows");
      }
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

int find_weight(const csr_t &graph, int vertex, int neighbor) {
  const int begin = graph.row_offsets[static_cast<std::size_t>(vertex)];
  const int end = graph.row_offsets[static_cast<std::size_t>(vertex) + 1];
  const auto first = graph.columns.begin() + begin;
  const auto last = graph.columns.begin() + end;
  const auto found = std::lower_bound(first, last, neighbor);
  if (found == last || *found != neighbor) {
    return -1;
  }
  return graph.weights[static_cast<std::size_t>(found - graph.columns.begin())];
}

csr_t prune(const csr_t &graph, int maximum_intersection_degree,
            const csr_t *protected_base, algorithm_t algorithm) {
  const auto start = std::chrono::steady_clock::now();
  std::vector<unsigned char> keep(graph.columns.size(), 1);
  std::uint64_t removed = 0;
  std::uint64_t protected_shortcuts = 0;

#pragma omp parallel for schedule(dynamic, 128) reduction(+ : removed, protected_shortcuts)
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    const int vertex_degree = degree(graph, vertex);
    for (int edge = graph.row_offsets[static_cast<std::size_t>(vertex)];
         edge < graph.row_offsets[static_cast<std::size_t>(vertex) + 1];
         ++edge) {
      const int neighbor = graph.columns[static_cast<std::size_t>(edge)];
      if (neighbor == vertex) {
        continue;
      }
      const int target_weight = graph.weights[static_cast<std::size_t>(edge)];
      if (protected_base != nullptr &&
          find_weight(*protected_base, vertex, neighbor) != target_weight) {
        ++protected_shortcuts;
        continue;
      }
      const int neighbor_degree = degree(graph, neighbor);
      const int smaller =
          vertex_degree <= neighbor_degree ? vertex : neighbor;
      const int larger = smaller == vertex ? neighbor : vertex;
      if (degree(graph, smaller) > maximum_intersection_degree) {
        continue;
      }
      bool dominated = false;
      for (int candidate =
               graph.row_offsets[static_cast<std::size_t>(smaller)];
           candidate <
           graph.row_offsets[static_cast<std::size_t>(smaller) + 1];
           ++candidate) {
        const int middle =
            graph.columns[static_cast<std::size_t>(candidate)];
        const int first_weight =
            graph.weights[static_cast<std::size_t>(candidate)];
        if (middle == vertex || middle == neighbor) {
          continue;
        }
        if ((is_minimum(algorithm) &&
             first_weight >= target_weight) ||
            (algorithm == algorithm_t::sswp &&
             first_weight <= target_weight)) {
          continue;
        }
        const int second_weight = find_weight(graph, larger, middle);
        const bool dominates =
            is_minimum(algorithm)
                ? second_weight > 0 &&
                      static_cast<std::int64_t>(first_weight) + second_weight <=
                          target_weight
                : second_weight > target_weight;
        if (dominates) {
          dominated = true;
          break;
        }
      }
      if (dominated) {
        keep[static_cast<std::size_t>(edge)] = 0;
        ++removed;
      }
    }
  }

  csr_t output;
  output.vertices = graph.vertices;
  output.row_offsets.resize(static_cast<std::size_t>(graph.vertices) + 1);
  std::uint64_t output_edges = 0;
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    output.row_offsets[static_cast<std::size_t>(vertex)] =
        static_cast<int>(output_edges);
    for (int edge = graph.row_offsets[static_cast<std::size_t>(vertex)];
         edge < graph.row_offsets[static_cast<std::size_t>(vertex) + 1];
         ++edge) {
      output_edges += keep[static_cast<std::size_t>(edge)] != 0;
    }
  }
  output.row_offsets.back() = static_cast<int>(output_edges);
  output.columns.resize(static_cast<std::size_t>(output_edges));
  output.weights.resize(static_cast<std::size_t>(output_edges));

#pragma omp parallel for schedule(dynamic, 1024)
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    std::size_t position =
        static_cast<std::size_t>(output.row_offsets[vertex]);
    for (int edge = graph.row_offsets[static_cast<std::size_t>(vertex)];
         edge < graph.row_offsets[static_cast<std::size_t>(vertex) + 1];
         ++edge) {
      if (keep[static_cast<std::size_t>(edge)] == 0) {
        continue;
      }
      output.columns[position] = graph.columns[static_cast<std::size_t>(edge)];
      output.weights[position] = graph.weights[static_cast<std::size_t>(edge)];
      ++position;
    }
  }
  std::cout << "prune_ms=" << elapsed_ms(start) << " removed=" << removed
            << " protected_shortcuts=" << protected_shortcuts
            << " output_edges=" << output_edges << '\n';
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
    const auto start = std::chrono::steady_clock::now();
    const auto graph = load_csr(options.input_dir);
    csr_t protected_base;
    const csr_t *protected_base_pointer = nullptr;
    if (!options.protected_base_dir.empty()) {
      protected_base = load_csr(options.protected_base_dir);
      if (protected_base.vertices != graph.vertices) {
        throw std::runtime_error("protected base vertex count mismatch");
      }
      protected_base_pointer = &protected_base;
    }
    const auto output =
        prune(graph, options.maximum_intersection_degree,
              protected_base_pointer, options.algorithm);
    write_csr(options.output_dir, output);
    std::cout << "max_intersection_degree="
              << options.maximum_intersection_degree
              << " protect_shortcuts="
              << (protected_base_pointer != nullptr ? 1 : 0)
              << " algorithm="
              << algorithm_name(options.algorithm)
              << " total_ms=" << elapsed_ms(start) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
