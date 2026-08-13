#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr std::int32_t kReorderMagic = 0x52454f52;

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
  fs::path sources_path;
  fs::path mapping_path;
  fs::path forest_input_dir;
  int query_count = 256;
  std::vector<int> shortcut_hops{2};
  bool root_only = false;
  bool directed = false;
  bool pull_directed = false;
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
  if (argc < 4) {
    throw std::invalid_argument(
        "Usage: add_forest_shortcuts <input-dir> <output-dir> <sources> "
        "[--mapping=path] [--forest-input=path] [--queries=256] "
        "[--hops=2,4,8] [--root-only] "
        "[--directed|--pull-directed] [--algorithm=bfs|sssp|sswp]");
  }
  options_t options;
  options.input_dir = argv[1];
  options.output_dir = argv[2];
  options.sources_path = argv[3];
  for (int i = 4; i < argc; ++i) {
    const std::string argument = argv[i];
    auto value_after = [&](const std::string &prefix) {
      return argument.substr(prefix.size());
    };
    if (argument.rfind("--mapping=", 0) == 0) {
      options.mapping_path = value_after("--mapping=");
    } else if (argument.rfind("--forest-input=", 0) == 0) {
      options.forest_input_dir = value_after("--forest-input=");
    } else if (argument.rfind("--queries=", 0) == 0) {
      options.query_count = std::stoi(value_after("--queries="));
    } else if (argument.rfind("--hops=", 0) == 0) {
      options.shortcut_hops.clear();
      std::stringstream values(value_after("--hops="));
      std::string value;
      while (std::getline(values, value, ',')) {
        options.shortcut_hops.push_back(std::stoi(value));
      }
    } else if (argument == "--root-only") {
      options.shortcut_hops.clear();
      options.root_only = true;
    } else if (argument == "--directed") {
      options.directed = true;
    } else if (argument == "--pull-directed") {
      options.directed = true;
      options.pull_directed = true;
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
  if (options.query_count <= 0 ||
      (options.shortcut_hops.empty() && !options.root_only) ||
      std::any_of(options.shortcut_hops.begin(), options.shortcut_hops.end(),
                  [](int hops) { return hops < 2; })) {
    throw std::invalid_argument("queries and shortcut hops must be positive");
  }
  std::sort(options.shortcut_hops.begin(), options.shortcut_hops.end());
  options.shortcut_hops.erase(
      std::unique(options.shortcut_hops.begin(), options.shortcut_hops.end()),
      options.shortcut_hops.end());
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
  std::cout << "load_ms=" << elapsed_ms(start)
            << " vertices=" << graph.vertices
            << " edges=" << graph.columns.size() << '\n';
  return graph;
}

std::vector<int> load_sources(const options_t &options, int vertices) {
  std::ifstream input(options.sources_path);
  if (!input) {
    throw std::runtime_error("cannot read sources: " +
                             options.sources_path.string());
  }
  std::vector<int> sources;
  std::string line;
  while (std::getline(input, line) &&
         static_cast<int>(sources.size()) < options.query_count) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    std::stringstream row(line);
    std::string field;
    std::vector<std::string> fields;
    while (std::getline(row, field, ',')) {
      fields.push_back(field);
    }
    try {
      sources.push_back(std::stoi(fields.size() > 1 ? fields[1] : fields[0]));
    } catch (const std::invalid_argument &) {
      if (sources.empty()) {
        continue;
      }
      throw;
    }
  }
  if (static_cast<int>(sources.size()) != options.query_count) {
    throw std::runtime_error("source count does not match --queries");
  }

  if (!options.mapping_path.empty()) {
    std::ifstream mapping(options.mapping_path, std::ios::binary);
    std::int32_t magic = 0;
    std::int32_t count = 0;
    std::int32_t strategy = 0;
    mapping.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    mapping.read(reinterpret_cast<char *>(&count), sizeof(count));
    mapping.read(reinterpret_cast<char *>(&strategy), sizeof(strategy));
    if (!mapping || magic != kReorderMagic || count != vertices) {
      throw std::runtime_error("mapping header does not match graph");
    }
    std::vector<int> new_to_old(static_cast<std::size_t>(vertices));
    mapping.read(reinterpret_cast<char *>(new_to_old.data()),
                 static_cast<std::streamsize>(new_to_old.size() * sizeof(int)));
    if (!mapping) {
      throw std::runtime_error("truncated mapping");
    }
    std::vector<int> old_to_new(static_cast<std::size_t>(vertices), -1);
    for (int new_id = 0; new_id < vertices; ++new_id) {
      old_to_new[static_cast<std::size_t>(new_to_old[new_id])] = new_id;
    }
    for (int &source : sources) {
      source = old_to_new[static_cast<std::size_t>(source)];
    }
  }
  for (int source : sources) {
    if (source < 0 || source >= vertices) {
      throw std::runtime_error("source outside graph");
    }
  }
  return sources;
}

class radix_heap_t {
public:
  using key_t = std::uint32_t;
  using item_t = std::pair<key_t, int>;

  bool empty() const { return size_ == 0; }

  void push(key_t key, int value) {
    if (key < last_) {
      throw std::logic_error("radix heap key decreased");
    }
    buckets_[bucket_index(key, last_)].emplace_back(key, value);
    ++size_;
  }

  item_t pop() {
    if (buckets_[0].empty()) {
      std::size_t bucket = 1;
      while (bucket < buckets_.size() && buckets_[bucket].empty()) {
        ++bucket;
      }
      if (bucket == buckets_.size()) {
        throw std::logic_error("pop from empty radix heap");
      }
      key_t next = std::numeric_limits<key_t>::max();
      for (const auto &item : buckets_[bucket]) {
        next = std::min(next, item.first);
      }
      last_ = next;
      auto items = std::move(buckets_[bucket]);
      buckets_[bucket].clear();
      for (const auto &item : items) {
        buckets_[bucket_index(item.first, last_)].push_back(item);
      }
    }
    const auto result = buckets_[0].back();
    buckets_[0].pop_back();
    --size_;
    return result;
  }

private:
  static std::size_t bucket_index(key_t key, key_t last) {
    const key_t difference = key ^ last;
    return difference == 0
               ? 0
               : static_cast<std::size_t>(32 - __builtin_clz(difference));
  }

  std::array<std::vector<item_t>, 33> buckets_;
  std::size_t size_ = 0;
  key_t last_ = 0;
};

struct forest_t {
  std::vector<std::uint32_t> distances;
  std::vector<int> parents;
  std::vector<int> parent_weights;
  std::vector<int> depths;
};

forest_t build_forest(const csr_t &graph, const std::vector<int> &sources,
                      algorithm_t algorithm) {
  const auto start = std::chrono::steady_clock::now();
  constexpr std::uint32_t infinity =
      std::numeric_limits<std::uint32_t>::max();
  forest_t forest;
  forest.distances.assign(static_cast<std::size_t>(graph.vertices),
                          is_minimum(algorithm) ? infinity : 0);
  forest.parents.assign(static_cast<std::size_t>(graph.vertices), -1);
  forest.parent_weights.assign(static_cast<std::size_t>(graph.vertices), 0);
  forest.depths.assign(static_cast<std::size_t>(graph.vertices), -1);
  std::vector<unsigned char> settled(static_cast<std::size_t>(graph.vertices));

  std::size_t reached = 0;
  int maximum_depth = 0;
  auto settle_vertex = [&](std::uint32_t value, int vertex) {
    const auto index = static_cast<std::size_t>(vertex);
    if (settled[index] != 0 || forest.distances[index] != value) {
      return false;
    }
    settled[index] = 1;
    ++reached;
    if (forest.parents[index] >= 0) {
      forest.depths[index] =
          forest.depths[static_cast<std::size_t>(forest.parents[index])] + 1;
      maximum_depth = std::max(maximum_depth, forest.depths[index]);
    }
    return true;
  };

  if (is_minimum(algorithm)) {
    radix_heap_t queue;
    for (int source : sources) {
      forest.distances[static_cast<std::size_t>(source)] = 0;
      forest.depths[static_cast<std::size_t>(source)] = 0;
      queue.push(0, source);
    }
    while (!queue.empty()) {
      const auto [distance, vertex] = queue.pop();
      if (!settle_vertex(distance, vertex)) {
        continue;
      }
      const auto index = static_cast<std::size_t>(vertex);
      for (int edge = graph.row_offsets[index];
           edge < graph.row_offsets[index + 1]; ++edge) {
        const int neighbor = graph.columns[static_cast<std::size_t>(edge)];
        const int weight = graph.weights[static_cast<std::size_t>(edge)];
        if (weight <= 0) {
          throw std::runtime_error("shortcuts require positive edge weights");
        }
        const std::uint64_t candidate64 =
            static_cast<std::uint64_t>(distance) + weight;
        const auto candidate =
            candidate64 >= infinity ? infinity
                                    : static_cast<std::uint32_t>(candidate64);
        const auto neighbor_index = static_cast<std::size_t>(neighbor);
        if (candidate < forest.distances[neighbor_index]) {
          forest.distances[neighbor_index] = candidate;
          forest.parents[neighbor_index] = vertex;
          forest.parent_weights[neighbor_index] = weight;
          queue.push(candidate, neighbor);
        }
      }
    }
  } else {
    using item_t = std::pair<std::uint32_t, int>;
    std::priority_queue<item_t> queue;
    for (int source : sources) {
      forest.distances[static_cast<std::size_t>(source)] = infinity;
      forest.depths[static_cast<std::size_t>(source)] = 0;
      queue.emplace(infinity, source);
    }
    while (!queue.empty()) {
      const auto [width, vertex] = queue.top();
      queue.pop();
      if (!settle_vertex(width, vertex)) {
        continue;
      }
      const auto index = static_cast<std::size_t>(vertex);
      for (int edge = graph.row_offsets[index];
           edge < graph.row_offsets[index + 1]; ++edge) {
        const int neighbor = graph.columns[static_cast<std::size_t>(edge)];
        const int weight = graph.weights[static_cast<std::size_t>(edge)];
        if (weight <= 0) {
          throw std::runtime_error("shortcuts require positive edge weights");
        }
        const auto candidate =
            std::min(width, static_cast<std::uint32_t>(weight));
        const auto neighbor_index = static_cast<std::size_t>(neighbor);
        if (candidate > forest.distances[neighbor_index]) {
          forest.distances[neighbor_index] = candidate;
          forest.parents[neighbor_index] = vertex;
          forest.parent_weights[neighbor_index] = weight;
          queue.emplace(candidate, neighbor);
        }
      }
    }
  }
  std::cout << "forest_ms=" << elapsed_ms(start) << " reached=" << reached
            << " max_depth=" << maximum_depth << " algorithm="
            << algorithm_name(algorithm) << '\n';
  return forest;
}

int ancestor_at(const std::vector<int> &parents, int vertex, int hops) {
  for (int step = 0; step < hops && vertex >= 0; ++step) {
    vertex = parents[static_cast<std::size_t>(vertex)];
  }
  return vertex;
}

int path_weight(const forest_t &forest, int vertex, int hops,
                algorithm_t algorithm) {
  if (is_minimum(algorithm)) {
    const int ancestor = ancestor_at(forest.parents, vertex, hops);
    if (ancestor < 0) {
      return -1;
    }
    const auto distance = forest.distances[static_cast<std::size_t>(vertex)] -
                          forest.distances[static_cast<std::size_t>(ancestor)];
    if (distance >
        static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("shortcut weight exceeds int range");
    }
    return static_cast<int>(distance);
  }
  int capacity = std::numeric_limits<int>::max();
  for (int step = 0; step < hops; ++step) {
    if (vertex < 0) {
      return -1;
    }
    capacity = std::min(
        capacity, forest.parent_weights[static_cast<std::size_t>(vertex)]);
    vertex = forest.parents[static_cast<std::size_t>(vertex)];
  }
  return capacity;
}

csr_t add_shortcuts(const csr_t &graph, const forest_t &forest,
                    const std::vector<int> &hops, bool root_only,
                    bool directed, bool pull_directed,
                    algorithm_t algorithm) {
  const auto start = std::chrono::steady_clock::now();
  std::vector<std::uint64_t> row_counts(
      static_cast<std::size_t>(graph.vertices));
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    row_counts[static_cast<std::size_t>(vertex)] =
        static_cast<std::uint64_t>(graph.row_offsets[vertex + 1] -
                                   graph.row_offsets[vertex]);
  }

  std::uint64_t shortcut_edges = 0;
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    for (int hop_count : hops) {
      if (forest.depths[static_cast<std::size_t>(vertex)] < hop_count) {
        continue;
      }
      const int ancestor = ancestor_at(forest.parents, vertex, hop_count);
      if (ancestor < 0) {
        continue;
      }
      if (!directed || !pull_directed) {
        ++row_counts[static_cast<std::size_t>(vertex)];
      }
      if (!directed || pull_directed) {
        ++row_counts[static_cast<std::size_t>(ancestor)];
      }
      shortcut_edges += directed ? 1 : 2;
    }
    if (root_only && forest.depths[static_cast<std::size_t>(vertex)] >= 2) {
      const int root = ancestor_at(
          forest.parents, vertex,
          forest.depths[static_cast<std::size_t>(vertex)]);
      if (root >= 0) {
        if (!directed || !pull_directed) {
          ++row_counts[static_cast<std::size_t>(vertex)];
        }
        if (!directed || pull_directed) {
          ++row_counts[static_cast<std::size_t>(root)];
        }
        shortcut_edges += directed ? 1 : 2;
      }
    }
  }

  csr_t output;
  output.vertices = graph.vertices;
  output.row_offsets.resize(static_cast<std::size_t>(graph.vertices) + 1);
  std::uint64_t edge_total = 0;
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    output.row_offsets[static_cast<std::size_t>(vertex)] =
        static_cast<int>(edge_total);
    edge_total += row_counts[static_cast<std::size_t>(vertex)];
    if (edge_total > static_cast<std::uint64_t>(
                         std::numeric_limits<int>::max())) {
      throw std::runtime_error("shortcut graph exceeds 32-bit CSR limit");
    }
  }
  output.row_offsets.back() = static_cast<int>(edge_total);
  output.columns.resize(static_cast<std::size_t>(edge_total));
  output.weights.resize(static_cast<std::size_t>(edge_total));
  std::vector<int> positions(output.row_offsets.begin(),
                             output.row_offsets.end() - 1);

  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    for (int edge = graph.row_offsets[vertex];
         edge < graph.row_offsets[vertex + 1]; ++edge) {
      const int position = positions[static_cast<std::size_t>(vertex)]++;
      output.columns[static_cast<std::size_t>(position)] =
          graph.columns[static_cast<std::size_t>(edge)];
      output.weights[static_cast<std::size_t>(position)] =
          graph.weights[static_cast<std::size_t>(edge)];
    }
  }
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    for (int hop_count : hops) {
      if (forest.depths[static_cast<std::size_t>(vertex)] < hop_count) {
        continue;
      }
      const int ancestor = ancestor_at(forest.parents, vertex, hop_count);
      if (ancestor < 0) {
        continue;
      }
      const int weight = path_weight(forest, vertex, hop_count, algorithm);
      if (weight <= 0) {
        throw std::runtime_error("invalid shortcut path weight");
      }
      if (!directed || !pull_directed) {
        int position = positions[static_cast<std::size_t>(vertex)]++;
        output.columns[static_cast<std::size_t>(position)] = ancestor;
        output.weights[static_cast<std::size_t>(position)] = weight;
      }
      if (!directed || pull_directed) {
        int position = positions[static_cast<std::size_t>(ancestor)]++;
        output.columns[static_cast<std::size_t>(position)] = vertex;
        output.weights[static_cast<std::size_t>(position)] = weight;
      }
    }
    if (root_only && forest.depths[static_cast<std::size_t>(vertex)] >= 2) {
      const int root = ancestor_at(
          forest.parents, vertex,
          forest.depths[static_cast<std::size_t>(vertex)]);
      if (root < 0) {
        continue;
      }
      const int weight = path_weight(
          forest, vertex, forest.depths[static_cast<std::size_t>(vertex)],
          algorithm);
      if (weight <= 0) {
        throw std::runtime_error("invalid root shortcut path weight");
      }
      if (!directed || !pull_directed) {
        int position = positions[static_cast<std::size_t>(vertex)]++;
        output.columns[static_cast<std::size_t>(position)] = root;
        output.weights[static_cast<std::size_t>(position)] = weight;
      }
      if (!directed || pull_directed) {
        int position = positions[static_cast<std::size_t>(root)]++;
        output.columns[static_cast<std::size_t>(position)] = vertex;
        output.weights[static_cast<std::size_t>(position)] = weight;
      }
    }
  }
  std::cout << "shortcut_ms=" << elapsed_ms(start)
            << " shortcut_edges=" << shortcut_edges
            << " output_edges=" << edge_total << '\n';
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
    const auto total_start = std::chrono::steady_clock::now();
    const auto graph = load_csr(options.input_dir);
    const auto sources = load_sources(options, graph.vertices);
    forest_t forest;
    if (options.forest_input_dir.empty()) {
      forest = build_forest(graph, sources, options.algorithm);
    } else {
      const auto forest_graph = load_csr(options.forest_input_dir);
      if (forest_graph.vertices != graph.vertices) {
        throw std::runtime_error(
            "forest input vertex count does not match output base graph");
      }
      forest = build_forest(forest_graph, sources, options.algorithm);
    }
    const auto output = add_shortcuts(graph, forest, options.shortcut_hops,
                                      options.root_only, options.directed,
                                      options.pull_directed,
                                      options.algorithm);
    write_csr(options.output_dir, output);
    std::cout << "hops=";
    if (options.root_only) {
      std::cout << "root";
    }
    for (std::size_t i = 0; i < options.shortcut_hops.size(); ++i) {
      std::cout << (i == 0 ? "" : ",") << options.shortcut_hops[i];
    }
    std::cout << " directed=" << (options.directed ? 1 : 0)
              << " pull_directed=" << (options.pull_directed ? 1 : 0)
              << " algorithm="
              << algorithm_name(options.algorithm)
              << " total_ms=" << elapsed_ms(total_start) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
