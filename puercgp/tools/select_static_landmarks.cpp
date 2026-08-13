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

constexpr std::int32_t kReorderMagic = 0x52454f52;

struct options_t {
  fs::path graph_dir;
  fs::path mapping_path;
  fs::path output_path;
  int landmark_count = 16;
  int farthest_count = 4;
};

struct csr_t {
  int vertices = 0;
  std::vector<int> rows;
  std::vector<int> columns;
};

options_t parse_options(int argc, char** argv) {
  if (argc < 4) {
    throw std::invalid_argument(
        "Usage: select_static_landmarks <graph-dir> <rabbit-mapping> "
        "<output.csv> [--count=16] [--farthest=4]");
  }
  options_t options;
  options.graph_dir = argv[1];
  options.mapping_path = argv[2];
  options.output_path = argv[3];
  for (int i = 4; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument.rfind("--count=", 0) == 0) {
      options.landmark_count = std::stoi(argument.substr(8));
    } else if (argument.rfind("--farthest=", 0) == 0) {
      options.farthest_count = std::stoi(argument.substr(11));
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  if (options.landmark_count <= 0 || options.farthest_count <= 0 ||
      options.farthest_count > options.landmark_count) {
    throw std::invalid_argument("invalid landmark counts");
  }
  return options;
}

template <typename value_t>
std::vector<value_t> read_binary_vector(const fs::path& path) {
  if (!fs::is_regular_file(path)) {
    throw std::runtime_error("missing input file: " + path.string());
  }
  const auto bytes = fs::file_size(path);
  if (bytes % sizeof(value_t) != 0) {
    throw std::runtime_error("invalid binary file size: " + path.string());
  }
  std::vector<value_t> values(bytes / sizeof(value_t));
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(bytes));
  if (!input) {
    throw std::runtime_error("failed to read: " + path.string());
  }
  return values;
}

csr_t load_graph(const fs::path& directory) {
  csr_t graph;
  graph.rows = read_binary_vector<int>(directory / "csr_vlist.bin");
  graph.columns = read_binary_vector<int>(directory / "csr_elist.bin");
  if (graph.rows.size() < 2 ||
      graph.rows.size() - 1 >
          static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("invalid 32-bit CSR");
  }
  graph.vertices = static_cast<int>(graph.rows.size() - 1);
  if (graph.rows.front() != 0 ||
      graph.rows.back() != static_cast<int>(graph.columns.size())) {
    throw std::runtime_error("inconsistent CSR offsets");
  }
  return graph;
}

std::vector<int> load_mapping(const fs::path& path, int vertices) {
  std::ifstream input(path, std::ios::binary);
  std::int32_t magic = 0;
  std::int32_t count = 0;
  std::int32_t strategy = 0;
  input.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  input.read(reinterpret_cast<char*>(&count), sizeof(count));
  input.read(reinterpret_cast<char*>(&strategy), sizeof(strategy));
  if (!input || magic != kReorderMagic || count != vertices) {
    throw std::runtime_error("mapping header does not match graph");
  }
  std::vector<int> order(static_cast<std::size_t>(vertices));
  input.read(reinterpret_cast<char*>(order.data()),
             static_cast<std::streamsize>(order.size() * sizeof(int)));
  if (!input) {
    throw std::runtime_error("truncated mapping");
  }
  return order;
}

int degree(const csr_t& graph, int vertex) {
  return graph.rows[static_cast<std::size_t>(vertex) + 1] -
         graph.rows[static_cast<std::size_t>(vertex)];
}

std::vector<int> bfs_distances(const csr_t& graph, int source,
                               std::vector<int>& queue) {
  std::vector<int> distances(static_cast<std::size_t>(graph.vertices), -1);
  queue.clear();
  queue.push_back(source);
  distances[static_cast<std::size_t>(source)] = 0;
  for (std::size_t head = 0; head < queue.size(); ++head) {
    const int vertex = queue[head];
    const int next_distance = distances[static_cast<std::size_t>(vertex)] + 1;
    for (int edge = graph.rows[static_cast<std::size_t>(vertex)];
         edge < graph.rows[static_cast<std::size_t>(vertex) + 1]; ++edge) {
      const int neighbor = graph.columns[static_cast<std::size_t>(edge)];
      if (distances[static_cast<std::size_t>(neighbor)] >= 0) {
        continue;
      }
      distances[static_cast<std::size_t>(neighbor)] = next_distance;
      queue.push_back(neighbor);
    }
  }
  return distances;
}

std::vector<int> select_landmarks(const csr_t& graph,
                                  const std::vector<int>& rabbit_order,
                                  int landmark_count, int farthest_count) {
  std::vector<int> selected;
  selected.reserve(static_cast<std::size_t>(landmark_count));
  std::vector<unsigned char> used(static_cast<std::size_t>(graph.vertices));
  std::vector<int> nearest(static_cast<std::size_t>(graph.vertices),
                           std::numeric_limits<int>::max());
  std::vector<int> queue;
  queue.reserve(static_cast<std::size_t>(graph.vertices));

  int next = 0;
  for (int vertex = 1; vertex < graph.vertices; ++vertex) {
    if (degree(graph, vertex) > degree(graph, next)) {
      next = vertex;
    }
  }
  for (int iteration = 0; iteration < farthest_count; ++iteration) {
    selected.push_back(next);
    used[static_cast<std::size_t>(next)] = 1;
    const auto distances = bfs_distances(graph, next, queue);
    int farthest = -1;
    for (int vertex = 0; vertex < graph.vertices; ++vertex) {
      const int distance = distances[static_cast<std::size_t>(vertex)];
      if (distance >= 0) {
        nearest[static_cast<std::size_t>(vertex)] =
            std::min(nearest[static_cast<std::size_t>(vertex)], distance);
      }
      if (used[static_cast<std::size_t>(vertex)] != 0) {
        continue;
      }
      if (nearest[static_cast<std::size_t>(vertex)] ==
          std::numeric_limits<int>::max()) {
        continue;
      }
      if (farthest < 0 || nearest[static_cast<std::size_t>(vertex)] >
                              nearest[static_cast<std::size_t>(farthest)] ||
          (nearest[static_cast<std::size_t>(vertex)] ==
               nearest[static_cast<std::size_t>(farthest)] &&
           degree(graph, vertex) > degree(graph, farthest))) {
        farthest = vertex;
      }
    }
    if (farthest < 0) {
      break;
    }
    next = farthest;
    std::cout << "farthest_iteration=" << iteration + 1
              << " landmark=" << selected.back()
              << " reached=" << queue.size()
              << " next_distance="
              << nearest[static_cast<std::size_t>(next)] << '\n';
  }

  const int remaining = landmark_count - static_cast<int>(selected.size());
  for (int segment = 0; segment < remaining; ++segment) {
    const std::size_t begin = static_cast<std::size_t>(segment) *
                              rabbit_order.size() / remaining;
    const std::size_t end = static_cast<std::size_t>(segment + 1) *
                            rabbit_order.size() / remaining;
    int best = -1;
    for (std::size_t index = begin; index < end; ++index) {
      const int vertex = rabbit_order[index];
      if (used[static_cast<std::size_t>(vertex)] == 0 &&
          (best < 0 || degree(graph, vertex) > degree(graph, best))) {
        best = vertex;
      }
    }
    if (best >= 0) {
      selected.push_back(best);
      used[static_cast<std::size_t>(best)] = 1;
    }
  }
  for (int vertex : rabbit_order) {
    if (static_cast<int>(selected.size()) == landmark_count) {
      break;
    }
    if (used[static_cast<std::size_t>(vertex)] == 0) {
      selected.push_back(vertex);
      used[static_cast<std::size_t>(vertex)] = 1;
    }
  }
  return selected;
}

void write_landmarks(const fs::path& path, const std::vector<int>& landmarks) {
  if (path.has_parent_path()) {
    fs::create_directories(path.parent_path());
  }
  std::ofstream output(path);
  output << "query,source\n";
  for (std::size_t index = 0; index < landmarks.size(); ++index) {
    output << index << ',' << landmarks[index] << '\n';
  }
  if (!output) {
    throw std::runtime_error("failed to write landmarks");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    const auto graph = load_graph(options.graph_dir);
    if (options.landmark_count > graph.vertices) {
      throw std::invalid_argument("landmark count exceeds vertex count");
    }
    const auto rabbit_order = load_mapping(options.mapping_path, graph.vertices);
    const auto landmarks = select_landmarks(
        graph, rabbit_order, options.landmark_count, options.farthest_count);
    write_landmarks(options.output_path, landmarks);
    std::cout << "vertices=" << graph.vertices
              << " edges=" << graph.columns.size()
              << " landmarks=" << landmarks.size()
              << " output=" << options.output_path << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
