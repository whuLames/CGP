#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <filesystem>

#include <thrust/host_vector.h>

#include <puercgp/backend/csr_graph.hxx>

namespace puercgp_examples {

struct host_csr_graph {
  int vertices = 0;
  int edges = 0;
  std::vector<int> row_offsets;
  std::vector<int> column_indices;
  std::vector<float> edge_weights;
  puercgp::csr_graph_storage<int, int, float> device_graph;

  puercgp::csr_graph_view<int, int, float> view() const {
    return device_graph.view();
  }
};

inline std::string lower_copy(std::string value) {
  for (char& c : value) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return value;
}

inline std::vector<int> parse_sources(const std::string& text) {
  std::vector<int> sources;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (!token.empty()) {
      sources.push_back(std::stoi(token));
    }
  }
  return sources;
}

inline host_csr_graph load_matrix_market(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("could not open matrix: " + path);
  }

  std::string header;
  std::getline(input, header);
  auto header_lower = lower_copy(header);
  bool pattern = header_lower.find("pattern") != std::string::npos;
  bool symmetric = header_lower.find("symmetric") != std::string::npos;

  std::string line;
  do {
    if (!std::getline(input, line)) {
      throw std::runtime_error("missing matrix dimensions: " + path);
    }
  } while (line.empty() || line[0] == '%');

  std::stringstream dims(line);
  int rows = 0;
  int cols = 0;
  int entries = 0;
  dims >> rows >> cols >> entries;
  int vertices = std::max(rows, cols);

  std::vector<std::tuple<int, int, float>> edges;
  edges.reserve(static_cast<std::size_t>(entries) * (symmetric ? 2 : 1));
  for (int i = 0; i < entries; ++i) {
    if (!std::getline(input, line)) {
      throw std::runtime_error("truncated matrix: " + path);
    }
    if (line.empty() || line[0] == '%') {
      --i;
      continue;
    }
    std::stringstream row(line);
    int src = 0;
    int dst = 0;
    float weight = 1.0f;
    row >> src >> dst;
    if (!pattern) {
      row >> weight;
    }
    --src;
    --dst;
    edges.emplace_back(src, dst, weight);
    if (symmetric && src != dst) {
      edges.emplace_back(dst, src, weight);
    }
  }

  std::sort(edges.begin(), edges.end());

  host_csr_graph graph;
  graph.vertices = vertices;
  graph.row_offsets.assign(static_cast<std::size_t>(vertices) + 1, 0);
  for (const auto& [src, dst, weight] : edges) {
    (void)dst;
    (void)weight;
    if (src < 0 || src >= vertices) {
      throw std::runtime_error("source vertex out of range in: " + path);
    }
    ++graph.row_offsets[static_cast<std::size_t>(src) + 1];
  }
  for (int v = 0; v < vertices; ++v) {
    graph.row_offsets[static_cast<std::size_t>(v) + 1] +=
        graph.row_offsets[static_cast<std::size_t>(v)];
  }
  graph.edges = static_cast<int>(edges.size());
  graph.column_indices.resize(edges.size());
  graph.edge_weights.resize(edges.size());
  std::vector<int> cursor = graph.row_offsets;
  for (const auto& [src, dst, weight] : edges) {
    int position = cursor[static_cast<std::size_t>(src)]++;
    graph.column_indices[static_cast<std::size_t>(position)] = dst;
    graph.edge_weights[static_cast<std::size_t>(position)] = weight;
  }

  thrust::host_vector<int> row_offsets(graph.row_offsets.begin(),
                                       graph.row_offsets.end());
  thrust::host_vector<int> column_indices(graph.column_indices.begin(),
                                          graph.column_indices.end());
  thrust::host_vector<float> edge_weights(graph.edge_weights.begin(),
                                          graph.edge_weights.end());
  graph.device_graph = puercgp::csr_graph_storage<int, int, float>(
      vertices, row_offsets, column_indices, edge_weights);
  return graph;
}

inline host_csr_graph load_binary_csr_directory(const std::string& path) {
  namespace fs = std::filesystem;
  fs::path dir(path);
  fs::path row_path = dir / "csr_vlist.bin";
  fs::path col_path = dir / "csr_elist.bin";
  if (!fs::is_regular_file(row_path) || !fs::is_regular_file(col_path)) {
    throw std::runtime_error("CSR directory must contain csr_vlist.bin and "
                             "csr_elist.bin: " + path);
  }

  auto row_bytes = fs::file_size(row_path);
  auto col_bytes = fs::file_size(col_path);
  if (row_bytes % sizeof(int) != 0 || col_bytes % sizeof(int) != 0) {
    throw std::runtime_error("CSR binary files must use 32-bit integers: " +
                             path);
  }

  host_csr_graph graph;
  graph.vertices = static_cast<int>(row_bytes / sizeof(int) - 1);
  graph.edges = static_cast<int>(col_bytes / sizeof(int));
  graph.row_offsets.resize(static_cast<std::size_t>(graph.vertices) + 1);
  graph.column_indices.resize(static_cast<std::size_t>(graph.edges));

  std::ifstream rows(row_path, std::ios::binary);
  std::ifstream cols(col_path, std::ios::binary);
  if (!rows || !cols) {
    throw std::runtime_error("could not open CSR binary files: " + path);
  }
  rows.read(reinterpret_cast<char*>(graph.row_offsets.data()),
            static_cast<std::streamsize>(row_bytes));
  cols.read(reinterpret_cast<char*>(graph.column_indices.data()),
            static_cast<std::streamsize>(col_bytes));
  if (!rows || !cols) {
    throw std::runtime_error("failed to read CSR binary files: " + path);
  }

  thrust::host_vector<int> row_offsets(graph.row_offsets.begin(),
                                       graph.row_offsets.end());
  thrust::host_vector<int> column_indices(graph.column_indices.begin(),
                                          graph.column_indices.end());
  graph.device_graph = puercgp::csr_graph_storage<int, int, float>(
      graph.vertices, row_offsets, column_indices);
  return graph;
}

inline host_csr_graph load_graph_auto(const std::string& path) {
  if (std::filesystem::is_directory(path)) {
    return load_binary_csr_directory(path);
  }
  return load_matrix_market(path);
}

inline std::vector<int> cpu_bfs(const host_csr_graph& graph, int source) {
  const int infinity = std::numeric_limits<int>::max();
  std::vector<int> distances(static_cast<std::size_t>(graph.vertices), infinity);
  std::queue<int> queue;
  distances[static_cast<std::size_t>(source)] = 0;
  queue.push(source);
  while (!queue.empty()) {
    int vertex = queue.front();
    queue.pop();
    for (int edge = graph.row_offsets[static_cast<std::size_t>(vertex)];
         edge < graph.row_offsets[static_cast<std::size_t>(vertex) + 1];
         ++edge) {
      int neighbor = graph.column_indices[static_cast<std::size_t>(edge)];
      if (distances[static_cast<std::size_t>(neighbor)] == infinity) {
        distances[static_cast<std::size_t>(neighbor)] =
            distances[static_cast<std::size_t>(vertex)] + 1;
        queue.push(neighbor);
      }
    }
  }
  return distances;
}

inline std::vector<float> cpu_sssp(const host_csr_graph& graph, int source) {
  constexpr float infinity = std::numeric_limits<float>::infinity();
  using item_t = std::pair<float, int>;
  std::priority_queue<item_t, std::vector<item_t>, std::greater<item_t>> queue;
  std::vector<float> distances(static_cast<std::size_t>(graph.vertices),
                               infinity);
  distances[static_cast<std::size_t>(source)] = 0.0f;
  queue.emplace(0.0f, source);
  while (!queue.empty()) {
    auto [distance, vertex] = queue.top();
    queue.pop();
    if (distance != distances[static_cast<std::size_t>(vertex)]) {
      continue;
    }
    for (int edge = graph.row_offsets[static_cast<std::size_t>(vertex)];
         edge < graph.row_offsets[static_cast<std::size_t>(vertex) + 1];
         ++edge) {
      int neighbor = graph.column_indices[static_cast<std::size_t>(edge)];
      float candidate =
          distance + graph.edge_weights[static_cast<std::size_t>(edge)];
      if (candidate < distances[static_cast<std::size_t>(neighbor)]) {
        distances[static_cast<std::size_t>(neighbor)] = candidate;
        queue.emplace(candidate, neighbor);
      }
    }
  }
  return distances;
}

inline float median(std::vector<float> values) {
  if (values.empty()) {
    return 0.0f;
  }
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

}  // namespace puercgp_examples
