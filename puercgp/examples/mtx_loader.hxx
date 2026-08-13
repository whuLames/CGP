#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
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

// 附加独立长链分量（双向 path），用于构造 WCC 长尾场景：
//   长链顶点 V_old..V_old+L-1，双向边 (j)<->(j+1)，不连主图（独立连通分量）
//   BFS/SSSP source 在主图 [0,V_old) 不进入长链，收敛快（O(主图直径)）
//   WCC 覆盖全图，label 沿长链逐跳传播，O(L) 轮收敛（长尾）
inline void attach_chain(host_csr_graph& g,
                         int chain_length,
                         bool build_pull_adjacency = false) {
  if (chain_length <= 1) return;
  const int V_old = g.vertices;
  const int E_old = g.edges;
  const int L = chain_length;
  const int chain_edges = 2 * (L - 1);  // 双向 path

  std::vector<int> new_row(static_cast<std::size_t>(V_old) + L + 1);
  for (int i = 0; i <= V_old; ++i) new_row[i] = g.row_offsets[i];
  int offset = E_old;
  for (int j = 0; j < L; ++j) {
    new_row[V_old + j] = offset;
    int deg = (j > 0 ? 1 : 0) + (j < L - 1 ? 1 : 0);
    offset += deg;
  }
  new_row[V_old + L] = offset;

  std::vector<int> new_col(static_cast<std::size_t>(E_old) + chain_edges);
  for (int i = 0; i < E_old; ++i) new_col[i] = g.column_indices[i];
  int pos = E_old;
  for (int j = 0; j < L; ++j) {
    if (j > 0) new_col[pos++] = V_old + (j - 1);
    if (j < L - 1) new_col[pos++] = V_old + (j + 1);
  }

  std::vector<float> new_w(static_cast<std::size_t>(E_old) + chain_edges, 1.0f);
  for (int i = 0; i < E_old && i < static_cast<int>(g.edge_weights.size()); ++i)
    new_w[i] = g.edge_weights[i];

  g.vertices = V_old + L;
  g.edges = E_old + chain_edges;
  g.row_offsets = std::move(new_row);
  g.column_indices = std::move(new_col);
  g.edge_weights = std::move(new_w);

  thrust::host_vector<int> row_offsets(g.row_offsets.begin(), g.row_offsets.end());
  thrust::host_vector<int> column_indices(g.column_indices.begin(), g.column_indices.end());
  thrust::host_vector<float> edge_weights(g.edge_weights.begin(), g.edge_weights.end());
  g.device_graph = puercgp::csr_graph_storage<int, int, float>(
      g.vertices, row_offsets, column_indices, edge_weights,
      build_pull_adjacency);
}

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

inline host_csr_graph load_matrix_market(const std::string& path,
                                         bool build_pull_adjacency = false) {
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
      vertices, row_offsets, column_indices, edge_weights,
      build_pull_adjacency);
  return graph;
}

inline host_csr_graph load_binary_csr_directory(
    const std::string& path,
    bool build_pull_adjacency = false) {
  namespace fs = std::filesystem;
  fs::path dir(path);
  fs::path row_path = dir / "csr_vlist.bin";
  fs::path col_path = dir / "csr_elist.bin";
  fs::path w_path = dir / "csr_weightlist.bin";
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
  graph.edge_weights.resize(static_cast<std::size_t>(graph.edges));

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

  // 权重文件可选：存在则读 int32 → float；不存在则填单位权（SSSP 退化为 BFS）
  const bool has_weights = fs::is_regular_file(w_path);
  if (has_weights) {
    auto w_bytes = fs::file_size(w_path);
    if (w_bytes != col_bytes) {
      throw std::runtime_error(
          "csr_weightlist.bin size must equal csr_elist.bin size: " + path);
    }
    std::vector<int> w_int(static_cast<std::size_t>(graph.edges));
    std::ifstream ws(w_path, std::ios::binary);
    if (!ws) {
      throw std::runtime_error("could not open csr_weightlist.bin: " + path);
    }
    ws.read(reinterpret_cast<char*>(w_int.data()),
            static_cast<std::streamsize>(w_bytes));
    for (std::size_t i = 0; i < w_int.size(); ++i) {
      graph.edge_weights[i] = static_cast<float>(w_int[i]);
    }
  } else {
    for (std::size_t i = 0; i < graph.edge_weights.size(); ++i) {
      graph.edge_weights[i] = 1.0f;
    }
  }

  thrust::host_vector<int> row_offsets(graph.row_offsets.begin(),
                                       graph.row_offsets.end());
  thrust::host_vector<int> column_indices(graph.column_indices.begin(),
                                          graph.column_indices.end());
  thrust::host_vector<float> edge_weights(graph.edge_weights.begin(),
                                          graph.edge_weights.end());
  graph.device_graph = puercgp::csr_graph_storage<int, int, float>(
      graph.vertices, row_offsets, column_indices, edge_weights,
      build_pull_adjacency);
  return graph;
}

inline std::uint64_t decode_little_endian(const unsigned char* bytes,
                                          unsigned width) {
  std::uint64_t value = 0;
  for (unsigned byte = 0; byte < width; ++byte) {
    value |= static_cast<std::uint64_t>(bytes[byte]) << (8 * byte);
  }
  return value;
}

inline host_csr_graph load_galois_gr(
    const std::string& path, bool build_pull_adjacency = false) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("could not open GR graph: " + path);
  }
  unsigned char header_bytes[32];
  input.read(reinterpret_cast<char*>(header_bytes), sizeof(header_bytes));
  if (!input) {
    throw std::runtime_error("truncated GR header: " + path);
  }
  const std::uint64_t version = decode_little_endian(header_bytes, 8);
  const std::uint64_t declared_width =
      decode_little_endian(header_bytes + 8, 8);
  const std::uint64_t vertices = decode_little_endian(header_bytes + 16, 8);
  const std::uint64_t edges = decode_little_endian(header_bytes + 24, 8);
  if (version != 1 ||
      !(declared_width == 0 || declared_width == 1 || declared_width == 2 ||
        declared_width == 4 || declared_width == 8)) {
    throw std::runtime_error("unsupported GR version or edge width: " + path);
  }
  if (vertices > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      edges > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("GR graph exceeds PuerCGP's 32-bit index limit");
  }
  const std::uint64_t destination_bytes = edges * sizeof(std::uint32_t);
  const std::uint64_t padding = (8 - destination_bytes % 8) % 8;
  const std::uint64_t weights_offset =
      sizeof(header_bytes) + vertices * sizeof(std::uint64_t) +
      destination_bytes + padding;
  const std::uint64_t canonical_size = weights_offset + edges * declared_width;
  const std::uint64_t file_size = std::filesystem::file_size(path);
  std::uint64_t physical_width = declared_width;
  if (file_size != canonical_size) {
    const std::uint64_t legacy_size = weights_offset + edges * 4;
    if ((declared_width != 1 && declared_width != 2) ||
        file_size != legacy_size) {
      throw std::runtime_error("GR file size does not match declared layout");
    }
    physical_width = 4;
  }

  host_csr_graph graph;
  graph.vertices = static_cast<int>(vertices);
  graph.edges = static_cast<int>(edges);
  graph.row_offsets.resize(static_cast<std::size_t>(vertices) + 1);
  graph.column_indices.resize(static_cast<std::size_t>(edges));
  graph.edge_weights.resize(static_cast<std::size_t>(edges));
  graph.row_offsets[0] = 0;

  constexpr std::size_t chunk_elements = 1U << 20;
  std::vector<std::uint64_t> offset_buffer(chunk_elements);
  std::uint64_t previous = 0;
  for (std::uint64_t first = 0; first < vertices;) {
    const auto count = static_cast<std::size_t>(
        std::min<std::uint64_t>(chunk_elements, vertices - first));
    input.read(reinterpret_cast<char*>(offset_buffer.data()),
               static_cast<std::streamsize>(count * sizeof(std::uint64_t)));
    if (!input) throw std::runtime_error("truncated GR offsets");
    for (std::size_t index = 0; index < count; ++index) {
      const std::uint64_t value = offset_buffer[index];
      if (value < previous || value > edges) {
        throw std::runtime_error("non-monotonic GR offset");
      }
      graph.row_offsets[static_cast<std::size_t>(first) + index + 1] =
          static_cast<int>(value);
      previous = value;
    }
    first += count;
  }
  if (previous != edges) {
    throw std::runtime_error("last GR offset does not equal E");
  }

  std::vector<std::uint32_t> destination_buffer(chunk_elements);
  for (std::uint64_t first = 0; first < edges;) {
    const auto count = static_cast<std::size_t>(
        std::min<std::uint64_t>(chunk_elements, edges - first));
    input.read(reinterpret_cast<char*>(destination_buffer.data()),
               static_cast<std::streamsize>(count * sizeof(std::uint32_t)));
    if (!input) throw std::runtime_error("truncated GR destinations");
    for (std::size_t index = 0; index < count; ++index) {
      if (destination_buffer[index] >= vertices) {
        throw std::runtime_error("GR destination outside [0,V)");
      }
      graph.column_indices[static_cast<std::size_t>(first) + index] =
          static_cast<int>(destination_buffer[index]);
    }
    first += count;
  }
  if (padding != 0) {
    unsigned char padding_bytes[8] = {};
    input.read(reinterpret_cast<char*>(padding_bytes), padding);
    if (!input) throw std::runtime_error("truncated GR padding");
    for (std::uint64_t byte = 0; byte < padding; ++byte) {
      if (padding_bytes[byte] != 0) {
        throw std::runtime_error("non-zero GR destination padding");
      }
    }
  }

  if (declared_width == 0) {
    std::fill(graph.edge_weights.begin(), graph.edge_weights.end(), 1.0f);
  } else {
    std::vector<unsigned char> weight_buffer(
        chunk_elements * static_cast<std::size_t>(physical_width));
    for (std::uint64_t first = 0; first < edges;) {
      const auto count = static_cast<std::size_t>(
          std::min<std::uint64_t>(chunk_elements, edges - first));
      const auto bytes = count * static_cast<std::size_t>(physical_width);
      input.read(reinterpret_cast<char*>(weight_buffer.data()),
                 static_cast<std::streamsize>(bytes));
      if (!input) throw std::runtime_error("truncated GR weights");
      for (std::size_t index = 0; index < count; ++index) {
        const auto value = decode_little_endian(
            weight_buffer.data() + index * physical_width,
            static_cast<unsigned>(physical_width));
        if (value > std::numeric_limits<std::uint32_t>::max() ||
            (physical_width == 4 && declared_width < 4 &&
             value >= (std::uint64_t{1} << (8 * declared_width)))) {
          throw std::runtime_error("GR weight exceeds its supported range");
        }
        graph.edge_weights[static_cast<std::size_t>(first) + index] =
            static_cast<float>(value);
      }
      first += count;
    }
  }

  thrust::host_vector<int> row_offsets(graph.row_offsets.begin(),
                                       graph.row_offsets.end());
  thrust::host_vector<int> column_indices(graph.column_indices.begin(),
                                          graph.column_indices.end());
  thrust::host_vector<float> edge_weights(graph.edge_weights.begin(),
                                          graph.edge_weights.end());
  graph.device_graph = puercgp::csr_graph_storage<int, int, float>(
      graph.vertices, row_offsets, column_indices, edge_weights,
      build_pull_adjacency);
  return graph;
}

inline host_csr_graph load_graph_auto(const std::string& path,
                                      bool build_pull_adjacency = false) {
  if (std::filesystem::is_directory(path)) {
    return load_binary_csr_directory(path, build_pull_adjacency);
  }
  const auto extension = lower_copy(std::filesystem::path(path).extension());
  if (extension == ".gr" || extension == ".ggr") {
    return load_galois_gr(path, build_pull_adjacency);
  }
  return load_matrix_market(path, build_pull_adjacency);
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

inline std::vector<float> cpu_sswp(const host_csr_graph& graph, int source) {
  const float unreachable = -std::numeric_limits<float>::infinity();
  using item_t = std::pair<float, int>;
  std::priority_queue<item_t> queue;
  std::vector<float> widths(static_cast<std::size_t>(graph.vertices),
                            unreachable);
  widths[static_cast<std::size_t>(source)] =
      std::numeric_limits<float>::infinity();
  queue.emplace(widths[static_cast<std::size_t>(source)], source);
  while (!queue.empty()) {
    auto [width, vertex] = queue.top();
    queue.pop();
    if (width != widths[static_cast<std::size_t>(vertex)]) {
      continue;
    }
    for (int edge = graph.row_offsets[static_cast<std::size_t>(vertex)];
         edge < graph.row_offsets[static_cast<std::size_t>(vertex) + 1];
         ++edge) {
      int neighbor = graph.column_indices[static_cast<std::size_t>(edge)];
      float candidate =
          std::min(width, graph.edge_weights[static_cast<std::size_t>(edge)]);
      if (candidate > widths[static_cast<std::size_t>(neighbor)]) {
        widths[static_cast<std::size_t>(neighbor)] = candidate;
        queue.emplace(candidate, neighbor);
      }
    }
  }
  return widths;
}

inline std::vector<float> cpu_personalized_pagerank(
    const host_csr_graph& graph,
    const std::vector<double>& personalization,
    double damping_factor,
    double tolerance = 1.0e-12,
    int max_iterations = 100000) {
  const std::size_t vertex_count =
      static_cast<std::size_t>(graph.vertices);
  if (personalization.size() != vertex_count) {
    throw std::invalid_argument("personalization size must match graph");
  }
  std::vector<double> rank = personalization;
  std::vector<double> next(vertex_count, 0.0);
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    double dangling_mass = 0.0;
    for (int vertex = 0; vertex < graph.vertices; ++vertex) {
      int begin = graph.row_offsets[static_cast<std::size_t>(vertex)];
      int end = graph.row_offsets[static_cast<std::size_t>(vertex) + 1];
      if (begin == end) {
        dangling_mass += rank[static_cast<std::size_t>(vertex)];
      }
    }
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
      next[vertex] =
          ((1.0 - damping_factor) + damping_factor * dangling_mass) *
          personalization[vertex];
    }
    for (int source = 0; source < graph.vertices; ++source) {
      int begin = graph.row_offsets[static_cast<std::size_t>(source)];
      int end = graph.row_offsets[static_cast<std::size_t>(source) + 1];
      int degree = end - begin;
      if (degree == 0) {
        continue;
      }
      double contribution = damping_factor *
          rank[static_cast<std::size_t>(source)] /
          static_cast<double>(degree);
      for (int edge = begin; edge < end; ++edge) {
        int destination =
            graph.column_indices[static_cast<std::size_t>(edge)];
        next[static_cast<std::size_t>(destination)] += contribution;
      }
    }
    double delta = 0.0;
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
      delta += std::fabs(next[vertex] - rank[vertex]);
    }
    rank.swap(next);
    if (delta <= tolerance) {
      break;
    }
  }
  return std::vector<float>(rank.begin(), rank.end());
}

inline std::vector<float> cpu_pagerank(const host_csr_graph& graph,
                                       double damping_factor) {
  const std::size_t vertex_count =
      static_cast<std::size_t>(graph.vertices);
  std::vector<double> personalization(
      vertex_count, 1.0 / static_cast<double>(vertex_count));
  return cpu_personalized_pagerank(graph, personalization, damping_factor);
}

inline std::vector<float> cpu_ppr(const host_csr_graph& graph,
                                 int source,
                                 double damping_factor) {
  std::vector<double> personalization(
      static_cast<std::size_t>(graph.vertices), 0.0);
  personalization[static_cast<std::size_t>(source)] = 1.0;
  return cpu_personalized_pagerank(graph, personalization, damping_factor);
}

// WCC via Label Propagation（异步 LP，与 GPU hybrid push 语义一致）
//   init : label[v] = v  （每个顶点初始为自己的连通分量代表）
//   iter : for each edge (v, u): label[v] = min(label[v], label[u])
//          异步更新——本轮修改立即影响后续顶点读取（与 GPU atomicMin push 同构）
//   conv : label[v] 收敛到 CC(v) 内的 min vertex id（min-reduce 单调减有下界）
//   CSR  : 假定双向存储（对称），遍历 row_offsets[v]..row_offsets[v+1] 即覆盖邻居
//   比对 : GPU 与 CPU 都用 min-reduce，收敛值 = CC 内 min vertex id，可直接 == 比对
inline std::vector<float> cpu_wcc(const host_csr_graph& graph,
                                  int max_iterations = 10000) {
  const int V = graph.vertices;
  std::vector<float> label(static_cast<std::size_t>(V));
  for (int v = 0; v < V; ++v) {
    label[static_cast<std::size_t>(v)] = static_cast<float>(v);
  }
  for (int iter = 0; iter < max_iterations; ++iter) {
    bool changed = false;
    for (int v = 0; v < V; ++v) {
      const float cur = label[static_cast<std::size_t>(v)];
      float best = cur;
      for (int edge = graph.row_offsets[static_cast<std::size_t>(v)];
           edge < graph.row_offsets[static_cast<std::size_t>(v) + 1]; ++edge) {
        float nb = label[static_cast<std::size_t>(
            graph.column_indices[static_cast<std::size_t>(edge)])];
        if (nb < best) {
          best = nb;
        }
      }
      if (best < cur) {
        label[static_cast<std::size_t>(v)] = best;
        changed = true;
      }
    }
    if (!changed) break;
  }
  return label;
}

inline float median(std::vector<float> values) {
  if (values.empty()) {
    return 0.0f;
  }
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

}  // namespace puercgp_examples
