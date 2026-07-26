#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>

#include "bench_types.cuh"

namespace ppbench {

namespace {

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(status));
  }
}

int popcount64(mask_t value) {
  return __builtin_popcountll(static_cast<unsigned long long>(value));
}

mask_t low_bits(int count, int offset = 0) {
  if (count <= 0) return 0;
  if (count >= 64) return ~mask_t{0};
  return ((mask_t{1} << count) - 1) << offset;
}

std::string lower_copy(std::string value) {
  for (char& c : value) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return value;
}

void add_weighted_edge(std::vector<std::tuple<int, int, float>>& edges,
                       int src,
                       int dst,
                       float weight) {
  if (src >= 0 && dst >= 0) edges.emplace_back(src, dst, weight);
}

host_graph_t build_csr_from_edges(int vertices,
                                  std::vector<std::tuple<int, int, float>> edges,
                                  std::string name) {
  std::sort(edges.begin(), edges.end());
  host_graph_t g;
  g.V = vertices;
  g.E = static_cast<int>(edges.size());
  g.name = std::move(name);
  g.row_offsets.assign(static_cast<std::size_t>(vertices) + 1, 0);
  for (auto [src, dst, w] : edges) {
    (void)dst;
    (void)w;
    if (src < 0 || src >= vertices) {
      throw std::runtime_error("source vertex out of range");
    }
    if (dst < 0 || dst >= vertices) {
      throw std::runtime_error("destination vertex out of range");
    }
    ++g.row_offsets[static_cast<std::size_t>(src) + 1];
  }
  for (int v = 0; v < vertices; ++v) {
    g.row_offsets[static_cast<std::size_t>(v) + 1] +=
        g.row_offsets[static_cast<std::size_t>(v)];
  }
  g.column_indices.resize(edges.size());
  g.weights.resize(edges.size());
  std::vector<int> cursor = g.row_offsets;
  for (auto [src, dst, w] : edges) {
    int pos = cursor[static_cast<std::size_t>(src)]++;
    g.column_indices[static_cast<std::size_t>(pos)] = dst;
    g.weights[static_cast<std::size_t>(pos)] = w;
  }

  std::vector<std::tuple<int, int, float>> rev;
  rev.reserve(edges.size());
  for (auto [src, dst, w] : edges) {
    rev.emplace_back(dst, src, w);
  }
  std::sort(rev.begin(), rev.end());
  g.in_row_offsets.assign(static_cast<std::size_t>(vertices) + 1, 0);
  for (auto [dst, src, w] : rev) {
    (void)src;
    (void)w;
    ++g.in_row_offsets[static_cast<std::size_t>(dst) + 1];
  }
  for (int v = 0; v < vertices; ++v) {
    g.in_row_offsets[static_cast<std::size_t>(v) + 1] +=
        g.in_row_offsets[static_cast<std::size_t>(v)];
  }
  g.in_column_indices.resize(rev.size());
  g.in_weights.resize(rev.size());
  cursor = g.in_row_offsets;
  for (auto [dst, src, w] : rev) {
    int pos = cursor[static_cast<std::size_t>(dst)]++;
    g.in_column_indices[static_cast<std::size_t>(pos)] = src;
    g.in_weights[static_cast<std::size_t>(pos)] = w;
  }
  return g;
}

host_graph_t load_matrix_market(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("could not open matrix: " + path);

  std::string header;
  std::getline(input, header);
  std::string header_lower = lower_copy(header);
  const bool pattern = header_lower.find("pattern") != std::string::npos;
  const bool symmetric = header_lower.find("symmetric") != std::string::npos;

  std::string line;
  do {
    if (!std::getline(input, line)) {
      throw std::runtime_error("missing matrix dimensions: " + path);
    }
  } while (line.empty() || line[0] == '%');

  std::stringstream dims(line);
  int rows = 0, cols = 0, entries = 0;
  dims >> rows >> cols >> entries;
  const int vertices = std::max(rows, cols);

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
    int src = 0, dst = 0;
    float weight = 1.0f;
    row >> src >> dst;
    if (!pattern) row >> weight;
    --src;
    --dst;
    add_weighted_edge(edges, src, dst, weight);
    if (symmetric && src != dst) add_weighted_edge(edges, dst, src, weight);
  }
  return build_csr_from_edges(vertices, std::move(edges), path);
}

host_graph_t load_binary_csr_directory(const std::string& path) {
  namespace fs = std::filesystem;
  fs::path dir(path);
  fs::path row_path = dir / "csr_vlist.bin";
  fs::path col_path = dir / "csr_elist.bin";
  fs::path w_path = dir / "csr_weightlist.bin";
  if (!fs::is_regular_file(row_path) || !fs::is_regular_file(col_path)) {
    throw std::runtime_error("CSR directory must contain csr_vlist.bin and "
                             "csr_elist.bin: " +
                             path);
  }
  const auto row_bytes = fs::file_size(row_path);
  const auto col_bytes = fs::file_size(col_path);
  if (row_bytes % sizeof(int) != 0 || col_bytes % sizeof(int) != 0) {
    throw std::runtime_error("CSR binary files must use int32: " + path);
  }
  host_graph_t raw;
  raw.V = static_cast<int>(row_bytes / sizeof(int) - 1);
  raw.E = static_cast<int>(col_bytes / sizeof(int));
  raw.name = path;
  raw.row_offsets.resize(static_cast<std::size_t>(raw.V) + 1);
  raw.column_indices.resize(static_cast<std::size_t>(raw.E));
  raw.weights.resize(static_cast<std::size_t>(raw.E), 1.0f);
  std::ifstream rows(row_path, std::ios::binary);
  std::ifstream cols(col_path, std::ios::binary);
  rows.read(reinterpret_cast<char*>(raw.row_offsets.data()),
            static_cast<std::streamsize>(row_bytes));
  cols.read(reinterpret_cast<char*>(raw.column_indices.data()),
            static_cast<std::streamsize>(col_bytes));
  if (!rows || !cols) throw std::runtime_error("failed to read CSR: " + path);
  if (fs::is_regular_file(w_path)) {
    if (fs::file_size(w_path) != col_bytes) {
      throw std::runtime_error("csr_weightlist.bin size mismatch: " + path);
    }
    std::vector<int> w_int(static_cast<std::size_t>(raw.E));
    std::ifstream ws(w_path, std::ios::binary);
    ws.read(reinterpret_cast<char*>(w_int.data()),
            static_cast<std::streamsize>(col_bytes));
    for (int i = 0; i < raw.E; ++i) raw.weights[i] = static_cast<float>(w_int[i]);
  }

  std::vector<std::tuple<int, int, float>> edges;
  edges.reserve(static_cast<std::size_t>(raw.E));
  for (int src = 0; src < raw.V; ++src) {
    for (int e = raw.row_offsets[src]; e < raw.row_offsets[src + 1]; ++e) {
      edges.emplace_back(src, raw.column_indices[static_cast<std::size_t>(e)],
                         raw.weights[static_cast<std::size_t>(e)]);
    }
  }
  return build_csr_from_edges(raw.V, std::move(edges), path);
}

template <typename T>
void read_exact(std::ifstream& in, T* dst, std::size_t count, const std::string& what) {
  in.read(reinterpret_cast<char*>(dst),
          static_cast<std::streamsize>(sizeof(T) * count));
  if (!in) throw std::runtime_error("failed reading " + what);
}

host_graph_t load_galois_gr(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("could not open Galois GR: " + path);

  std::uint64_t version = 0;
  std::uint64_t size_edge_ty = 0;
  std::uint64_t nvtxs64 = 0;
  std::uint64_t nedges64 = 0;
  read_exact(in, &version, 1, "GGR version");
  read_exact(in, &size_edge_ty, 1, "GGR edge type size");
  read_exact(in, &nvtxs64, 1, "GGR vertex count");
  read_exact(in, &nedges64, 1, "GGR edge count");
  if (version != 1) {
    throw std::runtime_error("unsupported GGR version " +
                             std::to_string(version) + ": " + path);
  }
  if (nvtxs64 > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      nedges64 > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("GGR graph exceeds int32 limits: " + path);
  }
  const int V = static_cast<int>(nvtxs64);
  const int E = static_cast<int>(nedges64);
  std::vector<std::int64_t> row_start(static_cast<std::size_t>(V));
  read_exact(in, row_start.data(), row_start.size(), "GGR row_start");
  std::vector<std::int32_t> dst(static_cast<std::size_t>(E));
  read_exact(in, dst.data(), dst.size(), "GGR edge_dst");

  std::vector<float> weights(static_cast<std::size_t>(E), 1.0f);
  if (size_edge_ty != 0) {
    if ((E % 2) == 1) {
      in.seekg(4, std::ios::cur);
      if (!in) throw std::runtime_error("failed skipping GGR weight padding: " + path);
    }
    // Local GGR files use a nonzero size_edge_ty as a weighted marker. In the
    // datasets under /home/zyl/data/ggr_data the stored adjwgt payload is int32
    // even when size_edge_ty is 1.
    std::vector<std::int32_t> w(static_cast<std::size_t>(E));
    read_exact(in, w.data(), w.size(), "GGR adjwgt");
    for (int i = 0; i < E; ++i) weights[static_cast<std::size_t>(i)] =
        static_cast<float>(w[static_cast<std::size_t>(i)]);
  }

  std::vector<std::tuple<int, int, float>> edges;
  edges.reserve(static_cast<std::size_t>(E));
  int prev = 0;
  for (int src = 0; src < V; ++src) {
    std::int64_t end64 = row_start[static_cast<std::size_t>(src)];
    if (end64 < prev || end64 > E) {
      throw std::runtime_error("invalid GGR row_start in: " + path);
    }
    int end = static_cast<int>(end64);
    for (int e = prev; e < end; ++e) {
      int d = static_cast<int>(dst[static_cast<std::size_t>(e)]);
      if (d < 0 || d >= V) {
        throw std::runtime_error("GGR edge destination out of range: " + path);
      }
      edges.emplace_back(src, d, weights[static_cast<std::size_t>(e)]);
    }
    prev = end;
  }
  if (prev != E) {
    throw std::runtime_error("GGR final row_start does not match edge count: " + path);
  }
  return build_csr_from_edges(V, std::move(edges), path);
}

host_graph_t make_toy_graph() {
  std::vector<std::tuple<int, int, float>> edges;
  edges.emplace_back(0, 1, 2.0f);
  edges.emplace_back(0, 2, 5.0f);
  edges.emplace_back(1, 3, 1.0f);
  edges.emplace_back(2, 3, 1.0f);
  return build_csr_from_edges(4, std::move(edges), "toy");
}

host_graph_t load_graph(const std::string& path) {
  namespace fs = std::filesystem;
  if (path.empty() || path == "toy") return make_toy_graph();
  if (fs::is_directory(path)) return load_binary_csr_directory(path);
  if (fs::path(path).extension() == ".gr") return load_galois_gr(path);
  return load_matrix_market(path);
}

template <typename T>
void cuda_copy_vector(T*& dst, const std::vector<T>& src, const char* name) {
  check_cuda(cudaMalloc(&dst, src.size() * sizeof(T)), name);
  check_cuda(cudaMemcpy(dst, src.data(), src.size() * sizeof(T),
                        cudaMemcpyHostToDevice),
             name);
}

device_graph_storage_t upload_graph(const host_graph_t& g) {
  device_graph_storage_t d;
  cuda_copy_vector(d.row_offsets, g.row_offsets, "row_offsets");
  cuda_copy_vector(d.column_indices, g.column_indices, "column_indices");
  cuda_copy_vector(d.in_row_offsets, g.in_row_offsets, "in_row_offsets");
  cuda_copy_vector(d.in_column_indices, g.in_column_indices,
                   "in_column_indices");
  d.view = device_graph_t{g.V,
                          g.E,
                          d.row_offsets,
                          d.column_indices,
                          nullptr,
                          d.in_row_offsets,
                          d.in_column_indices,
                          nullptr};
  return d;
}

void free_graph(device_graph_storage_t& d) {
  cudaFree(d.row_offsets);
  cudaFree(d.column_indices);
  cudaFree(d.in_row_offsets);
  cudaFree(d.in_column_indices);
  d = {};
}

frontier_host_t empty_frontier(int V, int Q) {
  frontier_host_t f;
  f.Q = Q;
  f.frontier_mask.assign(static_cast<std::size_t>(V), 0);
  return f;
}

void finalize_frontier(frontier_host_t& f) {
  f.frontier_vertices.clear();
  f.sum_per_query_frontier_size = 0;
  f.active_query_pairs = 0;
  for (std::size_t v = 0; v < f.frontier_mask.size(); ++v) {
    mask_t mask = f.frontier_mask[v];
    if (mask != 0) {
      f.frontier_vertices.push_back(static_cast<int>(v));
      int pc = popcount64(mask);
      f.active_query_pairs += static_cast<std::size_t>(pc);
      f.sum_per_query_frontier_size += static_cast<std::size_t>(pc);
    }
  }
  if (!f.frontier_vertices.empty()) {
    f.avg_popcount_per_active_vertex =
        static_cast<double>(f.active_query_pairs) /
        static_cast<double>(f.frontier_vertices.size());
  }
  const double denom = static_cast<double>(std::max<std::size_t>(
      1, f.sum_per_query_frontier_size));
  f.real_overlap =
      1.0 - static_cast<double>(f.frontier_vertices.size()) / denom;
  if (f.real_overlap < 0.0) f.real_overlap = 0.0;
}

std::vector<int> sample_vertices(int V, int count, std::mt19937& rng) {
  count = std::max(0, std::min(V, count));
  std::vector<int> vertices(static_cast<std::size_t>(V));
  std::iota(vertices.begin(), vertices.end(), 0);
  std::shuffle(vertices.begin(), vertices.end(), rng);
  vertices.resize(static_cast<std::size_t>(count));
  return vertices;
}

std::vector<int> sample_queries(int Q, int count, int offset, std::mt19937& rng) {
  count = std::max(0, std::min(Q, count));
  std::vector<int> qs(static_cast<std::size_t>(Q));
  for (int i = 0; i < Q; ++i) qs[static_cast<std::size_t>(i)] = offset + i;
  std::shuffle(qs.begin(), qs.end(), rng);
  qs.resize(static_cast<std::size_t>(count));
  return qs;
}

frontier_host_t generate_density_frontier(const host_graph_t& g,
                                          const bench_config_t& cfg) {
  std::mt19937 rng(static_cast<unsigned>(cfg.seed));
  frontier_host_t f = empty_frontier(g.V, cfg.Q);
  int unique_count =
      static_cast<int>(std::llround(cfg.rho_v * static_cast<double>(g.V)));
  unique_count = std::max(0, std::min(g.V, unique_count));
  int q_count =
      static_cast<int>(std::llround(cfg.rho_q * static_cast<double>(cfg.Q)));
  if (cfg.rho_q > 0.0) q_count = std::max(1, q_count);
  q_count = std::min(cfg.Q, q_count);

  for (int v : sample_vertices(g.V, unique_count, rng)) {
    mask_t mask = 0;
    for (int q : sample_queries(cfg.Q, q_count, 0, rng)) {
      mask |= (mask_t{1} << q);
    }
    f.frontier_mask[static_cast<std::size_t>(v)] = mask;
  }
  finalize_frontier(f);
  return f;
}

frontier_host_t generate_overlap_frontier(const host_graph_t& g,
                                          const bench_config_t& cfg) {
  std::mt19937 rng(static_cast<unsigned>(cfg.seed));
  frontier_host_t f = empty_frontier(g.V, cfg.Q);
  int per_q = cfg.frontier_size_per_query;
  if (per_q <= 0) {
    per_q = std::max(1, static_cast<int>(
                            std::llround(cfg.rho_v * static_cast<double>(g.V))));
  }
  per_q = std::max(0, std::min(g.V, per_q));
  int shared = static_cast<int>(std::llround(cfg.overlap * per_q));
  shared = std::max(0, std::min(per_q, shared));
  auto core = sample_vertices(g.V, shared, rng);
  std::vector<int> all_vertices(static_cast<std::size_t>(g.V));
  std::iota(all_vertices.begin(), all_vertices.end(), 0);

  for (int q = 0; q < cfg.Q; ++q) {
    std::unordered_set<int> chosen;
    for (int v : core) chosen.insert(v);
    std::shuffle(all_vertices.begin(), all_vertices.end(), rng);
    for (int v : all_vertices) {
      if (static_cast<int>(chosen.size()) >= per_q) break;
      chosen.insert(v);
    }
    for (int v : chosen) f.frontier_mask[static_cast<std::size_t>(v)] |= (mask_t{1} << q);
  }
  finalize_frontier(f);
  return f;
}

void add_group_density(frontier_host_t& f,
                       int V,
                       int q_offset,
                       int q_count,
                       double rho_v,
                       double rho_q,
                       std::mt19937& rng) {
  int unique_count =
      static_cast<int>(std::llround(rho_v * static_cast<double>(V)));
  unique_count = std::max(0, std::min(V, unique_count));
  int active_q =
      static_cast<int>(std::llround(rho_q * static_cast<double>(q_count)));
  if (rho_q > 0.0) active_q = std::max(1, active_q);
  active_q = std::min(q_count, active_q);
  for (int v : sample_vertices(V, unique_count, rng)) {
    for (int q : sample_queries(q_count, active_q, q_offset, rng)) {
      f.frontier_mask[static_cast<std::size_t>(v)] |= (mask_t{1} << q);
    }
  }
}

frontier_host_t generate_mixed_frontier(const host_graph_t& g,
                                        const bench_config_t& cfg) {
  std::mt19937 rng(static_cast<unsigned>(cfg.seed));
  frontier_host_t f = empty_frontier(g.V, cfg.Q);
  add_group_density(f, g.V, 0, cfg.Q_push, cfg.rho_push, 1.0, rng);
  add_group_density(f, g.V, cfg.Q_push, cfg.Q_pull, cfg.rho_pull, 1.0, rng);
  finalize_frontier(f);
  return f;
}

frontier_host_t generate_frontier(const host_graph_t& g,
                                  const bench_config_t& cfg) {
  if (cfg.Q < 1 || cfg.Q > 64) {
    throw std::invalid_argument("Q must be in [1,64]");
  }
  if (cfg.gen_mode == gen_mode_t::density) {
    return generate_density_frontier(g, cfg);
  }
  if (cfg.gen_mode == gen_mode_t::overlap) {
    return generate_overlap_frontier(g, cfg);
  }
  return generate_mixed_frontier(g, cfg);
}

device_state_t allocate_state(const host_graph_t& g, const bench_config_t& cfg) {
  device_state_t s;
  check_cuda(cudaMalloc(&s.frontier_mask, g.V * sizeof(mask_t)),
             "frontier_mask");
  int input_q = std::min(cfg.Q, 32);
  int output_q = cfg.Q;
  if (cfg.exec_mode == exec_mode_t::mixed_serial ||
      cfg.exec_mode == exec_mode_t::mixed_concurrent) {
    input_q = std::min(std::max(cfg.Q_push, cfg.Q_pull), 32);
    output_q = std::max(cfg.Q_push, cfg.Q_pull);
  }
  check_cuda(cudaMalloc(&s.spmm_input,
                        static_cast<std::size_t>(g.V) * input_q * sizeof(float)),
             "spmm_input");
  check_cuda(cudaMalloc(&s.spmm_output,
                        static_cast<std::size_t>(g.V) * output_q * sizeof(float)),
             "spmm_output");
  if (cfg.exec_mode == exec_mode_t::mixed_concurrent) {
    check_cuda(cudaMalloc(&s.spmm_pull_input,
                          static_cast<std::size_t>(g.V) * input_q * sizeof(float)),
               "spmm_pull_input");
    check_cuda(cudaMalloc(&s.spmm_pull_output,
                          static_cast<std::size_t>(g.V) * output_q * sizeof(float)),
               "spmm_pull_output");
  }
  check_cuda(cudaMalloc(&s.frontier_vertices, g.V * sizeof(int)),
             "frontier_vertices");
  return s;
}

void free_state(device_state_t& s) {
  cudaFree(s.frontier_mask);
  cudaFree(s.spmm_input);
  cudaFree(s.spmm_output);
  cudaFree(s.spmm_pull_input);
  cudaFree(s.spmm_pull_output);
  cudaFree(s.frontier_vertices);
  s = {};
}

void upload_state(const host_graph_t& g,
                  const bench_config_t& cfg,
                  const frontier_host_t& f,
                  device_state_t& s) {
  check_cuda(cudaMemcpy(s.frontier_mask, f.frontier_mask.data(),
                        g.V * sizeof(mask_t), cudaMemcpyHostToDevice),
             "copy frontier_mask");
  check_cuda(cudaMemcpy(s.frontier_vertices, f.frontier_vertices.data(),
                        f.frontier_vertices.size() * sizeof(int),
                        cudaMemcpyHostToDevice),
             "copy frontier_vertices");
}

}  // namespace

}  // namespace ppbench

using namespace ppbench;

__global__ void build_spmm_input_kernel(int V,
                                        int Q,
                                        const mask_t* frontier_mask,
                                        float* input,
                                        int q_offset,
                                        int M) {
  std::size_t total = static_cast<std::size_t>(V) * static_cast<std::size_t>(M);
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t i = tid; i < total; i += stride) {
    int v = static_cast<int>(i / static_cast<std::size_t>(M));
    int local_q = static_cast<int>(i % static_cast<std::size_t>(M));
    int q = q_offset + local_q;
    input[i] = (frontier_mask[v] & (mask_t{1} << q)) ? 1.0f : 0.0f;
  }
}

__global__ void push_scatter_sum_kernel(device_graph_t graph,
                                        int M,
                                        const int* frontier_vertices,
                                        int unique_count,
                                        const mask_t* frontier_mask,
                                        float* output,
                                        int q_offset) {
  for (int i = blockIdx.x; i < unique_count; i += gridDim.x) {
    int src = frontier_vertices[i];
    mask_t active = frontier_mask[src];
    int begin = graph.row_offsets[src];
    int end = graph.row_offsets[src + 1];
    for (int e = begin + threadIdx.x; e < end; e += blockDim.x) {
      int dst = graph.column_indices[e];
      mask_t bits = active >> q_offset;
      if (M < 64) bits &= ((mask_t{1} << M) - 1);
      for (int local_q = 0; local_q < M; ++local_q) {
        if ((bits & (mask_t{1} << local_q)) == 0) continue;
        atomicAdd(output + static_cast<std::size_t>(dst) * M + local_q, 1.0f);
      }
    }
  }
}

template <int M>
__global__ void ge_spmm_simple_kernel(int V,
                                      const int* row_offsets,
                                      const int* col_indices,
                                      const float* input,
                                      float* output,
                                      int tile_row) {
  int row = tile_row * blockIdx.x + threadIdx.y;
  if (row >= V) return;
  int col = threadIdx.x;
  int begin = row_offsets[row];
  int end = row_offsets[row + 1];
  float acc = 0.0f;
  for (int e = begin; e < end; ++e) {
    int nb = col_indices[e];
    acc += input[static_cast<std::size_t>(nb) * M + col];
  }
  output[static_cast<std::size_t>(row) * M + col] = acc;
}

template <int M, int CF>
__global__ void ge_spmm_smem_kernel(int V,
                                    const int* row_offsets,
                                    const int* col_indices,
                                    const float* input,
                                    float* output,
                                    int tile_row) {
  extern __shared__ int col_sh[];
  int shmem_offset = threadIdx.y << 5;
  int thread_idx = shmem_offset + threadIdx.x;
  int row = tile_row * blockIdx.x + threadIdx.y;
  if (row >= V) return;

  int cid = (blockIdx.y * (CF << 5)) + threadIdx.x;
  int begin = row_offsets[row];
  int end = row_offsets[row + 1];
  int ptr = begin + threadIdx.x;
  float acc[CF];
#pragma unroll
  for (int c = 0; c < CF; ++c) acc[c] = 0.0f;

  int nout = (M - cid + 31) / 32;
  if (nout < 0) nout = 0;
  if (nout > CF) nout = CF;

  for (int jj = begin; jj < end; jj += 32) {
    if (ptr < end) {
      col_sh[thread_idx] = col_indices[ptr] * M;
    }
    __syncwarp();
    ptr += 32;
    for (int kk = 0; kk < 32 && jj + kk < end; ++kk) {
      int offset = col_sh[shmem_offset + kk] + cid;
#pragma unroll
      for (int c = 0; c < CF; ++c) {
        if (c < nout) acc[c] += input[offset + c * 32];
      }
    }
    __syncwarp();
  }

  int out_base = row * M + cid;
#pragma unroll
  for (int c = 0; c < CF; ++c) {
    if (c < nout) output[out_base + c * 32] = acc[c];
  }
}

namespace ppbench {
namespace {

float elapsed_event_ms(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&ms, start, stop), "elapsed event");
  return ms;
}

void launch_build_spmm_input(const device_graph_t& graph,
                             const bench_config_t& cfg,
                             device_state_t& state,
                             int q_offset,
                             int M,
                             cudaStream_t stream);

void launch_push(const device_graph_t& graph,
                 const bench_config_t& cfg,
                 const frontier_host_t& f,
                 device_state_t& state,
                 cudaStream_t stream) {
  int blocks = std::max(1, std::min(65535, static_cast<int>(f.frontier_vertices.size())));
  int q_offset = 0;
  int M = cfg.Q;
  if (cfg.exec_mode == exec_mode_t::mixed_serial ||
      cfg.exec_mode == exec_mode_t::mixed_concurrent) {
    M = cfg.Q_push;
  }
  check_cuda(cudaMemsetAsync(state.spmm_output, 0,
                             static_cast<std::size_t>(graph.V) * M *
                                 sizeof(float),
                             stream),
             "clear spmm push output");
  push_scatter_sum_kernel<<<blocks, cfg.threads, 0, stream>>>(
      graph, M, state.frontier_vertices,
      static_cast<int>(f.frontier_vertices.size()), state.frontier_mask,
      state.spmm_output, q_offset);
}

void launch_build_spmm_input(const device_graph_t& graph,
                             const bench_config_t& cfg,
                             device_state_t& state,
                             int q_offset,
                             int M,
                             cudaStream_t stream) {
  std::size_t total = static_cast<std::size_t>(graph.V) * static_cast<std::size_t>(M);
  int blocks = std::max(1, std::min(65535, static_cast<int>((total + cfg.threads - 1) / cfg.threads)));
  build_spmm_input_kernel<<<blocks, cfg.threads, 0, stream>>>(
      graph.V, cfg.Q, state.frontier_mask, state.spmm_input, q_offset, M);
}

template <int M>
void launch_ge_spmm_m(device_graph_t graph,
                      const bench_config_t& cfg,
                      const float* input,
                      float* output,
                      cudaStream_t stream) {
  if constexpr (M < 32) {
    int tile_row = std::max(1, 128 / M);
    int grid_x = (graph.V + tile_row - 1) / tile_row;
    ge_spmm_simple_kernel<M><<<grid_x, dim3(M, tile_row), 0, stream>>>(
        graph.V, graph.in_row_offsets, graph.in_column_indices, input, output,
        tile_row);
  } else {
    constexpr int CF = (M + 31) / 32;
    int tile_row = 4;
    int grid_x = (graph.V + tile_row - 1) / tile_row;
    int grid_y = (M + (CF * 32) - 1) / (CF * 32);
    std::size_t smem = static_cast<std::size_t>(32 * tile_row) * sizeof(int);
    ge_spmm_smem_kernel<M, CF><<<dim3(grid_x, grid_y), dim3(32, tile_row),
                                smem, stream>>>(
        graph.V, graph.in_row_offsets, graph.in_column_indices, input, output,
        tile_row);
  }
}

void launch_ge_spmm(device_graph_t graph,
                    const bench_config_t& cfg,
                    int M,
                    const float* input,
                    float* output,
                    cudaStream_t stream) {
  switch (M) {
    case 1: launch_ge_spmm_m<1>(graph, cfg, input, output, stream); break;
    case 2: launch_ge_spmm_m<2>(graph, cfg, input, output, stream); break;
    case 4: launch_ge_spmm_m<4>(graph, cfg, input, output, stream); break;
    case 8: launch_ge_spmm_m<8>(graph, cfg, input, output, stream); break;
    case 16: launch_ge_spmm_m<16>(graph, cfg, input, output, stream); break;
    case 32: launch_ge_spmm_m<32>(graph, cfg, input, output, stream); break;
    case 64: launch_ge_spmm_m<64>(graph, cfg, input, output, stream); break;
    default:
      throw std::invalid_argument("spmm_sum supports M in {1,2,4,8,16,32,64}");
  }
}

void launch_pull(const device_graph_t& graph,
                 const bench_config_t& cfg,
                 device_state_t& state,
                 cudaStream_t stream) {
  int q_offset = 0;
  int M = cfg.Q;
  if (cfg.exec_mode == exec_mode_t::mixed_serial ||
      cfg.exec_mode == exec_mode_t::mixed_concurrent) {
    q_offset = cfg.Q_push;
    M = cfg.Q_pull;
  }
  float* input = state.spmm_pull_input ? state.spmm_pull_input : state.spmm_input;
  float* output = state.spmm_pull_output ? state.spmm_pull_output : state.spmm_output;
  constexpr int kPullChunkQ = 32;
  for (int chunk_offset = 0; chunk_offset < M; chunk_offset += kPullChunkQ) {
    int chunk_q = std::min(kPullChunkQ, M - chunk_offset);
    std::size_t total = static_cast<std::size_t>(graph.V) *
                        static_cast<std::size_t>(chunk_q);
    int blocks = std::max(1, std::min(
                                 65535,
                                 static_cast<int>((total + cfg.threads - 1) /
                                                  cfg.threads)));
    build_spmm_input_kernel<<<blocks, cfg.threads, 0, stream>>>(
        graph.V, cfg.Q, state.frontier_mask, input, q_offset + chunk_offset,
        chunk_q);
    check_cuda(cudaMemsetAsync(output, 0, total * sizeof(float), stream),
               "clear spmm pull output");
    launch_ge_spmm(graph, cfg, chunk_q, input, output, stream);
  }
}

bench_stats_t add_stats(const bench_stats_t& a, const bench_stats_t& b) {
  bench_stats_t out;
  out.vertices_scanned = a.vertices_scanned + b.vertices_scanned;
  out.edges_scanned = a.edges_scanned + b.edges_scanned;
  out.active_query_pairs = a.active_query_pairs + b.active_query_pairs;
  out.src_value_loads = a.src_value_loads + b.src_value_loads;
  out.dst_value_loads = a.dst_value_loads + b.dst_value_loads;
  out.weight_loads = a.weight_loads + b.weight_loads;
  out.relax_ops = a.relax_ops + b.relax_ops;
  out.atomic_ops = a.atomic_ops + b.atomic_ops;
  out.successful_updates = a.successful_updates + b.successful_updates;
  return out;
}

bench_stats_t estimate_push_spmm_stats(const host_graph_t& graph,
                                       const frontier_host_t& f,
                                       mask_t group_mask) {
  bench_stats_t s;
  for (int src : f.frontier_vertices) {
    mask_t active = f.frontier_mask[static_cast<std::size_t>(src)] & group_mask;
    if (active == 0) continue;
    int pc = popcount64(active);
    int deg = graph.row_offsets[static_cast<std::size_t>(src) + 1] -
              graph.row_offsets[static_cast<std::size_t>(src)];
    s.vertices_scanned += 1;
    s.edges_scanned += static_cast<unsigned long long>(deg);
    s.active_query_pairs += static_cast<unsigned long long>(pc);
    unsigned long long ops = static_cast<unsigned long long>(deg) *
                             static_cast<unsigned long long>(pc);
    s.src_value_loads += ops;
    s.relax_ops += ops;
    s.atomic_ops += ops;
  }
  return s;
}

bench_stats_t estimate_pull_spmm_stats(const host_graph_t& graph, int M) {
  bench_stats_t s;
  s.vertices_scanned = static_cast<unsigned long long>(graph.V);
  s.edges_scanned = static_cast<unsigned long long>(graph.E) *
                    static_cast<unsigned long long>(M);
  s.src_value_loads = s.edges_scanned;
  s.relax_ops = s.edges_scanned;
  return s;
}

bench_result_t run_once(const host_graph_t& graph,
                        const device_graph_t& d_graph,
                        const bench_config_t& cfg,
                        const frontier_host_t& f,
                        device_state_t& state,
                        mask_t push_mask,
                        mask_t pull_mask) {
  upload_state(graph, cfg, f, state);

  cudaEvent_t start, mid, stop;
  check_cuda(cudaEventCreate(&start), "create start");
  check_cuda(cudaEventCreate(&mid), "create mid");
  check_cuda(cudaEventCreate(&stop), "create stop");
  bench_result_t result;

  if (cfg.exec_mode == exec_mode_t::all_push) {
    check_cuda(cudaEventRecord(start), "record start");
    launch_push(d_graph, cfg, f, state, 0);
    check_cuda(cudaEventRecord(stop), "record stop");
    check_cuda(cudaEventSynchronize(stop), "sync stop");
    result.push_ms = elapsed_event_ms(start, stop);
    result.total_ms = result.push_ms;
  } else if (cfg.exec_mode == exec_mode_t::all_pull) {
    check_cuda(cudaEventRecord(start), "record start");
    launch_pull(d_graph, cfg, state, 0);
    check_cuda(cudaEventRecord(stop), "record stop");
    check_cuda(cudaEventSynchronize(stop), "sync stop");
    result.pull_ms = elapsed_event_ms(start, stop);
    result.total_ms = result.pull_ms;
  } else if (cfg.exec_mode == exec_mode_t::mixed_serial) {
    check_cuda(cudaEventRecord(start), "record start");
    launch_push(d_graph, cfg, f, state, 0);
    check_cuda(cudaEventRecord(mid), "record mid");
    launch_pull(d_graph, cfg, state, 0);
    check_cuda(cudaEventRecord(stop), "record stop");
    check_cuda(cudaEventSynchronize(stop), "sync stop");
    result.push_ms = elapsed_event_ms(start, mid);
    result.pull_ms = elapsed_event_ms(mid, stop);
    result.total_ms = elapsed_event_ms(start, stop);
  } else {
    cudaStream_t push_stream, pull_stream;
    check_cuda(cudaStreamCreate(&push_stream), "create push stream");
    check_cuda(cudaStreamCreate(&pull_stream), "create pull stream");
    cudaEvent_t push_done, pull_done;
    check_cuda(cudaEventCreate(&push_done), "create push_done");
    check_cuda(cudaEventCreate(&pull_done), "create pull_done");
    check_cuda(cudaEventRecord(start), "record start");
    check_cuda(cudaStreamWaitEvent(push_stream, start), "push wait start");
    check_cuda(cudaStreamWaitEvent(pull_stream, start), "pull wait start");
    launch_push(d_graph, cfg, f, state, push_stream);
    launch_pull(d_graph, cfg, state, pull_stream);
    check_cuda(cudaEventRecord(push_done, push_stream), "record push_done");
    check_cuda(cudaEventRecord(pull_done, pull_stream), "record pull_done");
    check_cuda(cudaEventSynchronize(push_done), "sync push_done");
    check_cuda(cudaEventSynchronize(pull_done), "sync pull_done");
    result.push_ms = elapsed_event_ms(start, push_done);
    result.pull_ms = elapsed_event_ms(start, pull_done);
    result.total_ms = std::max(result.push_ms, result.pull_ms);
    cudaEventDestroy(push_done);
    cudaEventDestroy(pull_done);
    cudaStreamDestroy(push_stream);
    cudaStreamDestroy(pull_stream);
  }

  check_cuda(cudaGetLastError(), "kernel launch");
  if (cfg.exec_mode == exec_mode_t::all_push) {
    result.stats = estimate_push_spmm_stats(graph, f, push_mask);
  } else if (cfg.exec_mode == exec_mode_t::all_pull) {
    result.stats = estimate_pull_spmm_stats(graph, cfg.Q);
  } else {
    result.stats = add_stats(estimate_push_spmm_stats(graph, f, push_mask),
                             estimate_pull_spmm_stats(graph, cfg.Q_pull));
  }
  cudaEventDestroy(start);
  cudaEventDestroy(mid);
  cudaEventDestroy(stop);
  return result;
}

float median(std::vector<float> values) {
  if (values.empty()) return 0.0f;
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

bench_result_t run_repeated(const host_graph_t& graph,
                            const device_graph_t& d_graph,
                            const bench_config_t& cfg,
                            const frontier_host_t& f,
                            device_state_t& state,
                            mask_t push_mask,
                            mask_t pull_mask) {
  for (int i = 0; i < cfg.warmup; ++i) {
    (void)run_once(graph, d_graph, cfg, f, state, push_mask, pull_mask);
  }
  std::vector<float> push_ms, pull_ms, total_ms;
  bench_result_t last;
  for (int i = 0; i < cfg.repeat; ++i) {
    last = run_once(graph, d_graph, cfg, f, state, push_mask, pull_mask);
    push_ms.push_back(last.push_ms);
    pull_ms.push_back(last.pull_ms);
    total_ms.push_back(last.total_ms);
  }
  last.push_ms = median(push_ms);
  last.pull_ms = median(pull_ms);
  last.total_ms = median(total_ms);
  return last;
}

void write_csv_header(std::ostream& out) {
  out << "graph,V,E,Q,gen_mode,exec_mode,update_mode,"
      << "Q_push,Q_pull,rho_v,rho_q,rho_push,rho_pull,"
      << "overlap,overlap_push,overlap_pull,cross_overlap,"
      << "unique_vertices,active_query_pairs,avg_popcount,real_overlap,"
      << "push_ms,pull_ms,total_ms,"
      << "vertices_scanned,edges_scanned,src_value_loads,dst_value_loads,"
      << "weight_loads,relax_ops,atomic_ops,successful_updates,seed\n";
}

void write_csv_row(std::ostream& out,
                   const host_graph_t& g,
                   const bench_config_t& cfg,
                   const frontier_host_t& f,
                   const bench_result_t& r) {
  int q_push_out = cfg.Q_push;
  int q_pull_out = cfg.Q_pull;
  if (cfg.exec_mode == exec_mode_t::all_push) {
    q_push_out = cfg.Q;
    q_pull_out = 0;
  } else if (cfg.exec_mode == exec_mode_t::all_pull) {
    q_push_out = 0;
    q_pull_out = cfg.Q;
  }
  out << g.name << "," << g.V << "," << g.E << "," << cfg.Q << ","
      << to_string(cfg.gen_mode) << "," << to_string(cfg.exec_mode) << ","
      << to_string(cfg.update_mode) << "," << q_push_out << ","
      << q_pull_out << "," << cfg.rho_v << "," << cfg.rho_q << ","
      << cfg.rho_push << "," << cfg.rho_pull << "," << cfg.overlap << ","
      << cfg.overlap_push << "," << cfg.overlap_pull << ","
      << cfg.cross_overlap << "," << f.frontier_vertices.size() << ","
      << f.active_query_pairs << "," << f.avg_popcount_per_active_vertex
      << "," << f.real_overlap << "," << r.push_ms << "," << r.pull_ms
      << "," << r.total_ms << "," << r.stats.vertices_scanned << ","
      << r.stats.edges_scanned << "," << r.stats.src_value_loads << ","
      << r.stats.dst_value_loads << "," << r.stats.weight_loads << ","
      << r.stats.relax_ops << "," << r.stats.atomic_ops << ","
      << r.stats.successful_updates << "," << cfg.seed << "\n";
}

bool nearly_equal(float a, float b) {
  return std::fabs(a - b) < 1e-4f;
}

std::vector<float> cpu_pull_spmm_sum(const host_graph_t& g,
                                     const frontier_host_t& f,
                                     int q_offset,
                                     int M) {
  std::vector<float> out(static_cast<std::size_t>(g.V) * M, 0.0f);
  for (int dst = 0; dst < g.V; ++dst) {
    for (int e = g.in_row_offsets[static_cast<std::size_t>(dst)];
         e < g.in_row_offsets[static_cast<std::size_t>(dst) + 1]; ++e) {
      int src = g.in_column_indices[static_cast<std::size_t>(e)];
      mask_t bits = f.frontier_mask[static_cast<std::size_t>(src)] >> q_offset;
      if (M < 64) bits &= ((mask_t{1} << M) - 1);
      while (bits != 0) {
        int local_q = __builtin_ffsll(static_cast<long long>(bits)) - 1;
        bits &= bits - 1;
        out[static_cast<std::size_t>(dst) * M + local_q] += 1.0f;
      }
    }
  }
  return out;
}

void run_self_test() {
  bench_config_t cfg;
  cfg.graph_path = "toy";
  cfg.Q = 4;
  cfg.repeat = 1;
  cfg.warmup = 0;
  cfg.gen_mode = gen_mode_t::density;
  cfg.exec_mode = exec_mode_t::all_pull;
  cfg.update_mode = update_mode_t::spmm_sum;
  host_graph_t g = make_toy_graph();
  frontier_host_t f = empty_frontier(g.V, cfg.Q);
  f.frontier_mask[0] = mask_t{1} << 0;
  f.frontier_mask[1] = (mask_t{1} << 1) | (mask_t{1} << 3);
  f.frontier_mask[2] = mask_t{1} << 2;
  finalize_frontier(f);
  auto d_graph = upload_graph(g);
  auto state = allocate_state(g, cfg);

  auto verify_pull_output = [&](const char* name, int q_offset, int M) {
    auto expected = cpu_pull_spmm_sum(g, f, q_offset, M);
    std::vector<float> got(expected.size());
    check_cuda(cudaMemcpy(got.data(), state.spmm_output,
                          got.size() * sizeof(float), cudaMemcpyDeviceToHost),
               "copy self-test pull output");
    for (std::size_t i = 0; i < got.size(); ++i) {
      if (!nearly_equal(got[i], expected[i])) {
        std::ostringstream os;
        os << name << " mismatch at output " << i << ": got " << got[i]
           << " expected " << expected[i];
        throw std::runtime_error(os.str());
      }
    }
  };

  cfg.exec_mode = exec_mode_t::all_pull;
  (void)run_repeated(g, d_graph.view, cfg, f, state, low_bits(cfg.Q),
                     low_bits(cfg.Q));
  verify_pull_output("all_pull_ge_spmm", 0, cfg.Q);

  cfg.exec_mode = exec_mode_t::mixed_serial;
  cfg.Q_push = 2;
  cfg.Q_pull = 2;
  (void)run_repeated(g, d_graph.view, cfg, f, state, low_bits(cfg.Q_push, 0),
                     low_bits(cfg.Q_pull, cfg.Q_push));
  verify_pull_output("mixed_serial_ge_spmm_pull", cfg.Q_push, cfg.Q_pull);

  free_state(state);
  free_graph(d_graph);
  std::cout << "self-test: PASS\n";
}

std::string graph_stem_for_output(const std::string& graph_path) {
  if (graph_path == "toy") return "toy";
  namespace fs = std::filesystem;
  fs::path p(graph_path);
  return p.stem().string();
}

void run_density_push_pull_q_sweep(const bench_config_t& base_cfg) {
  if (base_cfg.out_dir.empty()) {
    throw std::invalid_argument("--out-dir is required with --sweep=density_push_pull_q");
  }
  const std::vector<double> rho_v_list = {
      0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002,
      0.005,   0.01,   0.02,   0.05,   0.1,   0.2};
  const std::vector<double> rho_q_list = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7};
  const std::vector<int> pull_q_list = {1, 2, 4, 8, 16, 32, 64};
  const std::vector<int> seeds = {1, 2, 3, 4, 5};

  host_graph_t graph = load_graph(base_cfg.graph_path);
  bench_config_t alloc_cfg = base_cfg;
  alloc_cfg.Q = 64;
  auto d_graph = upload_graph(graph);
  auto state = allocate_state(graph, alloc_cfg);

  namespace fs = std::filesystem;
  fs::path out_dir = fs::path(base_cfg.out_dir) / graph_stem_for_output(graph.name);
  fs::create_directories(out_dir);
  fs::path push_path = out_dir / "push_density.csv";
  fs::path pull_path = out_dir / "pull_q.csv";
  std::ofstream push_csv(push_path, std::ios::trunc);
  std::ofstream pull_csv(pull_path, std::ios::trunc);
  if (!push_csv) throw std::runtime_error("could not open csv: " + push_path.string());
  if (!pull_csv) throw std::runtime_error("could not open csv: " + pull_path.string());
  write_csv_header(push_csv);
  write_csv_header(pull_csv);

  std::cout << "graph=" << graph.name << "\n";
  std::cout << "push_csv=" << push_path << "\n";
  std::cout << "pull_csv=" << pull_path << "\n";

  const int push_total = static_cast<int>(seeds.size() * rho_v_list.size() *
                                          rho_q_list.size());
  int push_done = 0;
  for (int seed : seeds) {
    for (double rho_v : rho_v_list) {
      for (double rho_q : rho_q_list) {
        bench_config_t cfg = base_cfg;
        cfg.gen_mode = gen_mode_t::density;
        cfg.exec_mode = exec_mode_t::all_push;
        cfg.update_mode = update_mode_t::spmm_sum;
        cfg.Q = 64;
        cfg.seed = seed;
        cfg.rho_v = rho_v;
        cfg.rho_q = rho_q;
        frontier_host_t frontier = generate_frontier(graph, cfg);
        bench_result_t result = run_repeated(
            graph, d_graph.view, cfg, frontier, state, low_bits(cfg.Q),
            low_bits(cfg.Q));
        write_csv_row(push_csv, graph, cfg, frontier, result);
        push_csv.flush();
        ++push_done;
        std::cout << "push " << push_done << "/" << push_total
                  << " seed=" << seed << " rho_v=" << rho_v
                  << " rho_q=" << rho_q << " ms=" << result.push_ms << "\n";
      }
    }
  }

  const int pull_total = static_cast<int>(seeds.size() * pull_q_list.size());
  int pull_done = 0;
  for (int seed : seeds) {
    for (int pull_q : pull_q_list) {
      bench_config_t cfg = base_cfg;
      cfg.gen_mode = gen_mode_t::density;
      cfg.exec_mode = exec_mode_t::all_pull;
      cfg.update_mode = update_mode_t::spmm_sum;
      cfg.Q = pull_q;
      cfg.seed = seed;
      cfg.rho_v = 0.001;
      cfg.rho_q = 0.5;
      frontier_host_t frontier = generate_frontier(graph, cfg);
      bench_result_t result = run_repeated(
          graph, d_graph.view, cfg, frontier, state, low_bits(cfg.Q),
          low_bits(cfg.Q));
      write_csv_row(pull_csv, graph, cfg, frontier, result);
      pull_csv.flush();
      ++pull_done;
      std::cout << "pull " << pull_done << "/" << pull_total
                << " seed=" << seed << " Q=" << pull_q
                << " ms=" << result.pull_ms << "\n";
    }
  }

  free_state(state);
  free_graph(d_graph);
}

gen_mode_t parse_gen_mode(const std::string& value) {
  if (value == "density") return gen_mode_t::density;
  if (value == "overlap") return gen_mode_t::overlap;
  if (value == "mixed") return gen_mode_t::mixed;
  throw std::invalid_argument("unknown --mode: " + value);
}

exec_mode_t parse_exec_mode(const std::string& value) {
  if (value == "all_push") return exec_mode_t::all_push;
  if (value == "all_pull") return exec_mode_t::all_pull;
  if (value == "mixed_serial") return exec_mode_t::mixed_serial;
  if (value == "mixed_concurrent") return exec_mode_t::mixed_concurrent;
  throw std::invalid_argument("unknown --exec: " + value);
}

update_mode_t parse_update_mode(const std::string& value) {
  if (value == "spmm_sum") return update_mode_t::spmm_sum;
  throw std::invalid_argument("unknown --update: " + value);
}

bool starts_with(const std::string& value, const std::string& prefix) {
  return value.rfind(prefix, 0) == 0;
}

std::string after_eq(const std::string& arg) {
  auto pos = arg.find('=');
  if (pos == std::string::npos) throw std::invalid_argument("expected key=value: " + arg);
  return arg.substr(pos + 1);
}

bench_config_t parse_args(int argc, char** argv) {
  bench_config_t cfg;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--self-test") cfg.self_test = true;
    else if (starts_with(a, "--graph=")) cfg.graph_path = after_eq(a);
    else if (starts_with(a, "--csv=")) cfg.csv_path = after_eq(a);
    else if (starts_with(a, "--out-dir=")) cfg.out_dir = after_eq(a);
    else if (starts_with(a, "--sweep=")) {
      std::string sweep = after_eq(a);
      if (sweep == "density_push_pull_q") {
        cfg.density_pull_sweep = true;
      } else {
        throw std::invalid_argument("unknown --sweep: " + sweep);
      }
    }
    else if (starts_with(a, "--mode=")) cfg.gen_mode = parse_gen_mode(after_eq(a));
    else if (starts_with(a, "--exec=")) cfg.exec_mode = parse_exec_mode(after_eq(a));
    else if (starts_with(a, "--update=")) cfg.update_mode = parse_update_mode(after_eq(a));
    else if (starts_with(a, "--Q=")) cfg.Q = std::stoi(after_eq(a));
    else if (starts_with(a, "--Q-push=")) cfg.Q_push = std::stoi(after_eq(a));
    else if (starts_with(a, "--Q-pull=")) cfg.Q_pull = std::stoi(after_eq(a));
    else if (starts_with(a, "--repeat=")) cfg.repeat = std::stoi(after_eq(a));
    else if (starts_with(a, "--warmup=")) cfg.warmup = std::stoi(after_eq(a));
    else if (starts_with(a, "--seed=")) cfg.seed = std::stoi(after_eq(a));
    else if (starts_with(a, "--rho-v=")) cfg.rho_v = std::stod(after_eq(a));
    else if (starts_with(a, "--rho-q=")) cfg.rho_q = std::stod(after_eq(a));
    else if (starts_with(a, "--rho-push=")) cfg.rho_push = std::stod(after_eq(a));
    else if (starts_with(a, "--rho-pull=")) cfg.rho_pull = std::stod(after_eq(a));
    else if (starts_with(a, "--overlap=")) cfg.overlap = std::stod(after_eq(a));
    else if (starts_with(a, "--overlap-push=")) cfg.overlap_push = std::stod(after_eq(a));
    else if (starts_with(a, "--overlap-pull=")) cfg.overlap_pull = std::stod(after_eq(a));
    else if (starts_with(a, "--cross-overlap=")) cfg.cross_overlap = std::stod(after_eq(a));
    else if (starts_with(a, "--frontier-size-per-query=")) cfg.frontier_size_per_query = std::stoi(after_eq(a));
    else if (starts_with(a, "--threads=")) cfg.threads = std::stoi(after_eq(a));
    else {
      throw std::invalid_argument("unknown argument: " + a);
    }
  }
  if (cfg.Q > 64) throw std::invalid_argument("--Q must be <= 64");
  if (cfg.gen_mode == gen_mode_t::mixed) {
    cfg.Q = cfg.Q_push + cfg.Q_pull;
    if (cfg.Q > 64) throw std::invalid_argument("--Q-push + --Q-pull must be <= 64");
  }
  if (cfg.graph_path.empty()) cfg.graph_path = "toy";
  return cfg;
}

}  // namespace
}  // namespace ppbench

int main(int argc, char** argv) {
  try {
    bench_config_t cfg = ppbench::parse_args(argc, argv);
    if (cfg.self_test) {
      ppbench::run_self_test();
      return 0;
    }
    if (cfg.density_pull_sweep) {
      ppbench::run_density_push_pull_q_sweep(cfg);
      return 0;
    }

    host_graph_t graph = ppbench::load_graph(cfg.graph_path);
    frontier_host_t frontier = ppbench::generate_frontier(graph, cfg);
    auto d_graph = ppbench::upload_graph(graph);
    auto state = ppbench::allocate_state(graph, cfg);

    mask_t all_mask = low_bits(cfg.Q);
    mask_t push_mask = all_mask;
    mask_t pull_mask = all_mask;
    if (cfg.exec_mode == exec_mode_t::mixed_serial ||
        cfg.exec_mode == exec_mode_t::mixed_concurrent) {
      push_mask = low_bits(cfg.Q_push, 0);
      pull_mask = low_bits(cfg.Q_pull, cfg.Q_push);
    }

    bench_result_t result = ppbench::run_repeated(
        graph, d_graph.view, cfg, frontier, state, push_mask, pull_mask);

    std::ostream* out = &std::cout;
    std::ofstream csv_file;
    bool need_header = true;
    if (!cfg.csv_path.empty()) {
      namespace fs = std::filesystem;
      need_header = !fs::exists(cfg.csv_path) || fs::file_size(cfg.csv_path) == 0;
      csv_file.open(cfg.csv_path, std::ios::app);
      if (!csv_file) throw std::runtime_error("could not open csv: " + cfg.csv_path);
      out = &csv_file;
    }
    if (need_header) ppbench::write_csv_header(*out);
    ppbench::write_csv_row(*out, graph, cfg, frontier, result);

    ppbench::free_state(state);
    ppbench::free_graph(d_graph);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
