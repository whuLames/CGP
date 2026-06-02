#include <gunrock/algorithms/cgp_sssp.hxx>
#include <gunrock/framework/benchmark.hxx>
#include <gunrock/io/parameters.hxx>
#include <gunrock/util/performance.hxx>

#include "sssp_cpu.hxx"
#include "nlohmann/json.hpp"

#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace gunrock;
using namespace memory;

namespace {

template <typename vertex_t, typename edge_t>
struct cgp_csr_graph_t {
  using vertex_type = vertex_t;
  using edge_type = edge_t;
  using weight_type = float;

  vertex_t number_of_vertices = 0;
  edge_t number_of_edges = 0;
  edge_t* row_offsets = nullptr;
  vertex_t* column_indices = nullptr;

  __host__ __device__ __forceinline__ vertex_t get_number_of_vertices() const {
    return number_of_vertices;
  }

  __host__ __device__ __forceinline__ edge_t get_number_of_edges() const {
    return number_of_edges;
  }

  __host__ __device__ __forceinline__ edge_t
  get_starting_edge(vertex_t const& v) const {
    return row_offsets[v];
  }

  __host__ __device__ __forceinline__ vertex_t
  get_destination_vertex(edge_t const& e) const {
    return column_indices[e];
  }

  __host__ __device__ __forceinline__ float get_edge_weight(
      edge_t const&) const {
    return 1.0f;
  }
};

template <typename vertex_t, typename edge_t>
struct loaded_gr_t {
  thrust::device_vector<edge_t> row_offsets;
  thrust::device_vector<vertex_t> column_indices;
  vertex_t n_vertices = 0;
  edge_t n_edges = 0;

  cgp_csr_graph_t<vertex_t, edge_t> view() {
    return {n_vertices, n_edges, thrust::raw_pointer_cast(row_offsets.data()),
            thrust::raw_pointer_cast(column_indices.data())};
  }
};

struct cgp_arguments_t {
  std::string filename;
  std::string source_string;
  std::string query_file;
  std::string json_dir = ".";
  std::string json_file;
  std::string traversal_mode = "hybrid";
  std::string push_strategy = "edge_balanced";
  std::string pull_strategy = "bitmap";
  double pull_frontier_ratio = 0.15;
  double pull_edge_ratio = 0.20;
  int num_runs = 1;
  bool validate = false;
  bool binary = false;
  bool profile_levels = false;
};

std::string trim(std::string value) {
  auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

cgp_arguments_t parse_arguments(int argc, char** argv) {
  cxxopts::Options options(argv[0], "CGP Concurrent Single Source Shortest Path");
  options.add_options()("help", "Print help")("m,market", "Matrix file",
                                              cxxopts::value<std::string>())(
      "s,src", "Source(s); comma-separated string of ints",
      cxxopts::value<std::string>())("query_file",
                                     "Query source file, one source per line",
                                     cxxopts::value<std::string>())(
      "n,num_runs", "Number of repeated runs when a single --src is passed",
      cxxopts::value<int>())("validate", "CPU validation for last query")(
      "d,json_dir", "JSON output directory", cxxopts::value<std::string>())(
      "f,json_file", "JSON output file", cxxopts::value<std::string>())(
      "traversal_mode", "Traversal mode: push, pull, or hybrid",
      cxxopts::value<std::string>())(
      "push_strategy",
      "Push strategy: edge_balanced, shared_node, or "
      "shared_node_query_parallel",
      cxxopts::value<std::string>())(
      "pull_strategy", "Pull strategy: bitmap or ge_spmm",
      cxxopts::value<std::string>())(
      "pull_frontier_ratio",
      "Hybrid pull threshold as active (query,vertex) frontier ratio",
      cxxopts::value<double>())(
      "pull_edge_ratio", "Hybrid pull threshold as active frontier-edge ratio",
      cxxopts::value<double>())("profile_levels",
                                "Record per-level CUDA event timings");

  auto parsed = options.parse(argc, argv);
  if (parsed.count("help") || parsed.count("market") == 0) {
    std::cout << options.help({""}) << std::endl;
    std::exit(0);
  }

  cgp_arguments_t args;
  args.filename = parsed["market"].as<std::string>();
  auto ext = std::filesystem::path(args.filename).extension().string();
  if (ext == ".gr" || ext == ".ggr") {
    args.binary = false;
  } else if (util::is_binary_csr(args.filename)) {
    args.binary = true;
  } else if (!util::is_market(args.filename)) {
    std::cout << options.help({""}) << std::endl;
    std::exit(1);
  }

  if (parsed.count("src")) {
    args.source_string = parsed["src"].as<std::string>();
  }
  if (parsed.count("query_file")) {
    args.query_file = parsed["query_file"].as<std::string>();
  }
  if (parsed.count("num_runs")) {
    args.num_runs = parsed["num_runs"].as<int>();
  }
  if (parsed.count("validate")) {
    args.validate = true;
  }
  if (parsed.count("json_dir")) {
    args.json_dir = parsed["json_dir"].as<std::string>();
  }
  if (parsed.count("json_file")) {
    args.json_file = parsed["json_file"].as<std::string>();
  }
  if (parsed.count("traversal_mode")) {
    args.traversal_mode = parsed["traversal_mode"].as<std::string>();
    if (args.traversal_mode != "push" && args.traversal_mode != "pull" &&
        args.traversal_mode != "hybrid") {
      std::cerr << "Error: --traversal_mode must be push, pull, or hybrid\n";
      std::exit(1);
    }
  }
  if (parsed.count("push_strategy")) {
    args.push_strategy = parsed["push_strategy"].as<std::string>();
    if (args.push_strategy != "edge_balanced" &&
        args.push_strategy != "shared_node" &&
        args.push_strategy != "shared_node_query_parallel") {
      std::cerr
          << "Error: --push_strategy must be edge_balanced, shared_node, or "
             "shared_node_query_parallel\n";
      std::exit(1);
    }
  }
  if (parsed.count("pull_strategy")) {
    args.pull_strategy = parsed["pull_strategy"].as<std::string>();
    if (args.pull_strategy != "bitmap" && args.pull_strategy != "ge_spmm") {
      std::cerr << "Error: --pull_strategy must be bitmap or ge_spmm\n";
      std::exit(1);
    }
  }
  if (parsed.count("pull_frontier_ratio")) {
    args.pull_frontier_ratio = parsed["pull_frontier_ratio"].as<double>();
  }
  if (parsed.count("pull_edge_ratio")) {
    args.pull_edge_ratio = parsed["pull_edge_ratio"].as<double>();
  }
  if (parsed.count("profile_levels")) {
    args.profile_levels = true;
  }
  return args;
}

template <typename T>
void read_exact(std::ifstream& input,
                T* destination,
                std::size_t count,
                const std::string& field) {
  input.read(reinterpret_cast<char*>(destination),
             static_cast<std::streamsize>(sizeof(T) * count));
  if (!input) {
    std::cerr << "Error: Failed reading " << field << "\n";
    std::exit(1);
  }
}

template <typename vertex_t, typename edge_t>
loaded_gr_t<vertex_t, edge_t> load_gr(const std::string& filename) {
  std::ifstream input(filename, std::ios::binary);
  if (!input) {
    std::cerr << "Error: Could not open graph " << filename << "\n";
    std::exit(1);
  }

  uint64_t version = 0;
  uint64_t size_edge_ty = 0;
  uint64_t n_vertices64 = 0;
  uint64_t n_edges64 = 0;
  read_exact(input, &version, 1, "GGR version");
  read_exact(input, &size_edge_ty, 1, "GGR edge type size");
  read_exact(input, &n_vertices64, 1, "GGR vertex count");
  read_exact(input, &n_edges64, 1, "GGR edge count");

  if (version != 1) {
    std::cerr << "Error: Unsupported GGR version " << version << "\n";
    std::exit(1);
  }
  if (n_vertices64 >
          static_cast<uint64_t>(std::numeric_limits<vertex_t>::max()) ||
      n_edges64 > static_cast<uint64_t>(std::numeric_limits<edge_t>::max())) {
    std::cerr << "Error: GGR graph exceeds int vertex/edge limits\n";
    std::exit(1);
  }

  loaded_gr_t<vertex_t, edge_t> graph;
  graph.n_vertices = static_cast<vertex_t>(n_vertices64);
  graph.n_edges = static_cast<edge_t>(n_edges64);

  std::vector<int64_t> row_end(static_cast<std::size_t>(graph.n_vertices));
  read_exact(input, row_end.data(), row_end.size(), "GGR row offsets");

  thrust::host_vector<edge_t> h_row_offsets(
      static_cast<std::size_t>(graph.n_vertices) + 1);
  h_row_offsets[0] = 0;
  for (vertex_t v = 0; v < graph.n_vertices; ++v) {
    if (row_end[v] < 0 ||
        row_end[v] > static_cast<int64_t>(std::numeric_limits<edge_t>::max())) {
      std::cerr << "Error: GGR row offset out of range at vertex " << v << "\n";
      std::exit(1);
    }
    h_row_offsets[static_cast<std::size_t>(v) + 1] =
        static_cast<edge_t>(row_end[v]);
  }
  if (h_row_offsets.back() != graph.n_edges) {
    std::cerr << "Error: GGR final row offset does not match edge count\n";
    std::exit(1);
  }

  thrust::host_vector<vertex_t> h_column_indices(graph.n_edges);
  read_exact(input, h_column_indices.data(), h_column_indices.size(),
             "GGR destinations");

  graph.row_offsets = h_row_offsets;
  graph.column_indices = h_column_indices;
  std::cout << "Loaded GGR graph with edge value size " << size_edge_ty
            << " bytes\n";
  return graph;
}

template <typename vertex_t>
std::vector<vertex_t> parse_query_file(const std::string& query_file,
                                       vertex_t n_vertices) {
  std::ifstream input(query_file);
  if (!input) {
    std::cerr << "Error: Could not open query_file " << query_file << "\n";
    std::exit(1);
  }

  std::vector<vertex_t> sources;
  std::string line;
  int line_no = 0;
  while (std::getline(input, line)) {
    ++line_no;
    line = trim(line);
    if (line.empty()) {
      continue;
    }

    long long source = -1;
    try {
      source = std::stoll(line);
    } catch (...) {
      std::cerr << "Error: Invalid source in query_file at line " << line_no
                << "\n";
      std::exit(1);
    }

    if (source < 0 || source >= n_vertices) {
      std::cerr << "Error: Source out of range in query_file at line "
                << line_no << "\n";
      std::exit(1);
    }
    sources.push_back(static_cast<vertex_t>(source));
  }
  return sources;
}

template <typename vertex_t>
std::vector<vertex_t> parse_sources(const cgp_arguments_t& args,
                                    vertex_t n_vertices) {
  if (!args.query_file.empty()) {
    auto sources = parse_query_file<vertex_t>(args.query_file, n_vertices);
    if (sources.empty()) {
      std::cerr << "Error: query_file did not contain any sources\n";
      std::exit(1);
    }
    return sources;
  }

  std::vector<int> int_sources;
  gunrock::io::cli::parse_source_string(args.source_string, &int_sources,
                                        n_vertices, args.num_runs);

  std::vector<vertex_t> sources;
  sources.reserve(int_sources.size());
  for (auto source : int_sources) {
    if (source < 0 || source >= n_vertices) {
      std::cerr << "Error: Invalid source\n";
      std::exit(1);
    }
    sources.push_back(static_cast<vertex_t>(source));
  }
  return sources;
}

std::string command_line(int argc, char** argv) {
  std::string command;
  for (int i = 0; i < argc; ++i) {
    command += argv[i];
    command += " ";
  }
  if (!command.empty()) {
    command.pop_back();
  }
  return command;
}

gunrock::cgp_sssp::traversal_mode_t parse_traversal_mode(
    const std::string& mode) {
  if (mode == "push") {
    return gunrock::cgp_sssp::traversal_mode_t::push;
  }
  if (mode == "pull") {
    return gunrock::cgp_sssp::traversal_mode_t::pull;
  }
  return gunrock::cgp_sssp::traversal_mode_t::hybrid;
}

gunrock::cgp_sssp::push_strategy_t parse_push_strategy(
    const std::string& strategy) {
  if (strategy == "shared_node") {
    return gunrock::cgp_sssp::push_strategy_t::shared_node;
  }
  if (strategy == "shared_node_query_parallel") {
    return gunrock::cgp_sssp::push_strategy_t::shared_node_query_parallel;
  }
  return gunrock::cgp_sssp::push_strategy_t::edge_balanced;
}

gunrock::cgp_sssp::pull_strategy_t parse_pull_strategy(
    const std::string& strategy) {
  if (strategy == "ge_spmm") {
    return gunrock::cgp_sssp::pull_strategy_t::ge_spmm;
  }
  return gunrock::cgp_sssp::pull_strategy_t::bitmap;
}

template <typename vertex_t, typename result_t>
void export_json(const cgp_arguments_t& args,
                 int argc,
                 char** argv,
                 std::size_t n_vertices,
                 std::size_t n_edges,
                 const std::vector<vertex_t>& sources,
                 const result_t& result) {
  nlohmann::json query_results = nlohmann::json::array();
  for (const auto& query : result.queries) {
    query_results.push_back({
        {"query_id", query.query_id},
        {"source", query.source},
        {"completion_level", query.completion_level},
        {"completion_wall_time_ms", query.completion_wall_time_ms},
    });
  }

  nlohmann::json jsn;
  jsn["engine"] = "Essentials";
  jsn["primitive"] = "cgp_sssp";
  jsn["graph_type"] = "market";
  jsn["graph_file"] = args.filename;
  jsn["num_vertices"] = n_vertices;
  jsn["num_edges"] = n_edges;
  jsn["srcs"] = sources;
  jsn["query_results"] = query_results;
  jsn["frontier_sizes"] = result.frontier_sizes;
  jsn["unique_frontier_sizes"] = result.unique_frontier_sizes;
  jsn["level_edge_counts"] = result.level_edge_counts;
  jsn["actual_level_edge_counts"] = result.actual_level_edge_counts;
  jsn["virtual_level_edge_counts"] = result.virtual_level_edge_counts;
  jsn["level_modes"] = result.level_modes;
  jsn["level_wall_times_ms"] = result.level_wall_times_ms;
  jsn["iterations"] = result.iterations;
  jsn["gpu_time_ms"] = result.gpu_time_ms;
  jsn["wall_time_ms"] = result.wall_time_ms;
  jsn["shared_push_kernel_ms"] = result.shared_push_kernel_ms;
  jsn["traversal_mode"] = args.traversal_mode;
  jsn["push_strategy"] = args.push_strategy;
  jsn["pull_strategy"] = args.pull_strategy;
  jsn["effective_query_dim"] = result.effective_query_dim;
  jsn["pull_frontier_ratio"] = args.pull_frontier_ratio;
  jsn["pull_edge_ratio"] = args.pull_edge_ratio;
  jsn["profile_levels"] = args.profile_levels;
  jsn["pull_frontier_threshold"] = args.pull_frontier_ratio *
                                   static_cast<double>(sources.size()) *
                                   static_cast<double>(n_vertices);
  jsn["pull_edge_threshold"] = args.pull_edge_ratio *
                               static_cast<double>(sources.size()) *
                               static_cast<double>(n_edges);
  nlohmann::json level_profiles = nlohmann::json::array();
  for (const auto& profile : result.level_profiles) {
    level_profiles.push_back({
        {"level", profile.level},
        {"mode", profile.mode},
        {"frontier_size", profile.frontier_size},
        {"edge_count", profile.edge_count},
        {"unique_frontier_size", profile.unique_frontier_size},
        {"actual_edge_count", profile.actual_edge_count},
        {"virtual_edge_count", profile.virtual_edge_count},
        {"pull_frontier_threshold", profile.pull_frontier_threshold},
        {"pull_edge_threshold", profile.pull_edge_threshold},
        {"level_wall_ms", profile.level_wall_ms},
        {"degree_scan_ms", profile.degree_scan_ms},
        {"push_kernel_ms", profile.push_kernel_ms},
        {"shared_push_kernel_ms", profile.shared_push_kernel_ms},
        {"bitmap_build_ms", profile.bitmap_build_ms},
        {"pull_kernel_ms", profile.pull_kernel_ms},
        {"ge_spmm_pull_kernel_ms", profile.ge_spmm_pull_kernel_ms},
        {"dense_build_ms", profile.dense_build_ms},
        {"dense_compact_ms", profile.dense_compact_ms},
        {"ge_spmm_postprocess_ms", profile.ge_spmm_postprocess_ms},
        {"compact_ms", profile.compact_ms},
        {"count_sync_ms", profile.count_sync_ms},
    });
  }
  jsn["level_profiles"] = level_profiles;
  jsn["optimization"] =
      "edge_balanced_expand, block_local_output_count, "
      "relaxation_frontier, hybrid_push_pull, shared_node_push, "
      "min_plus_ge_spmm_pull, shared_node_query_parallel_push";
  jsn["command_line"] = command_line(argc, argv);
  jsn["git_commit_sha"] = gunrock::io::git_commit_sha1();
  gunrock::util::stats::get_gpu_info(&jsn);
  gunrock::util::stats::system_info_t sys;
  sys.get_sys_info(&jsn);

  std::string output = args.json_file;
  if (output.empty()) {
    auto graph = std::filesystem::path(args.filename).filename().string();
    auto last_dot = graph.find_last_of('.');
    if (last_dot != std::string::npos) {
      graph = graph.substr(0, last_dot);
    }
    output = "cgp_sssp_" + graph + ".json";
  }

  std::filesystem::create_directories(args.json_dir);
  std::ofstream file(args.json_dir + "/" + output);
  file << jsn.dump(4);
}

template <typename graph_t, typename csr_t>
void run_loaded_graph(graph_t& G,
                      csr_t* validation_csr,
                      const cgp_arguments_t& args,
                      int argc,
                      char** argv) {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  auto context = std::make_shared<gcuda::multi_context_t>(0);
  auto n_vertices = static_cast<vertex_t>(G.get_number_of_vertices());
  auto sources = parse_sources<vertex_t>(args, n_vertices);
  if ((args.push_strategy == "shared_node" ||
       args.push_strategy == "shared_node_query_parallel") &&
      sources.size() > 32) {
    std::cerr
        << "Error: shared-node push strategies support at most 32 queries\n";
    std::exit(1);
  }
  if ((args.traversal_mode == "pull" || args.traversal_mode == "hybrid") &&
      sources.size() > 32) {
    std::cerr
        << "Error: pull and hybrid traversal modes support at most 32 queries\n";
    std::exit(1);
  }
  gunrock::cgp_sssp::options_t run_options;
  run_options.traversal_mode = parse_traversal_mode(args.traversal_mode);
  run_options.push_strategy = parse_push_strategy(args.push_strategy);
  run_options.pull_strategy = parse_pull_strategy(args.pull_strategy);
  run_options.pull_frontier_ratio = args.pull_frontier_ratio;
  run_options.pull_edge_ratio = args.pull_edge_ratio;
  run_options.profile_levels = args.profile_levels;
  auto result = gunrock::cgp_sssp::run(G, sources, context, run_options);

  std::cout << "Primitive : cgp_sssp\n";
  std::cout << "Sources : ";
  for (std::size_t i = 0; i < sources.size(); ++i) {
    std::cout << sources[i] << (i + 1 == sources.size() ? "\n" : ",");
  }
  std::cout << "Wall Time : " << result.wall_time_ms << " (ms)\n";
  std::cout << "GPU Time : " << result.gpu_time_ms << " (ms)\n";
  std::cout << "Iterations : " << result.iterations << "\n";
  std::cout << "Traversal Mode : " << args.traversal_mode << "\n";
  std::cout << "Push Strategy : " << args.push_strategy << "\n";
  std::cout << "Pull Strategy : " << args.pull_strategy << "\n";
  std::cout << "Effective Query Dim : " << result.effective_query_dim << "\n";
  std::cout << "Frontier Sizes : ";
  for (std::size_t i = 0; i < result.frontier_sizes.size(); ++i) {
    std::cout << result.frontier_sizes[i]
              << (i + 1 == result.frontier_sizes.size() ? "\n" : ",");
  }
  std::cout << "Unique Frontier Sizes : ";
  for (std::size_t i = 0; i < result.unique_frontier_sizes.size(); ++i) {
    std::cout << result.unique_frontier_sizes[i]
              << (i + 1 == result.unique_frontier_sizes.size() ? "\n" : ",");
  }
  std::cout << "Level Modes : ";
  for (std::size_t i = 0; i < result.level_modes.size(); ++i) {
    std::cout << result.level_modes[i]
              << (i + 1 == result.level_modes.size() ? "\n" : ",");
  }

  for (const auto& query : result.queries) {
    std::cout << "Query " << query.query_id << " Source " << query.source
              << " Completion Level " << query.completion_level
              << " Completion Wall Time " << query.completion_wall_time_ms
              << " (ms)\n";
  }

  export_json(args, argc, argv, G.get_number_of_vertices(),
              G.get_number_of_edges(), sources, result);

  if (args.validate) {
    if (validation_csr == nullptr) {
      std::cerr << "Error: --validate is only supported for Matrix Market/CSR "
                   "inputs in this driver\n";
      std::exit(1);
    }

    thrust::host_vector<weight_t> h_gpu_distances(result.distances);
    thrust::host_vector<weight_t> h_cpu_distances(n_vertices);
    thrust::host_vector<vertex_t> h_predecessors(n_vertices);

    auto last_query = static_cast<vertex_t>(sources.size() - 1);
    auto last_source = sources.back();
    float cpu_elapsed = sssp_cpu::run<csr_t, vertex_t, edge_t, weight_t>(
        *validation_csr, last_source, h_cpu_distances.data(),
        h_predecessors.data());

    int errors = 0;
    auto query_offset = static_cast<std::size_t>(last_query) * n_vertices;
    for (vertex_t v = 0; v < n_vertices; ++v) {
      auto gpu = h_gpu_distances[query_offset + v];
      auto cpu = h_cpu_distances[v];
      auto inf = std::numeric_limits<weight_t>::max();
      if ((gpu == inf || cpu == inf) ? gpu != cpu
                                     : std::abs(gpu - cpu) > 1e-4f) {
        ++errors;
      }
    }

    print::head(h_cpu_distances, 40, "CPU Distances");
    std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)\n";
    std::cout << "Number of errors : " << errors << "\n";
  }
}

}  // namespace

void test_cgp_sssp(int argc, char** argv) {
  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;
  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  auto args = parse_arguments(argc, argv);
  auto ext = std::filesystem::path(args.filename).extension().string();
  if (ext == ".gr" || ext == ".ggr") {
    auto loaded = load_gr<vertex_t, edge_t>(args.filename);
    auto G = loaded.view();
    run_loaded_graph<decltype(G), csr_t>(G, nullptr, args, argc, argv);
    return;
  }

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto [properties, coo] = mm.load(args.filename);

  csr_t csr;
  if (args.binary) {
    csr.read_binary(args.filename);
  } else {
    csr.from_coo(coo);
  }

  auto G = graph::build<memory_space_t::device>(properties, csr);
  run_loaded_graph<decltype(G), csr_t>(G, &csr, args, argc, argv);
}

int main(int argc, char** argv) {
  test_cgp_sssp(argc, argv);
}
