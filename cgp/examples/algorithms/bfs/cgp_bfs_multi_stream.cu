#include <gunrock/algorithms/bfs.hxx>
#include <gunrock/framework/benchmark.hxx>
#include <gunrock/io/parameters.hxx>
#include <gunrock/util/performance.hxx>

#include "bfs_cpu.hxx"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
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

  __host__ __device__ __forceinline__ edge_t* get_row_offsets() const {
    return row_offsets;
  }

  __host__ __device__ __forceinline__ edge_t
  get_starting_edge(vertex_t const& v) const {
    return row_offsets[v];
  }

  __host__ __device__ __forceinline__ vertex_t
  get_destination_vertex(edge_t const& e) const {
    return column_indices[e];
  }

  __host__ __device__ __forceinline__ edge_t
  get_number_of_neighbors(vertex_t const& v) const {
    return get_starting_edge(v + 1) - get_starting_edge(v);
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

struct arguments_t {
  std::string filename;
  std::string source_string;
  std::string query_file;
  std::string json_dir = ".";
  std::string json_file;
  int num_runs = 1;
  bool validate = false;
  bool binary = false;
};

struct query_timing_t {
  int query_id = 0;
  int source = 0;
  float gpu_time_ms = 0;
};

struct baseline_result_t {
  thrust::device_vector<int> distances;
  std::vector<query_timing_t> queries;
  float wall_time_ms = 0;
  float summed_gpu_time_ms = 0;
};

std::string trim(std::string value) {
  auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

arguments_t parse_arguments(int argc, char** argv) {
  cxxopts::Options options(argv[0], "Gunrock BFS multi-stream baseline");
  options.add_options()("help", "Print help")("m,market", "Matrix file",
                                              cxxopts::value<std::string>())(
      "s,src", "Source(s); comma-separated string of ints",
      cxxopts::value<std::string>())("query_file",
                                     "Query source file, one source per line",
                                     cxxopts::value<std::string>())(
      "n,num_runs", "Number of repeated runs when a single --src is passed",
      cxxopts::value<int>())("validate", "CPU validation for last query")(
      "d,json_dir", "JSON output directory", cxxopts::value<std::string>())(
      "f,json_file", "JSON output file", cxxopts::value<std::string>());

  auto parsed = options.parse(argc, argv);
  if (parsed.count("help") || parsed.count("market") == 0) {
    std::cout << options.help({""}) << std::endl;
    std::exit(0);
  }

  arguments_t args;
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
std::vector<vertex_t> parse_sources(const arguments_t& args,
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

float elapsed_ms(std::chrono::high_resolution_clock::time_point start) {
  auto stop = std::chrono::high_resolution_clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
                .count();
  return static_cast<float>(us) / 1000.0f;
}

template <typename graph_t>
baseline_result_t run_multi_stream_baseline(
    graph_t& G,
    const std::vector<typename graph_t::vertex_type>& sources) {
  using vertex_t = typename graph_t::vertex_type;

  auto n_vertices = static_cast<std::size_t>(G.get_number_of_vertices());
  baseline_result_t result;
  result.distances.resize(n_vertices * sources.size());
  thrust::device_vector<vertex_t> predecessors(n_vertices * sources.size());

  std::vector<hipStream_t> streams(sources.size(), nullptr);
  std::vector<std::shared_ptr<gcuda::multi_context_t>> contexts;
  contexts.reserve(sources.size());

  for (std::size_t i = 0; i < sources.size(); ++i) {
    hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking);
    contexts.push_back(std::make_shared<gcuda::multi_context_t>(0, streams[i]));
  }

  auto wall_start = std::chrono::high_resolution_clock::now();
  for (std::size_t i = 0; i < sources.size(); ++i) {
    auto distance_ptr =
        thrust::raw_pointer_cast(result.distances.data()) + i * n_vertices;
    auto predecessor_ptr =
        thrust::raw_pointer_cast(predecessors.data()) + i * n_vertices;
    gunrock::options_t options;
    options.advance_load_balance =
        gunrock::operators::load_balance_t::merge_path;
    gunrock::bfs::param_t<vertex_t> param(sources[i], options);
    gunrock::bfs::result_t<vertex_t> bfs_result(distance_ptr, predecessor_ptr);
    float gpu_time = gunrock::bfs::run(G, param, bfs_result, contexts[i]);
    result.queries.push_back(
        {static_cast<int>(i), static_cast<int>(sources[i]), gpu_time});
    result.summed_gpu_time_ms += gpu_time;
  }

  for (auto stream : streams) {
    hipStreamSynchronize(stream);
  }
  result.wall_time_ms = elapsed_ms(wall_start);

  for (auto stream : streams) {
    hipStreamDestroy(stream);
  }
  return result;
}

template <typename vertex_t>
void export_json(const arguments_t& args,
                 int argc,
                 char** argv,
                 std::size_t n_vertices,
                 std::size_t n_edges,
                 const std::vector<vertex_t>& sources,
                 const baseline_result_t& result) {
  nlohmann::json query_results = nlohmann::json::array();
  for (const auto& query : result.queries) {
    query_results.push_back({
        {"query_id", query.query_id},
        {"source", query.source},
        {"gpu_time_ms", query.gpu_time_ms},
    });
  }

  nlohmann::json jsn;
  jsn["engine"] = "Essentials";
  jsn["primitive"] = "cgp_bfs_multi_stream_baseline";
  jsn["baseline"] = "gunrock_bfs_per_query_nonblocking_stream";
  jsn["note"] =
      "Gunrock BFS internals may synchronize their own context stream; wall "
      "time is measured around the batch launch loop.";
  jsn["graph_type"] = "market";
  jsn["graph_file"] = args.filename;
  jsn["num_vertices"] = n_vertices;
  jsn["num_edges"] = n_edges;
  jsn["srcs"] = sources;
  jsn["query_results"] = query_results;
  jsn["summed_gpu_time_ms"] = result.summed_gpu_time_ms;
  jsn["wall_time_ms"] = result.wall_time_ms;
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
    output = "cgp_bfs_multi_stream_" + graph + ".json";
  }

  std::filesystem::create_directories(args.json_dir);
  std::ofstream file(args.json_dir + "/" + output);
  file << jsn.dump(4);
}

template <typename graph_t, typename csr_t>
void run_loaded_graph(graph_t& G,
                      csr_t* validation_csr,
                      const arguments_t& args,
                      int argc,
                      char** argv) {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;

  auto n_vertices = static_cast<vertex_t>(G.get_number_of_vertices());
  auto sources = parse_sources<vertex_t>(args, n_vertices);
  auto result = run_multi_stream_baseline(G, sources);

  std::cout << "Primitive : cgp_bfs_multi_stream_baseline\n";
  std::cout << "Sources : ";
  for (std::size_t i = 0; i < sources.size(); ++i) {
    std::cout << sources[i] << (i + 1 == sources.size() ? "\n" : ",");
  }
  std::cout << "Wall Time : " << result.wall_time_ms << " (ms)\n";
  std::cout << "Summed GPU Time : " << result.summed_gpu_time_ms << " (ms)\n";
  for (const auto& query : result.queries) {
    std::cout << "Query " << query.query_id << " Source " << query.source
              << " GPU Time " << query.gpu_time_ms << " (ms)\n";
  }

  export_json(args, argc, argv, G.get_number_of_vertices(),
              G.get_number_of_edges(), sources, result);

  if (args.validate) {
    if (validation_csr == nullptr) {
      std::cerr << "Error: --validate is only supported for Matrix Market/CSR "
                   "inputs in this driver\n";
      std::exit(1);
    }

    thrust::host_vector<vertex_t> h_gpu_distances(result.distances);
    thrust::host_vector<vertex_t> h_cpu_distances(n_vertices);
    thrust::host_vector<vertex_t> h_predecessors(n_vertices);

    auto last_query = sources.size() - 1;
    auto last_source = sources.back();
    float cpu_elapsed = bfs_cpu::run<csr_t, vertex_t, edge_t>(
        *validation_csr, last_source, h_cpu_distances.data(),
        h_predecessors.data());

    int errors = 0;
    auto query_offset = last_query * static_cast<std::size_t>(n_vertices);
    for (vertex_t v = 0; v < n_vertices; ++v) {
      if (h_gpu_distances[query_offset + v] != h_cpu_distances[v]) {
        ++errors;
      }
    }

    print::head(h_cpu_distances, 40, "CPU Distances");
    std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)\n";
    std::cout << "Number of errors : " << errors << "\n";
  }
}

}  // namespace

void test_cgp_bfs_multi_stream(int argc, char** argv) {
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
  test_cgp_bfs_multi_stream(argc, argv);
}
