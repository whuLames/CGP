#include <algorithm>
#include <cstddef>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <cxxopts.hpp>

#include <gunrock/algorithms/bfs.hxx>
#include <gunrock/io/gr.hxx>
#include <gunrock/io/parameters.hxx>
#include <gunrock/util/compare.hxx>
#include <gunrock/util/print.hxx>

#include "bfs_cpu.hxx"

using namespace gunrock;
using namespace memory;

namespace {

struct concurrent_arguments_t {
  std::string filename;
  std::string source_string;
  std::string query_file;
  std::string json_dir = ".";
  std::string json_file;
  int num_streams = 1;
  bool num_streams_provided = false;
  float memory_budget_percent = 75.0f;
  std::size_t memory_reserve_mb = 512;
  bool validate = false;
  bool export_metrics = false;
  operators::load_balance_t advance_load_balance =
      operators::load_balance_t::block_mapped;
  operators::filter_algorithm_t filter_algorithm =
      operators::filter_algorithm_t::compact;
  bool enable_filter = false;
  bool enable_uniquify = false;
  operators::uniquify_algorithm_t uniquify_algorithm =
      operators::uniquify_algorithm_t::unique;
  bool best_effort_uniquify = true;
  float uniquify_percent = 100.0f;
};

struct query_result_t {
  int query_id = 0;
  int source = 0;
  float gpu_ms = 0.0f;
};

struct batch_result_t {
  int batch_id = 0;
  float wall_ms = 0.0f;
  std::vector<query_result_t> queries;
};

struct memory_estimate_t {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  std::size_t reserve_bytes = 0;
  std::size_t usable_bytes = 0;
  std::size_t per_query_bytes = 0;
  std::size_t estimated_max_concurrency = 1;
  std::size_t effective_batch_size = 1;
};

operators::uniquify_algorithm_t parse_uniquify_algorithm(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (str == "unique_copy") return operators::uniquify_algorithm_t::unique_copy;
  return operators::uniquify_algorithm_t::unique;
}

concurrent_arguments_t parse_arguments(int argc, char** argv) {
  cxxopts::Options options(argv[0], "Concurrent Breadth First Search example");
  options.add_options()
      ("help", "Print help")
      ("m,market", "Matrix, binary CSR, or GR file",
       cxxopts::value<std::string>())
      ("s,src", "Source(s), comma-separated", cxxopts::value<std::string>())
      ("query_file", "Query file, one source per line",
       cxxopts::value<std::string>())
      ("num_streams", "Maximum concurrent queries",
       cxxopts::value<int>())
      ("memory_budget_percent",
       "Percentage of current free GPU memory used for automatic batch sizing",
       cxxopts::value<float>()->default_value("75"))
      ("memory_reserve_mb",
       "GPU memory reserved outside automatic batch sizing",
       cxxopts::value<std::size_t>()->default_value("512"))
      ("d,json_dir", "JSON output directory",
       cxxopts::value<std::string>()->default_value("."))
      ("f,json_file", "JSON output file", cxxopts::value<std::string>())
      ("validate", "CPU validation for the last query")
      ("export_metrics", "Accepted for compatibility; ignored")
      ("advance_load_balance",
       "Load balancing technique for advance operator",
       cxxopts::value<std::string>())
      ("filter_algorithm", "Filter algorithm",
       cxxopts::value<std::string>())
      ("enable_filter", "Enable filter operator")
      ("enable_uniquify", "Enable uniquify operator")
      ("uniquify_algorithm", "Uniquify algorithm",
       cxxopts::value<std::string>())
      ("best_effort_uniquify", "Best-effort uniquification")
      ("uniquify_percent", "Percentage of elements to uniquify",
       cxxopts::value<float>());

  auto parsed = options.parse(argc, argv);
  if (parsed.count("help") || parsed.count("market") == 0) {
    std::cout << options.help({""}) << std::endl;
    std::exit(0);
  }

  concurrent_arguments_t args;
  args.filename = parsed["market"].as<std::string>();
  if (!gunrock::util::is_binary_csr(args.filename) &&
      !gunrock::util::is_gr(args.filename) &&
      !gunrock::util::is_market(args.filename)) {
    std::cout << options.help({""}) << std::endl;
    std::exit(0);
  }

  if (parsed.count("src")) {
    args.source_string = parsed["src"].as<std::string>();
  }
  if (parsed.count("query_file")) {
    args.query_file = parsed["query_file"].as<std::string>();
  }
  if (parsed.count("json_file")) {
    args.json_file = parsed["json_file"].as<std::string>();
  }
  args.json_dir = parsed["json_dir"].as<std::string>();
  args.num_streams_provided = parsed.count("num_streams") == 1;
  if (args.num_streams_provided) {
    args.num_streams = std::max(1, parsed["num_streams"].as<int>());
  }
  args.memory_budget_percent =
      std::max(1.0f, std::min(100.0f,
                              parsed["memory_budget_percent"].as<float>()));
  args.memory_reserve_mb = parsed["memory_reserve_mb"].as<std::size_t>();
  args.validate = parsed.count("validate") == 1;
  args.export_metrics = parsed.count("export_metrics") == 1;

  if (parsed.count("advance_load_balance")) {
    args.advance_load_balance = gunrock::io::cli::parse_load_balance(
        parsed["advance_load_balance"].as<std::string>());
  }
  if (parsed.count("filter_algorithm")) {
    args.filter_algorithm = gunrock::io::cli::parse_filter_algorithm(
        parsed["filter_algorithm"].as<std::string>());
  }
  args.enable_filter = parsed.count("enable_filter") == 1;
  args.enable_uniquify = parsed.count("enable_uniquify") == 1;
  if (parsed.count("uniquify_algorithm")) {
    args.uniquify_algorithm =
        parse_uniquify_algorithm(parsed["uniquify_algorithm"].as<std::string>());
  }
  if (parsed.count("best_effort_uniquify")) {
    args.best_effort_uniquify = true;
  }
  if (parsed.count("uniquify_percent")) {
    args.uniquify_percent = parsed["uniquify_percent"].as<float>();
  }

  return args;
}

std::vector<int> parse_source_list(const std::string& source_string) {
  std::vector<int> sources;
  std::stringstream ss(source_string);
  while (ss.good()) {
    std::string token;
    std::getline(ss, token, ',');
    if (token.empty()) continue;
    try {
      sources.push_back(std::stoi(token));
    } catch (...) {
      std::cerr << "Error: Invalid source: " << token << std::endl;
      std::exit(1);
    }
  }
  return sources;
}

std::vector<int> read_query_file(const std::string& filename) {
  std::ifstream input(filename);
  if (!input.is_open()) {
    std::cerr << "Error: Could not open query file: " << filename << std::endl;
    std::exit(1);
  }

  std::vector<int> sources;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    int source = 0;
    ss >> source;
    if (ss.fail()) {
      std::cerr << "Error: Invalid source in query file: " << line
                << std::endl;
      std::exit(1);
    }
    sources.push_back(source);
  }
  return sources;
}

void validate_sources(const std::vector<int>& sources, int n_vertices) {
  if (sources.empty()) {
    std::cerr << "Error: No query sources provided" << std::endl;
    std::exit(1);
  }
  for (auto source : sources) {
    if (source < 0 || source >= n_vertices) {
      std::cerr << "Error: Invalid source: " << source << std::endl;
      std::exit(1);
    }
  }
}

std::string graph_type(const std::string& filename) {
  if (gunrock::util::is_gr(filename)) return "gr";
  if (gunrock::util::is_binary_csr(filename)) return "csr";
  return "market";
}

std::string json_path(const concurrent_arguments_t& args) {
  if (!args.json_file.empty()) {
    return args.json_dir + "/" + args.json_file;
  }
  return args.json_dir + "/bfs_concurrent.json";
}

std::size_t checked_add(std::size_t a, std::size_t b) {
  const auto max_value = std::numeric_limits<std::size_t>::max();
  if (max_value - a < b) return max_value;
  return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b) {
  const auto max_value = std::numeric_limits<std::size_t>::max();
  if (a != 0 && b > max_value / a) return max_value;
  return a * b;
}

template <typename vertex_t, typename edge_t>
std::size_t estimate_bfs_query_bytes(std::size_t n_vertices,
                                     std::size_t n_edges) {
  std::size_t bytes = 0;

  // User-owned query output.
  bytes = checked_add(bytes, checked_mul(n_vertices, sizeof(vertex_t)));

  // Enactor scan workspace.
  bytes = checked_add(bytes, checked_mul(n_vertices + 1, sizeof(edge_t)));

  // BFS pre-reserves two |V| frontier buffers.
  bytes = checked_add(bytes, checked_mul(2 * n_vertices, sizeof(vertex_t)));

  // Advance may resize the output frontier to the current work-domain output.
  // For a conservative OOM-avoidance estimate, budget one |E| vertex buffer.
  bytes = checked_add(bytes, checked_mul(n_edges, sizeof(vertex_t)));

  // Runtime overhead from Thrust/CUB temporaries, events, stream state, and
  // allocator alignment. Keep this fixed padding small relative to large graphs.
  bytes = checked_add(bytes, 64ull * 1024ull * 1024ull);
  return bytes;
}

template <typename vertex_t, typename edge_t, typename graph_t>
memory_estimate_t estimate_bfs_concurrency(const graph_t& G,
                                           const concurrent_arguments_t& args,
                                           std::size_t submitted_queries) {
  memory_estimate_t estimate;
  error::error_t status = hipMemGetInfo(&estimate.free_bytes,
                                        &estimate.total_bytes);
  error::throw_if_exception(status);

  estimate.reserve_bytes = checked_mul(args.memory_reserve_mb,
                                       1024ull * 1024ull);
  const auto budgeted_bytes = static_cast<std::size_t>(
      static_cast<double>(estimate.free_bytes) *
      static_cast<double>(args.memory_budget_percent) / 100.0);
  estimate.usable_bytes =
      budgeted_bytes > estimate.reserve_bytes
          ? budgeted_bytes - estimate.reserve_bytes
          : 0;
  estimate.per_query_bytes =
      estimate_bfs_query_bytes<vertex_t, edge_t>(
          static_cast<std::size_t>(G.get_number_of_vertices()),
          static_cast<std::size_t>(G.get_number_of_edges()));

  if (estimate.per_query_bytes > 0 && estimate.usable_bytes > 0) {
    estimate.estimated_max_concurrency =
        std::max<std::size_t>(1, estimate.usable_bytes /
                                     estimate.per_query_bytes);
  }
  estimate.estimated_max_concurrency =
      std::max<std::size_t>(1, std::min(estimate.estimated_max_concurrency,
                                        submitted_queries));
  estimate.effective_batch_size =
      args.num_streams_provided
          ? static_cast<std::size_t>(args.num_streams)
          : estimate.estimated_max_concurrency;
  return estimate;
}

void write_json(const concurrent_arguments_t& args,
                const std::string& graph_kind,
                int n_vertices,
                int n_edges,
                float total_wall_ms,
                const memory_estimate_t& memory_estimate,
                std::size_t submitted_queries,
                const std::vector<batch_result_t>& batches) {
  if (args.json_file.empty() && args.json_dir == ".") return;

  std::ofstream out(json_path(args));
  if (!out.is_open()) {
    std::cerr << "Error: Could not open JSON output file" << std::endl;
    std::exit(1);
  }

  out << std::fixed << std::setprecision(4);
  out << "{\n";
  out << "  \"primitive\": \"bfs_concurrent\",\n";
  out << "  \"graph\": \"" << args.filename << "\",\n";
  out << "  \"graph_type\": \"" << graph_kind << "\",\n";
  out << "  \"num_vertices\": " << n_vertices << ",\n";
  out << "  \"num_edges\": " << n_edges << ",\n";
  out << "  \"num_streams\": " << args.num_streams << ",\n";
  out << "  \"submitted_queries\": " << submitted_queries << ",\n";
  out << "  \"user_specified_num_streams\": "
      << (args.num_streams_provided ? "true" : "false") << ",\n";
  out << "  \"estimated_max_concurrency\": "
      << memory_estimate.estimated_max_concurrency << ",\n";
  out << "  \"effective_batch_size\": "
      << memory_estimate.effective_batch_size << ",\n";
  out << "  \"memory_budget_percent\": "
      << args.memory_budget_percent << ",\n";
  out << "  \"memory_reserve_mb\": "
      << args.memory_reserve_mb << ",\n";
  out << "  \"free_memory_after_graph_bytes\": "
      << memory_estimate.free_bytes << ",\n";
  out << "  \"total_memory_bytes\": "
      << memory_estimate.total_bytes << ",\n";
  out << "  \"usable_memory_bytes\": "
      << memory_estimate.usable_bytes << ",\n";
  out << "  \"estimated_per_query_bytes\": "
      << memory_estimate.per_query_bytes << ",\n";
  out << "  \"total_wall_ms\": " << total_wall_ms << ",\n";
  out << "  \"batches\": [\n";
  for (std::size_t i = 0; i < batches.size(); ++i) {
    const auto& batch = batches[i];
    out << "    {\n";
    out << "      \"batch_id\": " << batch.batch_id << ",\n";
    out << "      \"wall_ms\": " << batch.wall_ms << ",\n";
    out << "      \"queries\": [\n";
    for (std::size_t j = 0; j < batch.queries.size(); ++j) {
      const auto& query = batch.queries[j];
      out << "        {\"query_id\": " << query.query_id
          << ", \"source\": " << query.source
          << ", \"gpu_ms\": " << query.gpu_ms << "}";
      out << (j + 1 == batch.queries.size() ? "\n" : ",\n");
    }
    out << "      ]\n";
    out << "    }" << (i + 1 == batches.size() ? "\n" : ",\n");
  }
  out << "  ]\n";
  out << "}\n";
}

template <typename graph_t, typename csr_t, typename vertex_t>
std::vector<batch_result_t> run_queries(graph_t& G,
                                        csr_t& csr,
                                        const concurrent_arguments_t& args,
                                        const std::vector<int>& sources,
                                        std::size_t batch_size_limit,
                                        float* total_wall_ms) {
  using clock_t = std::chrono::high_resolution_clock;
  const std::size_t n_vertices = G.get_number_of_vertices();
  std::vector<batch_result_t> batches;
  auto total_start = clock_t::now();
  batch_size_limit = std::max<std::size_t>(1, batch_size_limit);

  for (std::size_t batch_begin = 0, batch_id = 0; batch_begin < sources.size();
       batch_begin += batch_size_limit, ++batch_id) {
    const std::size_t batch_end =
        std::min(batch_begin + batch_size_limit, sources.size());
    const std::size_t batch_size = batch_end - batch_begin;

    {
      std::vector<hipStream_t> streams(batch_size, nullptr);
      std::vector<std::shared_ptr<gcuda::multi_context_t>> contexts;
      std::vector<std::unique_ptr<thrust::device_vector<vertex_t>>> distances;
      std::vector<query_result_t> query_results(batch_size);
      std::vector<std::thread> workers;

      contexts.reserve(batch_size);
      distances.reserve(batch_size);
      workers.reserve(batch_size);

      for (std::size_t i = 0; i < batch_size; ++i) {
        hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking);
        contexts.push_back(
            std::make_shared<gcuda::multi_context_t>(0, streams[i]));
        distances.push_back(
            std::make_unique<thrust::device_vector<vertex_t>>(n_vertices));
      }

      auto batch_start = clock_t::now();
      for (std::size_t i = 0; i < batch_size; ++i) {
        const int query_id = static_cast<int>(batch_begin + i);
        workers.emplace_back([&, i, query_id]() {
          vertex_t source = static_cast<vertex_t>(sources[query_id]);
          auto& distances_i = *distances[i];
          float gpu_ms = gunrock::bfs::run(
              G, source, distances_i.data().get(), nullptr, contexts[i],
              args.advance_load_balance);
          query_results[i] = {query_id, static_cast<int>(source), gpu_ms};
        });
      }

      for (auto& worker : workers) {
        worker.join();
      }
      auto batch_stop = clock_t::now();

      batch_result_t batch;
      batch.batch_id = static_cast<int>(batch_id);
      batch.wall_ms =
          std::chrono::duration<float, std::milli>(batch_stop - batch_start)
              .count();
      batch.queries = query_results;
      batches.push_back(batch);

      if (args.validate && batch_end == sources.size()) {
        thrust::host_vector<vertex_t> h_distances(n_vertices);
        thrust::host_vector<vertex_t> h_predecessors(n_vertices);
        vertex_t last_source = static_cast<vertex_t>(sources.back());
        float cpu_elapsed =
            bfs_cpu::run<csr_t, vertex_t, typename csr_t::offset_type>(
            csr, last_source, h_distances.data(), h_predecessors.data());
        int n_errors = util::compare(distances.back()->data().get(),
                                     h_distances.data(), n_vertices);
        print::head(h_distances, 40, "CPU Distances");
        std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)"
                  << std::endl;
        std::cout << "Number of errors : " << n_errors << std::endl;
      }

      contexts.clear();
      distances.clear();
      for (auto stream : streams) {
        hipStreamDestroy(stream);
      }
    }
  }

  auto total_stop = clock_t::now();
  *total_wall_ms =
      std::chrono::duration<float, std::milli>(total_stop - total_start)
          .count();
  return batches;
}

void test_bfs_concurrent(int argc, char** argv) {
  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;
  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  auto args = parse_arguments(argc, argv);
  gunrock::graph::graph_properties_t properties;
  csr_t csr;
  auto kind = graph_type(args.filename);

  if (kind == "gr") {
    auto [gr_properties, gr_csr] =
        io::load_gr<vertex_t, edge_t, weight_t>(args.filename);
    properties = gr_properties;
    csr = csr_t(gr_csr);
  } else if (kind == "csr") {
    csr.read_binary(args.filename);
    properties.directed = true;
    properties.symmetric = false;
    properties.weighted = true;
  } else {
    io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
    auto [mm_properties, coo] = mm.load(args.filename);
    properties = mm_properties;
    csr.from_coo(coo);
  }

  auto G = graph::build<memory_space_t::device>(properties, csr);
  auto sources = !args.query_file.empty() ? read_query_file(args.query_file)
                                          : parse_source_list(args.source_string);
  validate_sources(sources, G.get_number_of_vertices());
  auto memory_estimate =
      estimate_bfs_concurrency<vertex_t, edge_t>(G, args, sources.size());

  std::cout << "Graph type : " << kind << std::endl;
  std::cout << "Queries : " << sources.size() << std::endl;
  std::cout << "Num streams specified : "
            << (args.num_streams_provided ? "true" : "false") << std::endl;
  std::cout << "Requested Num streams : " << args.num_streams << std::endl;
  std::cout << "Free Memory After Graph : "
            << memory_estimate.free_bytes << " bytes" << std::endl;
  std::cout << "Estimated Per Query Memory : "
            << memory_estimate.per_query_bytes << " bytes" << std::endl;
  std::cout << "Estimated Max Concurrency : "
            << memory_estimate.estimated_max_concurrency << std::endl;
  std::cout << "Effective Batch Size : "
            << memory_estimate.effective_batch_size << std::endl;

  float total_wall_ms = 0.0f;
  auto batches =
      run_queries<decltype(G), csr_t, vertex_t>(G, csr, args, sources,
                                                memory_estimate.effective_batch_size,
                                                &total_wall_ms);

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Total Wall Time : " << total_wall_ms << " (ms)" << std::endl;
  for (const auto& batch : batches) {
    std::cout << "Batch " << batch.batch_id
              << " Wall Time : " << batch.wall_ms << " (ms)" << std::endl;
    for (const auto& query : batch.queries) {
      std::cout << "query_id=" << query.query_id << " source=" << query.source
                << " gpu_ms=" << query.gpu_ms << std::endl;
    }
  }

  write_json(args, kind, G.get_number_of_vertices(), G.get_number_of_edges(),
             total_wall_ms, memory_estimate, sources.size(), batches);
}

}  // namespace

int main(int argc, char** argv) {
  test_bfs_concurrent(argc, argv);
}
