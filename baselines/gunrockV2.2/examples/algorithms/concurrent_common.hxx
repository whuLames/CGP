#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <cxxopts.hpp>

#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/io/parameters.hxx>

namespace gunrock_concurrent {

struct arguments_t {
  std::string filename;
  std::string query_file;
  std::string json_dir = ".";
  std::string json_file;
  int num_streams = 1;
  int num_queries = 256;
  int max_iterations = 10;
  float alpha = 0.85f;
  float tolerance = 1e-6f;
  bool signatures = false;
  gunrock::operators::load_balance_t advance_load_balance =
      gunrock::operators::load_balance_t::block_mapped;
};

struct signature_t {
  std::size_t finite_count = 0;
  double sum = 0.0;
  double weighted_sum = 0.0;
};

struct query_result_t {
  int query_id = 0;
  int source = -1;
  float gpu_ms = 0.0f;
  signature_t signature;
};

struct batch_result_t {
  int batch_id = 0;
  float wall_ms = 0.0f;
  std::vector<query_result_t> queries;
};

inline arguments_t parse_arguments(int argc,
                                   char** argv,
                                   const std::string& description,
                                   bool requires_sources) {
  cxxopts::Options options(argv[0], description);
  options.add_options()
      ("help", "Print help")
      ("m,market", "Matrix, binary CSR, or GR file",
       cxxopts::value<std::string>())
      ("query_file", "Query file, one source per line",
       cxxopts::value<std::string>())
      ("num_streams", "Maximum concurrent queries",
       cxxopts::value<int>()->default_value("1"))
      ("num_queries", "Total number of queries",
       cxxopts::value<int>()->default_value("256"))
      ("max_iterations", "Maximum iterations",
       cxxopts::value<int>()->default_value("10"))
      ("alpha", "PageRank damping factor",
       cxxopts::value<float>()->default_value("0.85"))
      ("tolerance", "PageRank tolerance",
       cxxopts::value<float>()->default_value("1e-6"))
      ("advance_load_balance", "SSSP advance load-balancing policy",
       cxxopts::value<std::string>())
      ("d,json_dir", "JSON output directory",
       cxxopts::value<std::string>()->default_value("."))
      ("f,json_file", "JSON output file", cxxopts::value<std::string>())
      ("signatures", "Export output signatures for validation");

  auto parsed = options.parse(argc, argv);
  if (parsed.count("help") || parsed.count("market") == 0 ||
      (requires_sources && parsed.count("query_file") == 0)) {
    std::cout << options.help({""}) << std::endl;
    std::exit(parsed.count("help") ? 0 : 2);
  }

  arguments_t args;
  args.filename = parsed["market"].as<std::string>();
  if (parsed.count("query_file")) {
    args.query_file = parsed["query_file"].as<std::string>();
  }
  if (parsed.count("json_file")) {
    args.json_file = parsed["json_file"].as<std::string>();
  }
  args.json_dir = parsed["json_dir"].as<std::string>();
  args.num_streams = std::max(1, parsed["num_streams"].as<int>());
  args.num_queries = std::max(1, parsed["num_queries"].as<int>());
  args.max_iterations = std::max(1, parsed["max_iterations"].as<int>());
  args.alpha = parsed["alpha"].as<float>();
  args.tolerance = parsed["tolerance"].as<float>();
  args.signatures = parsed.count("signatures") == 1;
  if (parsed.count("advance_load_balance")) {
    args.advance_load_balance = gunrock::io::cli::parse_load_balance(
        parsed["advance_load_balance"].as<std::string>());
  }
  return args;
}

inline std::vector<int> read_sources(const std::string& filename,
                                     int expected_count,
                                     int vertex_count) {
  std::ifstream input(filename);
  if (!input) {
    throw std::runtime_error("cannot open query file: " + filename);
  }

  std::vector<int> sources;
  int source = 0;
  while (input >> source) {
    if (source < 0 || source >= vertex_count) {
      throw std::runtime_error("query source is outside the graph");
    }
    sources.push_back(source);
  }
  if (static_cast<int>(sources.size()) != expected_count) {
    std::ostringstream message;
    message << "expected " << expected_count << " sources, found "
            << sources.size();
    throw std::runtime_error(message.str());
  }
  return sources;
}

inline std::string graph_kind(const std::string& filename) {
  if (gunrock::util::is_gr(filename)) return "gr";
  if (gunrock::util::is_binary_csr(filename)) return "csr";
  return "market";
}

template <typename value_t>
signature_t make_signature(const thrust::device_vector<value_t>& values) {
  thrust::host_vector<value_t> host(values);
  signature_t signature;
  for (std::size_t i = 0; i < host.size(); ++i) {
    const double value = static_cast<double>(host[i]);
    if (!std::isfinite(value) ||
        value >= static_cast<double>(std::numeric_limits<value_t>::max()) / 2.0) {
      continue;
    }
    ++signature.finite_count;
    signature.sum += value;
    signature.weighted_sum += value * static_cast<double>((i % 104729) + 1);
  }
  return signature;
}

template <typename value_t, typename run_query_t>
std::vector<batch_result_t> run_batched(
    const arguments_t& args,
    const std::vector<int>& sources,
    std::size_t vertex_count,
    run_query_t run_query,
    float* total_wall_ms) {
  using clock_t = std::chrono::steady_clock;
  const std::size_t batch_limit =
      std::min<std::size_t>(args.num_streams, sources.size());
  std::vector<batch_result_t> batches;
  const auto total_start = clock_t::now();

  for (std::size_t begin = 0, batch_id = 0; begin < sources.size();
       begin += batch_limit, ++batch_id) {
    const std::size_t end = std::min(begin + batch_limit, sources.size());
    const std::size_t count = end - begin;
    std::vector<hipStream_t> streams(count, nullptr);
    std::vector<std::shared_ptr<gunrock::gcuda::multi_context_t>> contexts;
    std::vector<std::unique_ptr<thrust::device_vector<value_t>>> outputs;
    std::vector<query_result_t> query_results(count);
    std::vector<std::thread> workers;
    std::mutex error_mutex;
    std::exception_ptr worker_error;

    contexts.reserve(count);
    outputs.reserve(count);
    workers.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      gunrock::error::throw_if_exception(
          hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking));
      contexts.push_back(
          std::make_shared<gunrock::gcuda::multi_context_t>(0, streams[i]));
      outputs.push_back(
          std::make_unique<thrust::device_vector<value_t>>(vertex_count));
    }

    const auto batch_start = clock_t::now();
    for (std::size_t i = 0; i < count; ++i) {
      const int query_id = static_cast<int>(begin + i);
      workers.emplace_back([&, i, query_id]() {
        try {
          const int source = sources[query_id];
          const float gpu_ms = run_query(
              query_id, source, outputs[i]->data().get(), contexts[i]);
          query_results[i] = {query_id, source, gpu_ms, {}};
        } catch (...) {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (!worker_error) worker_error = std::current_exception();
        }
      });
    }
    for (auto& worker : workers) worker.join();
    if (worker_error) std::rethrow_exception(worker_error);

    const auto batch_stop = clock_t::now();
    if (args.signatures) {
      for (std::size_t i = 0; i < count; ++i) {
        query_results[i].signature = make_signature(*outputs[i]);
      }
    }

    batch_result_t batch;
    batch.batch_id = static_cast<int>(batch_id);
    batch.wall_ms =
        std::chrono::duration<float, std::milli>(batch_stop - batch_start)
            .count();
    batch.queries = std::move(query_results);
    batches.push_back(std::move(batch));

    contexts.clear();
    outputs.clear();
    for (auto stream : streams) {
      if (stream) hipStreamDestroy(stream);
    }
  }

  const auto total_stop = clock_t::now();
  *total_wall_ms =
      std::chrono::duration<float, std::milli>(total_stop - total_start).count();
  return batches;
}

inline void write_json(const arguments_t& args,
                       const std::string& algorithm,
                       const std::string& kind,
                       std::size_t vertex_count,
                       std::size_t edge_count,
                       float total_wall_ms,
                       const std::vector<batch_result_t>& batches) {
  std::filesystem::create_directories(args.json_dir);
  const auto filename =
      args.json_file.empty() ? algorithm + "_concurrent.json" : args.json_file;
  std::ofstream output(std::filesystem::path(args.json_dir) / filename);
  if (!output) throw std::runtime_error("cannot create JSON output");

  output << std::fixed << std::setprecision(8);
  output << "{\n";
  output << "  \"algorithm\": \"" << algorithm << "\",\n";
  output << "  \"graph\": \"" << args.filename << "\",\n";
  output << "  \"graph_type\": \"" << kind << "\",\n";
  output << "  \"num_vertices\": " << vertex_count << ",\n";
  output << "  \"num_edges\": " << edge_count << ",\n";
  output << "  \"submitted_queries\": " << args.num_queries << ",\n";
  output << "  \"requested_q\": " << args.num_streams << ",\n";
  output << "  \"effective_batch_size\": "
         << std::min<std::size_t>(args.num_streams, args.num_queries) << ",\n";
  output << "  \"max_iterations\": " << args.max_iterations << ",\n";
  output << "  \"total_wall_ms\": " << total_wall_ms << ",\n";
  output << "  \"batches\": [\n";
  for (std::size_t i = 0; i < batches.size(); ++i) {
    const auto& batch = batches[i];
    output << "    {\"batch_id\": " << batch.batch_id
           << ", \"wall_ms\": " << batch.wall_ms << ", \"queries\": [\n";
    for (std::size_t j = 0; j < batch.queries.size(); ++j) {
      const auto& query = batch.queries[j];
      output << "      {\"query_id\": " << query.query_id
             << ", \"source\": " << query.source
             << ", \"gpu_ms\": " << query.gpu_ms;
      if (args.signatures) {
        output << ", \"finite_count\": " << query.signature.finite_count
               << ", \"sum\": " << query.signature.sum
               << ", \"weighted_sum\": " << query.signature.weighted_sum;
      }
      output << "}" << (j + 1 == batch.queries.size() ? "\n" : ",\n");
    }
    output << "    ]}" << (i + 1 == batches.size() ? "\n" : ",\n");
  }
  output << "  ]\n";
  output << "}\n";
}

}  // namespace gunrock_concurrent
