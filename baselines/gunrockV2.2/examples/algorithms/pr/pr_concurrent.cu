#include <gunrock/algorithms/pr.hxx>
#include <gunrock/io/gr.hxx>

#include "../concurrent_common.hxx"

using namespace gunrock;
using namespace memory;

int main(int argc, char** argv) {
  try {
    using vertex_t = int;
    using edge_t = int;
    using weight_t = float;
    using csr_t =
        format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

    const auto args = gunrock_concurrent::parse_arguments(
        argc, argv, "Concurrent PageRank", false);
    const auto kind = gunrock_concurrent::graph_kind(args.filename);
    graph::graph_properties_t properties;
    csr_t csr;
    if (kind == "gr") {
      auto [loaded_properties, loaded_csr] =
          io::load_gr<vertex_t, edge_t, weight_t>(args.filename);
      properties = loaded_properties;
      csr = csr_t(loaded_csr);
    } else if (kind == "csr") {
      csr.read_binary(args.filename);
      properties.directed = true;
      properties.symmetric = false;
      properties.weighted = true;
    } else {
      io::matrix_market_t<vertex_t, edge_t, weight_t> matrix;
      auto [loaded_properties, coo] = matrix.load(args.filename);
      properties = loaded_properties;
      csr.from_coo(coo);
    }

    auto graph = graph::build<memory_space_t::device>(properties, csr);
    std::vector<int> queries(args.num_queries, -1);
    auto run_query = [&](int,
                         int,
                         weight_t* output,
                         std::shared_ptr<gcuda::multi_context_t> context) {
      return pr::run(graph, args.alpha, args.tolerance, args.max_iterations,
                     output, context, nullptr);
    };

    float total_wall_ms = 0.0f;
    auto batches = gunrock_concurrent::run_batched<weight_t>(
        args, queries, graph.get_number_of_vertices(), run_query,
        &total_wall_ms);
    gunrock_concurrent::write_json(
        args, "pagerank", kind, graph.get_number_of_vertices(),
        graph.get_number_of_edges(), total_wall_ms, batches);
    std::cout << "Total Wall Time : " << total_wall_ms << " (ms)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Concurrent PageRank failed: " << error.what() << '\n';
    return 1;
  }
}
