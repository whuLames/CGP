#include <type_traits>

#include <puercgp/puercgp.hxx>

struct smoke_graph {
  using vertex_type = int;
  using edge_type = int;
  using weight_type = float;

  __host__ __device__ int get_number_of_vertices() const { return 4; }

  __host__ __device__ int get_number_of_edges() const { return 4; }

  __host__ __device__ int get_starting_edge(int vertex) const { return vertex; }

  __host__ __device__ int get_destination_vertex(int edge) const {
    return (edge + 1) % 4;
  }

  __host__ __device__ float get_edge_weight(int /*edge*/) const { return 1.0f; }
};

int main(int argc, char** argv) {
  (void)argv;

  smoke_graph graph;
  puercgp::query_batch<int> queries({0, 1, 2, 3});
  puercgp::algorithms::pagerank_query_batch rank_queries({
      {0.85f, 1.0e-8f},
  });

  puercgp::run_options options;
  options.traversal_mode = puercgp::traversal_mode_t::hybrid;
  options.push_strategy = puercgp::push_strategy_t::shared_node_warp;
  options.pull_strategy = puercgp::pull_strategy_t::fused;
  options.profile_iterations = true;

  puercgp::execution_context context;
  puercgp::graph_adapter<smoke_graph> adapter(graph);
  (void)adapter.get_number_of_vertices();

  using bfs_result_t = decltype(puercgp::run<puercgp::algorithms::bfs_policy>(
      graph, queries, context, options));
  using pagerank_result_t =
      decltype(puercgp::run<puercgp::algorithms::pagerank_policy>(
          graph, rank_queries, context, options));
  static_assert(std::is_same<typename puercgp::algorithms::bfs_policy::value_type,
                             int>::value,
                "BFS smoke policy should use int distances");

  if (argc < 0) {
    bfs_result_t result = puercgp::run<puercgp::algorithms::bfs_policy>(
        graph, queries, context, options);
    (void)result;
  }
  if (argc < -1) {
    pagerank_result_t result =
        puercgp::run<puercgp::algorithms::pagerank_policy>(
            graph, rank_queries, context, options);
    (void)result;
  }

  return static_cast<int>(queries.size()) == 4 ? 0 : 1;
}
