/**
 * @file bfs.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Breadth-First Search algorithm.
 * @date 2020-11-23
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/util/iteration_profiler.hxx>

namespace gunrock {
namespace bfs {

template <typename vertex_t>
struct param_t {
  vertex_t single_source;
  operators::load_balance_t advance_load_balance;
  util::iteration_profiler::csv_writer_t* iter_profile;
  param_t(vertex_t _single_source,
          operators::load_balance_t _advance_load_balance =
              operators::load_balance_t::block_mapped,
          util::iteration_profiler::csv_writer_t* _iter_profile = nullptr)
      : single_source(_single_source),
        advance_load_balance(_advance_load_balance),
        iter_profile(_iter_profile) {}
};

template <typename vertex_t>
struct result_t {
  vertex_t* distances;
  vertex_t* predecessors;  /// @todo: implement this.
  result_t(vertex_t* _distances, vertex_t* _predecessors)
      : distances(_distances), predecessors(_predecessors) {}
};

template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  param_type param;
  result_type result;

  problem_t(graph_t& G,
            param_type& _param,
            result_type& _result,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(G, _context),
        param(_param),
        result(_result) {}

  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  thrust::device_vector<vertex_t> visited;  /// @todo not used.

  void init() override {}

  void reset() override {
    auto n_vertices = this->get_graph().get_number_of_vertices();
    auto d_distances = thrust::device_pointer_cast(this->result.distances);
    auto context = this->get_single_context();
    thrust::fill(context->execution_policy(),
                 d_distances + 0, d_distances + n_vertices,
                 std::numeric_limits<vertex_t>::max());
    thrust::fill(context->execution_policy(),
                 d_distances + this->param.single_source,
                 d_distances + this->param.single_source + 1, 0);
  }
};

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  static gunrock::enactor_properties_t make_properties() {
    gunrock::enactor_properties_t properties;
    properties.frontier_sizing_factor = 1.0f;
    properties.self_manage_frontiers = true;
    return properties;
  }

  enactor_t(problem_t* _problem,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::enactor_t<problem_t>(_problem,
                                      _context,
                                      make_properties()) {
    // BFS frontier is a set of discovered vertices, so a single frontier
    // buffer only needs space for |V| items, not |E| candidates.
    auto n_vertices = this->get_problem()->get_graph().get_number_of_vertices();
    for (auto& buffer : this->frontiers) {
      buffer.set_resizing_factor(1.0f);
      buffer.reserve(static_cast<std::size_t>(n_vertices));
    }
  }

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;
  using frontier_t = typename enactor_t<problem_t>::frontier_t;

  void prepare_frontier(frontier_t* f,
                        gcuda::multi_context_t& context) override {
    auto P = this->get_problem();
    f->push_back(P->param.single_source);
  }

  void loop(gcuda::multi_context_t& context) override {
    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto single_source = P->param.single_source;
    auto distances = P->result.distances;
    auto visited = P->visited.data().get();

    auto iteration = this->iteration;
    bool profile_enabled = P->param.iter_profile != nullptr &&
                           P->param.iter_profile->enabled();
    auto input_frontier = profile_enabled
                              ? E->get_input_frontier()->get_number_of_elements()
                              : 0;
    std::string range_name =
        profile_enabled
            ? util::iteration_profiler::make_range_name("bfs", iteration)
            : std::string();
    util::iteration_profiler::range_t range(range_name, profile_enabled);
    util::iteration_profiler::event_timer_t timer(profile_enabled);
    timer.start(context.get_context(0)->stream());

    auto search = [distances, single_source, iteration] __host__ __device__(
                      vertex_t const& source,    // ... source
                      vertex_t const& neighbor,  // neighbor
                      edge_t const& edge,        // edge
                      weight_t const& weight     // weight (tuple).
                      ) -> bool {
      // If the neighbor is not visited, update the distance. Returning false
      // here means that the neighbor is not added to the output frontier, and
      // instead an invalid vertex is added in its place. These invalides (-1 in
      // most cases) can be removed using a filter operator or uniquify.

      // if (distances[neighbor] != std::numeric_limits<vertex_t>::max())
      //   return false;
      // else
      //   return (math::atomic::cas(
      //               &distances[neighbor],
      //               std::numeric_limits<vertex_t>::max(), iteration + 1) ==
      //               std::numeric_limits<vertex_t>::max());

      // Simpler logic for the above.
      auto old_distance =
          math::atomic::min(&distances[neighbor], iteration + 1);
      return (iteration + 1 < old_distance);
    };

    auto remove_invalids =
        [] __host__ __device__(vertex_t const& vertex) -> bool {
      // Returning true here means that we keep all the valid vertices.
      // Internally, filter will automatically remove invalids and will never
      // pass them to this lambda function.
      return true;
    };

    // Execute advance operator on the provided lambda
    auto advance_load_balance = P->param.advance_load_balance;
    std::string advance_range_name =
        profile_enabled
            ? util::iteration_profiler::make_range_name("bfs", iteration,
                                                        "advance")
            : std::string();
    util::iteration_profiler::range_t advance_range(advance_range_name,
                                                    profile_enabled);
    util::iteration_profiler::event_timer_t advance_timer(profile_enabled);
    advance_timer.start(context.get_context(0)->stream());
    operators::advance::execute_runtime(G, E, search, advance_load_balance, context);
    auto advance_elapsed_ms = advance_timer.stop(context.get_context(0)->stream());
    advance_range.pop();

    // Execute filter operator to remove the invalids.
    // @todo: Add CLI option to enable or disable this.
    // operators::filter::execute<operators::filter_algorithm_t::compact>(
    // G, E, remove_invalids, context);

    auto elapsed_ms = timer.stop(context.get_context(0)->stream());
    range.pop();

    if (profile_enabled) {
      auto output_frontier = E->get_input_frontier()->get_number_of_elements();
      P->param.iter_profile->write(
          iteration,
          {range_name,
           static_cast<long long>(input_frontier),
           -1,
           static_cast<long long>(input_frontier),
           static_cast<long long>(output_frontier),
           elapsed_ms});
      P->param.iter_profile->write(
          iteration,
          {advance_range_name,
           static_cast<long long>(input_frontier),
           -1,
           static_cast<long long>(input_frontier),
           static_cast<long long>(output_frontier),
           advance_elapsed_ms});
    }
  }

};  // struct enactor_t

/**
 * @brief Run Breadth-First Search algorithm on a given graph, G, starting from
 * the source node, single_source. The resulting distances are stored in the
 * distances pointer. All data must be allocated by the user, on the device
 * (GPU) and passed in to this function.
 *
 * @tparam graph_t Graph type.
 * @param G Graph object.
 * @param single_source A vertex in the graph (integral type).
 * @param distances Pointer to the distances array of size number of vertices.
 * @param predecessors Pointer to the predecessors array of size number of
 * vertices. (optional, wip)
 * @param context Device context.
 * @return float Time taken to run the algorithm.
 */
template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type& single_source,  // Parameter
          typename graph_t::vertex_type* distances,      // Output
          typename graph_t::vertex_type* predecessors,   // Output
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0)),  // Context
          operators::load_balance_t advance_load_balance =
              operators::load_balance_t::block_mapped,
          util::iteration_profiler::csv_writer_t* iter_profile = nullptr
) {
  using vertex_t = typename graph_t::vertex_type;
  using param_type = param_t<vertex_t>;
  using result_type = result_t<vertex_t>;

  param_type param(single_source, advance_load_balance, iter_profile);
  result_type result(distances, predecessors);

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();

  enactor_type enactor(&problem, context);
  return enactor.enact();
}

}  // namespace bfs
}  // namespace gunrock
