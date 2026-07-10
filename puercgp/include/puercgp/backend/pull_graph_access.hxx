#pragma once

#include <type_traits>
#include <utility>

namespace puercgp {
namespace detail {

template <typename graph_t, typename vertex_t, typename = void>
struct has_pull_edge_api : std::false_type {};

template <typename graph_t, typename vertex_t>
struct has_pull_edge_api<
    graph_t,
    vertex_t,
    std::void_t<decltype(std::declval<graph_t>().get_starting_pull_edge(
                    std::declval<vertex_t>())),
                decltype(std::declval<graph_t>().get_pull_neighbor_vertex(
                    std::declval<typename graph_t::edge_type>())),
                decltype(std::declval<graph_t>().get_pull_edge_weight(
                    std::declval<typename graph_t::edge_type>()))>>
    : std::true_type {};

template <typename graph_t, typename = void>
struct has_pull_adjacency_query : std::false_type {};

template <typename graph_t>
struct has_pull_adjacency_query<
    graph_t,
    std::void_t<decltype(std::declval<graph_t>().has_pull_adjacency())>>
    : std::true_type {};

template <typename graph_t, typename vertex_t>
__host__ __device__ auto get_pull_starting_edge(const graph_t& graph,
                                                vertex_t vertex) {
  if constexpr (has_pull_edge_api<graph_t, vertex_t>::value) {
    return graph.get_starting_pull_edge(vertex);
  } else {
    return graph.get_starting_edge(vertex);
  }
}

template <typename graph_t>
__host__ __device__ auto get_pull_neighbor_vertex(
    const graph_t& graph,
    typename graph_t::edge_type edge) {
  if constexpr (has_pull_edge_api<graph_t, typename graph_t::vertex_type>::value) {
    return graph.get_pull_neighbor_vertex(edge);
  } else {
    return graph.get_destination_vertex(edge);
  }
}

template <typename graph_t>
__host__ __device__ auto get_pull_edge_weight(
    const graph_t& graph,
    typename graph_t::edge_type edge) {
  if constexpr (has_pull_edge_api<graph_t, typename graph_t::vertex_type>::value) {
    return graph.get_pull_edge_weight(edge);
  } else {
    return graph.get_edge_weight(edge);
  }
}

template <typename graph_t>
bool graph_has_pull_adjacency(const graph_t& graph) {
  if constexpr (has_pull_adjacency_query<graph_t>::value) {
    return graph.has_pull_adjacency();
  } else {
    return false;
  }
}

}  // namespace detail
}  // namespace puercgp
