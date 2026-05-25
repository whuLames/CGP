#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <fstream>
#include <iostream>
#include <tuple>

#include <thrust/fill.h>

#include <gunrock/formats/formats.hxx>
#include <gunrock/error.hxx>
#include <gunrock/graph/properties.hxx>
#include <gunrock/memory.hxx>
#include <gunrock/util/filepath.hxx>

namespace gunrock {
namespace io {

using namespace memory;

/**
 * @brief Reads a Galois GR binary graph into a CSR format.
 *
 * GR binary layout:
 *   [header: 4 x uint64]
 *     version    = 1
 *     sizeEdgeTy = 0 (unweighted) or 1 (weighted)
 *     nvtxs      = V
 *     nedges     = E
 *   [row_start: V x int64]
 *   [edge_dst:   E x int32]
 *   [padding:    4 bytes] if weighted and E is odd
 *   [adjwgt:     E x int32] if weighted
 */
template <typename vertex_t, typename edge_t, typename weight_t>
std::tuple<gunrock::graph::graph_properties_t,
           format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>>
load_gr(std::string filename) {
  std::ifstream fp(filename, std::ios::binary);
  if (!fp.is_open()) {
    std::cerr << "File could not be opened: " << filename << std::endl;
    std::exit(1);
  }

  uint64_t header[4];
  fp.read(reinterpret_cast<char*>(header), sizeof(header));
  if (!fp) {
    std::cerr << "Could not read GR header: " << filename << std::endl;
    std::exit(1);
  }

  if (header[0] != 1) {
    std::cerr << "Unsupported GR version: " << header[0] << std::endl;
    std::exit(1);
  }

  const uint64_t size_edge_ty = header[1];
  const uint64_t nvtxs = header[2];
  const uint64_t nedges = header[3];

  error::throw_if_exception(
      nvtxs >= std::numeric_limits<vertex_t>::max(), "vertex_t overflow");
  error::throw_if_exception(
      nedges >= std::numeric_limits<edge_t>::max(), "edge_t overflow");

  gunrock::graph::graph_properties_t properties;
  properties.directed = true;
  properties.symmetric = false;
  properties.weighted = (size_edge_ty != 0);

  format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t> csr(
      (vertex_t)nvtxs, (vertex_t)nvtxs, (edge_t)nedges);
  thrust::host_vector<edge_t> h_row_offsets(nvtxs + 1);
  thrust::host_vector<vertex_t> h_column_indices(nedges);
  thrust::host_vector<weight_t> h_nonzero_values(nedges);

  // row_start: V x int64. row_offsets[0] is implicitly 0.
  h_row_offsets[0] = 0;
  for (uint64_t i = 0; i < nvtxs; ++i) {
    int64_t row_start = 0;
    fp.read(reinterpret_cast<char*>(&row_start), sizeof(int64_t));
    if (!fp) {
      std::cerr << "Could not read GR row offsets from: " << filename
                << std::endl;
      std::exit(1);
    }
    h_row_offsets[i + 1] = static_cast<edge_t>(row_start);
  }

  // edge_dst: E x int32
  for (uint64_t i = 0; i < nedges; ++i) {
    int32_t dst = 0;
    fp.read(reinterpret_cast<char*>(&dst), sizeof(int32_t));
    if (!fp) {
      std::cerr << "Could not read GR edge destinations from: " << filename
                << std::endl;
      std::exit(1);
    }
    h_column_indices[i] = static_cast<vertex_t>(dst);
  }

  if (size_edge_ty != 0) {
    if (nedges % 2 == 1) {
      fp.seekg(4, std::ios::cur);
    }
    for (uint64_t i = 0; i < nedges; ++i) {
      int32_t w = 0;
      fp.read(reinterpret_cast<char*>(&w), sizeof(int32_t));
      if (!fp) {
        std::cerr << "Could not read GR edge weights from: " << filename
                  << std::endl;
        std::exit(1);
      }
      h_nonzero_values[i] = static_cast<weight_t>(w);
    }
  } else {
    thrust::fill(h_nonzero_values.begin(), h_nonzero_values.end(),
                 static_cast<weight_t>(1));
  }

  csr.row_offsets = h_row_offsets;
  csr.column_indices = h_column_indices;
  csr.nonzero_values = h_nonzero_values;

  return {properties, csr};
}

}  // namespace io
}  // namespace gunrock
