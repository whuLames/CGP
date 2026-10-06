#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

__global__ void clear_contiguous(uint32_t* data, uint64_t count, uint32_t value) {
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += uint64_t(blockDim.x) * gridDim.x)
    data[i] = value;
}

__global__ void clear_one_query(uint32_t* data, uint64_t vertices,
                                uint32_t query_count, uint32_t query,
                                uint32_t value) {
  for (uint64_t vertex = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       vertex < vertices; vertex += uint64_t(blockDim.x) * gridDim.x)
    data[vertex * query_count + query] = value;
}

__global__ void clear_query_group(uint32_t* data, uint64_t vertices,
                                  uint32_t query_count, uint32_t first_query,
                                  uint32_t value) {
  const uint64_t cells = vertices << 5;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < cells;
       i += uint64_t(blockDim.x) * gridDim.x) {
    const uint64_t vertex = i >> 5;
    const uint32_t query = first_query + uint32_t(i & 31);
    data[vertex * query_count + query] = value;
  }
}

template<class Launch>
float median_time_ms(Launch&& launch, int repetitions) {
  for (int i = 0; i < 5; ++i) launch(0x12340000u + uint32_t(i));
  check(cudaDeviceSynchronize(), "warmup synchronize");

  cudaEvent_t begin = nullptr, end = nullptr;
  check(cudaEventCreate(&begin), "create begin event");
  check(cudaEventCreate(&end), "create end event");
  std::vector<float> times;
  times.reserve(repetitions);
  for (int i = 0; i < repetitions; ++i) {
    check(cudaEventRecord(begin), "record begin event");
    launch(0x56780000u + uint32_t(i));
    check(cudaEventRecord(end), "record end event");
    check(cudaEventSynchronize(end), "synchronize end event");
    float elapsed = 0.0f;
    check(cudaEventElapsedTime(&elapsed, begin, end), "elapsed time");
    times.push_back(elapsed);
  }
  check(cudaEventDestroy(begin), "destroy begin event");
  check(cudaEventDestroy(end), "destroy end event");
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

int blocks_for(uint64_t elements) {
  constexpr uint64_t threads = 256;
  return int(std::min<uint64_t>((elements + threads - 1) / threads, 65535));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const uint64_t vertices = argc > 1 ? std::stoull(argv[1]) : 10000000ULL;
    const int gpu = argc > 2 ? std::stoi(argv[2]) : 0;
    const int repetitions = argc > 3 ? std::stoi(argv[3]) : 21;
    if (vertices == 0 || repetitions <= 0)
      throw std::runtime_error("V and repetitions must be positive");
    check(cudaSetDevice(gpu), "set device");

    constexpr uint32_t max_queries = 256;
    const uint64_t cells = vertices * max_queries;
    if (vertices != cells / max_queries)
      throw std::runtime_error("allocation size overflow");
    uint32_t* data = nullptr;
    check(cudaMalloc(&data, cells * sizeof(uint32_t)), "allocate data");
    check(cudaMemset(data, 0, cells * sizeof(uint32_t)), "initialize data");

    constexpr int threads = 256;
    const int blocks = blocks_for(vertices);
    const float contiguous_ms = median_time_ms(
        [&](uint32_t value) {
          clear_contiguous<<<blocks, threads>>>(data, vertices, value);
          check(cudaGetLastError(), "launch contiguous clear");
        }, repetitions);
    const double bytes = double(vertices) * sizeof(uint32_t);

    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, gpu), "get device properties");
    std::cout << "gpu=" << gpu << ",name=" << properties.name
              << ",V=" << vertices << ",repetitions=" << repetitions
              << ",allocation_gib=" << std::fixed << std::setprecision(3)
              << double(cells * sizeof(uint32_t)) / double(1ULL << 30) << '\n';
    std::cout << "layout,M,time_ms,logical_GBps,normalized_vs_same_bytes_contiguous\n";
    std::cout << "contiguous,1," << std::setprecision(6) << contiguous_ms << ','
              << bytes / (double(contiguous_ms) * 1.0e6) << ",1.000000\n";

    for (uint32_t queries : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u}) {
      const uint32_t query = queries == 1 ? 0 : queries / 2;
      const float elapsed = median_time_ms(
          [&](uint32_t value) {
            clear_one_query<<<blocks, threads>>>(data, vertices, queries, query, value);
            check(cudaGetLastError(), "launch strided clear");
          }, repetitions);
      std::cout << "vertex-major," << queries << ',' << elapsed << ','
                << bytes / (double(elapsed) * 1.0e6) << ','
                << elapsed / contiguous_ms << '\n';
    }

    const uint64_t tile_cells = vertices * 32;
    const int tile_blocks = blocks_for(tile_cells);
    for (uint32_t queries : {32u, 64u, 128u, 256u}) {
      const float elapsed = median_time_ms(
          [&](uint32_t value) {
            clear_query_group<<<tile_blocks, threads>>>(
                data, vertices, queries, 0, value);
            check(cudaGetLastError(), "launch vertex-major group clear");
          }, repetitions);
      std::cout << "vertex-major-clear-32," << queries << ',' << elapsed << ','
                << double(tile_cells * sizeof(uint32_t)) /
                       (double(elapsed) * 1.0e6)
                << ',' << elapsed / (32.0f * contiguous_ms) << '\n';
    }
    const float tile_kernel_ms = median_time_ms(
        [&](uint32_t value) {
          clear_contiguous<<<tile_blocks, threads>>>(data, tile_cells, value);
          check(cudaGetLastError(), "launch tile clear");
        }, repetitions);
    std::cout << "tile32-clear-32,32," << tile_kernel_ms << ','
              << double(tile_cells * sizeof(uint32_t)) /
                     (double(tile_kernel_ms) * 1.0e6)
              << ',' << tile_kernel_ms / (32.0f * contiguous_ms) << '\n';

    uint32_t observed = 0;
    check(cudaMemcpy(&observed, data + (vertices - 1) * 256 + 128,
                     sizeof(observed), cudaMemcpyDeviceToHost), "validate copy");
    const uint32_t expected = 0x56780000u + uint32_t(repetitions - 1);
    if (observed != expected)
      throw std::runtime_error("strided clear validation failed");
    std::cout << "validation=pass\n";
    check(cudaFree(data), "free data");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
