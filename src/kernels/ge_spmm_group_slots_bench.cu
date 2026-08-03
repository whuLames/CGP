/*
 * Pure GE-SpMM gather-min microbenchmark for the group-slots hypothesis.
 *
 * Cases:
 *   q64          : one M=64, CF=2 kernel on the full GPU.
 *   green_2x32   : two M=32, CF=1 kernels on disjoint Green Context SM sets.
 *
 * Only the GE-SpMM kernel is timed. Input initialization, Green Context
 * creation, correctness fingerprints, and buffer management are excluded.
 * Every warmup and measured round synchronizes all participating streams
 * before the next round is submitted.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <puercgp/core/green_context.hxx>

namespace {

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error__ = (call);                                               \
    if (error__ != cudaSuccess) {                                               \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ +      \
                               ":" + std::to_string(__LINE__) + ": " +         \
                               cudaGetErrorString(error__));                    \
    }                                                                          \
  } while (0)

constexpr int kTotalQueries = 64;
constexpr int kGroupQueries = 32;
constexpr int kFingerprintBlocks = 256;
constexpr int kFingerprintThreads = 256;

enum class case_t { q64, green_2x32, both };

struct arguments_t {
  std::string graph_path;
  std::string csv_path;
  case_t selected_case = case_t::both;
  int tile_row = 8;
  int iterations = 30;
  int warmup = 5;
  int gpu = 0;
  unsigned int seed = 42;
  bool skip_verify = false;
  bool self_test = false;
};

struct host_graph_t {
  std::string name;
  int vertices = 0;
  std::int64_t edges = 0;
  std::vector<std::int64_t> row_offsets;
  std::vector<int> column_indices;
};

struct device_graph_t {
  int vertices = 0;
  std::int64_t edges = 0;
  std::int64_t* row_offsets = nullptr;
  int* column_indices = nullptr;
};

struct round_t {
  int round = 0;
  double wall_ms = 0.0;
  float stream0_ms = 0.0f;
  float stream1_ms = 0.0f;
};

struct fingerprint_t {
  std::array<unsigned long long, kTotalQueries> xor_hash{};
  std::array<unsigned long long, kTotalQueries> bit_sum{};

  bool operator==(const fingerprint_t& other) const {
    return xor_hash == other.xor_hash && bit_sum == other.bit_sum;
  }
};

struct case_result_t {
  std::string name;
  int sm0 = 0;
  int sm1 = 0;
  std::vector<round_t> rounds;
  fingerprint_t fingerprint;
  std::vector<float> toy_output;
};

std::string usage(const char* program) {
  return std::string("Usage: ") + program +
      " <csr_dir> [--case=q64|green_2x32|both] [--tile-row=N]"
      " [--iters=N] [--warmup=N] [--gpu=ID] [--seed=N]"
      " [--csv=path] [--skip-verify]\n"
      "       " + program + " --self-test [same optional flags]\n";
}

arguments_t parse_arguments(int argc, char** argv) {
  arguments_t args;
  if (argc < 2) {
    throw std::invalid_argument(usage(argv[0]));
  }
  int begin = 2;
  if (std::string(argv[1]) == "--self-test") {
    args.self_test = true;
    args.graph_path = "toy";
  } else {
    args.graph_path = argv[1];
  }
  for (int i = begin; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--case=", 0) == 0) {
      const std::string value = arg.substr(7);
      if (value == "q64")
        args.selected_case = case_t::q64;
      else if (value == "green_2x32")
        args.selected_case = case_t::green_2x32;
      else if (value == "both")
        args.selected_case = case_t::both;
      else
        throw std::invalid_argument("unknown --case value: " + value);
    } else if (arg.rfind("--tile-row=", 0) == 0) {
      args.tile_row = std::stoi(arg.substr(11));
    } else if (arg.rfind("--iters=", 0) == 0) {
      args.iterations = std::stoi(arg.substr(8));
    } else if (arg.rfind("--warmup=", 0) == 0) {
      args.warmup = std::stoi(arg.substr(9));
    } else if (arg.rfind("--gpu=", 0) == 0) {
      args.gpu = std::stoi(arg.substr(6));
    } else if (arg.rfind("--seed=", 0) == 0) {
      args.seed = static_cast<unsigned int>(std::stoul(arg.substr(7)));
    } else if (arg.rfind("--csv=", 0) == 0) {
      args.csv_path = arg.substr(6);
    } else if (arg == "--skip-verify") {
      args.skip_verify = true;
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  if (args.tile_row <= 0 || args.tile_row > 32 ||
      32 * args.tile_row > 1024) {
    throw std::invalid_argument("--tile-row must produce at most 1024 threads");
  }
  if (args.iterations <= 0) {
    throw std::invalid_argument("--iters must be positive");
  }
  if (args.warmup < 0) {
    throw std::invalid_argument("--warmup must be non-negative");
  }
  if (args.self_test) {
    args.selected_case = case_t::both;
    args.skip_verify = false;
  }
  return args;
}

void read_binary_ints(const std::filesystem::path& path,
                      std::vector<int>* values) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open " + path.string());
  const auto bytes = input.tellg();
  if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(int)) != 0) {
    throw std::runtime_error("invalid int32 file size: " + path.string());
  }
  input.seekg(0);
  values->resize(static_cast<std::size_t>(bytes / sizeof(int)));
  input.read(reinterpret_cast<char*>(values->data()), bytes);
  if (!input) throw std::runtime_error("failed reading " + path.string());
}

host_graph_t load_graph(const std::string& graph_path) {
  namespace fs = std::filesystem;
  const fs::path directory(graph_path);
  const fs::path row_path = directory / "csr_vlist.bin";
  const fs::path column_path = directory / "csr_elist.bin";
  if (!fs::is_regular_file(row_path) || !fs::is_regular_file(column_path)) {
    throw std::runtime_error(
        "CSR directory requires csr_vlist.bin and csr_elist.bin: " +
        graph_path);
  }

  std::vector<int> row32;
  read_binary_ints(row_path, &row32);
  if (row32.empty()) throw std::runtime_error("empty csr_vlist.bin");

  host_graph_t graph;
  graph.name = directory.filename().string();
  graph.vertices = static_cast<int>(row32.size() - 1);
  graph.row_offsets.resize(row32.size());
  for (std::size_t i = 0; i < row32.size(); ++i) {
    if (row32[i] < 0) throw std::runtime_error("negative CSR row offset");
    graph.row_offsets[i] = static_cast<std::int64_t>(row32[i]);
  }
  read_binary_ints(column_path, &graph.column_indices);
  graph.edges = static_cast<std::int64_t>(graph.column_indices.size());
  if (graph.row_offsets.back() != graph.edges) {
    throw std::runtime_error("CSR final row offset does not equal edge count");
  }
  return graph;
}

host_graph_t make_toy_graph() {
  host_graph_t graph;
  graph.name = "toy";
  graph.vertices = 6;
  graph.row_offsets = {0, 3, 5, 7, 9, 10, 10};
  graph.column_indices = {1, 2, 4, 0, 3, 1, 5, 0, 2, 3};
  graph.edges = static_cast<std::int64_t>(graph.column_indices.size());
  return graph;
}

device_graph_t upload_graph(const host_graph_t& graph) {
  device_graph_t device;
  device.vertices = graph.vertices;
  device.edges = graph.edges;
  CUDA_CHECK(cudaMalloc(&device.row_offsets,
                        graph.row_offsets.size() * sizeof(std::int64_t)));
  CUDA_CHECK(cudaMalloc(&device.column_indices,
                        graph.column_indices.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(device.row_offsets, graph.row_offsets.data(),
                        graph.row_offsets.size() * sizeof(std::int64_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device.column_indices, graph.column_indices.data(),
                        graph.column_indices.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  return device;
}

void free_graph(device_graph_t* graph) {
  if (graph->column_indices) cudaFree(graph->column_indices);
  if (graph->row_offsets) cudaFree(graph->row_offsets);
  *graph = {};
}

__host__ __device__ unsigned int mix32(unsigned int value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return value;
}

__host__ __device__ float input_value(std::size_t vertex,
                                      int global_query,
                                      unsigned int seed) {
  unsigned int value =
      static_cast<unsigned int>(vertex) * 0x9e3779b9U ^
      static_cast<unsigned int>(global_query + 1) * 0x85ebca6bU ^ seed;
  value = mix32(value);
  return 0.001f +
      static_cast<float>(value & 0x00ffffffU) / 16777216.0f;
}

template <int M>
__global__ void initialize_input_kernel(float* input,
                                        std::size_t vertices,
                                        int query_offset,
                                        unsigned int seed) {
  const std::size_t total = vertices * static_cast<std::size_t>(M);
  std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t stride =
      gridDim.x * static_cast<std::size_t>(blockDim.x);
  for (; index < total; index += stride) {
    const std::size_t vertex = index / M;
    const int query = static_cast<int>(index % M);
    input[index] = input_value(vertex, query_offset + query, seed);
  }
}

template <int M, int CF>
__global__ void ge_spmm_gather_min_kernel(
    int vertices,
    const std::int64_t* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const float* __restrict__ input,
    float* __restrict__ output,
    int tile_row) {
  extern __shared__ int neighbor_tile[];
  const int row_slot = threadIdx.y;
  const int lane = threadIdx.x;
  const int row = tile_row * blockIdx.x + row_slot;
  if (row >= vertices) return;

  const int feature0 = lane;
  const std::int64_t begin = row_offsets[row];
  const std::int64_t end = row_offsets[row + 1];
  std::int64_t pointer = begin + lane;
  float accumulator[CF];
#pragma unroll
  for (int c = 0; c < CF; ++c) {
    accumulator[c] = __int_as_float(0x7f800000);
  }

  const int shared_base = row_slot * 32;
  for (std::int64_t tile = begin; tile < end; tile += 32) {
    if (pointer < end) {
      neighbor_tile[shared_base + lane] = column_indices[pointer];
    }
    __syncwarp();
    pointer += 32;

    const int tile_count =
        static_cast<int>((end - tile) < 32 ? (end - tile) : 32);
    for (int i = 0; i < tile_count; ++i) {
      const int neighbor = neighbor_tile[shared_base + i];
      const std::size_t input_base =
          static_cast<std::size_t>(neighbor) * M + feature0;
#pragma unroll
      for (int c = 0; c < CF; ++c) {
        const float value = input[input_base + c * 32];
        accumulator[c] = fminf(accumulator[c], value);
      }
    }
    __syncwarp();
  }

  const std::size_t output_base =
      static_cast<std::size_t>(row) * M + feature0;
#pragma unroll
  for (int c = 0; c < CF; ++c) {
    output[output_base + c * 32] = accumulator[c];
  }
}

template <int M>
void launch_ge_spmm_min(const device_graph_t& graph,
                        int tile_row,
                        const float* input,
                        float* output,
                        cudaStream_t stream) {
  static_assert(M == 32 || M == 64, "benchmark only supports M=32/64");
  constexpr int coarsening = M / 32;
  const int grid_x = (graph.vertices + tile_row - 1) / tile_row;
  const std::size_t shared_bytes =
      static_cast<std::size_t>(32 * tile_row) * sizeof(int);
  ge_spmm_gather_min_kernel<M, coarsening>
      <<<grid_x, dim3(32, tile_row), shared_bytes, stream>>>(
          graph.vertices, graph.row_offsets, graph.column_indices, input,
          output, tile_row);
}

__device__ unsigned long long mix64(unsigned long long value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

template <int M>
__global__ void fingerprint_kernel(
    const float* output,
    std::size_t vertices,
    int query_offset,
    unsigned long long* xor_hash,
    unsigned long long* bit_sum) {
  const int local_query = static_cast<int>(blockIdx.y);
  const int global_query = query_offset + local_query;
  unsigned long long local_xor = 0;
  unsigned long long local_sum = 0;
  for (std::size_t vertex =
           blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
       vertex < vertices;
       vertex += gridDim.x * static_cast<std::size_t>(blockDim.x)) {
    const float value =
        output[vertex * static_cast<std::size_t>(M) + local_query];
    const unsigned int bits = __float_as_uint(value);
    local_xor ^= mix64((static_cast<unsigned long long>(bits) << 32) ^
                       static_cast<unsigned long long>(vertex));
    local_sum += bits;
  }

  __shared__ unsigned long long shared_xor[kFingerprintThreads];
  __shared__ unsigned long long shared_sum[kFingerprintThreads];
  shared_xor[threadIdx.x] = local_xor;
  shared_sum[threadIdx.x] = local_sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      shared_xor[threadIdx.x] ^= shared_xor[threadIdx.x + offset];
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicXor(xor_hash + global_query, shared_xor[0]);
    atomicAdd(bit_sum + global_query, shared_sum[0]);
  }
}

template <int M>
void append_fingerprint(const float* output,
                        std::size_t vertices,
                        int query_offset,
                        unsigned long long* xor_hash,
                        unsigned long long* bit_sum,
                        cudaStream_t stream) {
  fingerprint_kernel<M>
      <<<dim3(kFingerprintBlocks, M), kFingerprintThreads, 0, stream>>>(
          output, vertices, query_offset, xor_hash, bit_sum);
}

fingerprint_t copy_fingerprint(unsigned long long* xor_hash,
                               unsigned long long* bit_sum) {
  fingerprint_t result;
  CUDA_CHECK(cudaMemcpy(result.xor_hash.data(), xor_hash,
                        sizeof(result.xor_hash), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(result.bit_sum.data(), bit_sum,
                        sizeof(result.bit_sum), cudaMemcpyDeviceToHost));
  return result;
}

template <int M>
std::vector<float> copy_output(const float* output, std::size_t vertices) {
  std::vector<float> host(vertices * static_cast<std::size_t>(M));
  CUDA_CHECK(cudaMemcpy(host.data(), output, host.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return host;
}

void initialize_fingerprint_buffers(unsigned long long** xor_hash,
                                    unsigned long long** bit_sum) {
  CUDA_CHECK(cudaMalloc(xor_hash, kTotalQueries * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(bit_sum, kTotalQueries * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(*xor_hash, 0,
                        kTotalQueries * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(*bit_sum, 0,
                        kTotalQueries * sizeof(unsigned long long)));
}

case_result_t run_q64(const arguments_t& args,
                      const device_graph_t& graph,
                      bool retain_toy_output) {
  case_result_t result;
  result.name = "q64";
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, args.gpu));
  result.sm0 = properties.multiProcessorCount;

  const std::size_t elements =
      static_cast<std::size_t>(graph.vertices) * kTotalQueries;
  float* input = nullptr;
  float* output = nullptr;
  CUDA_CHECK(cudaMalloc(&input, elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&output, elements * sizeof(float)));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  const int init_blocks = std::max(
      1, std::min(65535, static_cast<int>((elements + 255) / 256)));
  initialize_input_kernel<kTotalQueries>
      <<<init_blocks, 256, 0, stream>>>(
          input, graph.vertices, 0, args.seed);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));
  const int total_rounds = args.warmup + args.iterations;
  for (int round = 0; round < total_rounds; ++round) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto wall_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(start_event, stream));
    launch_ge_spmm_min<kTotalQueries>(
        graph, args.tile_row, input, output, stream);
    CUDA_CHECK(cudaEventRecord(stop_event, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto wall_stop = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaGetLastError());
    if (round >= args.warmup) {
      float kernel_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start_event, stop_event));
      result.rounds.push_back(
          {round - args.warmup,
           std::chrono::duration<double, std::milli>(
               wall_stop - wall_start).count(),
           kernel_ms,
           0.0f});
    }
  }

  unsigned long long* xor_hash = nullptr;
  unsigned long long* bit_sum = nullptr;
  initialize_fingerprint_buffers(&xor_hash, &bit_sum);
  append_fingerprint<kTotalQueries>(
      output, graph.vertices, 0, xor_hash, bit_sum, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  result.fingerprint = copy_fingerprint(xor_hash, bit_sum);
  if (retain_toy_output) {
    result.toy_output = copy_output<kTotalQueries>(output, graph.vertices);
  }

  cudaFree(bit_sum);
  cudaFree(xor_hash);
  cudaEventDestroy(stop_event);
  cudaEventDestroy(start_event);
  cudaStreamDestroy(stream);
  cudaFree(output);
  cudaFree(input);
  return result;
}

case_result_t run_green_2x32(const arguments_t& args,
                             const device_graph_t& graph,
                             bool retain_toy_output) {
  case_result_t result;
  result.name = "green_2x32";
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, args.gpu));
  if (properties.multiProcessorCount % 2 != 0) {
    throw std::runtime_error("Green 2x32 requires an even device SM count");
  }
  const unsigned int requested_sms =
      static_cast<unsigned int>(properties.multiProcessorCount / 2);
  puercgp::detail::green_context_pair contexts(args.gpu, requested_sms);
  result.sm0 = static_cast<int>(contexts.first_sm_count());
  result.sm1 = static_cast<int>(contexts.second_sm_count());
  if (result.sm0 != static_cast<int>(requested_sms) ||
      result.sm1 != static_cast<int>(requested_sms)) {
    throw std::runtime_error(
        "Green Context did not produce equal half-device SM partitions");
  }
  const cudaStream_t stream0 = contexts.first_stream();
  const cudaStream_t stream1 = contexts.second_stream();

  const std::size_t group_elements =
      static_cast<std::size_t>(graph.vertices) * kGroupQueries;
  std::array<float*, 2> input{nullptr, nullptr};
  std::array<float*, 2> output{nullptr, nullptr};
  for (int group = 0; group < 2; ++group) {
    CUDA_CHECK(cudaMalloc(&input[group], group_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output[group], group_elements * sizeof(float)));
  }
  const int init_blocks = std::max(
      1, std::min(65535,
                  static_cast<int>((group_elements + 255) / 256)));
  initialize_input_kernel<kGroupQueries>
      <<<init_blocks, 256, 0, stream0>>>(
          input[0], graph.vertices, 0, args.seed);
  initialize_input_kernel<kGroupQueries>
      <<<init_blocks, 256, 0, stream1>>>(
          input[1], graph.vertices, kGroupQueries, args.seed);
  CUDA_CHECK(cudaStreamSynchronize(stream0));
  CUDA_CHECK(cudaStreamSynchronize(stream1));

  std::array<cudaEvent_t, 2> start_events{nullptr, nullptr};
  std::array<cudaEvent_t, 2> stop_events{nullptr, nullptr};
  for (int group = 0; group < 2; ++group) {
    CUDA_CHECK(cudaEventCreate(&start_events[group]));
    CUDA_CHECK(cudaEventCreate(&stop_events[group]));
  }

  const int total_rounds = args.warmup + args.iterations;
  for (int round = 0; round < total_rounds; ++round) {
    CUDA_CHECK(cudaStreamSynchronize(stream0));
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    const auto wall_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(start_events[0], stream0));
    CUDA_CHECK(cudaEventRecord(start_events[1], stream1));
    launch_ge_spmm_min<kGroupQueries>(
        graph, args.tile_row, input[0], output[0], stream0);
    launch_ge_spmm_min<kGroupQueries>(
        graph, args.tile_row, input[1], output[1], stream1);
    CUDA_CHECK(cudaEventRecord(stop_events[0], stream0));
    CUDA_CHECK(cudaEventRecord(stop_events[1], stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream0));
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    const auto wall_stop = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaGetLastError());
    if (round >= args.warmup) {
      float stream0_ms = 0.0f;
      float stream1_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(
          &stream0_ms, start_events[0], stop_events[0]));
      CUDA_CHECK(cudaEventElapsedTime(
          &stream1_ms, start_events[1], stop_events[1]));
      result.rounds.push_back(
          {round - args.warmup,
           std::chrono::duration<double, std::milli>(
               wall_stop - wall_start).count(),
           stream0_ms,
           stream1_ms});
    }
  }

  unsigned long long* xor_hash = nullptr;
  unsigned long long* bit_sum = nullptr;
  initialize_fingerprint_buffers(&xor_hash, &bit_sum);
  append_fingerprint<kGroupQueries>(
      output[0], graph.vertices, 0, xor_hash, bit_sum, stream0);
  append_fingerprint<kGroupQueries>(
      output[1], graph.vertices, kGroupQueries, xor_hash, bit_sum, stream1);
  CUDA_CHECK(cudaStreamSynchronize(stream0));
  CUDA_CHECK(cudaStreamSynchronize(stream1));
  result.fingerprint = copy_fingerprint(xor_hash, bit_sum);

  if (retain_toy_output) {
    const auto first = copy_output<kGroupQueries>(output[0], graph.vertices);
    const auto second = copy_output<kGroupQueries>(output[1], graph.vertices);
    result.toy_output.resize(
        static_cast<std::size_t>(graph.vertices) * kTotalQueries);
    for (int vertex = 0; vertex < graph.vertices; ++vertex) {
      std::copy_n(first.data() +
                      static_cast<std::size_t>(vertex) * kGroupQueries,
                  kGroupQueries,
                  result.toy_output.data() +
                      static_cast<std::size_t>(vertex) * kTotalQueries);
      std::copy_n(second.data() +
                      static_cast<std::size_t>(vertex) * kGroupQueries,
                  kGroupQueries,
                  result.toy_output.data() +
                      static_cast<std::size_t>(vertex) * kTotalQueries +
                      kGroupQueries);
    }
  }

  cudaFree(bit_sum);
  cudaFree(xor_hash);
  for (int group = 0; group < 2; ++group) {
    cudaEventDestroy(stop_events[group]);
    cudaEventDestroy(start_events[group]);
    cudaFree(output[group]);
    cudaFree(input[group]);
  }
  return result;
}

std::vector<float> cpu_reference(const host_graph_t& graph,
                                 unsigned int seed) {
  std::vector<float> output(
      static_cast<std::size_t>(graph.vertices) * kTotalQueries,
      std::numeric_limits<float>::infinity());
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    for (std::int64_t edge = graph.row_offsets[vertex];
         edge < graph.row_offsets[vertex + 1]; ++edge) {
      const int neighbor = graph.column_indices[edge];
      for (int query = 0; query < kTotalQueries; ++query) {
        float& current =
            output[static_cast<std::size_t>(vertex) * kTotalQueries + query];
        current = std::min(
            current, input_value(neighbor, query, seed));
      }
    }
  }
  return output;
}

void verify_toy(const host_graph_t& graph,
                const arguments_t& args,
                const case_result_t& q64,
                const case_result_t& green) {
  const auto expected = cpu_reference(graph, args.seed);
  if (q64.toy_output.size() != expected.size() ||
      green.toy_output.size() != expected.size()) {
    throw std::runtime_error("toy output size mismatch");
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (q64.toy_output[i] != expected[i]) {
      throw std::runtime_error(
          "Q64 toy mismatch at element " + std::to_string(i));
    }
    if (green.toy_output[i] != expected[i]) {
      throw std::runtime_error(
          "Green 2x32 toy mismatch at element " + std::to_string(i));
    }
  }
}

double percentile(std::vector<double> values, double p) {
  std::sort(values.begin(), values.end());
  const double position = p * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(position);
  const std::size_t upper = std::min(lower + 1, values.size() - 1);
  const double fraction = position - static_cast<double>(lower);
  return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

struct summary_t {
  double mean = 0.0;
  double median = 0.0;
  double p95 = 0.0;
  double standard_deviation = 0.0;
};

summary_t summarize(const case_result_t& result) {
  std::vector<double> values;
  values.reserve(result.rounds.size());
  for (const auto& round : result.rounds) values.push_back(round.wall_ms);
  summary_t summary;
  summary.mean =
      std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  summary.median = percentile(values, 0.5);
  summary.p95 = percentile(values, 0.95);
  double variance = 0.0;
  for (double value : values) {
    const double difference = value - summary.mean;
    variance += difference * difference;
  }
  summary.standard_deviation = std::sqrt(variance / values.size());
  return summary;
}

void print_summary(const case_result_t& result) {
  const auto summary = summarize(result);
  std::cout << "case=" << result.name
            << " sm=" << result.sm0;
  if (result.sm1) std::cout << "+" << result.sm1;
  std::cout << " mean_ms=" << summary.mean
            << " median_ms=" << summary.median
            << " p95_ms=" << summary.p95
            << " stddev_ms=" << summary.standard_deviation << "\n";
}

void write_csv(const arguments_t& args,
               const host_graph_t& graph,
               const std::vector<case_result_t>& results,
               bool fingerprints_match) {
  if (args.csv_path.empty()) return;
  std::filesystem::path output(args.csv_path);
  if (!output.parent_path().empty()) {
    std::filesystem::create_directories(output.parent_path());
  }
  std::ofstream csv(output);
  if (!csv) throw std::runtime_error("cannot open CSV: " + args.csv_path);
  csv << "dataset,V,E,case,round,wall_ms,stream0_ms,stream1_ms,"
         "sm0,sm1,tile_row,seed,fingerprint_match\n";
  for (const auto& result : results) {
    for (const auto& round : result.rounds) {
      csv << graph.name << "," << graph.vertices << "," << graph.edges << ","
          << result.name << "," << round.round << "," << round.wall_ms << ","
          << round.stream0_ms << "," << round.stream1_ms << ","
          << result.sm0 << "," << result.sm1 << "," << args.tile_row << ","
          << args.seed << "," << (fingerprints_match ? 1 : 0) << "\n";
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const arguments_t args = parse_arguments(argc, argv);
    CUDA_CHECK(cudaSetDevice(args.gpu));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, args.gpu));

    host_graph_t host_graph =
        args.self_test ? make_toy_graph() : load_graph(args.graph_path);
    device_graph_t device_graph = upload_graph(host_graph);

    std::cout << "dataset=" << host_graph.name
              << " V=" << host_graph.vertices
              << " E=" << host_graph.edges
              << " gpu=" << args.gpu
              << " gpu_name=" << properties.name
              << " device_sms=" << properties.multiProcessorCount
              << " tile_row=" << args.tile_row
              << " warmup=" << args.warmup
              << " iters=" << args.iterations
              << " seed=" << args.seed << "\n";

    std::vector<case_result_t> results;
    if (args.selected_case == case_t::q64 ||
        args.selected_case == case_t::both) {
      results.push_back(
          run_q64(args, device_graph, args.self_test));
      print_summary(results.back());
    }
    if (args.selected_case == case_t::green_2x32 ||
        args.selected_case == case_t::both) {
      results.push_back(
          run_green_2x32(args, device_graph, args.self_test));
      print_summary(results.back());
    }

    bool fingerprints_match = true;
    if (results.size() == 2 && !args.skip_verify) {
      fingerprints_match =
          results[0].fingerprint == results[1].fingerprint;
      if (!fingerprints_match) {
        throw std::runtime_error(
            "Q64 and Green 2x32 output fingerprints differ");
      }
      if (args.self_test) {
        verify_toy(host_graph, args, results[0], results[1]);
      }
      std::cout << "verification=PASS\n";
    } else {
      std::cout << "verification="
                << (args.skip_verify ? "SKIPPED" : "SINGLE_CASE") << "\n";
    }

    if (results.size() == 2) {
      const auto q64_summary = summarize(results[0]);
      const auto green_summary = summarize(results[1]);
      std::cout << "speedup_q64_over_green_2x32="
                << q64_summary.median / green_summary.median << "\n";
    }
    write_csv(args, host_graph, results, fingerprints_match);
    free_graph(&device_graph);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
