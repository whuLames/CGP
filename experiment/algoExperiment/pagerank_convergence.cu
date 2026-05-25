#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::ostringstream oss__;                                         \
            oss__ << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err__);                              \
            throw std::runtime_error(oss__.str());                            \
        }                                                                    \
    } while (0)

struct Graph {
    std::string path;
    uint32_t nvtxs = 0;
    uint32_t nedges = 0;
    std::vector<uint32_t> row_offsets;
    std::vector<uint32_t> edge_dst;
};

struct Args {
    std::vector<std::string> inputs;
    std::string output = "pagerank_convergence.csv";
    std::vector<double> dampings = {0.80, 0.85, 0.90};
    std::vector<double> epsilons = {1e-2, 1e-3, 1e-4};
    int max_iters = 200;
    int device = 0;
};

struct Metrics {
    int l1_iterations = 0;
    int max_delta_iterations = 0;
    double final_l1_residual = 0.0;
    double final_max_delta = 0.0;
    bool converged_by_l1 = false;
    bool converged_by_max_delta = false;
    float runtime_ms = 0.0f;
};

static std::string basename(const std::string& path)
{
    size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

static std::vector<double> parse_double_list(const std::string& text)
{
    std::vector<double> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        char* end = nullptr;
        double value = std::strtod(item.c_str(), &end);
        if (end == item.c_str() || *end != '\0') {
            throw std::runtime_error("Invalid floating point list item: " + item);
        }
        values.push_back(value);
    }
    if (values.empty()) {
        throw std::runtime_error("List argument must contain at least one value.");
    }
    return values;
}

static void print_usage(const char* prog)
{
    std::cerr
        << "Usage: " << prog << " --input graph.gr [--input graph2.gr] [options]\n"
        << "Options:\n"
        << "  --output <path>             CSV output path (default: pagerank_convergence.csv)\n"
        << "  --damping <a,b,c>           Damping factors (default: 0.80,0.85,0.90)\n"
        << "  --epsilon <a,b,c>           Convergence epsilons (default: 1e-2,1e-3,1e-4)\n"
        << "  --max-iters <n>             Maximum iterations (default: 200)\n"
        << "  --device <id>               CUDA device id (default: 0)\n";
}

static Args parse_args(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; i++) {
        std::string key(argv[i]);
        if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for argument: " + key);
        }
        std::string value(argv[++i]);
        if (key == "--input") {
            args.inputs.push_back(value);
        } else if (key == "--output") {
            args.output = value;
        } else if (key == "--damping") {
            args.dampings = parse_double_list(value);
        } else if (key == "--epsilon") {
            args.epsilons = parse_double_list(value);
        } else if (key == "--max-iters") {
            args.max_iters = std::atoi(value.c_str());
        } else if (key == "--device") {
            args.device = std::atoi(value.c_str());
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }

    if (args.inputs.empty()) {
        throw std::runtime_error("--input is required and may be repeated.");
    }
    if (args.max_iters <= 0) {
        throw std::runtime_error("--max-iters must be positive.");
    }
    for (double damping : args.dampings) {
        if (!(damping > 0.0 && damping < 1.0)) {
            throw std::runtime_error("Each damping factor must be in (0, 1).");
        }
    }
    for (double epsilon : args.epsilons) {
        if (!(epsilon > 0.0)) {
            throw std::runtime_error("Each epsilon must be positive.");
        }
    }
    return args;
}

template <typename T>
static void read_exact(std::ifstream& in, T* dst, size_t count, const std::string& what)
{
    in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(sizeof(T) * count));
    if (!in) {
        throw std::runtime_error("Failed reading " + what);
    }
}

static Graph read_ggr(const std::string& path)
{
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open input graph: " + path);
    }

    uint64_t version = 0;
    uint64_t size_edge_ty = 0;
    uint64_t nvtxs64 = 0;
    uint64_t nedges64 = 0;
    read_exact(in, &version, 1, "GGR version");
    read_exact(in, &size_edge_ty, 1, "GGR edge type size");
    read_exact(in, &nvtxs64, 1, "GGR vertex count");
    read_exact(in, &nedges64, 1, "GGR edge count");

    if (version != 1) {
        throw std::runtime_error("Unsupported GGR version in " + path + ": " + std::to_string(version));
    }
    if (size_edge_ty != 0 && size_edge_ty != sizeof(int32_t)) {
        std::cerr << "Warning: weighted GGR edge type size is " << size_edge_ty
                  << " bytes in " << path
                  << "; PageRank convergence ignores edge weights and uses topology only."
                  << std::endl;
    }
    if (nvtxs64 > std::numeric_limits<uint32_t>::max() ||
        nedges64 > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Graph exceeds uint32 limits used by this experiment: " + path);
    }

    Graph graph;
    graph.path = path;
    graph.nvtxs = static_cast<uint32_t>(nvtxs64);
    graph.nedges = static_cast<uint32_t>(nedges64);
    graph.row_offsets.resize(static_cast<size_t>(graph.nvtxs) + 1);
    graph.edge_dst.resize(graph.nedges);
    graph.row_offsets[0] = 0;

    std::vector<int64_t> row_start(graph.nvtxs);
    read_exact(in, row_start.data(), graph.nvtxs, "GGR row_start");
    for (uint32_t i = 0; i < graph.nvtxs; i++) {
        if (row_start[i] < 0 ||
            static_cast<uint64_t>(row_start[i]) > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("Invalid or overflowing row offset in " + path);
        }
        graph.row_offsets[static_cast<size_t>(i) + 1] = static_cast<uint32_t>(row_start[i]);
    }
    if (graph.row_offsets.back() != graph.nedges) {
        throw std::runtime_error("Last row offset does not match edge count in " + path);
    }

    read_exact(in, graph.edge_dst.data(), graph.nedges, "GGR edge_dst");
    for (uint32_t dst : graph.edge_dst) {
        if (dst >= graph.nvtxs) {
            throw std::runtime_error("GGR edge destination out of range in " + path);
        }
    }
    return graph;
}

__global__ void init_rank_kernel(float* rank, uint32_t n)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    float init = 1.0f / static_cast<float>(n);
    for (uint32_t v = tid; v < n; v += stride) {
        rank[v] = init;
    }
}

__global__ void fill_next_kernel(float* next, uint32_t n, float value)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (uint32_t v = tid; v < n; v += stride) {
        next[v] = value;
    }
}

__global__ void scatter_pagerank_kernel(const uint32_t* row_offsets,
                                        const uint32_t* edge_dst,
                                        const float* rank,
                                        float* next,
                                        uint32_t n,
                                        float damping)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (uint32_t src = tid; src < n; src += stride) {
        uint32_t begin = row_offsets[src];
        uint32_t end = row_offsets[static_cast<size_t>(src) + 1];
        uint32_t degree = end - begin;
        if (degree == 0) {
            continue;
        }
        float contribution = damping * rank[src] / static_cast<float>(degree);
        for (uint32_t e = begin; e < end; e++) {
            atomicAdd(&next[edge_dst[e]], contribution);
        }
    }
}

struct DanglingContribution {
    const uint32_t* row_offsets;
    const float* rank;

    __host__ __device__ double operator()(const uint32_t v) const
    {
        return row_offsets[v] == row_offsets[static_cast<size_t>(v) + 1] ? static_cast<double>(rank[v]) : 0.0;
    }
};

struct L1Diff {
    const float* a;
    const float* b;

    __host__ __device__ double operator()(const uint32_t v) const
    {
        double diff = static_cast<double>(a[v]) - static_cast<double>(b[v]);
        return diff < 0.0 ? -diff : diff;
    }
};

struct AbsDiff {
    const float* a;
    const float* b;

    __host__ __device__ float operator()(const uint32_t v) const
    {
        float diff = a[v] - b[v];
        return diff < 0.0f ? -diff : diff;
    }
};

static Metrics run_pagerank(const Graph& graph, double damping, double epsilon, int max_iters)
{
    const int block_size = 256;
    const int grid_size = std::min<int>((graph.nvtxs + block_size - 1) / block_size, 65535);

    uint32_t* d_row_offsets = nullptr;
    uint32_t* d_edge_dst = nullptr;
    float* d_rank = nullptr;
    float* d_next = nullptr;

    CUDA_CHECK(cudaMalloc(&d_row_offsets, graph.row_offsets.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_edge_dst, graph.edge_dst.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_rank, static_cast<size_t>(graph.nvtxs) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next, static_cast<size_t>(graph.nvtxs) * sizeof(float)));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    try {
        CUDA_CHECK(cudaMemcpy(d_row_offsets, graph.row_offsets.data(),
                              graph.row_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_edge_dst, graph.edge_dst.data(),
                              graph.edge_dst.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

        init_rank_kernel<<<grid_size, block_size>>>(d_rank, graph.nvtxs);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));

        Metrics metrics;
        for (int iter = 1; iter <= max_iters; iter++) {
            double dangling_sum = thrust::transform_reduce(
                thrust::device,
                thrust::counting_iterator<uint32_t>(0),
                thrust::counting_iterator<uint32_t>(graph.nvtxs),
                DanglingContribution{d_row_offsets, d_rank},
                0.0,
                thrust::plus<double>());

            float base = static_cast<float>((1.0 - damping) / graph.nvtxs +
                                            damping * dangling_sum / graph.nvtxs);
            fill_next_kernel<<<grid_size, block_size>>>(d_next, graph.nvtxs, base);
            CUDA_CHECK(cudaGetLastError());

            scatter_pagerank_kernel<<<grid_size, block_size>>>(
                d_row_offsets, d_edge_dst, d_rank, d_next, graph.nvtxs, static_cast<float>(damping));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            double l1 = thrust::transform_reduce(
                thrust::device,
                thrust::counting_iterator<uint32_t>(0),
                thrust::counting_iterator<uint32_t>(graph.nvtxs),
                L1Diff{d_next, d_rank},
                0.0,
                thrust::plus<double>());

            float max_delta = thrust::transform_reduce(
                thrust::device,
                thrust::counting_iterator<uint32_t>(0),
                thrust::counting_iterator<uint32_t>(graph.nvtxs),
                AbsDiff{d_next, d_rank},
                0.0f,
                thrust::maximum<float>());

            metrics.final_l1_residual = l1;
            metrics.final_max_delta = static_cast<double>(max_delta);

            if (!metrics.converged_by_l1 && l1 < epsilon) {
                metrics.converged_by_l1 = true;
                metrics.l1_iterations = iter;
            }
            if (!metrics.converged_by_max_delta && max_delta < epsilon) {
                metrics.converged_by_max_delta = true;
                metrics.max_delta_iterations = iter;
            }

            std::swap(d_rank, d_next);
            if (metrics.converged_by_l1 && metrics.converged_by_max_delta) {
                break;
            }
        }

        if (!metrics.converged_by_l1) {
            metrics.l1_iterations = max_iters;
        }
        if (!metrics.converged_by_max_delta) {
            metrics.max_delta_iterations = max_iters;
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&metrics.runtime_ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        start = nullptr;
        CUDA_CHECK(cudaEventDestroy(stop));
        stop = nullptr;

        CUDA_CHECK(cudaFree(d_row_offsets));
        d_row_offsets = nullptr;
        CUDA_CHECK(cudaFree(d_edge_dst));
        d_edge_dst = nullptr;
        CUDA_CHECK(cudaFree(d_rank));
        d_rank = nullptr;
        CUDA_CHECK(cudaFree(d_next));
        d_next = nullptr;
        return metrics;
    } catch (...) {
        if (start) {
            cudaEventDestroy(start);
        }
        if (stop) {
            cudaEventDestroy(stop);
        }
        cudaFree(d_row_offsets);
        cudaFree(d_edge_dst);
        cudaFree(d_rank);
        cudaFree(d_next);
        throw;
    }
}

int main(int argc, char** argv)
{
    try {
        Args args = parse_args(argc, argv);
        CUDA_CHECK(cudaSetDevice(args.device));

        std::ofstream csv(args.output.c_str());
        if (!csv) {
            throw std::runtime_error("Cannot open output CSV: " + args.output);
        }
        csv << "dataset,nvtxs,nedges,damping,epsilon,l1_iterations,max_delta_iterations,"
               "final_l1_residual,final_max_delta,converged_by_l1,converged_by_max_delta,runtime_ms\n";

        for (const std::string& input : args.inputs) {
            std::cerr << "Reading graph: " << input << std::endl;
            Graph graph = read_ggr(input);
            std::cerr << "Graph loaded: " << graph.nvtxs << " vertices, "
                      << graph.nedges << " edges" << std::endl;

            for (double damping : args.dampings) {
                for (double epsilon : args.epsilons) {
                    std::cerr << "Running damping=" << damping
                              << " epsilon=" << epsilon << std::endl;
                    Metrics metrics = run_pagerank(graph, damping, epsilon, args.max_iters);

                    csv << basename(input) << ','
                        << graph.nvtxs << ','
                        << graph.nedges << ','
                        << std::setprecision(12) << damping << ','
                        << std::setprecision(12) << epsilon << ','
                        << metrics.l1_iterations << ','
                        << metrics.max_delta_iterations << ','
                        << std::setprecision(12) << metrics.final_l1_residual << ','
                        << std::setprecision(12) << metrics.final_max_delta << ','
                        << (metrics.converged_by_l1 ? 1 : 0) << ','
                        << (metrics.converged_by_max_delta ? 1 : 0) << ','
                        << std::setprecision(6) << metrics.runtime_ms << '\n';
                    csv.flush();
                }
            }
        }
        std::cerr << "Wrote CSV: " << args.output << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        print_usage(argv[0]);
        return 1;
    }
}
