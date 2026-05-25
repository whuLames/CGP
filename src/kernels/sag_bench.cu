/*
 * sag_bench.cu
 * Standalone single-GPU benchmark for GNNAdvisor-style SAG kernel.
 *
 * Ported from GNNAdvisor OSDI'21: warp-per-part + shared memory aggregation.
 * Stripped of PyTorch dependencies, adapted for TCRGraph CSR data (int64 row_ptr).
 *
 * Design (paper Section 4 + 5.2):
 *   - Each warp handles one "part" (fixed-size neighbor group)
 *   - Neighbor IDs cached in shared memory
 *   - Features accumulated in shared memory, single atomic write-back per part
 *   - dimWorker controls dimension-level parallelism
 *
 * Usage:
 *   ./sag_bench <csr_dir> --M=N [--partSize=N] [--dimWorker=N] [--warpPerBlock=N]
 *               [--iters=N] [--warmup=N] [--gpu=ID] [--csv-only]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cassert>
#include <random>

#include <cuda_runtime.h>

using namespace std;

/* ================================================================
 * CUDA error checking
 * ================================================================ */
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define WARP_SIZE 32

/* ================================================================
 * Custom atomic add (ported from GNNAdvisor)
 * ================================================================ */
__device__ inline void atomicAdd_F(float* address, float value) {
    float old = value;
    while ((old = atomicExch(address, atomicExch(address, 0.0f) + old)) != 0.0f);
}

/* ================================================================
 * SAG Kernel — GPU-side (ported from GNNAdvisor_kernel.cu:187-259)
 * ================================================================ */
template<int M>
__global__ void sag_kernel(
    const int64_t* __restrict__ row_ptr,
    const int*     __restrict__ columns,
    const float*   __restrict__ input,     // [n_verts * M], AoS layout
    float*         __restrict__ output,    // [n_verts * M]
    const int*     __restrict__ partPtr,   // [num_parts+1]
    const int*     __restrict__ part2Node, // [num_parts] 
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock)
{
    /*
    num_parts: 将邻居划分为nums_parts, 每个part的大小为 partSize
    实现思路:
    本质上是一个warp负责一段邻居节点的聚合
    一个warp中的不同thread负责这段邻居节点不同embedding维度的计算
    最后每个thread写回结果到global memory
    */
    unsigned int tid          = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warpId       = tid / WARP_SIZE;
    int block_warpId = threadIdx.x / WARP_SIZE;
    int laneid       = threadIdx.x % WARP_SIZE;

    extern __shared__ int part_meta[];
    int   *partial_ids     = part_meta;
    float *partial_results = (float*)&part_meta[partSize * warpPerBlock];

    if (warpId >= num_parts) return;

    int srcId   = part2Node[warpId];
    int partBeg = partPtr[warpId];  // 一个warp负责一段邻居节点的聚合
    int partEnd = partPtr[warpId + 1];

    // Phase 1: Warp-cooperative load of neighbor IDs → shared memory
    const int pindex_base = block_warpId * partSize;
#pragma unroll
    for (int nidx = partBeg + laneid; nidx < partEnd; nidx += WARP_SIZE) {
        partial_ids[pindex_base + nidx - partBeg] = columns[nidx];
    }
    __syncwarp();


    // Phase 2: Neighbor aggregation — accumulate in shared memory
    const int presult_base = block_warpId * M;
    for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++) {
        int nid = partial_ids[pindex_base + nIdx];
        int nid_base = nid * M;

        if (nIdx == 0) {
            if (laneid < dimWorker) {
#pragma unroll
                for (int d = laneid; d < M; d += dimWorker)
                    partial_results[presult_base + d] = 0.0f;
            }
        }

        if (laneid < dimWorker) {
#pragma unroll
            for (int d = laneid; d < M; d += dimWorker)
                partial_results[presult_base + d] += input[nid_base + d];
        }
    }

    // Phase 3: Write-back from shared memory → global memory (dim atomicAdds)
    if (laneid < dimWorker) {
        int out_base = srcId * M;
#pragma unroll
        for (int d = laneid; d < M; d += dimWorker)
            atomicAdd_F(&output[out_base + d], partial_results[presult_base + d]);
    }
}

/* ================================================================
 * SAG Kernel (nocache variant) — no Phase 1 index caching
 *   Phase 1 removed: reads columns[nidx] directly in aggregation loop
 *   Shared memory: only partial_results (no partial_ids)
 * ================================================================ */
template<int M>
__global__ void sag_kernel_nocache(
    const int64_t* __restrict__ row_ptr,
    const int*     __restrict__ columns,
    const float*   __restrict__ input,
    float*         __restrict__ output,
    const int*     __restrict__ partPtr,
    const int*     __restrict__ part2Node,
    const int num_parts,
    const int dimWorker,
    const int warpPerBlock)
{
    unsigned int tid          = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warpId       = tid / WARP_SIZE;
    int block_warpId = threadIdx.x / WARP_SIZE;
    int laneid       = threadIdx.x % WARP_SIZE;

    extern __shared__ float partial_results[];  // only partial_results, no partial_ids

    if (warpId >= num_parts) return;

    int srcId   = part2Node[warpId];
    int partBeg = partPtr[warpId];
    int partEnd = partPtr[warpId + 1];

    // Aggregation: read columns[] directly, no smem index cache
    const int presult_base = block_warpId * M;
    for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++) {
        int nid = columns[partBeg + nIdx];       // ← direct global read
        int nid_base = nid * M;

        if (nIdx == 0) {
            if (laneid < dimWorker) {
#pragma unroll
                for (int d = laneid; d < M; d += dimWorker)
                    partial_results[presult_base + d] = 0.0f;
            }
        }

        if (laneid < dimWorker) {
#pragma unroll
            for (int d = laneid; d < M; d += dimWorker)
                partial_results[presult_base + d] += input[nid_base + d];
        }
    }

    if (laneid < dimWorker) {
        int out_base = srcId * M;
#pragma unroll
        for (int d = laneid; d < M; d += dimWorker)
            atomicAdd_F(&output[out_base + d], partial_results[presult_base + d]);
    }
}

/* ================================================================
 * Build Part: CPU-side neighbor partitioning (ported from GNNAdvisor.cpp:210-251)
 * ================================================================ */
static void build_part(const vector<int64_t>& row_ptr,
                       int n_verts,
                       int partSize,
                       vector<int>& partPtr,
                       vector<int>& part2Node,
                       int& num_parts)
{
    int64_t total_parts = 0;
    for (int i = 0; i < n_verts; i++) {
        int64_t deg = row_ptr[i + 1] - row_ptr[i];
        int thisNumParts = (deg % partSize == 0) ? (int)(deg / partSize)
                                                  : (int)(deg / partSize) + 1;
        total_parts += thisNumParts;
    }
    if (total_parts >= INT32_MAX) {
        fprintf(stderr, "ERROR: num_parts (%lld) >= INT32_MAX, try larger partSize\n",
                (long long)total_parts);
        exit(1);
    }
    num_parts = (int)total_parts;

    partPtr.resize(num_parts + 1);
    part2Node.resize(num_parts);

    int part_counter = 0;
    for (int i = 0; i < n_verts; i++) {
        int64_t deg = row_ptr[i + 1] - row_ptr[i];
        int thisNumParts = (deg % partSize == 0) ? (int)(deg / partSize)
                                                  : (int)(deg / partSize) + 1;
        for (int pid = 0; pid < thisNumParts; pid++) {
            int64_t partBeg = row_ptr[i] + (int64_t)pid * partSize;
            int64_t partEnd = partBeg + partSize < row_ptr[i + 1]
                                  ? partBeg + partSize
                                  : row_ptr[i + 1];
            partPtr[part_counter]   = (int)partBeg;
            part2Node[part_counter] = i;
            part_counter++;
            if (i == n_verts - 1 && partEnd == row_ptr[i + 1])
                partPtr[part_counter] = (int)partEnd;
        }
    }
}

/* ================================================================
 * Graph I/O (reused from segment_sum_bench_vec.cu)
 * ================================================================ */
static void read_bin_int(const string& fpath, vector<int>& out) {
    ifstream f(fpath, ios::binary | ios::ate);
    if (!f) { cerr << "Cannot open " << fpath << "\n"; exit(1); }
    streamsize sz = f.tellg();
    if (sz % (streamsize)sizeof(int) != 0) {
        cerr << "File not int-aligned: " << fpath << "\n"; exit(1);
    }
    f.seekg(0);
    out.resize(sz / sizeof(int));
    f.read(reinterpret_cast<char*>(out.data()), sz);
}

static void read_graph(const string& path,
                       vector<int64_t>& row_ptr,
                       vector<int>& columns,
                       int& n_verts,
                       int64_t& n_edges) {
    {
        ifstream vf(path + "/csr_vlist.bin", ios::binary | ios::ate);
        if (!vf) { cerr << "Cannot open csr_vlist.bin\n"; exit(1); }
        streamsize vsz = vf.tellg();
        bool use_int64 = false;
        if (vsz % 8 == 0) {
            vf.seekg(0);
            int nv64 = (int)(vsz / 8);
            vector<int64_t> rp64(nv64);
            vf.read(reinterpret_cast<char*>(rp64.data()), vsz);
            int64_t max_val = *max_element(rp64.begin(), rp64.end());
            // Sanity: max_val should not exceed number of edges by absurd amount
            if (max_val > INT32_MAX && max_val < (int64_t)rp64.size() * 1000LL) {
                row_ptr.swap(rp64);
                use_int64 = true;
            }
        }
        if (!use_int64) {
            vf.seekg(0);
            int nv32 = (int)(vsz / sizeof(int));
            vector<int> rp32(nv32);
            vf.read(reinterpret_cast<char*>(rp32.data()), vsz);
            row_ptr.resize(nv32);
            for (int i = 0; i < nv32; i++)
                row_ptr[i] = static_cast<int64_t>(rp32[i]);
        }
    }
    read_bin_int(path + "/csr_elist.bin", columns);
    n_verts = (int)row_ptr.size() - 1;
    n_edges = (int64_t)columns.size();
}

/* ================================================================
 * CPU reference segment sum
 * ================================================================ */
static void cpu_segment_sum(const vector<int64_t>& row_ptr,
                            const vector<int>& columns,
                            const vector<float>& aggData,
                            vector<float>& aggRes,
                            int n_verts, int M) {
    fill(aggRes.begin(), aggRes.end(), 0.0f);
    for (int v = 0; v < n_verts; ++v) {
        for (int64_t e = row_ptr[v]; e < row_ptr[v + 1]; ++e) {
            int nb = columns[e];
            for (int c = 0; c < M; ++c)
                aggRes[v * M + c] += aggData[nb * M + c];
        }
    }
}

/* ================================================================
 * Argument parsing
 * ================================================================ */
struct Args {
    string csr_dir;
    int M            = 4;
    int partSize     = 32;
    int dimWorker    = 32;
    int warpPerBlock = 8;
    int iters        = 30;
    int warmup       = 5;
    int gpu_id       = 0;
    bool csv_only    = false;
    bool no_cache    = false;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <csr_dir> --M=N [--partSize=N] [--dimWorker=N] [--warpPerBlock=N] [--iters=N] [--warmup=N] [--gpu=ID] [--csv-only] [--no-cache]\n", argv[0]);
        exit(1);
    }
    a.csr_dir = argv[1];
    for (int i = 2; i < argc; i++) {
        string s = argv[i];
        if (s.rfind("--M=", 0) == 0)            a.M            = atoi(s.c_str() + 4);
        if (s.rfind("--partSize=", 0) == 0)     a.partSize     = atoi(s.c_str() + 11);
        if (s.rfind("--dimWorker=", 0) == 0)    a.dimWorker    = atoi(s.c_str() + 12);
        if (s.rfind("--warpPerBlock=", 0) == 0) a.warpPerBlock = atoi(s.c_str() + 14);
        if (s.rfind("--iters=", 0) == 0)        a.iters        = atoi(s.c_str() + 8);
        if (s.rfind("--warmup=", 0) == 0)       a.warmup       = atoi(s.c_str() + 9);
        if (s.rfind("--gpu=", 0) == 0)          a.gpu_id       = atoi(s.c_str() + 6);
        if (s == "--csv-only")                  a.csv_only     = true;
        if (s == "--no-cache")                 a.no_cache     = true;
    }
    return a;
}

/* ================================================================
 * Kernel dispatch by M
 * ================================================================ */
template<int M>
static void run_sag(const Args& args,
                    const vector<int64_t>& h_row_ptr,
                    const vector<int>& h_columns,
                    int n_verts, int64_t n_edges) {
    int N = n_verts * M;

    // ── Preprocess: build parts ──
    if (!args.csv_only)
        printf("  Building parts (partSize=%d)...\n", args.partSize);
    vector<int> h_partPtr, h_part2Node;
    int num_parts;
    build_part(h_row_ptr, n_verts, args.partSize, h_partPtr, h_part2Node, num_parts);
    if (!args.csv_only)
        printf("  num_parts = %d (partSize=%d)\n", num_parts, args.partSize);

    // ── Allocate GPU memory ──
    int64_t* d_row_ptr;
    int*     d_columns;
    float*   d_input;     // vertex_data
    float*   d_output;    // aggRes
    int*     d_partPtr;
    int*     d_part2Node;

    CUDA_CHECK(cudaMalloc(&d_row_ptr,  (size_t)(n_verts + 1) * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_columns,  (size_t)n_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_input,    (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output,   (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partPtr,  (size_t)(num_parts + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_part2Node,(size_t)num_parts * sizeof(int)));

    // ── Upload ──
    CUDA_CHECK(cudaMemcpy(d_row_ptr,  h_row_ptr.data(),  (size_t)(n_verts+1)*sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_columns,  h_columns.data(),  (size_t)n_edges*sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_partPtr,  h_partPtr.data(),  (size_t)(num_parts+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_part2Node,h_part2Node.data(),(size_t)num_parts*sizeof(int),     cudaMemcpyHostToDevice));

    // ── Initialize input (random [0,1), fixed seed) ──
    if (!args.csv_only)
        printf("  Initializing M=%d vector data (%zu floats, %.2f MB)...\n",
               M, (size_t)N, (double)N * sizeof(float) / 1e6);
    vector<float> h_input(N);
    {
        mt19937 rng(42);
        uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < N; ++i) h_input[i] = dist(rng);
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    // ── CPU reference ──
    vector<float> cpu_output(N);
    cpu_segment_sum(h_row_ptr, h_columns, h_input, cpu_output, n_verts, M);

    // ── Launch config ──
    int block = args.warpPerBlock * WARP_SIZE;
    int64_t total_warps = (int64_t)num_parts;
    int grid  = (int)((total_warps * WARP_SIZE + block - 1) / block);
    int shared_memory = args.no_cache
        ? args.warpPerBlock * M * sizeof(float)                          // only partial_results
        : args.partSize * args.warpPerBlock * sizeof(int)                // partial_ids + partial_results
          + args.warpPerBlock * M * sizeof(float);

    if (!args.csv_only) {
        printf("  grid=%d, block=%d, shared_memory=%d bytes (%.1f KB)%s\n",
               grid, block, shared_memory, shared_memory / 1024.0,
               args.no_cache ? " [nocache]" : "");
    }

    // ── CUDA events ──
    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    double total_ms = 0.0;
    int measured_iters = 0;
    int total_iters = args.warmup + args.iters;

    // ── Iterations ──
    for (int iter = 0; iter < total_iters; iter++) {
        CUDA_CHECK(cudaMemset(d_output, 0, (size_t)N * sizeof(float)));

        CUDA_CHECK(cudaEventRecord(ev_start));
        if (args.no_cache) {
            sag_kernel_nocache<M><<<grid, block, shared_memory>>>(
                d_row_ptr, d_columns, d_input, d_output,
                d_partPtr, d_part2Node,
                num_parts, args.dimWorker, args.warpPerBlock);
        } else {
            sag_kernel<M><<<grid, block, shared_memory>>>(
                d_row_ptr, d_columns, d_input, d_output,
                d_partPtr, d_part2Node,
                num_parts, args.partSize, args.dimWorker, args.warpPerBlock);
        }
        CUDA_CHECK(cudaEventRecord(ev_end));

        CUDA_CHECK(cudaEventSynchronize(ev_end));
        CUDA_CHECK(cudaGetLastError());

        if (iter >= args.warmup) {
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));
            total_ms += ms;
            measured_iters++;
        }
    }

    double avg_ms = total_ms / measured_iters;

    // ── Correctness ──
    vector<float> gpu_output(N);
    CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));

    double max_abs_err = 0.0, max_rel_err = 0.0;
    double sum_cpu = 0.0, sum_gpu = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = fabs((double)gpu_output[i] - (double)cpu_output[i]);
        if (err > max_abs_err) max_abs_err = err;
        double mag = fabs((double)cpu_output[i]);
        if (mag > 1e-6) { double rel = err / mag; if (rel > max_rel_err) max_rel_err = rel; }
        sum_cpu += cpu_output[i];
        sum_gpu += gpu_output[i];
    }
    bool pass = (max_rel_err < 1e-4);

    if (!args.csv_only) {
        printf("\nResults (SAG kernel, M=%d, %d measured iterations, %d warmup):\n",
               M, measured_iters, args.warmup);
        printf("  partSize=%d, dimWorker=%d, warpPerBlock=%d\n",
               args.partSize, args.dimWorker, args.warpPerBlock);
        printf("  avg sag_kernel   : %.4f ms\n", avg_ms);
        printf("  per-component avg: %.4f ms\n", avg_ms / M);
        printf("  max rel error    : %.6e\n", max_rel_err);
        printf("  CPU sum          : %.6f\n", sum_cpu);
        printf("  GPU sum          : %.6f\n", sum_gpu);
        printf("  PASS: %s\n", pass ? "YES" : "NO (ERROR)");
    }

    printf("\nCSV: %s,SAG,%d,%d,%d,%d,%d,%.4f,%.4f,%.6e\n",
           args.csr_dir.c_str(), M, args.partSize, args.dimWorker, args.warpPerBlock,
           measured_iters, avg_ms, avg_ms / M, max_rel_err);

    // ── Cleanup ──
    cudaFree(d_row_ptr);
    cudaFree(d_columns);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_partPtr);
    cudaFree(d_part2Node);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
}

static void dispatch(const Args& args,
                     const vector<int64_t>& h_row_ptr,
                     const vector<int>& h_columns,
                     int n_verts, int64_t n_edges) {
    switch (args.M) {
        case 1:  run_sag<1> (args, h_row_ptr, h_columns, n_verts, n_edges); break;
        case 2:  run_sag<2> (args, h_row_ptr, h_columns, n_verts, n_edges); break;
        case 4:  run_sag<4> (args, h_row_ptr, h_columns, n_verts, n_edges); break;
        case 8:  run_sag<8> (args, h_row_ptr, h_columns, n_verts, n_edges); break;
        case 16: run_sag<16>(args, h_row_ptr, h_columns, n_verts, n_edges); break;
        case 32: run_sag<32>(args, h_row_ptr, h_columns, n_verts, n_edges); break;
        case 64: run_sag<64>(args, h_row_ptr, h_columns, n_verts, n_edges); break;
        case 80: run_sag<80>(args, h_row_ptr, h_columns, n_verts, n_edges); break;
        case 96: run_sag<96>(args, h_row_ptr, h_columns, n_verts, n_edges); break;
        case 128:run_sag<128>(args, h_row_ptr, h_columns, n_verts, n_edges); break;
        default:
            fprintf(stderr, "Unsupported M=%d. Supported: 1,2,4,8,16,32,64,80,96,128\n", args.M);
            exit(1);
    }
}

/* ================================================================
 * Main
 * ================================================================ */
int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(args.gpu_id));

    vector<int64_t> h_row_ptr;
    vector<int> h_columns;
    int n_verts;
    int64_t n_edges;
    read_graph(args.csr_dir, h_row_ptr, h_columns, n_verts, n_edges);

    if (!args.csv_only) {
        printf("Graph: %s\n", args.csr_dir.c_str());
        printf("  V = %d, E = %lld\n", n_verts, (long long)n_edges);
        printf("  M = %d, partSize = %d, dimWorker = %d, warpPerBlock = %d\n",
               args.M, args.partSize, args.dimWorker, args.warpPerBlock);
    }

    dispatch(args, h_row_ptr, h_columns, n_verts, n_edges);
    return 0;
}
