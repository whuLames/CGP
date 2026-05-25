/*
 * segment_sum_bench_vec.cu
 * Standalone single-GPU benchmark for vectorized segment_sum kernel.
 *
 * Extends the scalar fixed-work-per-thread kernel to support M-component
 * vectors. Each vertex holds M floats; segment_sum reduces them independently.
 *
 * Memory layout: AoS (interleaved) — data[v * M + c] for component c of vertex v.
 *
 * Usage:
 *   ./segment_sum_bench_vec <csr_dir> --M=N [--iters=N] [--warmup=N] [--ept=N] [--gpu=ID] [--csv-only]
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

/* ================================================================
 * Vectorized segment_sum kernel
 * ================================================================ */

#define BLOCK_SIZE 256

/*
 * Template vector segment_sum kernel.
 * M = vector length per vertex (compile-time constant for unrolling).
 *
 * Each thread processes fixed vertexNumEachThread edges.
 * aggData and aggRes are laid out as AoS: [v*M + c].
 */
template<int M>
__global__ void segment_sum_fixed_work_kernel_vec(
    const int64_t* __restrict__ rowPtr,
    const int*     __restrict__ columns,
    const float*   __restrict__ aggData,
    float*         __restrict__ aggRes,
    const int*     __restrict__ threadMap,
    const int*     __restrict__ threadStart,
    int numThread,
    int vertexNumEachThread)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numThread) return;

    int node  = threadMap[tid];
    int64_t seg_start = rowPtr[node];
    int64_t seg_end   = rowPtr[node + 1];
    int64_t degree    = seg_end - seg_start;

    int first_step = (int)(degree % vertexNumEachThread);
    if (first_step == 0)
        first_step = vertexNumEachThread;

    int64_t start, end;
    if (tid == threadStart[tid]) {
        start = seg_start;
        end   = seg_start + first_step;
    } else {
        start = seg_start + first_step + (int64_t)(tid - threadStart[tid] - 1) * vertexNumEachThread;
        end   = start + vertexNumEachThread;
    }

    // M accumulator registers
    float sum[M];
#pragma unroll
    for (int c = 0; c < M; ++c)
        sum[c] = 0.0f;

    // Edge traversal: for each edge, accumulate all M components
    while (start < end) {
        int col = columns[start];
        int base = col * M;
#pragma unroll
        for (int c = 0; c < M; ++c)
            sum[c] += aggData[base + c];
        ++start;
    }

    // Atomic write-back: M independent atomicAdds
    int out_base = node * M;
#pragma unroll
    for (int c = 0; c < M; ++c)
        atomicAdd(&aggRes[out_base + c], sum[c]);
}

/*
 * Simple kernel to initialize aggRes to zero.
 */
__global__ void memset_vec_kernel(float* aggRes, int n_verts, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_verts * M;
    if (i >= total) return;
    aggRes[i] = 0.0f;
}

/* ================================================================
 * Graph I/O (int32/int64 auto-detect, same as scalar version)
 * ================================================================ */
static void read_bin_int(const string& fpath, vector<int>& out)
{
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
                       int64_t& n_edges)
{
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
 * Build thread mapping
 * ================================================================ */
static void build_thread_mapping(const vector<int64_t>& row_ptr,
                                 int n_verts,
                                 int ept,
                                 vector<int>& threadMap,
                                 vector<int>& threadStart,
                                 int& numThread)
{
    int64_t total = 0;
    for (int v = 0; v < n_verts; v++) {
        int64_t deg = row_ptr[v + 1] - row_ptr[v];
        int nthreads = (deg > 0) ? (int)((deg + ept - 1) / ept) : 0;
        total += nthreads;
    }
    if (total >= INT32_MAX) {
        fprintf(stderr, "ERROR: numThread (%lld) >= INT32_MAX, try larger --ept\n", (long long)total);
        exit(1);
    }
    numThread = (int)total;

    threadMap.resize(numThread);
    threadStart.resize(numThread);

    int tid = 0;
    for (int v = 0; v < n_verts; v++) {
        int64_t deg = row_ptr[v + 1] - row_ptr[v];
        if (deg == 0) continue;
        int nthreads = (int)((deg + ept - 1) / ept);
        int first_tid = tid;
        for (int t = 0; t < nthreads; t++) {
            threadMap[tid]   = v;
            threadStart[tid] = first_tid;
            tid++;
        }
    }
}

/* ================================================================
 * CPU reference: segment_sum for correctness verification
 * ================================================================ */
static void cpu_segment_sum(const vector<int64_t>& row_ptr,
                            const vector<int>& columns,
                            const vector<float>& aggData,
                            vector<float>& aggRes,
                            int n_verts,
                            int M)
{
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
    int    M       = 4;
    int    iters   = 30;
    int    warmup  = 5;
    int    ept     = 32;
    int    gpu_id  = 0;
    bool   csv_only = false;
};

static Args parse_args(int argc, char** argv)
{
    Args a;
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <csr_dir> --M=N [--iters=N] [--warmup=N] [--ept=N] [--gpu=ID] [--csv-only]\n", argv[0]);
        exit(1);
    }
    a.csr_dir = argv[1];
    for (int i = 2; i < argc; i++) {
        string s = argv[i];
        if (s.rfind("--M=", 0) == 0)        a.M        = atoi(s.c_str() + 4);
        if (s.rfind("--iters=", 0) == 0)    a.iters    = atoi(s.c_str() + 8);
        if (s.rfind("--warmup=", 0) == 0)   a.warmup   = atoi(s.c_str() + 9);
        if (s.rfind("--ept=", 0) == 0)      a.ept      = atoi(s.c_str() + 6);
        if (s.rfind("--gpu=", 0) == 0)      a.gpu_id   = atoi(s.c_str() + 6);
        if (s == "--csv-only")              a.csv_only = true;
    }
    return a;
}

/* ================================================================
 * Kernel dispatch by M (template instantiation)
 * ================================================================ */
template<int M>
static void run_segment_sum(const Args& args,
                             const vector<int64_t>& h_row_ptr,
                             const vector<int>& h_columns,
                             int n_verts, int64_t n_edges,
                             const vector<int>& h_threadMap,
                             const vector<int>& h_threadStart,
                             int numThread)
{
    int N = n_verts * M;

    // ── Allocate GPU memory ──
    int64_t* d_row_ptr;
    int*     d_columns;
    float*   d_vertex_data;
    float*   d_aggRes;
    int*     d_threadMap;
    int*     d_threadStart;

    CUDA_CHECK(cudaMalloc(&d_row_ptr,     (size_t)(n_verts + 1) * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_columns,     (size_t)n_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vertex_data, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_aggRes,      (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_threadMap,   (size_t)numThread * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_threadStart, (size_t)numThread * sizeof(int)));

    // ── Copy to GPU ──
    CUDA_CHECK(cudaMemcpy(d_row_ptr,     h_row_ptr.data(),     (size_t)(n_verts + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_columns,     h_columns.data(),     (size_t)n_edges * sizeof(int),          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_threadMap,   h_threadMap.data(),   (size_t)numThread * sizeof(int),        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_threadStart, h_threadStart.data(), (size_t)numThread * sizeof(int),        cudaMemcpyHostToDevice));

    // ── Initialize vector data (random values in [0,1)) ──
    if (!args.csv_only)
        printf("  Initializing M=%d vector data (%zu floats, %.2f MB)...\n", M, (size_t)N, (double)N * sizeof(float) / 1e6);

    vector<float> h_vertex_data(N);
    {
        mt19937 rng(42);  // fixed seed for reproducibility
        uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < N; ++i)
            h_vertex_data[i] = dist(rng);
    }
    CUDA_CHECK(cudaMemcpy(d_vertex_data, h_vertex_data.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    // ── CPU reference result (one-time computation) ──
    vector<float> cpu_aggRes(N);
    cpu_segment_sum(h_row_ptr, h_columns, h_vertex_data, cpu_aggRes, n_verts, M);

    // ── CUDA events for timing ──
    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    int grid_seg = (numThread + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_clr = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double total_ms = 0.0;
    int    measured_iters = 0;
    int    total_iters = args.warmup + args.iters;

    // ── Iterations ──
    for (int iter = 0; iter < total_iters; iter++) {
        // Clear aggRes
        memset_vec_kernel<<<grid_clr, BLOCK_SIZE>>>(d_aggRes, n_verts, M);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(ev_start));
        segment_sum_fixed_work_kernel_vec<M><<<grid_seg, BLOCK_SIZE>>>(
            d_row_ptr, d_columns, d_vertex_data, d_aggRes,
            d_threadMap, d_threadStart,
            numThread, args.ept);
        CUDA_CHECK(cudaEventRecord(ev_end));

        CUDA_CHECK(cudaEventSynchronize(ev_end));

        if (iter >= args.warmup) {
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));
            total_ms += ms;
            measured_iters++;
        }
    }

    double avg_ms = total_ms / measured_iters;

    // ── Correctness verification ──
    vector<float> gpu_aggRes(N);
    CUDA_CHECK(cudaMemcpy(gpu_aggRes.data(), d_aggRes, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));

    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    double sum_cpu = 0.0, sum_gpu = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = fabs((double)gpu_aggRes[i] - (double)cpu_aggRes[i]);
        if (err > max_abs_err) max_abs_err = err;
        double mag = fabs((double)cpu_aggRes[i]);
        if (mag > 1e-6) {
            double rel = err / mag;
            if (rel > max_rel_err) max_rel_err = rel;
        }
        sum_cpu += cpu_aggRes[i];
        sum_gpu += gpu_aggRes[i];
    }

    // Pass criterion: relative error < 1e-4 (float atomics order-dependent)
    bool pass = (max_rel_err < 1e-4);

    if (!args.csv_only) {
        printf("\nResults (M=%d, %d measured iterations, %d warmup):\n", M, measured_iters, args.warmup);
        printf("  avg segment_sum_vec : %.4f ms\n", avg_ms);
        printf("  per-component avg   : %.4f ms (total / M)\n", avg_ms / M);
        printf("  max abs error       : %.6e\n", max_abs_err);
        printf("  max rel error       : %.6e\n", max_rel_err);
        printf("  CPU sum             : %.6f\n", sum_cpu);
        printf("  GPU sum             : %.6f\n", sum_gpu);
        printf("  PASS: %s\n", pass ? "YES" : "NO (ERROR)");
    }

    // CSV output: dataset,M,iters,avg_segsum_ms,per_comp_ms,max_err
    printf("\nCSV: %s,%d,%d,%.4f,%.4f,%.6e\n",
           args.csr_dir.c_str(), M, measured_iters, avg_ms, avg_ms / M, max_rel_err);

    // ── Cleanup ──
    cudaFree(d_row_ptr);
    cudaFree(d_columns);
    cudaFree(d_vertex_data);
    cudaFree(d_aggRes);
    cudaFree(d_threadMap);
    cudaFree(d_threadStart);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
}

// Dispatch table: instantiate for common M values
static void dispatch(const Args& args,
                     const vector<int64_t>& h_row_ptr,
                     const vector<int>& h_columns,
                     int n_verts, int64_t n_edges,
                     const vector<int>& h_threadMap,
                     const vector<int>& h_threadStart,
                     int numThread)
{
    switch (args.M) { // 这里不应该使用switch语句，因为M的值可能是连续的你懂吗
        case 1:  run_segment_sum<1> (args, h_row_ptr, h_columns, n_verts, n_edges, h_threadMap, h_threadStart, numThread); break;
        case 2:  run_segment_sum<2> (args, h_row_ptr, h_columns, n_verts, n_edges, h_threadMap, h_threadStart, numThread); break;
        case 4:  run_segment_sum<4> (args, h_row_ptr, h_columns, n_verts, n_edges, h_threadMap, h_threadStart, numThread); break;
        case 8:  run_segment_sum<8> (args, h_row_ptr, h_columns, n_verts, n_edges, h_threadMap, h_threadStart, numThread); break;
        case 16: run_segment_sum<16>(args, h_row_ptr, h_columns, n_verts, n_edges, h_threadMap, h_threadStart, numThread); break;
        case 32: run_segment_sum<32>(args, h_row_ptr, h_columns, n_verts, n_edges, h_threadMap, h_threadStart, numThread); break;
        case 64: run_segment_sum<64>(args, h_row_ptr, h_columns, n_verts, n_edges, h_threadMap, h_threadStart, numThread); break;
        case 80: run_segment_sum<80>(args, h_row_ptr, h_columns, n_verts, n_edges, h_threadMap, h_threadStart, numThread); break;
        case 96: run_segment_sum<96>(args, h_row_ptr, h_columns, n_verts, n_edges, h_threadMap, h_threadStart, numThread); break;
        case 128:run_segment_sum<128>(args, h_row_ptr, h_columns, n_verts, n_edges, h_threadMap, h_threadStart, numThread); break;
        default:
            fprintf(stderr, "Unsupported M=%d. Supported: 1,2,4,8,16,32,64,80,96,128\n", args.M);
            exit(1);
    }
}

/* ================================================================
 * Main
 * ================================================================ */
int main(int argc, char** argv)
{
    Args args = parse_args(argc, argv);

    CUDA_CHECK(cudaSetDevice(args.gpu_id));

    // ── Read graph ──
    vector<int64_t> h_row_ptr;
    vector<int> h_columns;
    int n_verts;
    int64_t n_edges;
    read_graph(args.csr_dir, h_row_ptr, h_columns, n_verts, n_edges);

    if (!args.csv_only) {
        printf("Graph: %s\n", args.csr_dir.c_str());
        printf("  V = %d, E = %lld\n", n_verts, (long long)n_edges);
        printf("  M = %d, ept = %d, iters = %d, warmup = %d\n", args.M, args.ept, args.iters, args.warmup);
    }

    // ── Build thread mapping ──
    vector<int> h_threadMap, h_threadStart;
    int numThread;
    build_thread_mapping(h_row_ptr, n_verts, args.ept,
                         h_threadMap, h_threadStart, numThread);

    if (!args.csv_only) {
        printf("  numThread = %d (%.2fx edges)\n", numThread, (double)numThread / n_edges);
        printf("  vector footprint: %zu floats = %.2f MB\n",
               (size_t)n_verts * args.M, (double)(n_verts * args.M) * sizeof(float) / 1e6);
    }

    // ── Run (template dispatch) ──
    dispatch(args, h_row_ptr, h_columns, n_verts, n_edges,
             h_threadMap, h_threadStart, numThread);

    return 0;
}
