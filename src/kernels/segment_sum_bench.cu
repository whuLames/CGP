/*
 * segment_sum_bench.cu
 * Standalone single-GPU segment_sum kernel benchmark.
 *
 * Tests the unified fixed-work-per-thread segment_sum kernel (no small/large split)
 * across different CSR graph datasets.
 *
 * Kernel reference: operator.md V1 "均衡线程模型"
 *   Each thread processes a fixed number of edges (vertexNumEachThread).
 *   The first thread of each vertex handles the remainder edges.
 *
 * Usage:
 *   ./segment_sum_bench <csr_dir> [--iters=N] [--warmup=N] [--ept=N] [--gpu=ID] [--csv-only]
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
 * Kernels
 * ================================================================ */

#define BLOCK_SIZE 256

/*
 * Fixed-work-per-thread segment sum kernel.
 * Each thread processes exactly `vertexNumEachThread` edges,
 * except the first thread of each vertex handles the remainder.
 *
 * threadMap[tid]   -> vertex this thread belongs to
 * threadStart[tid]  -> first tid assigned to that vertex
 */
__global__ void segment_sum_fixed_work_kernel(
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
        // First thread: handle remainder (first_step edges)
        start = seg_start;
        end   = seg_start + first_step;
    } else {
        // Subsequent threads: handle exactly vertexNumEachThread edges
        start = seg_start + first_step + (int64_t)(tid - threadStart[tid] - 1) * vertexNumEachThread;
        end   = start + vertexNumEachThread;
    }

    float sum = 0.0f;
    while (start < end) {
        sum += aggData[columns[start]];
        ++start;
    }
    atomicAdd(&aggRes[node], sum);
}

/*
 * PageRank update kernel.
 *   new_pr[v] = teleport + damping * aggRes[v]
 *   vertex_data[v] = new_pr[v] / out_degree[v]
 */
__global__ void pagerank_update_kernel(
    float*       __restrict__ pr,
    float*       __restrict__ vertex_data,
    const float* __restrict__ aggRes,
    const int*   __restrict__ out_degrees,
    float teleport,
    float damping,
    int n_verts)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_verts) return;

    float new_pr = teleport + damping * aggRes[v];
    pr[v] = new_pr;

    int deg = out_degrees[v];
    vertex_data[v] = (deg > 0) ? (new_pr / (float)deg) : 0.0f;
}

/* ================================================================
 * Graph I/O (int32/int64 auto-detect, ported from V7)
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
    // Auto-detect int32 vs int64 vlist format
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
 * Build thread mapping (supports int64 degrees)
 * ================================================================ */
static void build_thread_mapping(const vector<int64_t>& row_ptr,
                                 int n_verts,
                                 int ept,  // edges per thread
                                 vector<int>& threadMap,
                                 vector<int>& threadStart,
                                 int& numThread)
{
    // First pass: count total threads
    int64_t total = 0;
    for (int v = 0; v < n_verts; v++) {
        int64_t deg = row_ptr[v + 1] - row_ptr[v];
        int nthreads = (deg > 0) ? (int)((deg + ept - 1) / ept) : 0;
        total += nthreads;
    }
    assert(total < INT32_MAX && "numThread exceeds INT32_MAX");
    numThread = (int)total;

    threadMap.resize(numThread);
    threadStart.resize(numThread);

    // Second pass: fill
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
 * Argument parsing
 * ================================================================ */
struct Args {
    string csr_dir;
    int    iters   = 30;
    int    warmup  = 5;
    int    ept     = 32;   // edges per thread
    int    gpu_id  = 0;
    bool   csv_only = false;
};

static Args parse_args(int argc, char** argv)
{
    Args a;
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <csr_dir> [--iters=N] [--warmup=N] [--ept=N] [--gpu=ID] [--csv-only]\n", argv[0]);
        exit(1);
    }
    a.csr_dir = argv[1];
    for (int i = 2; i < argc; i++) {
        string s = argv[i];
        if (s.rfind("--iters=", 0) == 0)   a.iters    = atoi(s.c_str() + 8);
        if (s.rfind("--warmup=", 0) == 0)  a.warmup   = atoi(s.c_str() + 9);
        if (s.rfind("--ept=", 0) == 0)     a.ept      = atoi(s.c_str() + 6);
        if (s.rfind("--gpu=", 0) == 0)     a.gpu_id   = atoi(s.c_str() + 6);
        if (s == "--csv-only")             a.csv_only = true;
    }
    return a;
}

/* ================================================================
 * Main
 * ================================================================ */
int main(int argc, char** argv)
{
    Args args = parse_args(argc, argv);

    CUDA_CHECK(cudaSetDevice(args.gpu_id));

    // -- Read graph --
    vector<int64_t> h_row_ptr;
    vector<int> h_columns;
    int n_verts;
    int64_t n_edges;
    read_graph(args.csr_dir, h_row_ptr, h_columns, n_verts, n_edges);

    if (!args.csv_only) {
        printf("Graph: %s\n", args.csr_dir.c_str());
        printf("  V = %d, E = %lld\n", n_verts, (long long)n_edges);
        printf("  ept = %d, iters = %d, warmup = %d\n", args.ept, args.iters, args.warmup);
    }

    // -- Compute out-degrees --
    vector<int> h_out_degrees(n_verts);
    for (int v = 0; v < n_verts; v++) {
        int64_t deg = h_row_ptr[v + 1] - h_row_ptr[v];
        h_out_degrees[v] = (int)deg;
    }

    // -- Build thread mapping --
    vector<int> h_threadMap, h_threadStart;
    int numThread;
    build_thread_mapping(h_row_ptr, n_verts, args.ept,
                         h_threadMap, h_threadStart, numThread);
    if (!args.csv_only) {
        printf("  numThread = %d (%.2fx edges)\n", numThread, (double)numThread / n_edges);
    }

    // -- Allocate GPU memory --
    int64_t* d_row_ptr;
    int*     d_columns;
    int*     d_out_degrees;
    float*   d_vertex_data;
    float*   d_aggRes;
    float*   d_pr;
    int*     d_threadMap;
    int*     d_threadStart;

    CUDA_CHECK(cudaMalloc(&d_row_ptr,     (size_t)(n_verts + 1) * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_columns,     (size_t)n_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_degrees, (size_t)n_verts * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vertex_data, (size_t)n_verts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_aggRes,      (size_t)n_verts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pr,          (size_t)n_verts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_threadMap,   (size_t)numThread * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_threadStart, (size_t)numThread * sizeof(int)));

    // -- Copy to GPU --
    CUDA_CHECK(cudaMemcpy(d_row_ptr,     h_row_ptr.data(),     (size_t)(n_verts + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_columns,     h_columns.data(),     (size_t)n_edges * sizeof(int),          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_degrees, h_out_degrees.data(), (size_t)n_verts * sizeof(int),          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_threadMap,   h_threadMap.data(),   (size_t)numThread * sizeof(int),        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_threadStart, h_threadStart.data(), (size_t)numThread * sizeof(int),        cudaMemcpyHostToDevice));

    // -- Initialize PR --
    float damping  = 0.85f;
    float init_pr  = 1.0f / (float)n_verts;
    float teleport = (1.0f - damping) / (float)n_verts;

    vector<float> h_vertex_data(n_verts);
    vector<float> h_pr(n_verts, init_pr);
    for (int v = 0; v < n_verts; v++) {
        int deg = h_out_degrees[v];
        h_vertex_data[v] = (deg > 0) ? (init_pr / (float)deg) : 0.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_vertex_data, h_vertex_data.data(), (size_t)n_verts * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pr,          h_pr.data(),          (size_t)n_verts * sizeof(float), cudaMemcpyHostToDevice));

    // -- CUDA events for timing --
    cudaEvent_t ev_seg_start, ev_seg_end, ev_upd_start, ev_upd_end;
    CUDA_CHECK(cudaEventCreate(&ev_seg_start));
    CUDA_CHECK(cudaEventCreate(&ev_seg_end));
    CUDA_CHECK(cudaEventCreate(&ev_upd_start));
    CUDA_CHECK(cudaEventCreate(&ev_upd_end));

    int grid_seg = (numThread + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_upd = (n_verts  + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double total_seg_ms = 0.0;
    double total_upd_ms = 0.0;
    int    measured_iters = 0;

    int total_iters = args.warmup + args.iters;

    // -- PageRank iterations --
    for (int iter = 0; iter < total_iters; iter++) {
        CUDA_CHECK(cudaMemset(d_aggRes, 0, (size_t)n_verts * sizeof(float)));

        // Segment sum kernel
        CUDA_CHECK(cudaEventRecord(ev_seg_start));
        segment_sum_fixed_work_kernel<<<grid_seg, BLOCK_SIZE>>>(
            d_row_ptr, d_columns, d_vertex_data, d_aggRes,
            d_threadMap, d_threadStart,
            numThread, args.ept);
        CUDA_CHECK(cudaEventRecord(ev_seg_end));

        // PR update kernel
        CUDA_CHECK(cudaEventRecord(ev_upd_start));
        pagerank_update_kernel<<<grid_upd, BLOCK_SIZE>>>(
            d_pr, d_vertex_data, d_aggRes, d_out_degrees,
            teleport, damping, n_verts);
        CUDA_CHECK(cudaEventRecord(ev_upd_end));

        CUDA_CHECK(cudaEventSynchronize(ev_upd_end));

        if (iter >= args.warmup) {
            float seg_ms = 0, upd_ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&seg_ms, ev_seg_start, ev_seg_end));
            CUDA_CHECK(cudaEventElapsedTime(&upd_ms, ev_upd_start, ev_upd_end));
            total_seg_ms += seg_ms;
            total_upd_ms += upd_ms;
            measured_iters++;
        }
    }

    double avg_seg_ms = total_seg_ms / measured_iters;
    double avg_upd_ms = total_upd_ms / measured_iters;
    double avg_tot_ms = avg_seg_ms + avg_upd_ms;

    // -- Verify PR sum --
    vector<float> h_pr_result(n_verts);
    CUDA_CHECK(cudaMemcpy(h_pr_result.data(), d_pr, (size_t)n_verts * sizeof(float), cudaMemcpyDeviceToHost));
    double pr_sum = 0.0;
    for (int v = 0; v < n_verts; v++) pr_sum += h_pr_result[v];

    if (!args.csv_only) {
        printf("\nResults (%d measured iterations, %d warmup):\n", measured_iters, args.warmup);
        printf("  avg segment_sum : %.4f ms\n", avg_seg_ms);
        printf("  avg pr_update   : %.4f ms\n", avg_upd_ms);
        printf("  avg total/iter  : %.4f ms\n", avg_tot_ms);
        printf("  PR sum          : %.6f (expected ~1.0)\n", pr_sum);
    }

    // CSV-friendly output
    printf("\nCSV: %s,%.4f,%.4f,%.4f,%.6f\n",
           args.csr_dir.c_str(), avg_seg_ms, avg_upd_ms, avg_tot_ms, pr_sum);

    // -- Cleanup --
    cudaFree(d_row_ptr);
    cudaFree(d_columns);
    cudaFree(d_out_degrees);
    cudaFree(d_vertex_data);
    cudaFree(d_aggRes);
    cudaFree(d_pr);
    cudaFree(d_threadMap);
    cudaFree(d_threadStart);
    cudaEventDestroy(ev_seg_start);
    cudaEventDestroy(ev_seg_end);
    cudaEventDestroy(ev_upd_start);
    cudaEventDestroy(ev_upd_end);

    return 0;
}
