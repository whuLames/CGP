/*
 * ge_spmm_bench.cu
 * Standalone single-GPU benchmark for GE-SpMM (SC'20) kernel.
 *
 * Ported from GE-SpMM (Huang et al., SC'20): warp-centric SpMM with
 * shared-memory column-index caching and adaptive column-fetching
 * (thread coarsening via multiple accumulators).
 *
 * Adaptations for TCRGraph:
 *   - int64_t row_ptr for large graph support
 *   - No edge values (all weights = 1), removing A_csrVal entirely
 *   - Template dispatch for M in {1,2,4,8,16,32,64,80,96,128}
 *
 * SpMM <-> GNN Aggregation / Vectorized PageRank:
 *   C[i][j] = sum_k A[i][k] * B[k][j]           (SpMM)
 *   output[v][c] = sum_{u in N(v)} input[u][c]   (GNN aggregation)
 *   r_new[v][c] = sum_{u->v} r[u][c]/outdeg[u]   (vectorized PageRank)
 *
 * Usage:
 *   ./ge_spmm_bench <csr_dir> --M=N [--tile-row=N] [--iters=N] [--warmup=N]
 *                    [--gpu=ID] [--mode=full|load_only|graph_only]
 *                    [--sm-count=N] [--skip-verify] [--csv-only]
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
#include <random>

#include <cuda_runtime.h>
#include <cuda.h>

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

#define CU_CHECK(call)                                                         \
    do {                                                                        \
        CUresult err = (call);                                                  \
        if (err != CUDA_SUCCESS) {                                              \
            const char* err_name = nullptr;                                     \
            const char* err_str = nullptr;                                      \
            cuGetErrorName(err, &err_name);                                     \
            cuGetErrorString(err, &err_str);                                    \
            fprintf(stderr, "CUDA driver error at %s:%d: %s (%s)\n",           \
                    __FILE__, __LINE__,                                         \
                    err_name ? err_name : "unknown",                           \
                    err_str ? err_str : "unknown");                            \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

enum BenchMode {
    MODE_FULL = 0,
    MODE_LOAD_ONLY = 1,
    MODE_GRAPH_ONLY = 2
};

static const char* mode_name(BenchMode mode) {
    switch (mode) {
        case MODE_FULL:       return "full";
        case MODE_LOAD_ONLY:  return "load_only";
        case MODE_GRAPH_ONLY: return "graph_only";
        default:              return "unknown";
    }
}

__device__ __forceinline__ void consume_float(float v) {
    asm volatile("" :: "f"(v));
}

__device__ __forceinline__ void consume_int(int v) {
    asm volatile("" :: "r"(v));
}

/* ================================================================
 * GE-SpMM Simple Kernel (M < 32)
 *
 * No shared memory. Each thread handles one (row, column) output element.
 * Block layout: dim3(M, tile_row) — blockDim.x = M columns per row.
 * ================================================================ */
template<int M, BenchMode MODE>
__global__ void ge_spmm_simple(
    int A_nrows,
    const int64_t* __restrict__ A_csrRowPtr,
    const int*     __restrict__ A_csrColInd,
    const float*   __restrict__ B_dnVal,     // [A_ncols * M], row-major
    float*         __restrict__ C_dnVal,     // [A_nrows * M], row-major
    int tile_row)
{
    /*
    tile_row: 看起来是对row切分，一个block负责特定的row
    */

    // 这个看起来就是, 一个thread负责一个节点的某个维度上的计算，在维度上的计算不需要做原子化操作
    // 但有个问题是, 本质上还是一个thread对应一个节点，这样的话，不同节点之间的workload差异很大，这里是不是存在进一步的优化空间呢？
    // 更进一步, 什么metric能够进一步的解决这个问题

    int rid = tile_row * blockIdx.x + threadIdx.y; // the continuous threadIdx.x threads one dimensional computation
    if (rid >= A_nrows) return;

    int cid = threadIdx.x;
    int64_t lb = A_csrRowPtr[rid];
    int64_t hb = A_csrRowPtr[rid + 1];

    float acc = 0.0f;
    for (int64_t ptr = lb; ptr < hb; ptr++) {
        int col = A_csrColInd[ptr];
        if constexpr (MODE == MODE_GRAPH_ONLY) {
            consume_int(col);
        } else {
            float val = B_dnVal[(int64_t)col * M + cid];
            if constexpr (MODE == MODE_FULL) {
                acc += val;
            } else {
                consume_float(val);
            }
        }
    }
    C_dnVal[(int64_t)rid * M + cid] = acc;
}

/* ================================================================
 * GE-SpMM Shared-Memory Kernel (M >= 32)
 *
 * Warp-cooperative: 32 threads per row, column indices cached in smem.
 * CF (coarsening factor) = ceil(M/32): each thread handles CF output
 * columns in separate accumulators, exposing ILP and amortizing smem
 * broadcast cost.
 *
 * Block layout: dim3(32, tile_row).
 * Shared memory: 32 * tile_row * sizeof(int) for column index cache.
 * ================================================================ */
template<int M, int CF, BenchMode MODE>
__global__ void ge_spmm_smem(
    int A_nrows,
    const int64_t* __restrict__ A_csrRowPtr,
    const int*     __restrict__ A_csrColInd,
    const float*   __restrict__ B_dnVal,
    float*         __restrict__ C_dnVal,
    int tile_row)
{
    extern __shared__ int colInd_sh[];
    int shmem_offset = threadIdx.y << 5;              // threadIdx.y * 32
    int thread_idx   = shmem_offset + threadIdx.x;   // thread idx for the current block

    int rid = tile_row * blockIdx.x + threadIdx.y;  // 依旧是一个warp负责一行(一个节点)，上面是blockDim.x个thread负责一行

    // 每个线程负责间隔warpSize的embedding的更新
    // 这样的话，本质上每个thread的工作量是尽可能相近的, 但是warp之间的工作量差距很大，但这个本质上不会引起imbalanced的问题, 因为我就是以warp为角度调度的，或者说
    // 我调度的基本单位是一个指令，根本不会出现不均衡的情况, 大不了工作量大的warp多调度几次，本质上不会造成资源的浪费
    // 如果我按照naive的实现方式，一个thread处理指定数目的元素，那么其实我每个thread的workload很大，这样的话，如果在一个warp内的workload差异很大，一次warp调度本质上只有个别线程在计算，这样的话warp调度的次数会更多
    // 这个理解需要记录下来

    if (rid >= A_nrows) return;

    // Column base: blockIdx.y * (CF * 32) + threadIdx.x
    // 如何理解这里的cid?
    int cid = (blockIdx.y * (CF << 5)) + threadIdx.x; // cid 指的是embedding 维度上的索引，或者说是起始索引,因为一个thread要处理 M 的多个维度
    int64_t lb  = A_csrRowPtr[rid];
    int64_t hb  = A_csrRowPtr[rid + 1];
    int64_t ptr = lb + threadIdx.x;

    float acc[CF];
    #pragma unroll
    for (int c = 0; c < CF; c++) acc[c] = 0.0f;

    if (blockIdx.y != gridDim.y - 1) {  // 不是最后一个block
        // ---- Fast path: all CF accumulators valid ----
        for (int64_t jj = lb; jj < hb; jj += 32) { // 这个循环是否不必须为32
            if (ptr < hb) {
                // A_csrColInd[ptr] 是邻居节点的id， * M 代表这个邻居节点的embedding数据的内存地址
                int col = A_csrColInd[ptr];
                colInd_sh[thread_idx] = col * M; // 这里为什么乘M ？
                if constexpr (MODE == MODE_GRAPH_ONLY) {
                    consume_int(col);
                }
            }
            __syncwarp();
            ptr += 32;  // ptr指向当前thread处理的邻居节点数目

            if constexpr (MODE != MODE_GRAPH_ONLY) {
                for (int kk = 0; kk < 32 && jj + kk < hb; kk++) {   // 每个线程都遍历这32条边，但每个线程负责的区域不同
                    int offset = colInd_sh[shmem_offset + kk] + cid;
                    #pragma unroll
                    for (int c = 0; c < CF; c++) {
                        float val = B_dnVal[offset + c * 32];
                        if constexpr (MODE == MODE_FULL) {
                            acc[c] += val;
                        } else {
                            consume_float(val);
                        }
                    }
                }
            }
            __syncwarp();
        }

        int out_base = rid * M + cid;
        #pragma unroll
        for (int c = 0; c < CF; c++) {
            C_dnVal[out_base + c * 32] = acc[c]; // 这里不用原子操作的本质原因是因为我们上面做了同步，每次写回32条边的结果
        }

        // 这里是否可以先不写回数据？等到外层循环全部做完再写回？
    } else {
        // ---- Boundary block: some accumulators may be invalid ----
        int nout = (M - cid + 31) / 32;
        if (nout < 0)  nout = 0;
        if (nout > CF) nout = CF;

        for (int64_t jj = lb; jj < hb; jj += 32) {
            if (ptr < hb) {
                int col = A_csrColInd[ptr];
                colInd_sh[thread_idx] = col * M;
                if constexpr (MODE == MODE_GRAPH_ONLY) {
                    consume_int(col);
                }
            }
            __syncwarp();
            ptr += 32;

            if constexpr (MODE != MODE_GRAPH_ONLY) {
                for (int kk = 0; kk < 32 && jj + kk < hb; kk++) {
                    int offset = colInd_sh[shmem_offset + kk] + cid;
                    #pragma unroll
                    for (int c = 0; c < CF; c++) {
                        if (c < nout) {
                            float val = B_dnVal[offset + c * 32];
                            if constexpr (MODE == MODE_FULL) {
                                acc[c] += val;
                            } else {
                                consume_float(val);
                            }
                        }
                    }
                }
            }
            __syncwarp();
        }

        int out_base = rid * M + cid;
        #pragma unroll
        for (int c = 0; c < CF; c++) {
            if (c < nout) {
                C_dnVal[out_base + c * 32] = acc[c];
            }
        }
    }
}

/* ================================================================
 * Graph I/O
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
 * CPU reference: segment sum for correctness verification
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
    int tile_row     = 8;
    int iters        = 30;
    int warmup       = 5;
    int gpu_id       = 0;
    int sm_count     = 0;
    BenchMode mode   = MODE_FULL;
    bool skip_verify = false;
    bool csv_only    = false;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <csr_dir> --M=N [--tile-row=N] [--iters=N] "
            "[--warmup=N] [--gpu=ID] [--mode=full|load_only|graph_only] "
            "[--sm-count=N] [--skip-verify] [--csv-only]\n", argv[0]);
        exit(1);
    }
    a.csr_dir = argv[1];
    for (int i = 2; i < argc; i++) {
        string s = argv[i];
        if (s.rfind("--M=", 0) == 0)           a.M        = atoi(s.c_str() + 4);
        if (s.rfind("--tile-row=", 0) == 0)    a.tile_row = atoi(s.c_str() + 11);
        if (s.rfind("--iters=", 0) == 0)       a.iters    = atoi(s.c_str() + 8);
        if (s.rfind("--warmup=", 0) == 0)      a.warmup   = atoi(s.c_str() + 9);
        if (s.rfind("--gpu=", 0) == 0)         a.gpu_id   = atoi(s.c_str() + 6);
        if (s.rfind("--sm-count=", 0) == 0)    a.sm_count = atoi(s.c_str() + 11);
        if (s.rfind("--mode=", 0) == 0) {
            string mode = s.substr(7);
            if (mode == "full") {
                a.mode = MODE_FULL;
            } else if (mode == "load_only") {
                a.mode = MODE_LOAD_ONLY;
            } else if (mode == "graph_only") {
                a.mode = MODE_GRAPH_ONLY;
            } else {
                fprintf(stderr, "Unsupported mode=%s. Supported: full,load_only,graph_only\n",
                        mode.c_str());
                exit(1);
            }
        }
        if (s == "--skip-verify")             a.skip_verify = true;
        if (s == "--csv-only")                 a.csv_only  = true;
    }
    return a;
}

static void setup_cuda_context(const Args& args) {
    if (args.sm_count <= 0) {
        CUDA_CHECK(cudaSetDevice(args.gpu_id));
        return;
    }

    CUdevice dev;
    CUcontext ctx;
    int supported = 0;

    CU_CHECK(cuInit(0));
    CU_CHECK(cuDeviceGet(&dev, args.gpu_id));
    CU_CHECK(cuDeviceGetExecAffinitySupport(&supported,
                                            CU_EXEC_AFFINITY_TYPE_SM_COUNT,
                                            dev));
    if (!supported) {
        fprintf(stderr, "SM-count execution affinity is not supported on gpu=%d\n",
                args.gpu_id);
        exit(EXIT_FAILURE);
    }

    CUexecAffinityParam affinity;
    memset(&affinity, 0, sizeof(affinity));
    affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
    affinity.param.smCount.val = args.sm_count;

    CU_CHECK(cuCtxCreate_v3(&ctx, &affinity, 1, 0, dev));

    CUexecAffinityParam actual;
    memset(&actual, 0, sizeof(actual));
    CU_CHECK(cuCtxGetExecAffinity(&actual, CU_EXEC_AFFINITY_TYPE_SM_COUNT));
    if (actual.param.smCount.val != args.sm_count) {
        fprintf(stderr, "Requested sm-count=%d, actual sm-count=%u\n",
                args.sm_count, actual.param.smCount.val);
    }
}

/* ================================================================
 * Kernel dispatch and benchmark runner (template on M)
 * ================================================================ */

// Compile-time CF selection: CF = ceil(M/32) for M >= 32
template<int M> struct CFSelector { static constexpr int value = (M + 31) / 32; };

template<int M, BenchMode MODE>
static void run_ge_spmm(const Args& args,
                         const vector<int64_t>& h_row_ptr,
                         const vector<int>& h_columns,
                         int n_verts, int64_t n_edges)
{
    int N = n_verts * M;

    // ── Allocate GPU memory ──
    int64_t* d_row_ptr;
    int*     d_columns;
    float*   d_input;
    float*   d_output;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (size_t)(n_verts + 1) * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_columns, (size_t)n_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_input,   (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output,  (size_t)N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr.data(),
                           (size_t)(n_verts + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_columns, h_columns.data(),
                           (size_t)n_edges * sizeof(int), cudaMemcpyHostToDevice));

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
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), (size_t)N * sizeof(float),
                           cudaMemcpyHostToDevice));

    // ── Launch config ──
    int tile_row = args.tile_row;

    // CUDA events
    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    double total_ms = 0.0;
    int measured_iters = 0;
    int total_iters = args.warmup + args.iters;

    if constexpr (M < 32) {
        // ---- Simple kernel path ----
        // Ensure reasonable block size
        int eff_tile = max(tile_row, max(1, 128 / M));
        int block_x = M;
        int block_y = eff_tile;
        int grid_x  = (n_verts + eff_tile - 1) / eff_tile;

        if (!args.csv_only)
            printf("  [simple] grid=%d, block=(%d,%d), smem=0\n",
                   grid_x, block_x, block_y);

        for (int iter = 0; iter < total_iters; iter++) {
            CUDA_CHECK(cudaMemset(d_output, 0, (size_t)N * sizeof(float)));

            CUDA_CHECK(cudaEventRecord(ev_start));
            ge_spmm_simple<M, MODE><<<grid_x, dim3(block_x, block_y)>>>(
                n_verts, d_row_ptr, d_columns, d_input, d_output, eff_tile);
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
    } else {
        // ---- Shared-memory kernel path ----
        constexpr int CF = CFSelector<M>::value;  // CF = ceil(M / 32)

        /*
        这样的划分方法，本质上是：
        对于同一列的block(有着相同的blockIdx.x)，他们处理连续的一段长度为tile_row的节点
        对于同一列中的不同行的block, 他们处理
        */
        int block_x = 32;
        int block_y = tile_row;
        int grid_x  = (n_verts + tile_row - 1) / tile_row;
        int grid_y  = (M + CF * 32 - 1) / (CF * 32);  
        int smem    = 32 * tile_row * sizeof(int);

        
        if (!args.csv_only)
            printf("  [smem] grid=(%d,%d), block=(%d,%d), CF=%d, smem=%d B\n",
                   grid_x, grid_y, block_x, block_y, CF, smem);

        for (int iter = 0; iter < total_iters; iter++) {
            CUDA_CHECK(cudaMemset(d_output, 0, (size_t)N * sizeof(float)));

            CUDA_CHECK(cudaEventRecord(ev_start));
            ge_spmm_smem<M, CF, MODE><<<dim3(grid_x, grid_y), dim3(block_x, block_y), smem>>>(
                n_verts, d_row_ptr, d_columns, d_input, d_output, tile_row);
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
    }

    double avg_ms = total_ms / measured_iters;

    double max_abs_err = 0.0, max_rel_err = 0.0;
    double sum_cpu = 0.0, sum_gpu = 0.0;
    bool pass = true;

    if constexpr (MODE == MODE_FULL) {
        if (args.skip_verify) {
            pass = true;
        } else {
        vector<float> cpu_output(N);
        // ── CPU reference ──
        cpu_segment_sum(h_row_ptr, h_columns, h_input, cpu_output, n_verts, M);

        // ── Correctness verification ──
        vector<float> gpu_output(N);
        CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output,
                               (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; ++i) {
            double err = fabs((double)gpu_output[i] - (double)cpu_output[i]);
            if (err > max_abs_err) max_abs_err = err;
            double mag = fabs((double)cpu_output[i]);
            if (mag > 1e-6) {
                double rel = err / mag;
                if (rel > max_rel_err) max_rel_err = rel;
            }
            sum_cpu += cpu_output[i];
            sum_gpu += gpu_output[i];
        }
        pass = (max_rel_err < 1e-4);
        }
    }

    if (!args.csv_only) {
        printf("\nResults (GE-SpMM, mode=%s, M=%d, %d measured iters, %d warmup):\n",
               mode_name(MODE), M, measured_iters, args.warmup);
        printf("  avg ge_spmm       : %.4f ms\n", avg_ms);
        printf("  per-component avg  : %.4f ms\n", avg_ms / M);
        if constexpr (MODE == MODE_FULL) {
            if (args.skip_verify) {
                printf("  max rel error      : N/A (verification skipped)\n");
                printf("  PASS: SKIPPED\n");
            } else {
                printf("  max rel error      : %.6e\n", max_rel_err);
                printf("  CPU sum            : %.6f\n", sum_cpu);
                printf("  GPU sum            : %.6f\n", sum_gpu);
                printf("  PASS: %s\n", pass ? "YES" : "NO (ERROR)");
            }
        } else {
            printf("  max rel error      : N/A (mode does not compute SpMM)\n");
            printf("  PASS: SKIPPED\n");
        }
    }

    // CSV: dataset,kernel,mode,M,tile_row,iters,avg_ms,per_comp_ms,max_rel_err
    if constexpr (MODE == MODE_FULL) {
        if (args.skip_verify) {
            printf("\nCSV: %s,GESpMM,%s,%d,%d,%d,%.4f,%.4f,N/A\n",
                   args.csr_dir.c_str(), mode_name(MODE), M, args.tile_row,
                   measured_iters, avg_ms, avg_ms / M);
        } else {
        printf("\nCSV: %s,GESpMM,%s,%d,%d,%d,%.4f,%.4f,%.6e\n",
               args.csr_dir.c_str(), mode_name(MODE), M, args.tile_row,
               measured_iters, avg_ms, avg_ms / M, max_rel_err);
        }
    } else {
        printf("\nCSV: %s,GESpMM,%s,%d,%d,%d,%.4f,%.4f,N/A\n",
               args.csr_dir.c_str(), mode_name(MODE), M, args.tile_row,
               measured_iters, avg_ms, avg_ms / M);
    }

    // ── Cleanup ──
    cudaFree(d_row_ptr);
    cudaFree(d_columns);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
}

/* ================================================================
 * Dispatch by M value
 * ================================================================ */
static void dispatch(const Args& args,
                     const vector<int64_t>& h_row_ptr,
                     const vector<int>& h_columns,
                     int n_verts, int64_t n_edges) {
#define RUN_MODE(m)                                                            \
    do {                                                                       \
        switch (args.mode) {                                                   \
            case MODE_FULL:                                                    \
                run_ge_spmm<m, MODE_FULL>(args, h_row_ptr, h_columns,          \
                                          n_verts, n_edges);                   \
                break;                                                         \
            case MODE_LOAD_ONLY:                                               \
                run_ge_spmm<m, MODE_LOAD_ONLY>(args, h_row_ptr, h_columns,     \
                                               n_verts, n_edges);              \
                break;                                                         \
            case MODE_GRAPH_ONLY:                                              \
                run_ge_spmm<m, MODE_GRAPH_ONLY>(args, h_row_ptr, h_columns,    \
                                                n_verts, n_edges);             \
                break;                                                         \
            default:                                                           \
                fprintf(stderr, "Unsupported mode=%d\n", (int)args.mode);      \
                exit(1);                                                       \
        }                                                                      \
    } while (0)

    switch (args.M) {
        case 1:   RUN_MODE(1);   break;
        case 2:   RUN_MODE(2);   break;
        case 4:   RUN_MODE(4);   break;
        case 8:   RUN_MODE(8);   break;
        case 16:  RUN_MODE(16);  break;
        case 32:  RUN_MODE(32);  break;
        case 64:  RUN_MODE(64);  break;
        case 80:  RUN_MODE(80);  break;
        case 96:  RUN_MODE(96);  break;
        case 128: RUN_MODE(128); break;
        default:
            fprintf(stderr, "Unsupported M=%d. Supported: 1,2,4,8,16,32,64,80,96,128\n",
                    args.M);
            exit(1);
    }

#undef RUN_MODE
}

/* ================================================================
 * Main
 * ================================================================ */
int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    setup_cuda_context(args);

    vector<int64_t> h_row_ptr;
    vector<int> h_columns;
    int n_verts;
    int64_t n_edges;
    read_graph(args.csr_dir, h_row_ptr, h_columns, n_verts, n_edges);

    if (!args.csv_only) {
        printf("Graph: %s\n", args.csr_dir.c_str());
        printf("  V = %d, E = %lld\n", n_verts, (long long)n_edges);
        printf("  M = %d, tile_row = %d, iters = %d, warmup = %d, mode = %s, sm_count = %d\n",
               args.M, args.tile_row, args.iters, args.warmup,
               mode_name(args.mode), args.sm_count);
    }

    dispatch(args, h_row_ptr, h_columns, n_verts, n_edges);
    return 0;
}
