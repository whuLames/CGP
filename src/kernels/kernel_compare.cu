/*
 * kernel_compare.cu
 * Unified SpMM kernel benchmark: SegSumVec / GE-SpMM / SAG comparison.
 *
 * Three kernel implementations:
 *   - SegSumVec  : fixed-work-per-thread vectorized segment_sum
 *   - GE-SpMM    : SC'20 warp-centric with column-index caching
 *   - SAG        : GNNAdvisor OSDI'21 warp-per-part with shared memory
 *
 * Usage:
 *   ./kernel_compare <csr_dir> --kernel=segsum_vec|ge_spmm|sag|all
 *       --M=N [--iters=N] [--warmup=N] [--gpu=ID] [--csv-only]
 *       [SegSumVec: --ept=N]
 *       [GE-SpMM:   --tile-row=N]
 *       [SAG:       --partSize=N --dimWorker=N --warpPerBlock=N --no-cache]
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
 * Common macros and utilities
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

#define WARP_SIZE  32
#define BLOCK_SIZE 256

__device__ inline void atomicAdd_F(float* address, float value) {
    float old = value;
    while ((old = atomicExch(address, atomicExch(address, 0.0f) + old)) != 0.0f);
}

/* ================================================================
 * Common I/O and reference implementation
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
 * Unified Args and argument parsing
 * ================================================================ */
enum KernelType { K_SEGSUM_VEC, K_GE_SPMM, K_SAG, K_ALL };

struct Args {
    string csr_dir;
    KernelType kernel = K_ALL;
    // Common
    int M            = 4;
    int iters        = 30;
    int warmup       = 5;
    int gpu_id       = 0;
    bool csv_only    = false;
    // SegSumVec
    int ept          = 32;
    // GE-SpMM
    int tile_row     = 8;
    // SAG
    int partSize     = 32;
    int dimWorker    = 32;
    int warpPerBlock = 8;
    bool no_cache    = false;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <csr_dir> --kernel=segsum_vec|ge_spmm|sag|all\n"
            "       --M=N [--iters=N] [--warmup=N] [--gpu=ID] [--csv-only]\n"
            "       [SegSumVec: --ept=N]\n"
            "       [GE-SpMM:   --tile-row=N]\n"
            "       [SAG:       --partSize=N --dimWorker=N --warpPerBlock=N --no-cache]\n",
            argv[0]);
        exit(1);
    }
    a.csr_dir = argv[1];
    for (int i = 2; i < argc; i++) {
        string s = argv[i];
        if (s.rfind("--kernel=", 0) == 0) {
            string k = s.substr(9);
            if (k == "segsum_vec")           a.kernel = K_SEGSUM_VEC;
            else if (k == "ge_spmm")         a.kernel = K_GE_SPMM;
            else if (k == "sag")             a.kernel = K_SAG;
            else if (k == "all")             a.kernel = K_ALL;
            else { fprintf(stderr, "Unknown kernel: %s\n", k.c_str()); exit(1); }
        }
        if (s.rfind("--M=", 0) == 0)            a.M            = atoi(s.c_str() + 4);
        if (s.rfind("--ept=", 0) == 0)          a.ept          = atoi(s.c_str() + 6);
        if (s.rfind("--tile-row=", 0) == 0)     a.tile_row     = atoi(s.c_str() + 11);
        if (s.rfind("--partSize=", 0) == 0)     a.partSize     = atoi(s.c_str() + 11);
        if (s.rfind("--dimWorker=", 0) == 0)    a.dimWorker    = atoi(s.c_str() + 12);
        if (s.rfind("--warpPerBlock=", 0) == 0) a.warpPerBlock = atoi(s.c_str() + 14);
        if (s.rfind("--iters=", 0) == 0)        a.iters        = atoi(s.c_str() + 8);
        if (s.rfind("--warmup=", 0) == 0)       a.warmup       = atoi(s.c_str() + 9);
        if (s.rfind("--gpu=", 0) == 0)          a.gpu_id       = atoi(s.c_str() + 6);
        if (s == "--csv-only")                  a.csv_only     = true;
        if (s == "--no-cache")                  a.no_cache     = true;
    }
    return a;
}

/* ================================================================
 * Helper: generate input and verify
 * ================================================================ */
static void generate_input(int N, vector<float>& h_input) {
    h_input.resize(N);
    mt19937 rng(42);
    uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) h_input[i] = dist(rng);
}

static void verify_result(const vector<float>& gpu_output,
                          const vector<float>& cpu_output,
                          int N, double& max_rel_err) {
    double max_abs = 0.0;
    max_rel_err = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = fabs((double)gpu_output[i] - (double)cpu_output[i]);
        if (err > max_abs) max_abs = err;
        double mag = fabs((double)cpu_output[i]);
        if (mag > 1e-6) { double rel = err / mag; if (rel > max_rel_err) max_rel_err = rel; }
    }
}

/* ================================================================
 * Kernel — SegSumVec
 * ================================================================ */
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

    float sum[M];
#pragma unroll
    for (int c = 0; c < M; ++c)
        sum[c] = 0.0f;

    while (start < end) {
        int col = columns[start];
        int base = col * M;
#pragma unroll
        for (int c = 0; c < M; ++c)
            sum[c] += aggData[base + c];
        ++start;
    }

    int out_base = node * M;
#pragma unroll
    for (int c = 0; c < M; ++c)
        atomicAdd(&aggRes[out_base + c], sum[c]);
}

__global__ void memset_vec_kernel(float* aggRes, int n_verts, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_verts * M;
    if (i >= total) return;
    aggRes[i] = 0.0f;
}

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
        fprintf(stderr, "ERROR: numThread (%lld) >= INT32_MAX\n", (long long)total);
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
 * Kernel — GE-SpMM
 * ================================================================ */
template<int M>
__global__ void ge_spmm_simple(
    int A_nrows,
    const int64_t* __restrict__ A_csrRowPtr,
    const int*     __restrict__ A_csrColInd,
    const float*   __restrict__ B_dnVal,
    float*         __restrict__ C_dnVal,
    int tile_row)
{
    int rid = tile_row * blockIdx.x + threadIdx.y;
    if (rid >= A_nrows) return;

    int cid = threadIdx.x;
    int64_t lb = A_csrRowPtr[rid];
    int64_t hb = A_csrRowPtr[rid + 1];

    float acc = 0.0f;
    for (int64_t ptr = lb; ptr < hb; ptr++) {
        acc += B_dnVal[(int64_t)A_csrColInd[ptr] * M + cid];
    }
    C_dnVal[(int64_t)rid * M + cid] = acc;
}

template<int M, int CF>
__global__ void ge_spmm_smem(
    int A_nrows,
    const int64_t* __restrict__ A_csrRowPtr,
    const int*     __restrict__ A_csrColInd,
    const float*   __restrict__ B_dnVal,
    float*         __restrict__ C_dnVal,
    int tile_row)
{
    extern __shared__ int colInd_sh[];
    int shmem_offset = threadIdx.y << 5;
    int thread_idx   = shmem_offset + threadIdx.x;

    int rid = tile_row * blockIdx.x + threadIdx.y;
    if (rid >= A_nrows) return;

    int cid = (blockIdx.y * (CF << 5)) + threadIdx.x;
    int64_t lb  = A_csrRowPtr[rid];
    int64_t hb  = A_csrRowPtr[rid + 1];
    int64_t ptr = lb + threadIdx.x;

    float acc[CF];
    #pragma unroll
    for (int c = 0; c < CF; c++) acc[c] = 0.0f;

    if (blockIdx.y != gridDim.y - 1) {
        for (int64_t jj = lb; jj < hb; jj += 32) {
            if (ptr < hb) {
                colInd_sh[thread_idx] = A_csrColInd[ptr] * M;
            }
            __syncwarp();
            ptr += 32;

            for (int kk = 0; kk < 32 && jj + kk < hb; kk++) {
                int offset = colInd_sh[shmem_offset + kk] + cid;
                #pragma unroll
                for (int c = 0; c < CF; c++) {
                    acc[c] += B_dnVal[offset + c * 32];
                }
            }
            __syncwarp();
        }

        int out_base = rid * M + cid;
        #pragma unroll
        for (int c = 0; c < CF; c++) {
            C_dnVal[out_base + c * 32] = acc[c];
        }
    } else {
        int nout = (M - cid + 31) / 32;
        if (nout < 0)  nout = 0;
        if (nout > CF) nout = CF;

        for (int64_t jj = lb; jj < hb; jj += 32) {
            if (ptr < hb) {
                colInd_sh[thread_idx] = A_csrColInd[ptr] * M;
            }
            __syncwarp();
            ptr += 32;

            for (int kk = 0; kk < 32 && jj + kk < hb; kk++) {
                int offset = colInd_sh[shmem_offset + kk] + cid;
                #pragma unroll
                for (int c = 0; c < CF; c++) {
                    if (c < nout) {
                        acc[c] += B_dnVal[offset + c * 32];
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

template<int M> struct CFSelector { static constexpr int value = (M + 31) / 32; };

/* ================================================================
 * Kernel — SAG
 * ================================================================ */
template<int M>
__global__ void sag_kernel(
    const int64_t* __restrict__ row_ptr,
    const int*     __restrict__ columns,
    const float*   __restrict__ input,
    float*         __restrict__ output,
    const int*     __restrict__ partPtr,
    const int*     __restrict__ part2Node,
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock)
{
    unsigned int tid          = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warpId       = tid / WARP_SIZE;
    int block_warpId = threadIdx.x / WARP_SIZE;
    int laneid       = threadIdx.x % WARP_SIZE;

    extern __shared__ int part_meta[];
    int   *partial_ids     = part_meta;
    float *partial_results = (float*)&part_meta[partSize * warpPerBlock];

    if (warpId >= num_parts) return;

    int srcId   = part2Node[warpId];
    int partBeg = partPtr[warpId];
    int partEnd = partPtr[warpId + 1];

    const int pindex_base = block_warpId * partSize;
#pragma unroll
    for (int nidx = partBeg + laneid; nidx < partEnd; nidx += WARP_SIZE) {
        partial_ids[pindex_base + nidx - partBeg] = columns[nidx];
    }
    __syncwarp();

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

    if (laneid < dimWorker) {
        int out_base = srcId * M;
#pragma unroll
        for (int d = laneid; d < M; d += dimWorker)
            atomicAdd_F(&output[out_base + d], partial_results[presult_base + d]);
    }
}

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

    extern __shared__ float partial_results[];

    if (warpId >= num_parts) return;

    int srcId   = part2Node[warpId];
    int partBeg = partPtr[warpId];
    int partEnd = partPtr[warpId + 1];

    const int presult_base = block_warpId * M;
    for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++) {
        int nid = columns[partBeg + nIdx];
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

static void build_part(const vector<int64_t>& row_ptr,
                       int n_verts, int partSize,
                       vector<int>& partPtr, vector<int>& part2Node,
                       int& num_parts)
{
    int64_t total_parts = 0;
    for (int i = 0; i < n_verts; i++) {
        int64_t deg = row_ptr[i + 1] - row_ptr[i];
        int thisNumParts = (deg % partSize == 0) ? (int)(deg / partSize) : (int)(deg / partSize) + 1;
        total_parts += thisNumParts;
    }
    if (total_parts >= INT32_MAX) {
        fprintf(stderr, "ERROR: num_parts (%lld) >= INT32_MAX\n", (long long)total_parts);
        exit(1);
    }
    num_parts = (int)total_parts;
    partPtr.resize(num_parts + 1);
    part2Node.resize(num_parts);

    int part_counter = 0;
    for (int i = 0; i < n_verts; i++) {
        int64_t deg = row_ptr[i + 1] - row_ptr[i];
        int thisNumParts = (deg % partSize == 0) ? (int)(deg / partSize) : (int)(deg / partSize) + 1;
        for (int pid = 0; pid < thisNumParts; pid++) {
            int64_t partBeg = row_ptr[i] + (int64_t)pid * partSize;
            int64_t partEnd = partBeg + partSize < row_ptr[i + 1] ? partBeg + partSize : row_ptr[i + 1];
            partPtr[part_counter]   = (int)partBeg;
            part2Node[part_counter] = i;
            part_counter++;
            if (i == n_verts - 1 && partEnd == row_ptr[i + 1])
                partPtr[part_counter] = (int)partEnd;
        }
    }
}

/* ================================================================
 * Run — SegSumVec (external data)
 * ================================================================ */
template<int M>
static void run_segsum_vec(const Args& args,
                           const vector<int64_t>& h_row_ptr,
                           const vector<int>& h_columns,
                           int n_verts, int64_t n_edges,
                           const vector<float>& h_input,
                           const vector<float>& cpu_output)
{
    int N = n_verts * M;

    // Build thread mapping
    vector<int> h_threadMap, h_threadStart;
    int numThread;
    build_thread_mapping(h_row_ptr, n_verts, args.ept,
                         h_threadMap, h_threadStart, numThread);

    if (!args.csv_only)
        printf("  numThread=%d, ept=%d\n", numThread, args.ept);

    // GPU alloc
    int64_t* d_row_ptr; int* d_columns; float* d_input; float* d_output;
    int* d_threadMap; int* d_threadStart;

    CUDA_CHECK(cudaMalloc(&d_row_ptr,     (size_t)(n_verts + 1) * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_columns,     (size_t)n_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_input,       (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output,      (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_threadMap,   (size_t)numThread * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_threadStart, (size_t)numThread * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr,     h_row_ptr.data(),     (size_t)(n_verts+1)*sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_columns,     h_columns.data(),     (size_t)n_edges*sizeof(int),        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_threadMap,   h_threadMap.data(),   (size_t)numThread*sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_threadStart, h_threadStart.data(), (size_t)numThread*sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input,       h_input.data(),       (size_t)N*sizeof(float),             cudaMemcpyHostToDevice));

    int grid_seg = (numThread + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_clr = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    double total_ms = 0.0;
    int measured_iters = 0;
    int total_iters = args.warmup + args.iters;

    for (int iter = 0; iter < total_iters; iter++) {
        memset_vec_kernel<<<grid_clr, BLOCK_SIZE>>>(d_output, n_verts, M);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(ev_start));
        segment_sum_fixed_work_kernel_vec<M><<<grid_seg, BLOCK_SIZE>>>(
            d_row_ptr, d_columns, d_input, d_output,
            d_threadMap, d_threadStart, numThread, args.ept);
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

    // Verify
    vector<float> gpu_output(N);
    CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    double max_rel_err;
    verify_result(gpu_output, cpu_output, N, max_rel_err);
    bool pass = (max_rel_err < 1e-4);

    if (!args.csv_only) {
        printf("\nResults (SegSumVec, M=%d, %d iters):\n", M, measured_iters);
        printf("  avg kernel    : %.4f ms\n", avg_ms);
        printf("  per-comp      : %.4f ms\n", avg_ms / M);
        printf("  max rel error : %.6e\n", max_rel_err);
        printf("  PASS: %s\n", pass ? "YES" : "NO");
    }
    printf("CSV: %s,SegSumVec,%d,%d,-,-,-,%d,%.4f,%.4f,%.6e\n",
           args.csr_dir.c_str(), M, args.ept,
           measured_iters, avg_ms, avg_ms / M, max_rel_err);

    cudaFree(d_row_ptr); cudaFree(d_columns); cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_threadMap); cudaFree(d_threadStart);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_end);
}

/* ================================================================
 * Run — GE-SpMM (external data)
 * ================================================================ */
template<int M>
static void run_ge_spmm(const Args& args,
                         const vector<int64_t>& h_row_ptr,
                         const vector<int>& h_columns,
                         int n_verts, int64_t n_edges,
                         const vector<float>& h_input,
                         const vector<float>& cpu_output)
{
    int N = n_verts * M;

    int64_t* d_row_ptr; int* d_columns; float* d_input; float* d_output;
    CUDA_CHECK(cudaMalloc(&d_row_ptr,  (size_t)(n_verts + 1) * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_columns,  (size_t)n_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_input,    (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output,   (size_t)N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr.data(), (size_t)(n_verts+1)*sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_columns, h_columns.data(), (size_t)n_edges*sizeof(int),          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input,   h_input.data(),   (size_t)N*sizeof(float),              cudaMemcpyHostToDevice));

    int tile_row = args.tile_row;

    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    double total_ms = 0.0;
    int measured_iters = 0;
    int total_iters = args.warmup + args.iters;

    if constexpr (M < 32) {
        int eff_tile = max(tile_row, max(1, 128 / M));
        int block_x = M, block_y = eff_tile;
        int grid_x  = (n_verts + eff_tile - 1) / eff_tile;

        if (!args.csv_only)
            printf("  [simple] grid=%d, block=(%d,%d)\n", grid_x, block_x, block_y);

        for (int iter = 0; iter < total_iters; iter++) {
            CUDA_CHECK(cudaMemset(d_output, 0, (size_t)N * sizeof(float)));
            CUDA_CHECK(cudaEventRecord(ev_start));
            ge_spmm_simple<M><<<grid_x, dim3(block_x, block_y)>>>(
                n_verts, d_row_ptr, d_columns, d_input, d_output, eff_tile);
            CUDA_CHECK(cudaEventRecord(ev_end));
            CUDA_CHECK(cudaEventSynchronize(ev_end));
            CUDA_CHECK(cudaGetLastError());
            if (iter >= args.warmup) {
                float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));
                total_ms += ms; measured_iters++;
            }
        }
    } else {
        constexpr int CF = CFSelector<M>::value;
        int block_x = 32, block_y = tile_row;
        int grid_x  = (n_verts + tile_row - 1) / tile_row;
        int grid_y  = (M + CF * 32 - 1) / (CF * 32);
        int smem    = 32 * tile_row * sizeof(int);

        if (!args.csv_only)
            printf("  [smem] grid=(%d,%d), block=(%d,%d), CF=%d, smem=%d B\n",
                   grid_x, grid_y, block_x, block_y, CF, smem);

        for (int iter = 0; iter < total_iters; iter++) {
            CUDA_CHECK(cudaMemset(d_output, 0, (size_t)N * sizeof(float)));
            CUDA_CHECK(cudaEventRecord(ev_start));
            ge_spmm_smem<M, CF><<<dim3(grid_x, grid_y), dim3(block_x, block_y), smem>>>(
                n_verts, d_row_ptr, d_columns, d_input, d_output, tile_row);
            CUDA_CHECK(cudaEventRecord(ev_end));
            CUDA_CHECK(cudaEventSynchronize(ev_end));
            CUDA_CHECK(cudaGetLastError());
            if (iter >= args.warmup) {
                float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));
                total_ms += ms; measured_iters++;
            }
        }
    }

    double avg_ms = total_ms / measured_iters;

    vector<float> gpu_output(N);
    CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    double max_rel_err;
    verify_result(gpu_output, cpu_output, N, max_rel_err);
    bool pass = (max_rel_err < 1e-4);

    if (!args.csv_only) {
        printf("\nResults (GE-SpMM, M=%d, %d iters):\n", M, measured_iters);
        printf("  avg kernel    : %.4f ms\n", avg_ms);
        printf("  per-comp      : %.4f ms\n", avg_ms / M);
        printf("  max rel error : %.6e\n", max_rel_err);
        printf("  PASS: %s\n", pass ? "YES" : "NO");
    }
    printf("CSV: %s,GESpMM,%d,-,%d,-,-,%d,%.4f,%.4f,%.6e\n",
           args.csr_dir.c_str(), M, args.tile_row,
           measured_iters, avg_ms, avg_ms / M, max_rel_err);

    cudaFree(d_row_ptr); cudaFree(d_columns); cudaFree(d_input); cudaFree(d_output);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_end);
}

/* ================================================================
 * Run — SAG (external data)
 * ================================================================ */
template<int M>
static void run_sag(const Args& args,
                    const vector<int64_t>& h_row_ptr,
                    const vector<int>& h_columns,
                    int n_verts, int64_t n_edges,
                    const vector<float>& h_input,
                    const vector<float>& cpu_output)
{
    int N = n_verts * M;

    if (!args.csv_only) printf("  Building parts (partSize=%d)...\n", args.partSize);
    vector<int> h_partPtr, h_part2Node;
    int num_parts;
    build_part(h_row_ptr, n_verts, args.partSize, h_partPtr, h_part2Node, num_parts);
    if (!args.csv_only) printf("  num_parts=%d\n", num_parts);

    int64_t* d_row_ptr; int* d_columns; float* d_input; float* d_output;
    int* d_partPtr; int* d_part2Node;
    CUDA_CHECK(cudaMalloc(&d_row_ptr,   (size_t)(n_verts + 1) * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_columns,   (size_t)n_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_input,     (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output,    (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partPtr,   (size_t)(num_parts + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_part2Node, (size_t)num_parts * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr,   h_row_ptr.data(),   (size_t)(n_verts+1)*sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_columns,   h_columns.data(),   (size_t)n_edges*sizeof(int),        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_partPtr,   h_partPtr.data(),   (size_t)(num_parts+1)*sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_part2Node, h_part2Node.data(), (size_t)num_parts*sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input,     h_input.data(),     (size_t)N*sizeof(float),             cudaMemcpyHostToDevice));

    int block = args.warpPerBlock * WARP_SIZE;
    int64_t total_warps = (int64_t)num_parts;
    int grid  = (int)((total_warps * WARP_SIZE + block - 1) / block);
    int shared_memory = args.no_cache
        ? args.warpPerBlock * M * sizeof(float)
        : args.partSize * args.warpPerBlock * sizeof(int) + args.warpPerBlock * M * sizeof(float);

    if (!args.csv_only)
        printf("  grid=%d, block=%d, smem=%d bytes%s\n", grid, block, shared_memory,
               args.no_cache ? " [nocache]" : "");

    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    double total_ms = 0.0;
    int measured_iters = 0;
    int total_iters = args.warmup + args.iters;

    for (int iter = 0; iter < total_iters; iter++) {
        CUDA_CHECK(cudaMemset(d_output, 0, (size_t)N * sizeof(float)));
        CUDA_CHECK(cudaEventRecord(ev_start));
        if (args.no_cache) {
            sag_kernel_nocache<M><<<grid, block, shared_memory>>>(
                d_row_ptr, d_columns, d_input, d_output,
                d_partPtr, d_part2Node, num_parts, args.dimWorker, args.warpPerBlock);
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
            float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));
            total_ms += ms; measured_iters++;
        }
    }

    double avg_ms = total_ms / measured_iters;

    vector<float> gpu_output(N);
    CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    double max_rel_err;
    verify_result(gpu_output, cpu_output, N, max_rel_err);
    bool pass = (max_rel_err < 1e-4);

    if (!args.csv_only) {
        printf("\nResults (SAG, M=%d, %d iters):\n", M, measured_iters);
        printf("  partSize=%d, dimWorker=%d, warpPerBlock=%d\n",
               args.partSize, args.dimWorker, args.warpPerBlock);
        printf("  avg kernel    : %.4f ms\n", avg_ms);
        printf("  per-comp      : %.4f ms\n", avg_ms / M);
        printf("  max rel error : %.6e\n", max_rel_err);
        printf("  PASS: %s\n", pass ? "YES" : "NO");
    }
    printf("CSV: %s,SAG,%d,-,-,%d,%d,%d,%.4f,%.4f,%.6e\n",
           args.csr_dir.c_str(), M, args.warpPerBlock, args.partSize,
           measured_iters, avg_ms, avg_ms / M, max_rel_err);

    cudaFree(d_row_ptr); cudaFree(d_columns); cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_partPtr); cudaFree(d_part2Node);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_end);
}

/* ================================================================
 * Dispatch
 * ================================================================ */
static void check_M(int M) {
    if (M < 1 || M > 128 ||
        (M > 56 && M < 64) ||
        (M > 64 && M < 80) ||
        (M > 80 && M < 96) ||
        (M > 96 && M < 128)) {
        fprintf(stderr, "Unsupported M=%d. Supported: 1-56,64,80,96,128\n", M);
        exit(1);
    }
}

#define DISPATCH_SINGLE_CASE(m) \
    case m: \
        switch (args.kernel) { \
            case K_SEGSUM_VEC: run_segsum_vec<m>(args, h_row_ptr, h_columns, n_verts, n_edges, h_input, cpu_output); break; \
            case K_GE_SPMM:    run_ge_spmm<m>(args, h_row_ptr, h_columns, n_verts, n_edges, h_input, cpu_output); break; \
            case K_SAG:        run_sag<m>(args, h_row_ptr, h_columns, n_verts, n_edges, h_input, cpu_output); break; \
            default: break; \
        } break;

static void dispatch_single(const Args& args,
                             const vector<int64_t>& h_row_ptr,
                             const vector<int>& h_columns,
                             int n_verts, int64_t n_edges,
                             const vector<float>& h_input,
                             const vector<float>& cpu_output)
{
    switch (args.M) {
        DISPATCH_SINGLE_CASE(1)   DISPATCH_SINGLE_CASE(2)   DISPATCH_SINGLE_CASE(3)
        DISPATCH_SINGLE_CASE(4)   DISPATCH_SINGLE_CASE(5)   DISPATCH_SINGLE_CASE(6)
        DISPATCH_SINGLE_CASE(7)   DISPATCH_SINGLE_CASE(8)   DISPATCH_SINGLE_CASE(9)
        DISPATCH_SINGLE_CASE(10)  DISPATCH_SINGLE_CASE(11)  DISPATCH_SINGLE_CASE(12)
        DISPATCH_SINGLE_CASE(13)  DISPATCH_SINGLE_CASE(14)  DISPATCH_SINGLE_CASE(15)
        DISPATCH_SINGLE_CASE(16)  DISPATCH_SINGLE_CASE(17)  DISPATCH_SINGLE_CASE(18)
        DISPATCH_SINGLE_CASE(19)  DISPATCH_SINGLE_CASE(20)  DISPATCH_SINGLE_CASE(21)
        DISPATCH_SINGLE_CASE(22)  DISPATCH_SINGLE_CASE(23)  DISPATCH_SINGLE_CASE(24)
        DISPATCH_SINGLE_CASE(25)  DISPATCH_SINGLE_CASE(26)  DISPATCH_SINGLE_CASE(27)
        DISPATCH_SINGLE_CASE(28)  DISPATCH_SINGLE_CASE(29)  DISPATCH_SINGLE_CASE(30)
        DISPATCH_SINGLE_CASE(31)  DISPATCH_SINGLE_CASE(32)  DISPATCH_SINGLE_CASE(33)
        DISPATCH_SINGLE_CASE(34)  DISPATCH_SINGLE_CASE(35)  DISPATCH_SINGLE_CASE(36)
        DISPATCH_SINGLE_CASE(37)  DISPATCH_SINGLE_CASE(38)  DISPATCH_SINGLE_CASE(39)
        DISPATCH_SINGLE_CASE(40)  DISPATCH_SINGLE_CASE(41)  DISPATCH_SINGLE_CASE(42)
        DISPATCH_SINGLE_CASE(43)  DISPATCH_SINGLE_CASE(44)  DISPATCH_SINGLE_CASE(45)
        DISPATCH_SINGLE_CASE(46)  DISPATCH_SINGLE_CASE(47)  DISPATCH_SINGLE_CASE(48)
        DISPATCH_SINGLE_CASE(49)  DISPATCH_SINGLE_CASE(50)  DISPATCH_SINGLE_CASE(51)
        DISPATCH_SINGLE_CASE(52)  DISPATCH_SINGLE_CASE(53)  DISPATCH_SINGLE_CASE(54)
        DISPATCH_SINGLE_CASE(55)  DISPATCH_SINGLE_CASE(56)
        DISPATCH_SINGLE_CASE(64)
        DISPATCH_SINGLE_CASE(80)  DISPATCH_SINGLE_CASE(96)  DISPATCH_SINGLE_CASE(128)
    }
}
#undef DISPATCH_SINGLE_CASE

#define DISPATCH_ALL_CASE(m) \
    case m: \
        { int64_t _n = (int64_t)n_verts * m; \
          h_input.resize(_n); cpu_output.resize(_n); \
          generate_input((int)_n, h_input); \
          cpu_segment_sum(h_row_ptr, h_columns, h_input, cpu_output, n_verts, m); } \
        if (!args.csv_only) printf("\n===== SegSumVec =====\n"); \
        CUDA_CHECK(cudaDeviceSynchronize()); \
        run_segsum_vec<m>(args, h_row_ptr, h_columns, n_verts, n_edges, h_input, cpu_output); \
        if (!args.csv_only) printf("\n===== GE-SpMM =====\n"); \
        CUDA_CHECK(cudaDeviceSynchronize()); \
        run_ge_spmm<m>(args, h_row_ptr, h_columns, n_verts, n_edges, h_input, cpu_output); \
        if (!args.csv_only) printf("\n===== SAG =====\n"); \
        CUDA_CHECK(cudaDeviceSynchronize()); \
        run_sag<m>(args, h_row_ptr, h_columns, n_verts, n_edges, h_input, cpu_output); \
        break;

static void dispatch_all(const Args& args,
                         const vector<int64_t>& h_row_ptr,
                         const vector<int>& h_columns,
                         int n_verts, int64_t n_edges)
{
    vector<float> h_input, cpu_output;
    switch (args.M) {
        DISPATCH_ALL_CASE(1)   DISPATCH_ALL_CASE(2)   DISPATCH_ALL_CASE(3)
        DISPATCH_ALL_CASE(4)   DISPATCH_ALL_CASE(5)   DISPATCH_ALL_CASE(6)
        DISPATCH_ALL_CASE(7)   DISPATCH_ALL_CASE(8)   DISPATCH_ALL_CASE(9)
        DISPATCH_ALL_CASE(10)  DISPATCH_ALL_CASE(11)  DISPATCH_ALL_CASE(12)
        DISPATCH_ALL_CASE(13)  DISPATCH_ALL_CASE(14)  DISPATCH_ALL_CASE(15)
        DISPATCH_ALL_CASE(16)  DISPATCH_ALL_CASE(17)  DISPATCH_ALL_CASE(18)
        DISPATCH_ALL_CASE(19)  DISPATCH_ALL_CASE(20)  DISPATCH_ALL_CASE(21)
        DISPATCH_ALL_CASE(22)  DISPATCH_ALL_CASE(23)  DISPATCH_ALL_CASE(24)
        DISPATCH_ALL_CASE(25)  DISPATCH_ALL_CASE(26)  DISPATCH_ALL_CASE(27)
        DISPATCH_ALL_CASE(28)  DISPATCH_ALL_CASE(29)  DISPATCH_ALL_CASE(30)
        DISPATCH_ALL_CASE(31)  DISPATCH_ALL_CASE(32)  DISPATCH_ALL_CASE(33)
        DISPATCH_ALL_CASE(34)  DISPATCH_ALL_CASE(35)  DISPATCH_ALL_CASE(36)
        DISPATCH_ALL_CASE(37)  DISPATCH_ALL_CASE(38)  DISPATCH_ALL_CASE(39)
        DISPATCH_ALL_CASE(40)  DISPATCH_ALL_CASE(41)  DISPATCH_ALL_CASE(42)
        DISPATCH_ALL_CASE(43)  DISPATCH_ALL_CASE(44)  DISPATCH_ALL_CASE(45)
        DISPATCH_ALL_CASE(46)  DISPATCH_ALL_CASE(47)  DISPATCH_ALL_CASE(48)
        DISPATCH_ALL_CASE(49)  DISPATCH_ALL_CASE(50)  DISPATCH_ALL_CASE(51)
        DISPATCH_ALL_CASE(52)  DISPATCH_ALL_CASE(53)  DISPATCH_ALL_CASE(54)
        DISPATCH_ALL_CASE(55)  DISPATCH_ALL_CASE(56)
        DISPATCH_ALL_CASE(64)
        DISPATCH_ALL_CASE(80)  DISPATCH_ALL_CASE(96)  DISPATCH_ALL_CASE(128)
    }
}
#undef DISPATCH_ALL_CASE

/* ================================================================
 * Main
 * ================================================================ */
int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(args.gpu_id));
    check_M(args.M);

    vector<int64_t> h_row_ptr;
    vector<int> h_columns;
    int n_verts;
    int64_t n_edges;
    read_graph(args.csr_dir, h_row_ptr, h_columns, n_verts, n_edges);

    if (!args.csv_only) {
        printf("Graph: %s\n", args.csr_dir.c_str());
        printf("  V = %d, E = %lld\n", n_verts, (long long)n_edges);
        printf("  M = %d, kernel = %s\n", args.M,
               args.kernel == K_SEGSUM_VEC ? "SegSumVec" :
               args.kernel == K_GE_SPMM    ? "GE-SpMM" :
               args.kernel == K_SAG        ? "SAG" : "ALL");
    }

    if (args.kernel == K_ALL) {
        dispatch_all(args, h_row_ptr, h_columns, n_verts, n_edges);
    } else {
        int N = n_verts * args.M;
        vector<float> h_input(N), cpu_output(N);
        generate_input(N, h_input);
        cpu_segment_sum(h_row_ptr, h_columns, h_input, cpu_output, n_verts, args.M);
        dispatch_single(args, h_row_ptr, h_columns, n_verts, n_edges, h_input, cpu_output);
    }

    return 0;
}
