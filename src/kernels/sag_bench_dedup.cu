/*
 * sag_bench_dedup.cu
 * SAG kernel with block-level neighbor dedup cache.
 *
 * Extension of sag_bench.cu: pre-deduplicates neighbor embeddings at block level,
 * caches unique embeddings in shared memory, reads from smem during aggregation.
 *
 * Key optimization: global reads reduced from M_edges × dim to N_unique × dim
 *   where N_unique ≤ M_edges = warpPerBlock × partSize
 *   Benefit: dedup_ratio = 1 - N_unique / M_edges
 *
 * Adaptive fallback: blocks with dedup_ratio < 10% or smem overflow
 *   use original SAG path (global reads directly).
 *
 * Usage:
 *   ./sag_bench_dedup <csr_dir> --M=N [--partSize=N] [--dimWorker=N] [--warpPerBlock=N]
 *                      [--iters=N] [--warmup=N] [--gpu=ID] [--csv-only]
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
#include <unordered_map>
#include <unordered_set>

#include <cuda_runtime.h>

using namespace std;

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
#define MAX_SMEM  (96 * 1024)  // V100 per-SM shared memory

/* ================================================================
 * Custom atomic add
 * ================================================================ */
__device__ inline void atomicAdd_F(float* address, float value) {
    float old = value;
    while ((old = atomicExch(address, atomicExch(address, 0.0f) + old)) != 0.0f);
}

/* ================================================================
 * Dedup Kernel
 * ================================================================ */
template<int M>
__global__ void sag_kernel_dedup(
    const int64_t* __restrict__ row_ptr,
    const int*     __restrict__ columns,
    const float*   __restrict__ input,
    float*         __restrict__ output,
    const int*     __restrict__ partPtr,
    const int*     __restrict__ part2Node,
    const int*     __restrict__ blockUniqIds,        // [num_blocks * maxUq]
    const int*     __restrict__ blockUniqCounts,      // [num_blocks]
    const short*   __restrict__ blockRemap,           // [num_parts * partSize] int16
    const int*     __restrict__ blockNeedDedupArr,    // [num_blocks] per-block flag
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock,
    const int maxUq)                                  // max unique per block (for smem stride)
{
    unsigned int tid          = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warpId       = tid / WARP_SIZE;
    int block_warpId = threadIdx.x / WARP_SIZE;
    int laneid       = threadIdx.x % WARP_SIZE;

    extern __shared__ int part_meta[];
    int   *partial_ids     = part_meta;
    // shared_emb starts after partial_ids
    float *shared_emb      = (float*)&part_meta[partSize * warpPerBlock];
    // partial_results starts after shared_emb
    float *partial_results = (float*)&shared_emb[maxUq * M];

    if (warpId >= num_parts) return;

    int srcId   = part2Node[warpId];
    int partBeg = partPtr[warpId];
    int partEnd = partPtr[warpId + 1];

    // Phase 1: cooperative load neighbor IDs → partial_ids
    const int pindex_base = block_warpId * partSize;
#pragma unroll
    for (int nidx = partBeg + laneid; nidx < partEnd; nidx += WARP_SIZE) {
        partial_ids[pindex_base + nidx - partBeg] = columns[nidx];
    }
    __syncwarp();

    const int presult_base = block_warpId * M;
    int blockId = warpId / warpPerBlock;

    if (blockNeedDedupArr[blockId]) {
        // Phase 1.5: all-warps cooperative load of unique embeddings → shared_emb
        int uq_count = blockUniqCounts[blockId];
        int* uniq_ids = (int*)&blockUniqIds[blockId * maxUq];
        for (int i = threadIdx.x; i < uq_count; i += blockDim.x) {
            int nid = uniq_ids[i];
            int nid_base = nid * M;
            float* dst = &shared_emb[i * M];
#pragma unroll
            for (int d = 0; d < M; d++)
                dst[d] = input[nid_base + d];
        }
        __syncthreads();

        // Phase 2: aggregation — read from shared_emb via remap
        int remap_base = (blockId * warpPerBlock + block_warpId) * partSize;

        for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++) {
            // Remap: original position → shared_emb offset
            short emb_off = blockRemap[remap_base + nIdx];
            int   emb_base = emb_off * M;

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
                    partial_results[presult_base + d] += shared_emb[emb_base + d];
            }
        }
    } else {
        // Fallback: original SAG path (global reads)
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
    }

    // Phase 3: atomic write-back
    if (laneid < dimWorker) {
        int out_base = srcId * M;
#pragma unroll
        for (int d = laneid; d < M; d += dimWorker)
            atomicAdd_F(&output[out_base + d], partial_results[presult_base + d]);
    }
}

/* ================================================================
 * build_part (same as sag_bench.cu)
 * ================================================================ */
static void build_part(const vector<int64_t>& row_ptr,
                       int n_verts, int partSize,
                       vector<int>& partPtr, vector<int>& part2Node, int& num_parts) {
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
            partPtr[part_counter] = (int)partBeg;
            part2Node[part_counter] = i;
            part_counter++;
            if (i == n_verts - 1 && partEnd == row_ptr[i + 1])
                partPtr[part_counter] = (int)partEnd;
        }
    }
}

/* ================================================================
 * build_block_cache: deduplicate neighbors per block
 * ================================================================ */
static void build_block_cache(
    const vector<int>& columns,
    const vector<int>& partPtr,
    const vector<int>& part2Node,
    int num_parts, int partSize, int warpPerBlock, int M,
    vector<int>&   blockUniqIds,       // [num_blocks * maxUq]
    vector<int>&   blockUniqCounts,    // [num_blocks]
    vector<short>& blockRemap,         // [num_parts * partSize]
    vector<int>&   blockNeedDedup,     // [num_blocks] 0 or 1
    int& maxUq)
{
    int num_blocks = (num_parts + warpPerBlock - 1) / warpPerBlock;
    maxUq = 0;
    int max_uq_allowed = (MAX_SMEM - partSize * warpPerBlock * 4 - warpPerBlock * M * 4) / (M * 4);
    if (max_uq_allowed > warpPerBlock * partSize) max_uq_allowed = warpPerBlock * partSize;

    blockUniqCounts.resize(num_blocks);
    blockNeedDedup.resize(num_blocks, 0);
    blockRemap.assign((size_t)num_parts * partSize, -1);

    // Count max unique first
    vector<int> block_max_uq(num_blocks, 0);
    vector<vector<int>> block_nid_set(num_blocks);
    for (int b = 0; b < num_blocks; b++) block_nid_set[b].reserve(warpPerBlock * partSize);

    for (int b = 0; b < num_blocks; b++) {
        int p_start = b * warpPerBlock;
        int p_end   = min(p_start + warpPerBlock, num_parts);
        unordered_set<int> nbr_set;
        int total_m = 0;
        for (int p = p_start; p < p_end; p++) {
            for (int e = partPtr[p]; e < partPtr[p + 1]; e++) {
                nbr_set.insert(columns[e]);
                total_m++;
            }
        }
        int uq = (int)nbr_set.size();
        blockUniqCounts[b] = uq;
        block_max_uq[b] = uq;
        if (uq > maxUq) maxUq = uq;

        // Decide whether to use dedup
        float dedup_ratio = (total_m > 0) ? (1.0f - (float)uq / (float)total_m) : 0.0f;
        if (dedup_ratio >= 0.10f && uq <= max_uq_allowed) {
            blockNeedDedup[b] = 1;
            block_nid_set[b].assign(nbr_set.begin(), nbr_set.end());
        }
    }

    // Fill blockUniqIds
    if (maxUq == 0) maxUq = 1;
    blockUniqIds.assign((size_t)num_blocks * maxUq, 0);
    for (int b = 0; b < num_blocks; b++) {
        if (blockNeedDedup[b]) {
            for (int j = 0; j < blockUniqCounts[b]; j++)
                blockUniqIds[b * maxUq + j] = block_nid_set[b][j];
        }
    }

    // Build remap table
    for (int b = 0; b < num_blocks; b++) {
        if (!blockNeedDedup[b]) continue;
        // Build nid → emb_offset (within this block's unique list)
        unordered_map<int, short> nid_to_off;
        const int* uq_ids = &blockUniqIds[b * maxUq];
        for (int j = 0; j < blockUniqCounts[b]; j++)
            nid_to_off[uq_ids[j]] = (short)j;

        int p_start = b * warpPerBlock;
        int p_end   = min(p_start + warpPerBlock, num_parts);
        for (int p = p_start; p < p_end; p++) {
            int warp_in_block = p - p_start;
            int remap_base = (b * warpPerBlock + warp_in_block) * partSize;
            for (int e = partPtr[p]; e < partPtr[p + 1]; e++) {
                int nid = columns[e];
                int pos = e - partPtr[p];
                blockRemap[remap_base + pos] = nid_to_off[nid];
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
                       vector<int64_t>& row_ptr, vector<int>& columns,
                       int& n_verts, int64_t& n_edges) {
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
                row_ptr.swap(rp64); use_int64 = true;
            }
        }
        if (!use_int64) {
            vf.seekg(0);
            int nv32 = (int)(vsz / sizeof(int));
            vector<int> rp32(nv32);
            vf.read(reinterpret_cast<char*>(rp32.data()), vsz);
            row_ptr.resize(nv32);
            for (int i = 0; i < nv32; i++) row_ptr[i] = static_cast<int64_t>(rp32[i]);
        }
    }
    read_bin_int(path + "/csr_elist.bin", columns);
    n_verts = (int)row_ptr.size() - 1;
    n_edges = (int64_t)columns.size();
}

/* ================================================================
 * CPU reference
 * ================================================================ */
static void cpu_segment_sum(const vector<int64_t>& row_ptr,
                            const vector<int>& columns,
                            const vector<float>& aggData,
                            vector<float>& aggRes, int n_verts, int M) {
    fill(aggRes.begin(), aggRes.end(), 0.0f);
    for (int v = 0; v < n_verts; ++v)
        for (int64_t e = row_ptr[v]; e < row_ptr[v + 1]; ++e) {
            int nb = columns[e];
            for (int c = 0; c < M; ++c) aggRes[v * M + c] += aggData[nb * M + c];
        }
}

/* ================================================================
 * Args
 * ================================================================ */
struct Args {
    string csr_dir;
    int M=4, partSize=32, dimWorker=32, warpPerBlock=8;
    int iters=30, warmup=5, gpu_id=0;
    bool csv_only = false;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <csr_dir> --M=N [--partSize=N] [--dimWorker=N] [--warpPerBlock=N] [--iters=N] [--warmup=N] [--gpu=ID] [--csv-only]\n", argv[0]);
        exit(1);
    }
    a.csr_dir = argv[1];
    for (int i = 2; i < argc; i++) {
        string s = argv[i];
        if (s.rfind("--M=", 0) == 0)            a.M = atoi(s.c_str() + 4);
        if (s.rfind("--partSize=", 0) == 0)     a.partSize = atoi(s.c_str() + 11);
        if (s.rfind("--dimWorker=", 0) == 0)    a.dimWorker = atoi(s.c_str() + 12);
        if (s.rfind("--warpPerBlock=", 0) == 0) a.warpPerBlock = atoi(s.c_str() + 14);
        if (s.rfind("--iters=", 0) == 0)        a.iters = atoi(s.c_str() + 8);
        if (s.rfind("--warmup=", 0) == 0)       a.warmup = atoi(s.c_str() + 9);
        if (s.rfind("--gpu=", 0) == 0)          a.gpu_id = atoi(s.c_str() + 6);
        if (s == "--csv-only")                  a.csv_only = true;
    }
    return a;
}

/* ================================================================
 * Run with template dispatch
 * ================================================================ */
template<int M>
static void run_sag_dedup(const Args& args,
                          const vector<int64_t>& h_row_ptr,
                          const vector<int>& h_columns,
                          int n_verts, int64_t n_edges) {
    int N = n_verts * M;

    // Preprocess
    if (!args.csv_only) printf("  Building parts...\n");
    vector<int> h_partPtr, h_part2Node;
    int num_parts;
    build_part(h_row_ptr, n_verts, args.partSize, h_partPtr, h_part2Node, num_parts);
    int num_blocks = (num_parts + args.warpPerBlock - 1) / args.warpPerBlock;

    if (!args.csv_only) printf("  Building block cache (dedup)...\n");
    vector<int>   h_uniqIds, h_uniqCounts, h_needDedup;
    vector<short> h_remap;
    int maxUq;
    build_block_cache(h_columns, h_partPtr, h_part2Node,
                      num_parts, args.partSize, args.warpPerBlock, M,
                      h_uniqIds, h_uniqCounts, h_remap, h_needDedup, maxUq);

    int n_dedup_blocks = 0;
    for (int v : h_needDedup) n_dedup_blocks += v;

    if (!args.csv_only) {
        printf("  num_parts=%d  num_blocks=%d  maxUq=%d  dedup_blocks=%d/%.0f%%\n",
               num_parts, num_blocks, maxUq, n_dedup_blocks,
               100.0 * n_dedup_blocks / max(num_blocks, 1));
    }

    int ps = args.partSize, wpb = args.warpPerBlock;
    int smem_req = ps * wpb * 4 + wpb * ps * M * 4 + wpb * M * 4;
    bool global_dedup = (smem_req <= MAX_SMEM);
    if (!global_dedup) maxUq = 0;  // no shared_emb when disabled

    if (!args.csv_only) {
        printf("  dedup smem/block=%d bytes (%.1f KB), %s\n",
               smem_req, smem_req / 1024.0,
               global_dedup ? "ENABLED" : "DISABLED (smem overflow, fallback to SAG)");
    }

    // GPU allocations
    int64_t* d_row_ptr; int* d_columns; float* d_input; float* d_output;
    int* d_partPtr; int* d_part2Node;
    int* d_uniqIds; int* d_uniqCounts; short* d_remap; int* d_needDedup;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (size_t)(n_verts+1)*sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_columns, (size_t)n_edges*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_input,   (size_t)N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output,  (size_t)N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partPtr, (size_t)(num_parts+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_part2Node,(size_t)num_parts*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_needDedup,  (size_t)num_blocks * sizeof(int)));
    int uq_alloc = max(1, maxUq);
    CUDA_CHECK(cudaMalloc(&d_uniqIds,   (size_t)num_blocks * uq_alloc * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_uniqCounts, (size_t)num_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_remap,      (size_t)num_parts * args.partSize * sizeof(short)));

    // Upload
    CUDA_CHECK(cudaMemcpy(d_row_ptr,   h_row_ptr.data(), (size_t)(n_verts+1)*sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_columns,   h_columns.data(), (size_t)n_edges*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_partPtr,   h_partPtr.data(), (size_t)(num_parts+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_part2Node, h_part2Node.data(), (size_t)num_parts*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_needDedup,  h_needDedup.data(),  (size_t)num_blocks*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uniqIds,   h_uniqIds.data(),   (size_t)num_blocks*uq_alloc*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uniqCounts, h_uniqCounts.data(), (size_t)num_blocks*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_remap,      h_remap.data(),      (size_t)num_parts*args.partSize*sizeof(short), cudaMemcpyHostToDevice));

    // Init input
    if (!args.csv_only)
        printf("  Initializing M=%d vector data (%.2f MB)...\n", M, (double)N*sizeof(float)/1e6);
    vector<float> h_input(N);
    { mt19937 rng(42); uniform_real_distribution<float> dist(0.0f,1.0f);
      for (int i=0;i<N;i++) h_input[i]=dist(rng); }
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), (size_t)N*sizeof(float), cudaMemcpyHostToDevice));

    // CPU ref
    vector<float> cpu_output(N);
    cpu_segment_sum(h_row_ptr, h_columns, h_input, cpu_output, n_verts, M);

    // Launch config
    int block = args.warpPerBlock * WARP_SIZE;
    int64_t tw = (int64_t)num_parts;
    int grid = (int)((tw * WARP_SIZE + block - 1) / block);
    int uq_smem = global_dedup ? maxUq : 0;
    int shared_mem = args.partSize * args.warpPerBlock * sizeof(int)
                   + uq_smem * M * sizeof(float)
                   + args.warpPerBlock * M * sizeof(float);

    if (!args.csv_only)
        printf("  grid=%d block=%d smem=%d bytes (%.1f KB)\n", grid, block, shared_mem, shared_mem/1024.0);

    // CUDA events
    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));
    double total_ms = 0.0;
    int measured_iters = 0, total_iters = args.warmup + args.iters;

    // Iterations
    for (int iter = 0; iter < total_iters; iter++) {
        CUDA_CHECK(cudaMemset(d_output, 0, (size_t)N*sizeof(float)));

        CUDA_CHECK(cudaEventRecord(ev_start));
        sag_kernel_dedup<M><<<grid, block, shared_mem>>>(
            d_row_ptr, d_columns, d_input, d_output,
            d_partPtr, d_part2Node,
            d_uniqIds, d_uniqCounts, d_remap, d_needDedup,
            num_parts, args.partSize, args.dimWorker, args.warpPerBlock, maxUq);
        CUDA_CHECK(cudaEventRecord(ev_end));
        CUDA_CHECK(cudaEventSynchronize(ev_end));
        CUDA_CHECK(cudaGetLastError());

        if (iter >= args.warmup) {
            float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));
            total_ms += ms; measured_iters++;
        }
    }
    double avg_ms = total_ms / measured_iters;

    // Verify
    vector<float> gpu_output(N);
    CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, (size_t)N*sizeof(float), cudaMemcpyDeviceToHost));
    double max_abs=0, max_rel=0, sum_c=0, sum_g=0;
    for (int i=0;i<N;i++) {
        double e=fabs((double)gpu_output[i]-(double)cpu_output[i]);
        if (e>max_abs) max_abs=e;
        double mag=fabs((double)cpu_output[i]);
        if (mag>1e-6) { double r=e/mag; if (r>max_rel) max_rel=r; }
        sum_c+=cpu_output[i]; sum_g+=gpu_output[i];
    }
    bool pass=(max_rel<1e-4);

    if (!args.csv_only) {
        printf("\nResults (SAG+Dedup, M=%d, %d iterations):\n", M, measured_iters);
        printf("  avg kernel  : %.4f ms\n", avg_ms);
        printf("  per-comp    : %.4f ms\n", avg_ms/M);
        printf("  max rel err : %.6e\n", max_rel);
        printf("  PASS: %s\n", pass?"YES":"NO");
    }
    printf("\nCSV: %s,SAG+DEDUP,%d,%d,%d,%d,%d,%.4f,%.4f,%.6e\n",
           args.csr_dir.c_str(), M, args.partSize, args.dimWorker, args.warpPerBlock,
           measured_iters, avg_ms, avg_ms/M, max_rel);

    // Cleanup
    cudaFree(d_row_ptr); cudaFree(d_columns); cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_partPtr); cudaFree(d_part2Node);
    cudaFree(d_uniqIds); cudaFree(d_uniqCounts); cudaFree(d_remap); cudaFree(d_needDedup);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_end);
}

static void dispatch(const Args& args,
                     const vector<int64_t>& h_row_ptr,
                     const vector<int>& h_columns, int n_verts, int64_t n_edges) {
    switch (args.M) {
        case 1: run_sag_dedup<1>(args,h_row_ptr,h_columns,n_verts,n_edges); break;
        case 2: run_sag_dedup<2>(args,h_row_ptr,h_columns,n_verts,n_edges); break;
        case 4: run_sag_dedup<4>(args,h_row_ptr,h_columns,n_verts,n_edges); break;
        case 8: run_sag_dedup<8>(args,h_row_ptr,h_columns,n_verts,n_edges); break;
        case 16:run_sag_dedup<16>(args,h_row_ptr,h_columns,n_verts,n_edges); break;
        case 32:run_sag_dedup<32>(args,h_row_ptr,h_columns,n_verts,n_edges); break;
        case 64:run_sag_dedup<64>(args,h_row_ptr,h_columns,n_verts,n_edges); break;
        case 80:run_sag_dedup<80>(args,h_row_ptr,h_columns,n_verts,n_edges); break;
        case 96:run_sag_dedup<96>(args,h_row_ptr,h_columns,n_verts,n_edges); break;
        case 128:run_sag_dedup<128>(args,h_row_ptr,h_columns,n_verts,n_edges); break;
        default:
            fprintf(stderr,"Unsupported M=%d\n",args.M); exit(1);
    }
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(args.gpu_id));
    vector<int64_t> h_row_ptr; vector<int> h_columns;
    int n_verts; int64_t n_edges;
    read_graph(args.csr_dir, h_row_ptr, h_columns, n_verts, n_edges);
    if (!args.csv_only)
        printf("Graph: %s  V=%d E=%lld  M=%d partSize=%d dimWorker=%d warpPerBlock=%d\n",
               args.csr_dir.c_str(), n_verts, (long long)n_edges,
               args.M, args.partSize, args.dimWorker, args.warpPerBlock);
    dispatch(args, h_row_ptr, h_columns, n_verts, n_edges);
    return 0;
}
