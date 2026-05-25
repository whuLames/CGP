/*
 * ge_spmm_push_bench.cu
 * Standalone single-GPU all-push SpMM benchmark.
 *
 * The input graph is read as the same sparse matrix used by ge_spmm_bench:
 *   A[dst, src] = 1
 *   C = A * B
 *
 * For undirected graphs stored with both directions, A == A^T.  This
 * benchmark directly interprets each input CSR row as one source vertex's
 * outgoing neighbor list:
 *   for src:
 *     for dst in out_neighbors(src):
 *       atomicAdd(C[dst, c], B[src, c])
 *
 * Supported inputs:
 *   1. CSR directory containing csr_vlist.bin and csr_elist.bin
 *   2. Galois .gr file:
 *      header: version(u64), sizeEdgeTy(u64), nvtxs(u64), nedges(u64)
 *      row_start[1..V] as u64, edge_dst[E] as int32
 *
 * Usage:
 *   ./ge_spmm_push_bench <csr_dir_or_gr_file> --M=N [--tile-row=N]
 *                        [--iters=N] [--warmup=N] [--gpu=ID] [--csv-only]
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

using namespace std;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* ================================================================
 * All-push kernels over CSR rows interpreted as source vertices.
 * ================================================================ */
template<int M>
__global__ void spmm_push_atomic_kernel(
    int n_src,
    const int64_t* __restrict__ adj_row_ptr,
    const int* __restrict__ adj_columns,
    const float* __restrict__ input,
    float* __restrict__ output,
    int tile_row)
{
    int src = tile_row * blockIdx.x + threadIdx.y;
    if (src >= n_src) return;

    int c = threadIdx.x;
    float x = input[(int64_t)src * M + c];

    int64_t begin = adj_row_ptr[src];
    int64_t end = adj_row_ptr[src + 1];
    for (int64_t e = begin; e < end; ++e) {
        int dst = adj_columns[e];
        atomicAdd(&output[(int64_t)dst * M + c], x);
    }
}

template<int M, int CF>
__global__ void spmm_push_atomic_warp_kernel(
    int n_src,
    const int64_t* __restrict__ adj_row_ptr,
    const int* __restrict__ adj_columns,
    const float* __restrict__ input,
    float* __restrict__ output,
    int tile_row)
{
    int src = tile_row * blockIdx.x + threadIdx.y;
    if (src >= n_src) return;

    int c0 = blockIdx.y * (CF << 5) + threadIdx.x;
    int valid = M - c0;
    if (valid <= 0) return;

    float x[CF];
#pragma unroll
    for (int c = 0; c < CF; ++c) {
        int col = c0 + c * 32;
        x[c] = (col < M) ? input[(int64_t)src * M + col] : 0.0f;
    }

    int64_t begin = adj_row_ptr[src];
    int64_t end = adj_row_ptr[src + 1];
    for (int64_t e = begin; e < end; ++e) {
        int dst = adj_columns[e];
        int64_t base = (int64_t)dst * M + c0;
#pragma unroll
        for (int c = 0; c < CF; ++c) {
            int col = c0 + c * 32;
            if (col < M) atomicAdd(&output[base + c * 32], x[c]);
        }
    }
}

/* ================================================================
 * Graph I/O
 * ================================================================ */
static bool has_suffix(const string& s, const string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

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

static void read_csr_dir(const string& path,
                         vector<int64_t>& row_ptr,
                         vector<int>& columns,
                         int& n_verts,
                         int64_t& n_edges) {
    {
        ifstream vf(path + "/csr_vlist.bin", ios::binary | ios::ate);
        if (!vf) { cerr << "Cannot open " << path << "/csr_vlist.bin\n"; exit(1); }
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

static void read_gr_file(const string& path,
                         vector<int64_t>& row_ptr,
                         vector<int>& columns,
                         int& n_verts,
                         int64_t& n_edges,
                         uint64_t& size_edge_ty) {
    ifstream f(path, ios::binary | ios::ate);
    if (!f) { cerr << "Cannot open " << path << "\n"; exit(1); }
    uint64_t file_size = (uint64_t)f.tellg();
    f.seekg(0);

    uint64_t header[4];
    f.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!f) { cerr << "Cannot read .gr header: " << path << "\n"; exit(1); }

    uint64_t version = header[0];
    size_edge_ty = header[1];
    uint64_t nv = header[2];
    uint64_t ne = header[3];
    if (version == 0 || nv > (uint64_t)INT32_MAX || ne > (uint64_t)INT64_MAX) {
        cerr << "Unsupported .gr header in " << path << "\n";
        exit(1);
    }

    uint64_t min_size = 32ULL + nv * 8ULL + ne * 4ULL;
    if (file_size < min_size) {
        cerr << "Truncated .gr file: " << path << "\n";
        exit(1);
    }

    n_verts = (int)nv;
    n_edges = (int64_t)ne;
    row_ptr.assign(n_verts + 1, 0);
    vector<uint64_t> row_end(n_verts);
    f.read(reinterpret_cast<char*>(row_end.data()), (streamsize)(nv * 8ULL));
    if (!f) { cerr << "Cannot read .gr row_start: " << path << "\n"; exit(1); }
    for (int i = 0; i < n_verts; ++i) row_ptr[i + 1] = (int64_t)row_end[i];

    columns.resize((size_t)n_edges);
    f.read(reinterpret_cast<char*>(columns.data()), (streamsize)(ne * 4ULL));
    if (!f) { cerr << "Cannot read .gr edge_dst: " << path << "\n"; exit(1); }
}

static void read_graph(const string& path,
                       vector<int64_t>& row_ptr,
                       vector<int>& columns,
                       int& n_verts,
                       int64_t& n_edges,
                       string& format_desc) {
    if (has_suffix(path, ".gr")) {
        uint64_t size_edge_ty = 0;
        read_gr_file(path, row_ptr, columns, n_verts, n_edges, size_edge_ty);
        format_desc = ".gr(sizeEdgeTy=" + to_string(size_edge_ty) + ")";
    } else {
        read_csr_dir(path, row_ptr, columns, n_verts, n_edges);
        format_desc = "csr_dir";
    }
}

/* ================================================================
 * Host helpers
 * ================================================================ */
static void cpu_pull_reference(const vector<int64_t>& row_ptr,
                               const vector<int>& columns,
                               const vector<float>& input,
                               vector<float>& output,
                               int n_verts, int M) {
    fill(output.begin(), output.end(), 0.0f);
    for (int dst = 0; dst < n_verts; ++dst) {
        for (int64_t e = row_ptr[dst]; e < row_ptr[dst + 1]; ++e) {
            int src = columns[e];
            for (int c = 0; c < M; ++c)
                output[(int64_t)dst * M + c] += input[(int64_t)src * M + c];
        }
    }
}

/* ================================================================
 * Argument parsing
 * ================================================================ */
struct Args {
    string graph_path;
    int M = 4;
    int tile_row = 8;
    int iters = 30;
    int warmup = 5;
    int gpu_id = 0;
    bool csv_only = false;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <csr_dir_or_gr_file> --M=N [--tile-row=N] [--iters=N] "
            "[--warmup=N] [--gpu=ID] [--csv-only]\n", argv[0]);
        exit(1);
    }
    a.graph_path = argv[1];
    for (int i = 2; i < argc; i++) {
        string s = argv[i];
        if (s.rfind("--M=", 0) == 0)        a.M = atoi(s.c_str() + 4);
        if (s.rfind("--tile-row=", 0) == 0) a.tile_row = atoi(s.c_str() + 11);
        if (s.rfind("--iters=", 0) == 0)    a.iters = atoi(s.c_str() + 8);
        if (s.rfind("--warmup=", 0) == 0)   a.warmup = atoi(s.c_str() + 9);
        if (s.rfind("--gpu=", 0) == 0)      a.gpu_id = atoi(s.c_str() + 6);
        if (s == "--csv-only")              a.csv_only = true;
    }
    return a;
}

template<int M> struct CFSelector { static constexpr int value = (M + 31) / 32; };

template<int M>
static void run_push_spmm(const Args& args,
                          const vector<int64_t>& h_row_ptr,
                          const vector<int>& h_columns,
                          int n_verts, int64_t n_edges) {
    int64_t N64 = (int64_t)n_verts * M;
    if (N64 < 0 || (uint64_t)N64 > (uint64_t)SIZE_MAX / sizeof(float)) {
        cerr << "Input/output vectors are too large\n";
        exit(1);
    }
    size_t N = (size_t)N64;

    int64_t* d_adj_row_ptr;
    int* d_adj_columns;
    float* d_input;
    float* d_output;

    CUDA_CHECK(cudaMalloc(&d_adj_row_ptr, (size_t)(n_verts + 1) * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_adj_columns, (size_t)n_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_adj_row_ptr, h_row_ptr.data(),
                          (size_t)(n_verts + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adj_columns, h_columns.data(),
                          (size_t)n_edges * sizeof(int), cudaMemcpyHostToDevice));

    if (!args.csv_only)
        printf("  Initializing M=%d vector data (%zu floats, %.2f MB)...\n",
               M, N, (double)N * sizeof(float) / 1e6);

    vector<float> h_input(N);
    {
        mt19937 rng(42);
        uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < N; ++i) h_input[i] = dist(rng);
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    vector<float> cpu_output(N);
    cpu_pull_reference(h_row_ptr, h_columns, h_input, cpu_output, n_verts, M);

    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    double total_ms = 0.0;
    int measured_iters = 0;
    int total_iters = args.warmup + args.iters;

    if constexpr (M < 32) {
        int eff_tile = max(args.tile_row, max(1, 128 / M));
        int grid_x = (n_verts + eff_tile - 1) / eff_tile;
        dim3 block(M, eff_tile);

        if (!args.csv_only)
            printf("  [push-atomic] grid=%d, block=(%d,%d)\n", grid_x, M, eff_tile);

        for (int iter = 0; iter < total_iters; ++iter) {
            CUDA_CHECK(cudaMemset(d_output, 0, N * sizeof(float)));
            CUDA_CHECK(cudaEventRecord(ev_start));
            spmm_push_atomic_kernel<M><<<grid_x, block>>>(
                n_verts, d_adj_row_ptr, d_adj_columns, d_input, d_output, eff_tile);
            CUDA_CHECK(cudaEventRecord(ev_end));
            CUDA_CHECK(cudaEventSynchronize(ev_end));
            CUDA_CHECK(cudaGetLastError());

            if (iter >= args.warmup) {
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));
                total_ms += ms;
                measured_iters++;
            }
        }
    } else {
        constexpr int CF = CFSelector<M>::value;
        int tile_row = args.tile_row;
        int grid_x = (n_verts + tile_row - 1) / tile_row;
        int grid_y = (M + CF * 32 - 1) / (CF * 32);
        dim3 block(32, tile_row);

        if (!args.csv_only)
            printf("  [push-atomic-warp] grid=(%d,%d), block=(32,%d), CF=%d\n",
                   grid_x, grid_y, tile_row, CF);

        for (int iter = 0; iter < total_iters; ++iter) {
            CUDA_CHECK(cudaMemset(d_output, 0, N * sizeof(float)));
            CUDA_CHECK(cudaEventRecord(ev_start));
            spmm_push_atomic_warp_kernel<M, CF><<<dim3(grid_x, grid_y), block>>>(
                n_verts, d_adj_row_ptr, d_adj_columns, d_input, d_output, tile_row);
            CUDA_CHECK(cudaEventRecord(ev_end));
            CUDA_CHECK(cudaEventSynchronize(ev_end));
            CUDA_CHECK(cudaGetLastError());

            if (iter >= args.warmup) {
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));
                total_ms += ms;
                measured_iters++;
            }
        }
    }

    double avg_ms = total_ms / measured_iters;

    vector<float> gpu_output(N);
    CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    double max_abs_err = 0.0, max_rel_err = 0.0;
    double sum_cpu = 0.0, sum_gpu = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double err = fabs((double)gpu_output[i] - (double)cpu_output[i]);
        max_abs_err = max(max_abs_err, err);
        double mag = fabs((double)cpu_output[i]);
        if (mag > 1e-6) max_rel_err = max(max_rel_err, err / mag);
        sum_cpu += cpu_output[i];
        sum_gpu += gpu_output[i];
    }
    bool pass = (max_rel_err < 1e-4);

    if (!args.csv_only) {
        printf("\nResults (All-Push SpMM, M=%d, %d measured iters, %d warmup):\n",
               M, measured_iters, args.warmup);
        printf("  avg push_spmm      : %.4f ms\n", avg_ms);
        printf("  per-component avg  : %.4f ms\n", avg_ms / M);
        printf("  max abs error      : %.6e\n", max_abs_err);
        printf("  max rel error      : %.6e\n", max_rel_err);
        printf("  CPU sum            : %.6f\n", sum_cpu);
        printf("  GPU sum            : %.6f\n", sum_gpu);
        printf("  PASS: %s\n", pass ? "YES" : "NO (ERROR)");
    }

    printf("\nCSV: %s,PushSpMMAtomic,%d,%d,%d,%.4f,%.4f,%.6e\n",
           args.graph_path.c_str(), M, args.tile_row,
           measured_iters, avg_ms, avg_ms / M, max_rel_err);

    cudaFree(d_adj_row_ptr);
    cudaFree(d_adj_columns);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
}

static void dispatch(const Args& args,
                     const vector<int64_t>& h_row_ptr,
                     const vector<int>& h_columns,
                     int n_verts, int64_t n_edges) {
    switch (args.M) {
        case 1:   run_push_spmm<1>  (args, h_row_ptr, h_columns, n_verts, n_edges); break;
        case 2:   run_push_spmm<2>  (args, h_row_ptr, h_columns, n_verts, n_edges); break;
        case 4:   run_push_spmm<4>  (args, h_row_ptr, h_columns, n_verts, n_edges); break;
        case 8:   run_push_spmm<8>  (args, h_row_ptr, h_columns, n_verts, n_edges); break;
        case 16:  run_push_spmm<16> (args, h_row_ptr, h_columns, n_verts, n_edges); break;
        case 32:  run_push_spmm<32> (args, h_row_ptr, h_columns, n_verts, n_edges); break;
        case 64:  run_push_spmm<64> (args, h_row_ptr, h_columns, n_verts, n_edges); break;
        case 80:  run_push_spmm<80> (args, h_row_ptr, h_columns, n_verts, n_edges); break;
        case 96:  run_push_spmm<96> (args, h_row_ptr, h_columns, n_verts, n_edges); break;
        case 128: run_push_spmm<128>(args, h_row_ptr, h_columns, n_verts, n_edges); break;
        default:
            fprintf(stderr, "Unsupported M=%d. Supported: 1,2,4,8,16,32,64,80,96,128\n",
                    args.M);
            exit(1);
    }
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(args.gpu_id));

    vector<int64_t> h_row_ptr;
    vector<int> h_columns;
    int n_verts = 0;
    int64_t n_edges = 0;
    string format_desc;
    read_graph(args.graph_path, h_row_ptr, h_columns, n_verts, n_edges, format_desc);

    if (!args.csv_only) {
        printf("Graph: %s\n", args.graph_path.c_str());
        printf("  format = %s\n", format_desc.c_str());
        printf("  V = %d, E = %lld\n", n_verts, (long long)n_edges);
        printf("  M = %d, tile_row = %d, iters = %d, warmup = %d\n",
               args.M, args.tile_row, args.iters, args.warmup);
        printf("  Treating input CSR rows as source outgoing neighbor lists (requires undirected/symmetric graph).\n");
    }

    dispatch(args, h_row_ptr, h_columns, n_verts, n_edges);
    return 0;
}
