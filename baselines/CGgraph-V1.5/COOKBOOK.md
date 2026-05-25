# CGgraph-V1.5 Cookbook

> 自动生成于 2026-05-18，由 /zcf:repro 生成

## 项目信息

| 项目 | 内容 |
|------|------|
| 仓库/路径 | `/home/zyl/Projects/ocgp/baselines/CGgraph-V1.5` |
| 描述 | CPU-GPU 协同图处理系统，论文 CGgraph: An Ultra-fast Graph Processing System on Modern Commodity CPU-GPU Co-processor 源码 |
| 论文 | CGgraph: An Ultra-fast Graph Processing System on Modern Commodity CPU-GPU Co-processor |
| 语言 | C++/CUDA |

## 环境要求

| 依赖 | 版本要求 | 本机版本 |
|------|---------|---------|
| CUDA | ≥ 12.0 | 12.8 |
| GCC | > 11.4.0 | 12.2.0 |
| CMake | ≥ 3.5.0 | 3.25.1 |
| OpenMP | 必需 | 4.5 |
| TBB | 必需 | 2021.8.0 |
| MPI | 必需 | OpenMPI 3.1 |
| gflags | 必需 | 2.2.2 (本地编译) |
| GPU | - | Tesla V100-SXM2-32GB × 2 (sm_70) |

## 编译步骤

```bash
# 设置环境（cicc 需要在 PATH 中）
export PATH="/home/zyl/.conda/envs/torch2.8/nvvm/bin:$PATH"

# 配置与编译
cd /home/zyl/Projects/ocgp/baselines/CGgraph-V1.5
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### 编译注意事项

- 需要将 `$CONDA_ENV/nvvm/bin` 添加到 PATH（conda 安装的 CUDA Toolkit 中 `cicc` 不在标准 bin 目录）
- 项目使用 CMake 的 `CUDA_DETECT_INSTALLED_GPUS` 自动检测 GPU 架构，无需手动设置 `-arch=sm_XX`
- 配置过程遇到并修复了多个问题，详见 [ISSUES.md](ISSUES.md)

## 运行示例

### 数据准备

CGgraph 支持两种输入格式：

#### 方式一：GGR 格式（推荐）

GGR (Galois GR) 是单文件二进制格式，可与其他 baseline（如 Subway）共用同一份数据。

```
文件布局:
[Header 32B] version(uint64) + sizeEdgeTy(uint64) + nvtxs(uint64) + nedges(uint64)
[Data]      V×int64 row_start[1..V] + E×int32 edge_dst [+ 4B padding if weighted & E odd] + E×int32 adjwgt
```

- `version` 必须为 1
- `sizeEdgeTy` = 0 表示无权图，非 0 表示有权图
- `row_start` 省略了 [0]（永远是 0），仅存储 [1..V]

使用 GGR 格式时，直接将文件路径传给 `-graphName`：

```bash
CGgraphV1.5 -graphName /path/to/graph.ggr -algorithm 0 -root 0 -runs 5
```

#### 方式二：三文件 CSR 格式（原始）

每个图需要 3 个文件：
- `native_csrOffset_u32.bin` — CSR offset 数组，类型 `countl_type`（uint32），长度为顶点数+1
- `native_csrDest_u32.bin` — CSR dest 数组，类型 `vertex_id_type`（uint32），长度为边数
- `native_csrWeight_u32.bin` — CSR weight 数组，类型 `edge_data_type`（uint32），长度为边数

数据路径需在 `src/Basic/Graph/graphFileList.hpp` 中注册。

### 运行命令

```bash
# 设置运行时库路径
export LD_LIBRARY_PATH="/home/zyl/.conda/envs/torch2.8/targets/x86_64-linux/lib:/home/zyl/.conda/envs/torch2.8/lib:$LD_LIBRARY_PATH"

# GGR 格式 - BFS
mpirun --allow-run-as-root -np 1 ./build/CGgraphV1.5 \
    -graphName /home/zyl/Projects/ocgp/baselines/CGgraph-V1.5/test_data/test.ggr \
    -algorithm 0 -gpuMemory 0 -root 0 -runs 5 -useDeviceId 0

# 三文件 CSR 格式 - BFS
mpirun --allow-run-as-root -np 1 ./build/CGgraphV1.5 \
    -graphName test -algorithm 0 -gpuMemory 0 -root 0 -runs 5 -useDeviceId 0

# SSSP（algorithm=1）
mpirun --allow-run-as-root -np 1 ./build/CGgraphV1.5 \
    -graphName test -algorithm 1 -gpuMemory 0 -root 0 -runs 5 -useDeviceId 0
```

### 参数说明

| 参数 | 含义 | 默认值 |
|------|------|--------|
| -algorithm | 算法选择：0=BFS, 1=SSSP | 0 |
| -graphName | 图数据名称或 GGR 文件路径（.ggr/.gr） | "friendster" |
| -gpuMemory | GPU 内存模式：0=GPU_MEM, 1=UVM, 2=ZERO_COPY | 0 |
| -root | BFS/SSSP 起始节点（或 PageRank 最大迭代次数） | 25689 |
| -runs | 算法运行次数 | 5 |
| -useDeviceId | 使用的 GPU 编号 | 0 |

## 输出格式

程序运行后输出：
- 图数据加载信息（顶点数、边数）
- CPU/GPU 速度测量结果（多次运行取平均）
- 每次迭代完成信息和耗时
- 最终选择 CPU-ONLY 或 CPU+GPU 协同模式
- 结果文件写入 `./CGgraphV1-5/` 目录（rank、old2new、重排后的 CSR 文件）
- 性能记录文件：`FPS_CPU_<graphName>_CGgraphRV1_5_<algo>.bin` 和 `FPS_GPU_<graphName>_CGgraphRV1_5_<algo>.bin`

## 数据格式

### 格式一：GGR（Galois GR）— 推荐

单文件二进制 CSR 格式，自带 header 元信息（顶点/边数），可与其他 baseline 共用。

```
偏移    内容                                    大小
0       version (uint64, 必须为1)                8B
8       sizeEdgeTy (uint64, 0=无权, 非0=有权)    8B
16      nvtxs (uint64, 顶点数 V)                 8B
24      nedges (uint64, 边数 E)                  8B
32      row_start[1..V] (int64, 省略 [0]=0)     V×8B
32+V×8  edge_dst[0..E-1] (int32)                E×4B
[+4B padding] (仅当有权且 E 为奇数)
        adjwgt[0..E-1] (int32, 可选)             E×4B
```

### 格式二：三文件 CSR — 原始格式

- **字节序**: Little-Endian
- **数据类型**: 由 `src/Basic/Type/data_type.hpp` 定义
  - `countl_type` (uint32) — CSR offset 的数据类型
  - `vertex_id_type` (uint32) — 顶点 ID 的数据类型
  - `edge_data_type` (uint32) — 边权重的数据类型

**注意**: 当总边数超过 uint32 范围时，需将 `countl_type` 改为 uint64；对应 offset 文件需使用 `native_csrOffset_u64.bin`。

## 可用算法/程序

| 程序名 | 算法 | 说明 |
|--------|------|------|
| CGgraphV1.5 | BFS | 广度优先搜索（-algorithm 0） |
| CGgraphV1.5 | SSSP | 单源最短路径（-algorithm 1） |
| CGgraphV1.5 | WCC | 连通分量（README 提及，待更新） |
| CGgraphV1.5 | PageRank | 页面排名（README 提及，待更新） |
