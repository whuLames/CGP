# Subway Cookbook

> 自动生成于 2026-05-17，由 /zcf:repro 生成

## 项目信息

| 项目 | 内容 |
|------|------|
| 仓库/路径 | `/home/zyl/Projects/ocgp/baselines/Subway` |
| 描述 | Out-of-GPU-memory 图处理框架，通过只传输活跃子图来减少 CPU-GPU 数据传输量 |
| 论文 | [EUROSYS'20] Subway: Minimizing Data Transfer during Out-of-GPU-Memory Graph Processing |
| 前作 | [ASPLOS'18] Tigr: Transforming Irregular Graphs for GPU-Friendly Graph Processing |
| 语言 | C++/CUDA (C++11) |

## 环境要求

| 依赖 | 版本要求 | 本机版本 |
|------|---------|---------|
| CUDA Toolkit | 任意 | 12.8 |
| GCC | 支持 C++11 | 12.2.0 |
| GPU | 任意 NVIDIA GPU | Tesla V100-SXM2-32GB (sm_70) |
| Make | GNU Make | 4.3 |

无其他外部依赖（无 NCCL/MPI/Boost 等）。

## 编译步骤

```bash
cd /home/zyl/Projects/ocgp/baselines/Subway

# 直接编译（必须串行，不能用 -j）
make
```

### 编译注意事项

1. **CUDA arch 必须匹配 GPU**：原始 Makefile 为 `-arch=sm_60`（Pascal），V100 需改为 `-arch=sm_70`。涉及 3 个 Makefile：
   - `Makefile`（根目录）
   - `shared/Makefile`
   - `subway/Makefile`
   - `tools/Makefile` **不需要修改**（只用 g++ 编译）

2. **不能并行编译**：根 Makefile 的 `all` 目标中 `make1/make2/make3` 与链接步骤之间缺少正确的依赖声明，`make -j` 会失败。必须使用 `make`（串行）。

3. **nvcc deprecation 警告**：CUDA 12.8 会提示 sm_70 在未来版本将被废弃，可忽略。

## 数据准备

### 输入格式

| 格式 | 扩展名 | 类型 | 说明 |
|------|--------|------|------|
| Edge List | `.el` | 文本 | 每行 `SOURCE DESTINATION` |
| Weighted Edge List | `.wel` | 文本 | 每行 `SOURCE DESTINATION WEIGHT` |
| Binary CSR | `.bcsr` | 二进制 | 由 converter 从 .el 生成（**推荐**） |
| Binary Weighted CSR | `.bwcsr` | 二进制 | 由 converter 从 .wel 生成（**推荐**） |

### 格式转换

```bash
# 无权图：.el → .bcsr
tools/converter path/to/graph.el

# 有权图：.wel → .bwcsr
tools/converter path/to/graph.wel
```

转换后二进制文件与原文件同目录，仅扩展名不同。

### 使用已有数据集

Subway 的 .bcsr 格式与常见 CSR 格式兼容。如果已有 CSR 数据（如 `/home/zyl/data/csr_data/tcr_data/tcr_data/` 中的 `cit-Patents`），需确认二进制格式是否匹配（Subway 期望前两个 uint32 为 num_nodes 和 num_edges）。

## 运行示例

### BFS（无权图）

```bash
./bfs-sync --input /path/to/graph.bcsr
./bfs-async --input /path/to/graph.bcsr
```

### PageRank（无权图）

```bash
./pr-sync --input /path/to/graph.bcsr
./pr-async --input /path/to/graph.bcsr
```

### Connected Components（无权图）

```bash
./cc-sync --input /path/to/graph.bcsr
./cc-async --input /path/to/graph.bcsr
```

### SSSP / SSWP（有权图）

```bash
./sssp-sync --input /path/to/graph.bwcsr --source 0
./sssp-async --input /path/to/graph.bwcsr --source 0
./sswp-sync --input /path/to/graph.bwcsr --source 0
./sswp-async --input /path/to/graph.bwcsr --source 0
```

### 参数说明

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--input` | 输入图文件路径（必选） | 无 |
| `--source` | 起始顶点 ID | 0 |

## 输出格式

```
Reading the input graph from the following file:
>> <input_path>
Done reading.
Number of nodes = <N>
Number of edges = <E>
Graph Reading finished in <time> (s).
Processing finished in <time> (s).
Number of iterations = <iter_count>
Results of first 5 nodes:
[0:<val> 1:<val> 2:<val> ...]
```

- BFS/CC：结果为整数（距离/组件 ID）
- PageRank：结果为浮点数（PR 值）
- SSSP/SSWP：结果为整数（最短路径权重）
- `4294967294` = UINT_MAX - 1，表示未访问到的节点

## 可用算法/程序

| 程序名 | 算法 | 计算模式 | 输入类型 |
|--------|------|---------|---------|
| `bfs-sync` | Breadth-First Search | 同步 | 无权 |
| `bfs-async` | Breadth-First Search | 异步 | 无权 |
| `cc-sync` | Connected Components | 同步 | 无权 |
| `cc-async` | Connected Components | 异步 | 无权 |
| `pr-sync` | PageRank | 同步 | 无权 |
| `pr-async` | PageRank | 异步 | 无权 |
| `sssp-sync` | Single-Source Shortest Path | 同步 | 有权 |
| `sssp-async` | Single-Source Shortest Path | 异步 | 有权 |
| `sswp-sync` | Single-Source Widest Path | 同步 | 有权 |
| `sswp-async` | Single-Source Widest Path | 异步 | 有权 |
| `tools/converter` | 格式转换工具 | - | .el/.wel |

sync = 每个 global iteration 处理完整图；async = 子图加载后异步处理，通常减少迭代次数。
