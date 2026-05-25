# Glign Cookbook

> 自动生成于 2026-05-18，由 /zcf:repro 生成

## 项目信息

| 项目 | 内容 |
|------|------|
| 仓库 | https://github.com/xyin014/Glign-AE |
| 本地路径 | /home/zyl/Projects/ocgp/baselines/Glign |
| 描述 | 并发图查询运行时系统，通过对齐多个图遍历来提升数据局部性 |
| 论文 | Glign: Taming Misaligned Graph Traversals in Concurrent Graph Processing (ASPLOS'23) |
| 语言 | C++ (基于 Ligra 框架) |
| 协议 | MIT |

## 环境要求

| 依赖 | 原始要求 | 本机版本 | 备注 |
|------|---------|---------|------|
| GCC | 5.3.0~7.4.0 (Cilk Plus) | 12.2.0 | **使用 OpenMP 替代 Cilk Plus** |
| CMake | 无 | N/A | 使用 Makefile 构建 |
| Make | GNU Make | ✓ | |
| OpenMP | 可选 | ✓ | 默认需要 Cilk Plus，已改为 OpenMP |
| Perf | 可选（profiling） | ✓ | 仅 `run_profilings.sh` 需要 |

## 编译步骤

### 1. 进入 apps 目录

```bash
cd /home/zyl/Projects/ocgp/baselines/Glign/apps
```

### 2. 编译所有 benchmark

```bash
# 单独编译
make OPENMP=1 EDGELONG=1 LONG=1 SSSP_Batch
make OPENMP=1 EDGELONG=1 LONG=1 BFS_Batch
make OPENMP=1 EDGELONG=1 LONG=1 SSWP_Batch
make OPENMP=1 EDGELONG=1 LONG=1 SSNP_Batch
make OPENMP=1 EDGELONG=1 LONG=1 Viterbi_Batch

# 或使用修改后的 compile.sh 一键编译
bash compile.sh
```

### 3. 清理

```bash
make clean
```

### 编译注意事项

1. **Cilk Plus → OpenMP**：原始代码要求 GCC 5.3~7.4 的 Cilk Plus 支持。当前 GCC 12.2 已移除 Cilk Plus，因此使用 `OPENMP=1` 替代。`ligra/parallel.h` 已内置 OpenMP 适配层。
2. **cilk_spawn/cilk_sync 退化为串行**：`quickSort.h` 和 `transpose.h` 中的 `cilk_spawn`/`cilk_sync` 在 OpenMP 模式下退化为普通函数调用（执行结果正确，但排序/转置不并行）。
3. **编译 warning**：`ligra.h:1395` 有一个 `LONG_MAX` 到 `int` 的隐式转换溢出警告，不影响功能。
4. **Heter_Batch 未编译**：原始 `compile.sh` 需要切换到 `ae-heter` 分支编译 `Heter_Batch`，当前分支不含该源文件。

## 运行示例

### 数据准备

Glign 使用 Ligra 的 **加权邻接表格式**（Weighted Adjacency Graph）。自带小数据集在 `inputs/` 目录下：

```bash
# 查看自带数据集
ls ../inputs/
# rMatGraph_J_5_100     (无权重)
# rMatGraph_WJ_5_100    (有权重)
```

大型图数据需要从 [Google Drive](https://drive.google.com/drive/folders/1VNzred7_cdvoyvwyQdBnQkLJZqaDFYuV?usp=sharing) 下载，放入 `ae-data/` 目录。

数据格式（Ligra Weighted Adjacency）：
```
WeightedAdjacencyGraph
<顶点数>
<边数>
<offset[0]>
<offset[1]>
...
<offset[n]>        (共 n+1 个 offset)
<邻居顶点> <边权重>  (共 m 对)
...
```

### 准备 query 文件

query 文件为纯文本，每行一个源顶点 ID：

```bash
# 示例：创建包含 4 个查询的文件
printf "0\n1\n2\n3\n" > /tmp/test_queries.txt
```

### 运行命令

**注意：graph 路径必须是最后一个参数。**

```bash
cd /home/zyl/Projects/ocgp/baselines/Glign/apps

# SSSP (单源最短路径) 并发查询
./SSSP_Batch -option glign -batch 2 -max_combination 4 -mode 3 \
    -qf /tmp/test_queries.txt ../inputs/rMatGraph_WJ_5_100

# BFS (广度优先搜索) 并发查询
./BFS_Batch -option glign -batch 2 -max_combination 4 -mode 3 \
    -qf /tmp/test_queries.txt ../inputs/rMatGraph_WJ_5_100

# SSWP (单源最宽路径) 并发查询
./SSWP_Batch -option glign -batch 2 -max_combination 4 -mode 3 \
    -qf /tmp/test_queries.txt ../inputs/rMatGraph_WJ_5_100

# SSNP (单源最窄路径) 并发查询
./SSNP_Batch -option glign -batch 2 -max_combination 4 -mode 3 \
    -qf /tmp/test_queries.txt ../inputs/rMatGraph_WJ_5_100

# Viterbi 并发查询
./Viterbi_Batch -option glign -batch 2 -max_combination 4 -mode 3 \
    -qf /tmp/test_queries.txt ../inputs/rMatGraph_WJ_5_100
```

### 运行完整实验

```bash
# 使用 LiveJournal 数据集（需先下载数据到 ae-data/）
bash run_experiments.sh LJ ../ae-data/soc-LiveJournal1.weighted.adj ../query_input/LJ_queries.txt

# 结果存储在 ../results/LJ/ 目录
```

### 参数说明

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `-option` | 运行模式：`glign`（同构查询）/ `glign-heter`（异构查询）/ `ligra-c`（Ligra-C 对比）/ `ground-truth`（验证） | 无 |
| `-batch` | 每批并发查询数 | 4 |
| `-max_combination` | 总查询数 | 256 |
| `-mode` | Glign 模式：1=顺序，2=Intra/Inter 对齐，3=完整 Glign | 1 |
| `-delay` | 启用 inter-iteration alignment（需配合 `-mode 2` 或 `3`） | 关 |
| `-qf` | query 源顶点文件路径 | 无（必选） |
| `-s` | 对称图 | 自动检测 |
| `-b` | 二进制图格式 | 关 |
| `-gr` | GGR (Galois GR) 二进制图格式 | 关 |
| `-m` | mmap 读取 | 关 |
| `[graph]` | 图文件路径（**必须为最后一个参数**） | 无（必选） |

### 变体参数速查

| 变体 | 参数 |
|------|------|
| Ligra-Seq | `-option glign -mode 1` |
| Ligra-C | `-option ligra-c` |
| Glign-Intra | `-option glign -mode 2` |
| Glign-Inter | `-option glign -mode 2 -delay` |
| Glign-Batch | `-option glign -mode 3` |
| **Glign (完整)** | `-option glign -mode 3 -delay` |

## 输出格式

程序输出到 stdout，格式示例：

```
running batched queries on glign
graph file name: <路径>
query file name: <路径>
number of random queries: <查询数>
asymmetric graph
n=<顶点数> m=<边数>
Profiling cost: <分析耗时秒数>

Glign-Batch evaluation..
Glign-Batch evaluation time: <评估耗时秒数>
Glign-Batch F: <前沿大小1>
Glign-Batch F: <前沿大小2>
```

完整实验结果存储在 `results/<图名>/` 目录，每个文件包含详细的计时和前沿大小信息。

## 数据格式

### 输入图格式：Ligra Weighted Adjacency

文本格式，首行标识 `WeightedAdjacencyGraph`，后面依次是顶点数、边数、offset 数组（n+1 个 int）、然后是 m 对 (邻居, 权重)。

### Query 文件格式

纯文本，每行一个整数，表示查询的源顶点 ID。

## 可用算法/程序

| 程序名 | 算法 | 说明 |
|--------|------|------|
| `SSSP_Batch` | 单源最短路径 (Single-Source Shortest Path) | Dijkstra 风格并发 SSSP |
| `BFS_Batch` | 广度优先搜索 (Breadth-First Search) | 并发 BFS |
| `SSWP_Batch` | 单源最宽路径 (Single-Source Widest Path) | 并发最宽路径 |
| `SSNP_Batch` | 单源最窄路径 (Single-Source Narrowest Path) | 并发最窄路径 |
| `Viterbi_Batch` | Viterbi 译码 | 并发 Viterbi 算法 |
| `Heter_Batch` | 异构混合查询 | 需 `ae-heter` 分支，混合 BFS/SSSP/SSWP/SSNP |

## 工具程序

位于 `utils/` 目录，用于数据格式转换：

```bash
cd /home/zyl/Projects/ocgp/baselines/Glign/utils

# SNAP 格式 → Ligra 邻接表
make OPENMP=1 SNAPtoAdj
./SNAPtoAdj <input.snap> <output.adj>

# 邻接表添加随机权重
make OPENMP=1 adjGraphAddWeights
./adjGraphAddWeights <input.adj> <output.weighted.adj>

# 邻接表 → 二进制
make OPENMP=1 adjToBinary
./adjToBinary <input.adj>

# Ligra 邻接表 → GGR 二进制格式
g++ -std=c++14 -O3 -o adjToGGR adjToGGR.C
./adjToGGR <input.adj> <output.gr>
```

### GGR 格式使用

GGR 是 Galois GR 二进制图格式，相比文本邻接表可节省 3~5 倍磁盘空间。使用 `-gr` 参数启用：

```bash
# 用 GGR 格式运行
./SSSP_Batch -option glign -batch 64 -max_combination 512 -mode 3 -delay \
    -gr -qf ../query_input/LJ_queries.txt ../ae-data/LJ.gr

# 用转换工具将 Ligra 邻接表转为 GGR
cd ../utils && g++ -std=c++14 -O3 -o adjToGGR adjToGGR.C
./adjToGGR ../ae-data/soc-LiveJournal1.weighted.adj ../ae-data/LJ.gr
```

GGR 格式规范：
```
[32B 头部] version(u64) | sizeEdgeTy(u64) | nvtxs(u64) | nedges(u64)
[V*8B]     row_start[1..V] (int64, 省略 row_start[0]=0)
[E*4B]     edge_dst[E] (int32)
[可选 4B]  padding (仅 weighted 且 E 为奇数)
[E*4B]     adjwgt[E] (int32, 仅 weighted)
```

## 修改记录

| 文件 | 修改内容 |
|------|---------|
| `ligra/IO.h` | 新增 `readGraphFromGGR()` 函数，支持 GGR 二进制格式读取 |
| `ligra/ligra.h` | 7 处 `bool binary` 后增加 `bool gr = P.getOptionValue("-gr")`，8 处 `readGraph` 调用传入 `gr` 参数 |
| `utils/adjToGGR.C` | 新增 Ligra 邻接表 → GGR 格式转换工具 |

## 配置过程

配置过程顺利，无已知问题。编译使用 `OPENMP=1` 替代已废弃的 Cilk Plus，功能验证通过。
