# Subway 环境配置问题记录

> 自动生成于 2026-05-17，由 /zcf:repro 生成

## 问题 1: CUDA arch flag 不匹配 V100

- **现象**: 原始 Makefile 指定 `-arch=sm_60`（Pascal），V100 GPU 需 `sm_70`
- **原因**: 仓库原始目标硬件为 Pascal 架构 GPU
- **解决方案**: 修改 3 个 Makefile 中的 `NFLAGS=-arch=sm_60` 为 `NFLAGS=-arch=sm_70`
  - `Makefile`（根目录）
  - `shared/Makefile`yi
  - `subway/Makefile`
- **适用条件**: 所有 sm_70+ 的 GPU（V100, T4, etc.）。A100 需改为 sm_80。

## 问题 2: 并行编译失败 (make -j)

- **现象**: `make -j$(nproc)` 报错 `No rule to make target 'subway/bfs-sync.o'`
- **原因**: 根 Makefile 的 `all` 目标中，`make1/make2/make3`（编译 .o 文件）与链接步骤之间缺少显式依赖关系。并行模式下 make 可能在 .o 文件生成前尝试链接。
- **解决方案**: 使用 `make` 串行编译。不要使用 `make -j`。
- **适用条件**: 所有环境。这是 Makefile 的结构性问题。

## 问题 3: converter 工具 segfault（已修复）

- **现象**: `tools/converter /tmp/test_graph.el` 产生 Segmentation fault (exit code 139)
- **原因**: `converter.cpp` 第 76 行（el 分支）和第 144 行（wel 分支）中 `outDegreeCounter = new uint[num_nodes]` 未初始化为 0。`new uint[]` 不会自动清零，导致读取垃圾值计算错误的内存地址。
- **解决方案**: 将 `new uint[num_nodes]` 改为 `new uint[num_nodes]()`（值初始化，自动清零）。同时删除了未使用的 `uint location;` 声明。
- **适用条件**: 所有环境。这是原始代码的 bug。
- **修改文件**: `tools/converter.cpp`（备份在 `tools/converter.cpp.bak`）

## 问题 4: 小图 GPU OOM

- **现象**: SSSP/SSWP 在 5 节点测试图上报 `GPUassert: out of memory subgraph.cu 41`
- **原因**: 测试图过小，GPU 内存分配的最小粒度导致实际分配失败。Subway 设计用于处理大规模图（百万节点以上）。
- **解决方案**: 使用真实数据集（如 cit-Patents, soc-LiveJournal1 等）。这不是 bug，是使用场景不匹配。
- **适用条件**: 仅在输入图节点数极少（< 100）时出现。
