# CGgraph-V1.5 环境配置与功能扩展记录

> 自动生成于 2026-05-18，由 /zcf:repro 生成，GGR 格式支持于 2026-05-18 添加

## 问题 1: CUDA 头文件路径硬编码

- **现象**: 编译报错 `fatal error: /usr/local/cuda-11.7/include/cuda_runtime.h: No such file or directory`
- **原因**: `src/Basic/CUDA/cuda_include.cuh` 中硬编码了 CUDA 11.7 的绝对路径，且通过 `#define CUDA_INCLUDE_TEMP` 启用
- **解决方案**: 注释掉 `#define CUDA_INCLUDE_TEMP`，使用标准 `#include <cuda_runtime.h>` 等系统路径
- **修改文件**: `src/Basic/CUDA/cuda_include.cuh`（已备份为 `.bak`）
- **适用条件**: CUDA Toolkit 不安装在 `/usr/local/cuda-11.7/` 的所有环境

## 问题 2: gflags 路径硬编码

- **现象**: CMakeLists.txt 中 gflags 的 include 和 lib 路径指向 `/home/omnisky/cpj_app/gflag/gflags-master/build/`
- **原因**: 原始开发环境路径硬编码，且系统无 sudo 权限安装 libgflags-dev
- **解决方案**: 将路径改为本机已有 gflags 编译产物：`/home/zyl/Projects/baselines/groute/build/deps/gflags/`
- **修改文件**: `CMakeLists.txt` 第 108-109 行和第 153 行（已备份为 `.bak`）
- **适用条件**: 非原始开发环境

## 问题 3: TBB 库路径硬编码

- **现象**: CMakeLists.txt 中 TBB 路径指向 `/usr/local/tbb-2019_U8/lib/libtbb.so`（不存在）
- **原因**: 原始开发环境使用旧版 TBB，本机 TBB 通过 apt 安装在不同路径
- **解决方案**: 将路径改为系统 TBB 库路径 `/usr/lib/x86_64-linux-gnu/libtbb.so`
- **修改文件**: `CMakeLists.txt` 第 154 行（已备份为 `.bak`）
- **适用条件**: TBB 通过系统包管理器安装的环境

## 问题 4: CUDA 库路径错误 (lib64 vs lib)

- **现象**: 链接报错 `No rule to make target '.../lib64/stubs/libcuda.so'`
- **原因**: CMakeLists.txt 中 CUDA 库路径使用 `lib64/` 后缀，但 conda 安装的 CUDA Toolkit 库在 `lib/` 目录下
- **解决方案**: 将 `lib64/` 改为 `lib/`，并使用 `link_directories` + 库名的方式替代绝对路径；nvToolsExt 使用 `.so.1` 全路径
- **修改文件**: `CMakeLists.txt` 第 64-67 行和第 152 行（已备份为 `.bak`）
- **适用条件**: CUDA Toolkit 通过 conda 安装的环境

## 问题 5: cicc 编译器不在 PATH 中

- **现象**: 编译报错 `sh: 1: cicc: not found`
- **原因**: conda 安装的 CUDA Toolkit 中 `cicc`（CUDA 中间编译器）位于 `nvvm/bin/` 子目录，不在默认 `bin/` 中
- **解决方案**: 编译前执行 `export PATH="/home/zyl/.conda/envs/torch2.8/nvvm/bin:$PATH"`
- **修改文件**: 无源码修改，仅需环境变量设置
- **适用条件**: CUDA Toolkit 通过 conda 安装的环境

## 问题 6: 链接库名称不匹配

- **现象**: 链接报错 `cannot find -lomp`、`cannot find -lnvToolsExt`、`cannot find -lucp`
- **原因**:
  - `-lomp`: 系统 OpenMP 运行时库为 `libgomp.so`（GCC），不是 `libomp.so`（Intel/LLVM）
  - `-lnvToolsExt`: 库路径不在链接搜索路径中
  - `-lucp`: 系统 `libucp.so` 缺少无版本号的符号链接
- **解决方案**:
  - `omp` → `gomp`
  - nvToolsExt 使用绝对路径 `${CUDA_TOOLKIT_TARGET_DIR}/lib/libnvToolsExt.so.1`
  - ucp 使用绝对路径 `/usr/lib/x86_64-linux-gnu/libucp.so.0`
- **修改文件**: `CMakeLists.txt` 第 152 行（已备份为 `.bak`）
- **适用条件**: GCC 工具链 + 系统库缺少版本号符号链接的环境

## 问题 7: console_V3_3.hpp 缺少 cstring 头文件

- **现象**: 编译报错 `namespace "std" has no member "strlen"`
- **原因**: `src/Basic/Console/console_V3_3.hpp` 使用了 `std::strlen` 但未 `#include <cstring>`
- **解决方案**: 在文件头部添加 `#include <cstring>`
- **修改文件**: `src/Basic/Console/console_V3_3.hpp` 第 15 行（已备份为 `.bak`）
- **适用条件**: CUDA 12.x + GCC 12 组合（旧版编译器可能更宽松）

## 功能扩展: GGR 格式支持

- **描述**: 新增 GGR (Galois GR) 单文件二进制图格式读取支持，允许与其他 baseline（如 Subway）共用同一份图数据
- **新增文件**: `src/Basic/IO/io_ggr.hpp`（GGR 格式解析器）
- **修改文件**:
  - `src/Basic/Graph/graphFileList.hpp` — 添加 GGR 文件检测和加载分支
  - `project/CGgraphV1.5.hpp` — 从 GGR 文件路径提取 stem 作为 displayName
- **使用方式**: `-graphName /path/to/graph.ggr`（自动检测文件扩展名和格式）
- **Reorder 缓存**: 首次从 GGR 加载后执行 reorder，缓存写入 `./CGgraphV1-5/{stem}_*.bin`，后续运行自动复用
- **类型转换**: GGR 的 int64 offset → countl_type (uint32)，int32 dest/weight → vertex_id_type/edge_data_type，含溢出检查
