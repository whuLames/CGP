# LCCG: A Locality-Centric Hardware Accelerator for High Throughput of Concurrent Graph Processing

- 文献：Zhao et al., SC 2021
- 阅读状态：未取得公开全文；以下仅依据 BibTeX 与公开元数据
- DOI：https://doi.org/10.1145/3458817.3480854

## 主要内容

LCCG 为并发图处理设计 locality-centric 专用加速器。公开摘要显示，它在多核系统旁增加拓扑感知的硬件机制，在线规整不规则遍历，使并发作业既能复用图结构，又能合并对顶点状态的访问；论文基于 64 核模拟平台评估，并与纯软件及多种图加速器比较。

## 与当前工作的关系

LCCG 与“减少访问次数 + 降低访问成本”的两轴划分概念上最接近。关键区别是它依赖专用硬件和模拟环境，而当前工作面向现成 GPU，以软件运行时、query grouping、vertex sharing 和 warp-level 访存映射实现类似目标。论文应明确比较优化对象、硬件假设和可部署性。
