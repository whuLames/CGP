# GraphWeft SIGMOD 2027 初稿

本目录包含一份可编译的英文 SIGMOD 扩展初稿。正文按删减前的工作稿组织，目标是先达到约 14--15 页，再围绕核心 claim 压缩到投稿页限。论文不是把当前项目包装成已经完成的结果，而是把系统主线、设计、实现、已有证据和投稿前缺口统一到一个版本中。

当前 `main.pdf` 共 15 页：正文约 14.5 页，参考文献从第 15 页开始；正文包含 8 张图和 4 个算法。

## 编译

```bash
cd /home/zyl/Projects/ocgp/puercgp/paper-writing/sigmod27
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

清理 LaTeX 中间文件：

```bash
latexmk -c
```

## 导出矢量图

将 `figures/` 中的 TikZ/PGFPlots 图批量编译为不含 caption 的独立矢量 PDF：

```bash
./figures/export_all.sh
```

生成的 PDF 与源文件同名，并按正文 section 保存在
`figures/01_introduction/`--`figures/06_evaluation/` 对应子目录中。修改图形的
`.tex` 文件后，重新执行该命令即可更新全部 TikZ/PGFPlots PDF。

## 目录内容

- `main.tex`：SIGMOD/ACM 双栏入口、匿名审稿设置和摘要。
- `sections/`：完整正文，包括 Introduction、Background、Overview、Design、Implementation、Evaluation、Related Work、Discussion、Conclusion。
- `figures/`：按 section 编号组织的论文图片；TikZ/PGFPlots 源文件与导出的
  PDF 成对放置，全体系统运行时间对比图位于 `figures/06_evaluation/`。
- `algorithms/`：四个算法浮动体，覆盖批处理主循环、两种访问策略和在线调度器。
- `references.bib`：正文使用的参考文献。
- `CLAIM_EVIDENCE.md`：核心 claim 与代码、实验、缺失证据的映射。
- `EXPERIMENT_TODO.md`：按投稿优先级排列的实验清单。

## 当前论文主张

GraphWeft 将 CGQ 视为 GPU 上的内存访问问题，统一原则是“共享重复访问，并规整剩余访问”：

1. 通过 exact query-aware state 和 VertexShare 减少重复邻居访问次数。
2. 通过 V×Q 状态布局和 AlignedGather 降低剩余 query-state 访问的有效代价。
3. 通过 phase index、batching 和 bounded offset 增加同一迭代中的可共享访问。

GPU 的具体并行映射集中放在 Implementation，Introduction、Motivation、Overview 和 Design 优先描述 CGQ 计算模型、内存访问问题与逻辑机制。当前最需要补齐的是 selector 的校准和固定策略端到端结果。现有 Q=64 hybrid BFS 对 iBFS 的几何平均吞吐仅为 0.49×；初稿已如实写入，不能在提交版本中隐藏或用 best-of 模式替代。

## 投稿格式提醒

SIGMOD 2027 research paper 使用 ACM `sample-sigconf` 双栏格式，审稿稿正文最多 12 页（参考文献不计入），并要求双盲。提交前还需根据当时有效的 ACM 生成式 AI 政策复核披露方式；作者必须逐条核验本文中的陈述、数字和引用。
