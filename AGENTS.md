# 项目级 Agent 指南

本文件用于约束 coding agent 在本项目中的行为。除非用户在当前对话中给出更高优先级的明确指令，agent 应遵守本文档要求。

## 适用范围

- 本文件适用于其所在目录及所有子目录。
- 如果子目录存在更具体的 `AGENTS.md`，则子目录规则优先适用于该子目录内的文件。
- 后续新增规则时，应尽量放入对应章节；如无合适章节，可在“扩展规则”中新增小节。

## 通用原则

- 修改文件前，先理解项目结构、现有脚本和已有约定。
- 不要随意改动与当前任务无关的文件。
- 不要删除或覆盖用户已有结果、数据、日志、模型权重或实验产物，除非用户明确要求。
- 运行测试、实验、benchmark 或评测时，必须记录可复现信息。
- 最终回复应简要说明修改内容、验证方式、关键结果和相关文件路径。

## 实验与测试记录协议

当执行任何实验、测试、benchmark、消融实验、性能评测、模型评测或数据处理验证时，必须将本次运行的结果统一写入独立目录：

```text
experiments/<YYYYMMDD-HHMMSS>_<short-purpose>/
```

其中：

- `<YYYYMMDD-HHMMSS>` 使用本地时间。
- `<short-purpose>` 使用简短英文或拼音短语，描述实验目的，例如 `baseline_eval`、`compare_cache`、`ablation_topk`。
- 不允许将实验输出散落在项目根目录。
- 如果现有脚本默认输出到其他位置，应将关键输出复制或移动到本次实验目录，并在 `README.md` 中记录原始路径。

每个实验目录至少包含：

```text
README.md
config.json
metrics.json
stdout.log
stderr.log
artifacts/
```

### README.md 必填内容

`README.md` 必须记录：

- 实验目的：本次实验要比较什么、验证什么。
- 实验假设：预期观察到什么结果。
- 测试数据集：数据集名称、来源、规模、切分方式、预处理方式。
- 测试场景：任务类型、输入规模、工作负载、baseline、candidate、对比维度。
- 运行环境：硬件、操作系统、关键依赖版本；能自动采集则自动采集。
- 运行命令：完整命令、工作目录、关键环境变量。
- 结果指标：主要指标表格，例如准确率、延迟、吞吐、显存、内存、错误数等。
- 观察现象：异常、失败、不可比因素、性能波动等。
- 结论：本次实验是否支持假设。
- 下一步：建议继续验证的问题。

推荐模板：

````md
# Experiment: <short title>

## Purpose

## Hypothesis

## Dataset

- Name:
- Source:
- Size:
- Split:
- Preprocessing:

## Scenario

- Task:
- Workload:
- Baseline:
- Candidate:
- Comparison:

## Environment

- OS:
- Hardware:
- Runtime:
- Dependencies:
- Git commit:

## Command

```bash
...
```

## Results

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|

## Observations

## Conclusion

## Issues

## Next Steps
````

### config.json 必填内容

`config.json` 应尽量包含：

- 实验名称和目的。
- 运行时间。
- Git commit 或代码版本。
- 数据集配置。
- 关键参数。
- 随机种子。
- baseline 和 candidate 配置。
- 运行命令。
- 环境信息。

### metrics.json 必填内容

`metrics.json` 应保存结构化指标，便于后续自动汇总。示例：

```json
{
  "status": "success",
  "metrics": {
    "accuracy": null,
    "latency_ms": null,
    "throughput": null,
    "memory_mb": null,
    "error_count": null
  },
  "notes": ""
}
```

### stdout.log 和 stderr.log

- 所有实验命令的标准输出应写入 `stdout.log`。
- 所有实验命令的错误输出应写入 `stderr.log`。
- 如果命令失败，也必须保留日志，并在 `README.md` 的 `Issues` 中说明失败原因。

### artifacts/

`artifacts/` 用于保存：

- 图表。
- 截图。
- 中间结果。
- 预测输出。
- profile 文件。
- 生成报告。
- 其他可复查产物。

### result.csv
`result.csv` 用于保存：

- 整理好的实验对比结果
- 每个test case 在 核心展示指标下的对比结果


## 测试与验证

- 如果修改代码，应优先运行与修改范围相关的最小测试。
- 如果运行完整测试成本很高，应说明未运行完整测试的原因，并运行可行的替代验证。
- 测试结果也应遵守“实验与测试记录协议”，除非只是非常轻量的只读检查命令。

## 数据与产物

- 不要改动原始数据集，除非任务明确要求。
- 派生数据应写入实验目录或项目约定的数据输出目录。
- 大文件、模型权重、压缩包、临时缓存不应随意提交或移动。

## 代码修改

后续可在此补充项目代码风格、命名规范、目录约定、格式化命令和测试命令。

## 依赖与环境

后续可在此补充依赖安装方式、虚拟环境、容器、编译选项和常用环境变量。

## 安全与权限

后续可在此补充敏感文件、密钥、网络访问、外部服务调用和权限申请规则。

## 文档与输出格式

后续可在此补充论文、报告、README、实验总结或 PR 描述的格式要求。

## 扩展规则

### 一些环境变量
- ncu路径：/home/zyl/.conda/envs/torch2.8/bin/ncu
- nvcc路径：/home/zyl/.conda/envs/torch2.8/bin/nvcc
- nsys路径：/home/zyl/.conda/envs/torch2.8/nsight-compute-2025.1.1/host/target-linux-x64/nsys
