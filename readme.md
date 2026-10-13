# CodePilot

CodePilot 是一个用于研究编码智能体的实验平台，基于 LangGraph 实现智能体循环，在 Docker 沙箱中执行 SWE-bench Verified 编码任务，收集代码补丁、执行轨迹和独立评测结果。

目前主要比较相同模型决策预算下的两种工具配置：

- `two`：`shell`、`finish`。
- `five`：`read`、`edit`、`write`、`shell`、`finish`。

每次正常行。
返回的模型响应计为一次决策；智能体调用 `finish` 表示主动结束，任务是否解决由 SWE-bench 评测器判断。

## 环境准备

需要 Python 3.11+、可用的 Docker，以及支持工具调用的 OpenAI 兼容模型服务。首次运行需要下载数据集和任务镜像。以下命令均在项目根目录执
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install langgraph langchain-openai datasets swebench docker pytest
```

## 快速开始

开发入口默认使用固定的 50 个开发任务，连接 `http://localhost:8000/v1`，模型为 `Qwen/Qwen3.5-35B-A3B`，默认启用 thinking。先启动对应模型服务，再运行小规模实验：

```bash
python -m scripts.run_dev --tool-config five --limit 2 --max-model-calls 20
```

可通过 `--model`、`--base-url` 调整模型和服务地址；需要认证时，使用 `--api-key-env` 指定保存密钥的环境变量名称。

使用相同任务、模型参数和决策预算运行两组实验，再比较结果：

```bash
python -m scripts.run_dev --tool-config two --limit 2 --max-model-calls 20
python -m scripts.run_dev --tool-config five --limit 2 --max-model-calls 20
python -m scripts.analyze_dev
```

去掉 `--limit` 可运行全部 50 个开发任务。每次运行结束后会自动评测；再次执行相同命令会跳过已有补丁。修改实验参数时，需用 `--output` 指定新的输出路径。

结果保存在 `outputs/dev/two/` 和 `outputs/dev/five/`，包含补丁、实验配置、评测结果和 `trajectories/` 执行轨迹；比较报告为 `outputs/dev/comparison.json`。

## 项目结构

```text
codepilot/   智能体、任务适配、Docker 沙箱、补丁和轨迹管理
tools/       智能体工具实现
scripts/     实验运行、任务筛选和结果分析入口
tests/       测试
data/        固定开发任务清单
docs/        实验协议、开发说明和架构决策
```

运行测试：

```bash
python -m pytest tests -q
```