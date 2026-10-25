# CodePilot

CodePilot 基于 LangGraph 实现编码 agent，使用 Docker 执行任务、收集补丁和完整轨迹。主 agent 提供 `read`、`edit`、`write`、`shell`、`finish`、`think`、`delegate` 七个工具；子 agent 提供前六个工具。

`think` 保存当前模型生成的计划。`delegate` 并行执行独立子任务，子任务使用相同仓库快照和独立上下文。子 agent 的所有仓库写入，包括 shell，必须通过文件授权检查。

## 安装

需要 Python 3.10+、本地 Docker 守护进程，以及 Linux FUSE 3（`/dev/fuse`、`fusermount3`、`libfuse3`）。多子 agent 使用宿主机 FUSE 代理，因此不支持远程 Docker 守护进程；缺少执行层权限机制时会拒绝子任务。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,evaluation]'
```


## 运行

默认开发入口连接 `http://localhost:8000/v1`，可以使用自定义API：

```bash
python -m scripts.run_dev --limit 1 --max-model-calls 20 --max-parallel-subagents 4
python -m scripts.analyze_dev
```

`--max-model-calls` 是模型调用阈值。已启动子任务不会因为达到阈值而停止；整批返回后，主 agent 进行一次不提供工具的最终分析并结束。因此实际模型调用数可能超过阈值。`finish` 表示 agent 主动结束。

## 轨迹

所有 agent 轨迹默认位于 `trajectory/`，可用 `--trajectory-dir` 指定另一个根目录：

```text
trajectory/
  failures.jsonl
  evaluation.jsonl
  <任务ID>/
    attempt-<次数>-<唯一ID>/
      events.jsonl
      plan.md
      subagents/<子任务ID>/
        events.jsonl
        plan.md
        report.json
        changes.json
        patch.diff
```

计划只在模型调用 think 时创建或更新。`changes.json` 保存可精确恢复的内容增量；`patch.diff` 供人阅读。子模型失败时，部分修改留档且不自动接纳。主子轨迹通过运行 ID、子任务 ID 和父工具调用 ID 关联。实际模型输入输出始终记录。

预测及评测结果默认位于 `outputs/dev/`。分析脚本汇总单次运行及所有子 agent 的实际成本。

## 验证

```bash
python -m pytest tests -q
python -m mypy codepilot scripts
```

文件权限测试会创建临时 FUSE 挂载，需要主机允许 `/dev/fuse`。Docker 集成测试默认跳过；指定一个已安装、包含 Python 和 Git 的本地镜像可执行实际容器验证：

```bash
CODEPILOT_DOCKER_TEST_IMAGE=<本地镜像> python -m pytest tests/test_docker_subagents.py -q
```
