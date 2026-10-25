import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from test_repository_workspace import LocalRepository

from codepilot.agent.core import AgentCore
from codepilot.sandbox.file_policy import RestrictedWorkspace


class ForkableRepository(LocalRepository):
    @contextmanager
    def fork(self, baseline, allowed_files):
        with TemporaryDirectory() as directory:
            backing = Path(directory) / "private"
            baseline.materialize(backing)
            with RestrictedWorkspace(backing, allowed_files) as mount:
                yield LocalRepository(mount)


class ParallelModel:
    def __init__(self, tasks):
        self.tasks = tasks
        self.barrier = Barrier(2)
        self.final_messages = None

    def bind_tools(self, tools, **kwargs):
        parent = any(tool.name == "delegate" for tool in tools)
        owner = self

        class Bound:
            def invoke(self, messages):
                if parent:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "delegate", "id": "batch-1", "args": {"tasks": owner.tasks}}
                        ],
                    )
                task = json.loads(next(m.content for m in messages if isinstance(m, HumanMessage)))
                if isinstance(messages[-1], HumanMessage):
                    owner.barrier.wait(timeout=10)
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write",
                                "id": "write-1",
                                "args": {"path": task["allowed_files"][0], "content": task["goal"]},
                            }
                        ],
                    )
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "finish",
                            "id": "done-1",
                            "args": {"summary": "Implemented " + task["goal"]},
                        }
                    ],
                )

        return Bound()

    def invoke(self, messages):
        self.final_messages = messages
        return AIMessage(content="Final report without tools")


def test_parallel_subgraphs_finish_after_threshold_and_invalid_task_does_not_block_peers(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    (root / "parent.txt").write_text("unfinished parent work")
    tasks = [
        {
            "goal": "fix A",
            "context": "",
            "allowed_files": ["a.txt"],
            "acceptance_criteria": "A fixed",
        },
        {
            "goal": "fix B",
            "context": "",
            "allowed_files": ["b.txt"],
            "acceptance_criteria": "B fixed",
        },
        {"goal": "invalid missing required fields"},
    ]
    model = ParallelModel(tasks)
    state = AgentCore(
        model,
        ForkableRepository(root),
        max_model_calls=1,
        max_parallel_subagents=2,
        trajectory_dir=tmp_path / "trajectory",
    ).run("parent")
    reports = json.loads(
        next(
            m.content
            for m in model.final_messages
            if isinstance(m, ToolMessage) and m.name == "delegate"
        )
    )
    assert [report["status"] for report in reports] == ["completed", "completed", "rejected"]
    assert (root / "a.txt").read_text() == "fix A"
    assert (root / "b.txt").read_text() == "fix B"
    assert (root / "parent.txt").read_text() == "unfinished parent work"
    assert state["decision_counts"] == {"main": 1, "subagent": 4, "final_analysis": 1, "total": 6}
    assert state["model_iterations"] == 6
    assert state["summary"] == "Final report without tools"
    assert len(list((tmp_path / "trajectory/subagents").glob("*/events.jsonl"))) == 2


def test_failed_subagent_keeps_partial_patch_without_applying_it_or_cancelling_peer(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    tasks = [
        {
            "goal": "fix A",
            "context": "",
            "allowed_files": ["a.txt"],
            "acceptance_criteria": "A fixed",
        },
        {
            "goal": "fix B",
            "context": "",
            "allowed_files": ["b.txt"],
            "acceptance_criteria": "B fixed",
        },
    ]

    class FailingModel(ParallelModel):
        def bind_tools(self, tools, **kwargs):
            bound = super().bind_tools(tools, **kwargs)
            original = bound.invoke

            def invoke(messages):
                if isinstance(messages[-1], ToolMessage) and messages[-1].name == "write":
                    task = json.loads(
                        next(m.content for m in messages if isinstance(m, HumanMessage))
                    )
                    if task["goal"] == "fix B":
                        raise ConnectionError("Model unavailable")
                return original(messages)

            bound.invoke = invoke
            return bound

    model = FailingModel(tasks)
    AgentCore(
        model, ForkableRepository(root), max_model_calls=1, trajectory_dir=tmp_path / "trajectory"
    ).run("parent")
    reports = json.loads(
        next(
            m.content
            for m in model.final_messages
            if isinstance(m, ToolMessage) and m.name == "delegate"
        )
    )
    assert [report["status"] for report in reports] == ["completed", "failed"]
    assert (root / "a.txt").read_text() == "fix A"
    assert not (root / "b.txt").exists()
    changes = json.loads((Path(reports[1]["artifact_directory"]) / "changes.json").read_text())
    assert [change["path"] for change in changes] == ["b.txt"]


def test_all_conflicting_tasks_are_rejected_while_independent_task_runs(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    (root / "shared.txt").write_text("unchanged")
    tasks = [
        {"goal": goal, "context": "", "allowed_files": [name], "acceptance_criteria": "verified"}
        for goal, name in (("A", "shared.txt"), ("B", "shared.txt"), ("C", "independent.txt"))
    ]
    model = ParallelModel(tasks)
    model.barrier = Barrier(1)
    AgentCore(
        model, ForkableRepository(root), max_model_calls=1, trajectory_dir=tmp_path / "trajectory"
    ).run("parent")
    reports = json.loads(
        next(
            m.content
            for m in model.final_messages
            if isinstance(m, ToolMessage) and m.name == "delegate"
        )
    )
    assert [report["status"] for report in reports] == ["rejected", "rejected", "completed"]
    assert (root / "shared.txt").read_text() == "unchanged"
    assert (root / "independent.txt").read_text() == "C"


def test_more_tasks_than_workers_queue_and_all_finish_beyond_soft_budget(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    tasks = [
        {
            "goal": str(index),
            "context": "",
            "allowed_files": [f"{index}.txt"],
            "acceptance_criteria": "verified",
        }
        for index in range(4)
    ]
    model = ParallelModel(tasks)
    state = AgentCore(
        model,
        ForkableRepository(root),
        max_model_calls=1,
        max_parallel_subagents=2,
        trajectory_dir=tmp_path / "trajectory",
    ).run("parent")
    assert [(root / f"{index}.txt").read_text() for index in range(4)] == ["0", "1", "2", "3"]
    assert state["decision_counts"] == {"main": 1, "subagent": 8, "final_analysis": 1, "total": 10}
