from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from codepilot.agent.runner import run_task
from codepilot.benchmark import Task
from codepilot.sandbox.base import ExecResult, RepositorySandbox


class FakeSandbox(RepositorySandbox):
    def __init__(self):
        self.started = False
        self.stopped = False
        self.commands = []

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def execute(self, command, *, timeout=120, check=False, workdir=None):
        assert self.started and not self.stopped
        self.commands.append(command)
        return ExecResult(command, 0, "/testbed\n", "", 0.01)

    def prepare_patch_snapshot(self):
        assert self.started and not self.stopped

    def git_diff(self, *, binary=True):
        assert binary
        return "diff --git a/a.py b/a.py\n"

    def changed_files(self):
        return ["a.py"]


class FakeModel:
    def bind_tools(self, tools, *, parallel_tool_calls=None):
        assert parallel_tool_calls is True
        return self

    def invoke(self, messages):
        if isinstance(messages[-1], HumanMessage):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "shell",
                        "args": {"command": "pwd"},
                        "id": "shell-1",
                    }
                ],
            )

        assert isinstance(messages[-1], ToolMessage)
        assert "/testbed" in messages[-1].content
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "finish",
                    "args": {"summary": "done"},
                    "id": "finish-1",
                }
            ],
        )


def test_runner_uses_task_and_sandbox_interfaces():
    task = Task("org__repo-1", "org/repo", "abc123", "fix me")
    sandbox = FakeSandbox()

    state, patch = run_task(task, FakeModel(), lambda received: sandbox)

    assert sandbox.commands == ["pwd"]
    assert sandbox.stopped
    assert patch.changed_files == ("a.py",)
    assert state["tool_call_iterations"] == 1
