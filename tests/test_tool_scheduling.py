import threading

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END

from codepilot.agent.core import AgentCore
from codepilot.agent.runner import run_task
from codepilot.benchmark import Task
from codepilot.sandbox.base import ExecResult
from tests.test_agent_runner import FakeSandbox
from tests.test_file_tools import LocalSandbox


class Model:
    def __init__(self, responses=()):
        self.responses = iter(responses)
        self.invoke_count = 0

    def bind_tools(self, tools, **kwargs):
        self.names = [tool.name for tool in tools]
        return self

    def invoke(self, messages):
        self.invoke_count += 1
        return next(self.responses)


def call(name, args=None, number=1):
    return {"name": name, "args": args or {}, "id": f"call-{number}"}


def state(calls, iterations=0, *, model_iterations=0):
    return {
        "messages": [AIMessage(content="", tool_calls=calls)],
        "tool_call_iterations": iterations,
        "model_iterations": model_iterations,
    }


def test_sequential_failure_skips_remaining_calls(tmp_path):
    agent = AgentCore(
        Model(), LocalSandbox(tmp_path), tool_configuration="five", max_model_calls=1
    )
    result = agent.tools_node(
        state(
            [
                call("read", {"path": "missing"}),
                call("write", {"path": "file", "content": "x"}, 2),
            ],
            model_iterations=1,
        )
    )
    assert not (tmp_path / "file").exists()
    assert [message.status for message in result.update["messages"]] == [
        "error",
        "error",
    ]
    assert "Skipped" in result.update["messages"][1].content
    assert result.update["tool_call_iterations"] == 1


def test_nonzero_shell_exit_continues_but_timeout_skips(tmp_path):
    agent = AgentCore(Model(), LocalSandbox(tmp_path), tool_configuration="five")
    calls = [
        call("shell", {"command": "exit 1"}),
        call("write", {"path": "file", "content": "x"}, 2),
    ]
    result = agent.tools_node(state(calls))
    assert all(message.status == "success" for message in result.update["messages"])
    assert (tmp_path / "file").read_text() == "x"

    class TimedOut(LocalSandbox):
        def execute(self, command, **kwargs):
            return ExecResult(command, 124, "", "", 1, timed_out=True)

    agent = AgentCore(Model(), TimedOut(tmp_path), tool_configuration="five")
    result = agent.tools_node(state(calls))
    assert "timed out" in result.update["messages"][0].content
    assert "Skipped" in result.update["messages"][1].content


def test_all_reads_run_concurrently_and_results_keep_call_order(tmp_path):
    agent = AgentCore(
        Model(), LocalSandbox(tmp_path), tool_configuration="five", max_model_calls=1
    )
    barrier = threading.Barrier(2)

    @tool
    def read(path: str) -> str:
        """Read a file."""
        barrier.wait(timeout=5)
        if path == "missing":
            raise FileNotFoundError(path)
        return path

    read.metadata = {"executionMode": "parallel"}
    agent.tools_by_name["read"] = read
    result = agent.tools_node(
        state(
            [
                call("read", {"path": "missing"}),
                call("read", {"path": "good"}, 2),
            ],
            model_iterations=1,
        )
    )
    messages = result.update["messages"]
    assert [message.tool_call_id for message in messages] == ["call-1", "call-2"]
    assert [message.status for message in messages] == ["error", "success"]
    assert messages[1].content == "good"


@pytest.mark.parametrize(
    "names",
    [
        ["write", "finish", "read"],
        ["finish", "write", "read"],
        ["write", "read", "finish"],
        ["finish", "finish"],
    ],
)
def test_finish_mixed_batch_rejected_before_mutation(tmp_path, monkeypatch, names):
    agent = AgentCore(Model(), LocalSandbox(tmp_path), tool_configuration="five")
    arguments = {
        "write": {"path": "file", "content": "x"},
        "finish": {"summary": "done"},
        "read": {"path": "file"},
    }
    calls = [
        call(name, arguments[name], number) for number, name in enumerate(names, 1)
    ]
    observed = []
    original_observation = agent.observation

    def observation(request, content, *, failed=False, skipped=False):
        observed.append((request["id"], failed, skipped))
        return original_observation(request, content, failed=failed, skipped=skipped)

    def must_not_execute(*args, **kwargs):
        pytest.fail("A rejected batch must not invoke tools or execution helpers")

    monkeypatch.setattr(agent, "observation", observation)
    monkeypatch.setattr(agent, "invoke_tool", must_not_execute)
    monkeypatch.setattr(agent, "execute_sequential", must_not_execute)
    monkeypatch.setattr(agent, "execute_parallel", must_not_execute)
    result = agent.tools_node(state(calls, iterations=4, model_iterations=1))
    messages = result.update["messages"]
    assert not (tmp_path / "file").exists()
    assert [message.tool_call_id for message in messages] == [
        request["id"] for request in calls
    ]
    assert [message.name for message in messages] == names
    assert all(message.status == "error" for message in messages)
    assert all("finish" in message.content.lower() for message in messages)
    assert observed == [(request["id"], True, False) for request in calls]
    assert "termination_reason" not in result.update
    assert "summary" not in result.update
    assert result.update.get("tool_call_iterations", 4) == 4
    assert result.update.get("model_iterations", 1) == 1
    assert result.goto == "agent_node"


def test_invalid_finish_and_unknown_tool_return_failures(tmp_path):
    agent = AgentCore(Model(), LocalSandbox(tmp_path))
    assert agent.tools_node(state([call("finish")])).goto == "agent_node"
    result = agent.tools_node(state([call("unknown")]))
    assert result.update["messages"][0].status == "error"


def test_last_decision_processes_entire_batch_without_another_model_call(tmp_path):
    model = Model(
        [
            AIMessage(
                content="",
                tool_calls=[
                    call("shell", {"command": "touch first"}),
                    call("shell", {"command": "touch second"}, 2),
                ],
            )
        ]
    )
    result = AgentCore(model, LocalSandbox(tmp_path), max_model_calls=1).run("Fix")
    assert result["termination_reason"] == "model_call_limit"
    assert result["model_iterations"] == 1
    assert result["tool_call_iterations"] == 2
    assert model.invoke_count == 1
    assert (tmp_path / "first").exists()
    assert (tmp_path / "second").exists()


def test_exhausted_budget_prevents_invocation(tmp_path):
    model = Model()
    agent = AgentCore(model, LocalSandbox(tmp_path), max_model_calls=2)
    result = agent.agent_node(state([], model_iterations=2))
    assert result.goto == END
    assert result.update["termination_reason"] == "model_call_limit"
    assert model.invoke_count == 0


def test_finish_on_last_decision_preserves_voluntary_completion(tmp_path):
    model = Model(
        [AIMessage(content="", tool_calls=[call("finish", {"summary": "done"})])]
    )
    result = AgentCore(model, LocalSandbox(tmp_path), max_model_calls=1).run("Fix")
    assert result["termination_reason"] == "voluntary_completion"
    assert result["model_iterations"] == 1
    assert result["summary"] == "done"
    assert model.invoke_count == 1


def test_runner_collects_patch_after_model_budget_exhaustion():
    task = Task("org__repo-1", "org/repo", "base", "Fix")
    sandbox = FakeSandbox()
    model = Model(
        [AIMessage(content="", tool_calls=[call("shell", {"command": "pwd"})])]
    )
    result, patch = run_task(task, model, lambda received: sandbox, max_model_calls=1)
    assert result["termination_reason"] == "model_call_limit"
    assert result["model_iterations"] == model.invoke_count == 1
    assert patch.text == "diff --git a/a.py b/a.py\n"
    assert patch.changed_files == ("a.py",)
    assert sandbox.stopped


def test_graph_limit_accommodates_model_budget(tmp_path):
    allowance = 20
    model = Model(
        [
            AIMessage(
                content="", tool_calls=[call("shell", {"command": "true"}, number)]
            )
            for number in range(allowance)
        ]
    )
    result = AgentCore(model, LocalSandbox(tmp_path), max_model_calls=allowance).run(
        "Fix"
    )
    assert result["termination_reason"] == "model_call_limit"
    assert result["model_iterations"] == model.invoke_count == allowance
    assert result["tool_call_iterations"] == allowance


def test_graph_runs_five_tools_and_records_voluntary_completion(tmp_path):
    model = Model(
        [
            AIMessage(
                content="",
                tool_calls=[call("write", {"path": "file", "content": "first"})],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    call(
                        "edit",
                        {
                            "path": "file",
                            "edits": [{"old_text": "first", "new_text": "second"}],
                        },
                        2,
                    )
                ],
            ),
            AIMessage(content="", tool_calls=[call("finish", {"summary": "done"}, 3)]),
        ]
    )
    agent = AgentCore(
        model, LocalSandbox(tmp_path), tool_configuration="five", max_model_calls=3
    )
    result = agent.run("Fix the task")
    assert model.names == ["read", "edit", "write", "shell", "finish"]
    assert result["termination_reason"] == "voluntary_completion"
    assert result["summary"] == "done"
    assert result["model_iterations"] == model.invoke_count == 3
    assert (tmp_path / "file").read_text() == "second"


def test_context_limit_is_distinct_from_service_failure(tmp_path):
    class ContextError(Exception):
        code = "context_length_exceeded"

    class FailingModel(Model):
        def invoke(self, messages):
            raise ContextError()

    result = AgentCore(FailingModel(), LocalSandbox(tmp_path)).run("Fix")
    assert result["termination_reason"] == "context_limit"


def test_task_instructions_identical_between_configurations(tmp_path):
    two = AgentCore(Model(), LocalSandbox(tmp_path), tool_configuration="two")
    five = AgentCore(Model(), LocalSandbox(tmp_path), tool_configuration="five")
    before_two, tools_two = two.system_prompt.split("<Available Tools>")
    before_five, tools_five = five.system_prompt.split("<Available Tools>")
    assert before_two == before_five
    assert (
        tools_two.split("</Available Tools>")[1]
        == tools_five.split("</Available Tools>")[1]
    )


def test_plain_response_has_distinct_termination(tmp_path):
    model = Model([AIMessage(content="done")])
    result = AgentCore(model, LocalSandbox(tmp_path), max_model_calls=1).run("Fix")
    assert result["termination_reason"] == "no_tool_call"
    assert result["model_iterations"] == model.invoke_count == 1


def test_rejected_batches_are_bounded(tmp_path):
    calls = [
        call("shell", {"command": "touch unexpected"}),
        call("finish", {"summary": "done"}, 2),
    ]
    model = Model([AIMessage(content="", tool_calls=calls) for _ in range(3)])
    result = AgentCore(model, LocalSandbox(tmp_path), max_model_calls=3).run("Fix")
    assert result["termination_reason"] == "model_call_limit"
    assert result["model_iterations"] == model.invoke_count == 3
    assert result["tool_call_iterations"] == 0
    assert not (tmp_path / "unexpected").exists()


def test_model_service_failure_propagates_for_runner_retry(tmp_path):
    class FailingModel(Model):
        def invoke(self, messages):
            raise ConnectionError("Model service unavailable")

    with pytest.raises(ConnectionError):
        AgentCore(FailingModel(), LocalSandbox(tmp_path)).run("Fix")
