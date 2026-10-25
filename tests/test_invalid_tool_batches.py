import json

import pytest
from langchain_core.messages import AIMessage

from codepilot.agent.core import AgentCore
from codepilot.trajectory import TrajectoryLogger
from tests.test_file_tools import LocalSandbox
from tests.test_tool_scheduling import Model, call


def response(valid_names, invalid_names):
    arguments = {
        "read": {"path": "file"},
        "write": {"path": "file", "content": "x"},
        "shell": {"command": "touch file"},
        "finish": {"summary": "done"},
    }
    return AIMessage(
        content="",
        tool_calls=[
            call(name, arguments[name], number) for number, name in enumerate(valid_names, 1)
        ],
        invalid_tool_calls=[
            {
                "id": f"invalid-{number}",
                "name": name,
                "args": "{",
                "error": "Invalid JSON arguments",
            }
            for number, name in enumerate(invalid_names, 1)
        ],
    )


@pytest.mark.parametrize(
    ("valid_names", "invalid_names"),
    [
        ([], ["read"]),
        ([], ["finish", "read"]),
        (["write"], ["read"]),
        (["finish"], ["read"]),
        (["shell"], ["finish"]),
        (["read", "write"], ["read", "finish"]),
    ],
)
def test_any_invalid_json_rejects_every_call_without_execution(
    tmp_path, monkeypatch, valid_names, invalid_names
):
    path = tmp_path / "events.jsonl"
    agent = AgentCore(
        Model(),
        LocalSandbox(tmp_path),
        logger=TrajectoryLogger(path),
    )

    def must_not_execute(*args, **kwargs):
        pytest.fail("A JSON parsing failure must reject the batch before execution")

    monkeypatch.setattr(agent, "invoke_tool", must_not_execute)
    monkeypatch.setattr(agent, "execute_sequential", must_not_execute)
    monkeypatch.setattr(agent, "execute_parallel", must_not_execute)
    message = response(valid_names, invalid_names)
    incoming = {
        "messages": [message],
        "tool_call_iterations": 4,
        "model_iterations": 1,
    }
    result = agent.tools_node(incoming)
    requests = [*message.tool_calls, *message.invalid_tool_calls]
    observations = result.update["messages"]
    assert result.goto == "agent_node"
    assert [item.tool_call_id for item in observations] == [item["id"] for item in requests]
    assert [item.name for item in observations] == [item["name"] for item in requests]
    assert all(item.status == "error" for item in observations)
    assert all("invalid JSON" in item.content for item in observations)
    assert "summary" not in result.update
    assert "termination_reason" not in result.update
    assert result.update.get("tool_call_iterations", 4) == 4
    assert result.update.get("model_iterations", 1) == 1
    assert incoming["messages"] == [message]
    assert not (tmp_path / "file").exists()
    events = [json.loads(line) for line in path.read_text().splitlines()]
    logged_requests = [event for event in events if event["event_type"] == "tool_call"]
    logged_results = [event for event in events if event["event_type"] == "tool_result"]
    assert [event["tool_call_id"] for event in logged_requests] == [item["id"] for item in requests]
    assert [event["tool_call_id"] for event in logged_results] == [item["id"] for item in requests]
    assert all(event["status"] == "error" for event in logged_results)


@pytest.mark.parametrize("can_retry", [False, True])
def test_invalid_json_consumes_a_decision_and_retries_only_with_budget(tmp_path, can_retry):
    messages = [response(["write"], ["finish"])]
    if can_retry:
        messages.append(response(["finish"], []))
    model = Model(messages)
    result = AgentCore(
        model,
        LocalSandbox(tmp_path),
        max_model_calls=2 if can_retry else 1,
    ).run("Fix the task")
    assert model.invoke_count == result["model_iterations"] == len(messages) + (not can_retry)
    assert result["tool_call_iterations"] == 0
    assert result["termination_reason"] == (
        "voluntary_completion" if can_retry else "model_call_limit"
    )
    assert not (tmp_path / "file").exists()
