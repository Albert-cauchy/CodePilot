import json

import pytest
from langchain_core.messages import AIMessage

import codepilot.agent.core as core_module
from codepilot.agent.core import AgentCore
from codepilot.trajectory import TrajectoryLogger
from tests.test_file_tools import LocalSandbox
from tests.test_tool_scheduling import Model, call


@pytest.mark.parametrize("ending", ["finish", "rejected", "plain", "context"])
def test_trajectory_records_requests_results_and_normal_termination(
    tmp_path, monkeypatch, ending
):
    calls = []
    reason = "no_tool_call"
    decisions = 1
    if ending == "finish":
        calls = [call("finish", {"summary": "done"})]
        reason = "voluntary_completion"
    elif ending == "rejected":
        calls = [
            call("write", {"path": "file", "content": "x"}),
            call("finish", {"summary": "done"}, 2),
        ]
        reason = "model_call_limit"
    elif ending == "context":
        reason = "context_limit"
        decisions = 0

    model = Model([AIMessage(content="", tool_calls=calls)])
    if ending == "context":
        class ContextError(Exception):
            code = "context_length_exceeded"

        def invoke(messages):
            raise ContextError()

        monkeypatch.setattr(model, "invoke", invoke)

    clock = iter([10.0, 12.5])
    monkeypatch.setattr(core_module, "perf_counter", lambda: next(clock))
    path = tmp_path / "events.jsonl"
    logger = TrajectoryLogger(path)
    agent = AgentCore(
        model,
        LocalSandbox(tmp_path),
        tool_configuration="five",
        max_model_calls=1,
        logger=logger,
    )
    state = agent.run("Fix the task")

    events = [json.loads(line) for line in path.read_text().splitlines()]
    requests = [event for event in events if event["event_type"] == "tool_call"]
    results = [event for event in events if event["event_type"] == "tool_result"]
    assert [(event["tool_call_id"], event["name"], event["args"]) for event in requests] == [
        (request["id"], request["name"], request["args"]) for request in calls
    ]
    assert [event["tool_call_id"] for event in results] == [
        request["id"] for request in calls
    ]
    assert all(
        event["status"] == ("error" if ending == "rejected" else "success")
        for event in results
    )
    if calls:
        last_request = max(events.index(event) for event in requests)
        first_result = min(events.index(event) for event in results)
        assert last_request < first_result

    terminations = [event for event in events if event["event_type"] == "termination"]
    assert terminations == [events[-1]]
    assert events[-1]["termination_reason"] == state["termination_reason"] == reason
    assert events[-1]["model_iterations"] == state["model_iterations"] == decisions
    assert events[-1]["tool_call_iterations"] == state["tool_call_iterations"] == 0
    assert events[-1]["elapsed_seconds"] == 2.5
    assert not (tmp_path / "file").exists()


def test_service_failure_does_not_record_normal_termination(tmp_path):
    class FailingModel(Model):
        def invoke(self, messages):
            raise ConnectionError("Model service unavailable")

    path = tmp_path / "events.jsonl"
    logger = TrajectoryLogger(path)
    agent = AgentCore(
        FailingModel(),
        LocalSandbox(tmp_path),
        logger=logger,
    )
    with pytest.raises(ConnectionError):
        agent.run("Fix the task")

    assert not path.exists()
