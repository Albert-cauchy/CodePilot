import json

from langchain_core.messages import AIMessage

from codepilot.agent.core import AgentCore
from codepilot.trajectory import TrajectoryLogger
from tests.test_file_tools import LocalSandbox


class FakeModel:
    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.calls > 1:
            return AIMessage(content="Final report")
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "shell", "args": {"command": "pwd"}, "id": "one"},
                {"name": "shell", "args": {"command": "ls"}, "id": "two"},
            ],
        )


def test_agent_records_actual_input_response_and_all_tool_calls(tmp_path):
    path = tmp_path / "trajectory/events.jsonl"
    logger = TrajectoryLogger(path, instance_id="org__repo-1", attempt=2)
    AgentCore(FakeModel(), LocalSandbox(tmp_path), logger=logger, max_model_calls=1).run(
        "Fix the issue"
    )
    events = [json.loads(line) for line in path.read_text().splitlines()]
    inputs = [event for event in events if event["event_type"] == "model_input"]
    responses = [event for event in events if event["event_type"] == "model_response"]
    assert inputs[0]["messages"][-1]["data"]["content"] == "Fix the issue"
    assert len(responses[0]["message"]["data"]["tool_calls"]) == 2
    assert [event["role"] for event in inputs] == ["main", "final_analysis"]
    assert all(event["instance_id"] == "org__repo-1" and event["attempt"] == 2 for event in events)
    assert {event["agent_id"] for event in events} == {"main"}
