import json

from langchain_core.messages import AIMessage, HumanMessage

from codepilot.trajectory import TrajectoryLogger
from scripts.run_benchmark import LoggedModel


class FakeModel:
    def bind_tools(self, tools, **kwargs):
        assert kwargs == {"parallel_tool_calls": False}
        return self

    def invoke(self, messages):
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "shell", "args": {"command": "pwd"}, "id": "one"},
                {"name": "shell", "args": {"command": "ls"}, "id": "two"},
            ],
        )


def test_logged_model_records_input_and_all_tool_calls(tmp_path):
    path = tmp_path / "events.jsonl"
    logger = TrajectoryLogger(path)
    model = LoggedModel(FakeModel(), logger, "org__repo-1", 2).bind_tools(
        [], parallel_tool_calls=False
    )

    model.invoke([HumanMessage(content="Fix the issue")])

    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["event_type"] for event in events] == [
        "model_input", "model_response",
    ]
    assert events[0]["messages"][0]["data"]["content"] == "Fix the issue"
    assert len(events[1]["message"]["data"]["tool_calls"]) == 2
    assert all(
        event["instance_id"] == "org__repo-1"
        and event["attempt"] == 2
        and event["step"] == 1
        for event in events
    )
