import json

from langchain_core.messages import AIMessage, ToolMessage
from test_file_tools import LocalSandbox

from codepilot.agent.core import AgentCore


class PlanModel:
    def __init__(self):
        self.inputs = []
        self.bound_tools = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = [tool.name for tool in tools]
        return self

    def invoke(self, messages):
        self.inputs.append(messages)
        if not isinstance(messages[-1], ToolMessage):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "think",
                        "id": "plan-1",
                        "args": {
                            "steps": [
                                {"id": "1", "description": "检查修复", "status": "in_progress"}
                            ]
                        },
                    }
                ],
            )
        return AIMessage(content="Final analysis: work remains")


def test_think_is_a_normal_decision_then_one_final_analysis_without_tools(tmp_path):
    model = PlanModel()
    agent = AgentCore(
        model, LocalSandbox(tmp_path), max_model_calls=1, trajectory_dir=tmp_path / "trajectory"
    )
    state = agent.run("Fix it")
    assert set(model.bound_tools) == {
        "read",
        "edit",
        "write",
        "shell",
        "finish",
        "think",
        "delegate",
    }
    assert state["termination_reason"] == "model_call_limit"
    assert state["model_iterations"] == 2
    assert state["final_analysis_calls"] == 1
    assert state["summary"] == "Final analysis: work remains"
    assert (tmp_path / "trajectory/plan.md").exists()
    # Exactly one plan copy in the next model input, in its tool result.
    assert (
        sum(
            isinstance(message, ToolMessage) and "# Task plan" in message.content
            for message in model.inputs[-1]
        )
        == 1
    )
    events = [
        json.loads(line) for line in (tmp_path / "trajectory/events.jsonl").read_text().splitlines()
    ]
    assert [e["role"] for e in events if e["event_type"] == "model_input"] == [
        "main",
        "final_analysis",
    ]
