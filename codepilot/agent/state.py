from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # Attempted repository operations; statistics only.
    tool_call_iterations: NotRequired[int]
    # Normally returned responses; the decision budget.
    model_iterations: NotRequired[int]
    summary: NotRequired[str]
    # Only populate this when routing to END, recording the reason the run terminated.
    termination_reason: NotRequired[str]
