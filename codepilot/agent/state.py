from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # Attempted repository operations; statistics only.
    tool_call_iterations: NotRequired[int]
    # Normally returned responses; the decision budget.
    model_iterations: NotRequired[int]
    summary: NotRequired[str]
    # Only populate this when routing to END, recording the reason the run terminated.
    termination_reason: NotRequired[str]
    unresolved_issues: NotRequired[list[str]]
