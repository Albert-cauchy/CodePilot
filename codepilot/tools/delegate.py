from typing import Annotated, Any

from langchain_core.tools import InjectedToolCallId, tool
from pydantic import BaseModel, ConfigDict, Field


class SubtaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    goal: str = Field(min_length=1)
    context: str
    allowed_files: list[str]
    acceptance_criteria: str = Field(min_length=1)


def create_delegate_tool(callback):
    @tool
    def delegate(
        tasks: Annotated[
            list[Any],
            Field(
                min_length=1,
                description="Independent atomic subtasks. File scopes must be disjoint.",
                json_schema_extra={"items": SubtaskSpec.model_json_schema()},
            ),
        ],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> str:
        """Delegate independent atomic tasks to parallel subagents and wait for all
        reports. Supply each goal, necessary context, exact writable repository
        files (empty for read-only), and acceptance criteria. Each subagent sees
        the same current repository snapshot and only the context you supply.
        Do not assign overlapping files. Invalid tasks are rejected individually.
        No budget allocation is required. Call alone.
        """
        return callback(tasks, tool_call_id)

    delegate.metadata = {"executionMode": "sequential"}
    return delegate
