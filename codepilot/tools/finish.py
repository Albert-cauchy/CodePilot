from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field


@tool
def finish(
    summary: Annotated[
        str,
        Field(
            description="Briefly summarize the fix that was implemented and the validation that was performed."
        ),
    ],
    unresolved_issues: list[str] | None = None,
) -> str:
    """Declare that no further repository actions are needed. This does not establish evaluation success. Call finish alone."""
    return summary


finish.metadata = {"executionMode": "sequential"}
