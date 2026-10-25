from typing import Annotated

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from codepilot.sandbox.base import BaseSandbox
from codepilot.tools._files import execute_file_operation


class Edit(BaseModel):
    old_text: str = Field(
        min_length=1,
        description="Existing text identifying a unique region; include enough context",
    )
    new_text: str = Field(
        description="Replacement text; an empty string deletes the matched region"
    )


def create_edit_tool(sandbox: BaseSandbox):
    @tool
    def edit(
        path: Annotated[
            str,
            Field(
                min_length=1,
                description="Existing file path relative to the repository root, or an absolute sandbox path",
            ),
        ],
        edits: Annotated[
            list[Edit],
            Field(
                min_length=1,
                description="Non-overlapping replacements, all matched against the original file",
            ),
        ],
    ) -> str:
        """Replace unique text regions in one UTF-8 file. Try exact matches first, then normalize trailing whitespace, quotes, dashes, special spaces, and line endings. Multiple matches or overlapping regions fail. If any edit fails, write nothing."""
        return execute_file_operation(
            sandbox,
            "edit",
            path=path,
            edits=[entry.model_dump() for entry in edits],
        )

    edit.metadata = {"executionMode": "sequential"}
    return edit
