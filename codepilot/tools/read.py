from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

from codepilot.sandbox.base import BaseSandbox
from codepilot.tools._files import execute_file_operation


def create_read_tool(sandbox: BaseSandbox):
    @tool
    def read(
        path: Annotated[
            str,
            Field(
                min_length=1,
                description="File path relative to the repository root, or an absolute sandbox path",
            ),
        ],
        offset: Annotated[
            int,
            Field(
                ge=1,
                strict=True,
                description="First line to read, using one-based line numbers",
            ),
        ] = 1,
        limit: Annotated[
            int,
            Field(ge=1, strict=True, description="Maximum number of lines to return"),
        ] = 200,
    ) -> str:
        """Read a UTF-8 file with line numbers and an optional line range. Reads may run in parallel."""
        return execute_file_operation(
            sandbox, "read", path=path, offset=offset, limit=limit
        )

    read.metadata = {"executionMode": "parallel"}
    return read
