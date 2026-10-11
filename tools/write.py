from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

from codepilot.sandbox.base import BaseSandbox
from tools._files import execute_file_operation


def create_write_tool(sandbox: BaseSandbox):
    @tool
    def write(
        path: Annotated[
            str,
            Field(
                min_length=1,
                description="File path relative to the repository root, or an absolute sandbox path",
            ),
        ],
        content: Annotated[
            str,
            Field(
                description="Complete UTF-8 contents to write, replacing any existing contents"
            ),
        ],
    ) -> str:
        """Write an entire file, creating parent directories and the file if absent. Existing contents are replaced."""
        return execute_file_operation(sandbox, "write", path=path, content=content)

    write.metadata = {"executionMode": "sequential"}
    return write
