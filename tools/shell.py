from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

from codepilot.sandbox.base import BaseSandbox, SandboxError


def create_shell_tool(sandbox: BaseSandbox):

    @tool
    def shell(
        command: Annotated[
            str,
            Field(
                description="Shell command to execute inside the repository workspace"
            ),
        ],
    ) -> str:
        """Execute a shell command inside the current repository"""
        result = sandbox.execute(command)
        if result.timed_out:
            raise SandboxError("Shell command timed out.")
        parts = [
            f"Exit code:{result.exit_code}",
        ]
        if result.stdout:
            parts.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            parts.append(f"STDERR:\n{result.stderr}")

        return "\n\n".join(parts)

    shell.metadata = {"executionMode": "sequential"}
    return shell
