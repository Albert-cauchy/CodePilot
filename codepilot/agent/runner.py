from __future__ import annotations

from collections.abc import Callable

from codepilot.benchmark.base import Task
from codepilot.patch import PatchManager, PatchSnapshot
from codepilot.sandbox.base import RepositorySandbox

from .core import AgentCore


def run_task(
    task: Task,
    model,
    sandbox_factory: Callable[[Task], RepositorySandbox],
    *,
    tool_configuration: str = "two",
    max_model_calls: int = 500,
    logger=None,
) -> tuple[dict, PatchSnapshot]:
    """Run one task using only the task and sandbox interfaces."""
    with sandbox_factory(task) as sandbox:
        agent = AgentCore(
            model=model,
            sandbox=sandbox,
            max_model_calls=max_model_calls,
            tool_configuration=tool_configuration,
            logger=logger,
        )
        final_state = agent.run(task.problem_statement)
        patch = PatchManager(sandbox).collect()

    return final_state, patch
