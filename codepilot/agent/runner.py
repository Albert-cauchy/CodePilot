from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from codepilot.benchmark.base import Task
from codepilot.patch import PatchManager, PatchSnapshot
from codepilot.sandbox.base import RepositorySandbox

from .core import AgentCore


def run_task(
    task: Task,
    model,
    sandbox_factory: Callable[[Task], RepositorySandbox],
    *,
    max_model_calls: int = 500,
    logger=None,
    trajectory_dir: Path | str | None = None,
    max_parallel_subagents: int = 4,
) -> tuple[dict, PatchSnapshot]:
    """Run one task using only the task and sandbox interfaces."""
    directory = (
        Path(trajectory_dir)
        if trajectory_dir is not None
        else (
            Path(logger.path).parent
            if logger is not None
            else Path("trajectory") / task.instance_id / uuid4().hex
        )
    )
    with sandbox_factory(task) as sandbox:
        agent = AgentCore(
            model=model,
            sandbox=sandbox,
            max_model_calls=max_model_calls,
            logger=logger,
            trajectory_dir=directory,
            max_parallel_subagents=max_parallel_subagents,
        )
        final_state = agent.run(task.problem_statement)
        patch = PatchManager(sandbox).collect()

    return final_state, patch
