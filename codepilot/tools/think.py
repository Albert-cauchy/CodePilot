"""Persist the current model's plan without invoking another model."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from codepilot.tools._file_operations import execute_operation


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    status: Literal["pending", "in_progress", "completed"]

    @model_validator(mode="after")
    def single_line(self):
        if any(char in self.id + self.description for char in "\r\n"):
            raise ValueError("Plan IDs and descriptions must be single-line strings")
        return self


class ThinkInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: list[PlanStep] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({step.id for step in self.steps}) != len(self.steps):
            raise ValueError("Plan step IDs must be unique")
        return self


def create_think_tool(trajectory_dir: Path):
    directory = Path(trajectory_dir).resolve()
    plan_path = directory / "plan.md"

    @tool(args_schema=ThinkInput)
    def think(steps: list[PlanStep]) -> str:
        """Create or revise your execution plan. Analyze the task and available evidence,
        then provide the complete current list of steps, including unchanged steps.
        Use stable IDs and update statuses as work progresses. This tool saves your
        plan.md; it does not call a planning model or execute the plan. Call alone.
        """
        content = "# Task plan\n\n" + "".join(
            f"- {step.id} | {step.status} | {step.description}\n" for step in steps
        )
        directory.mkdir(parents=True, exist_ok=True)
        if plan_path.is_symlink():
            raise ValueError("plan.md must not be a symbolic link")
        exists = plan_path.exists()
        with TemporaryDirectory(prefix=".think-", dir=directory) as staging:
            candidate = Path(staging) / "plan.md"
            if exists:
                original = plan_path.read_text(encoding="utf-8")
                if not original:
                    raise ValueError("Existing plan is empty; cannot apply an edit")
                candidate.write_text(original, encoding="utf-8")
                execute_operation(
                    {
                        "operation": "edit",
                        "path": str(candidate),
                        "edits": [{"old_text": original, "new_text": content}],
                    }
                )
            else:
                execute_operation(
                    {
                        "operation": "write",
                        "path": str(candidate),
                        "content": content,
                    }
                )
            os.replace(candidate, plan_path)
        return json.dumps(
            {"operation": "edit" if exists else "write", "plan": content}, ensure_ascii=False
        )

    think.metadata = {"executionMode": "sequential"}
    return think
