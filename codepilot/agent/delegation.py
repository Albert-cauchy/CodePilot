"""Batch delegation: validate independently, run child graphs, then accept deltas."""

import base64
import difflib
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from codepilot.sandbox.base import ChildExecutionError
from codepilot.trajectory import TrajectoryLogger
from codepilot.tools.delegate import SubtaskSpec


def describe_patch(changes):
    lines: list[str] = []
    for change in changes:
        name = change["path"]
        try:

            def text(entry):
                if entry is None:
                    return []
                if entry["kind"] == "symlink":
                    return [entry["target"] + "\n"]
                return base64.b64decode(entry["data"]).decode("utf-8").splitlines(True)

            lines.extend(
                difflib.unified_diff(
                    text(change["before"]),
                    text(change["after"]),
                    fromfile="a/" + name,
                    tofile="b/" + name,
                )
            )
        except UnicodeDecodeError:
            lines.append("Binary file changed: " + name + "\n")
    return "".join(lines)


def run_delegation_batch(parent, tasks, tool_call_id):
    from .core import AgentCore

    reports: list[dict[str, Any]] = [
        {"subtask_id": uuid4().hex, "status": "pending"} for _ in tasks
    ]
    specs = {}
    for index, raw in enumerate(tasks):
        try:
            specs[index] = SubtaskSpec.model_validate(raw)
        except ValidationError as exc:
            reports[index].update(status="rejected", error=str(exc))
    if not specs:
        parent.log("delegation_result", tool_call_id=tool_call_id, reports=reports)
        return json.dumps(reports, ensure_ascii=False)

    # One baseline for the ENTIRE batch, including tasks waiting for a worker.
    baseline = parent.sandbox.snapshot()
    scopes = {}
    for index, spec in specs.items():
        try:
            scopes[index] = baseline.validate_scope(spec.allowed_files)
        except ValueError as exc:
            reports[index].update(status="rejected", error=str(exc))
    conflicts: dict[int, set[str]] = {}
    for first, files in scopes.items():
        for second, others in scopes.items():
            if second <= first:
                continue
            overlap = {
                a
                for a in files
                for b in others
                if a == b or a.startswith(b + "/") or b.startswith(a + "/")
            }
            if overlap:
                conflicts.setdefault(first, set()).update(overlap)
                conflicts.setdefault(second, set()).update(overlap)
    for index, paths in conflicts.items():
        reports[index].update(
            status="rejected",
            error="Overlapping file authorization",
            conflicting_files=sorted(paths),
            conflicting_subtasks=[
                reports[other]["subtask_id"]
                for other in conflicts
                if other != index
                and any(
                    a == b or a.startswith(b + "/") or b.startswith(a + "/")
                    for a in scopes[index]
                    for b in scopes[other]
                )
            ],
        )
    accepted = [index for index in specs if reports[index]["status"] != "rejected"]
    parent.log(
        "delegation_started",
        tool_call_id=tool_call_id,
        requested=len(tasks),
        accepted=len(accepted),
        max_parallel_subagents=parent.max_parallel_subagents,
        rejections=[report for report in reports if report["status"] == "rejected"],
    )

    def run_child(index):
        spec = specs[index]
        report = reports[index]
        directory = parent.trajectory_dir / "subagents" / report["subtask_id"]
        directory.mkdir(parents=True, exist_ok=True)
        child_agent = None
        changes = []
        try:
            with parent.sandbox.fork(baseline, scopes[index]) as child:
                child_agent = AgentCore(
                    parent.model,
                    child,
                    max_model_calls=parent.max_model_calls,
                    trajectory_dir=directory,
                    budget=parent.budget,
                    is_subagent=True,
                    agent_id=report["subtask_id"],
                    run_id=parent.run_id,
                    parent_tool_call_id=tool_call_id,
                    logger=TrajectoryLogger(directory / "events.jsonl"),
                )
                try:
                    state = child_agent.run(json.dumps(spec.model_dump(), ensure_ascii=False))
                    report.update(
                        status="completed"
                        if state["termination_reason"] == "voluntary_completion"
                        else "incomplete",
                        termination_reason=state["termination_reason"],
                        summary=state.get("summary", ""),
                        verification=child_agent.verification,
                        unresolved_issues=state.get("unresolved_issues", []),
                        model_iterations=state["own_model_iterations"],
                    )
                    if state["termination_reason"] == "context_limit":
                        report.update(status="failed", error="Subagent context limit")
                except Exception as exc:
                    report.update(
                        status="failed",
                        termination_reason="subagent_error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                finally:
                    # Collect before the context manager removes the private copy.
                    # Production backends stop background processes first.
                    try:
                        changes = child.changes_since(baseline)
                    except ChildExecutionError as exc:
                        changes = exc.changes
                        raise
                if any(change["path"] not in scopes[index] for change in changes):
                    report.update(
                        status="failed",
                        error="Unauthorized repository changes",
                        termination_reason="authorization_error",
                    )
        except Exception as exc:
            report.update(
                status="failed",
                termination_reason="subagent_error",
                error=f"{type(exc).__name__}: {exc}",
            )
            # Preserve inspectable partial work even if a model request failed.
            # A stopped/failed environment may not permit a snapshot; never merge it.
        report.setdefault("summary", "")
        report.setdefault("verification", child_agent.verification if child_agent else [])
        report.setdefault("unresolved_issues", [report["error"]] if "error" in report else [])
        report["model_iterations"] = child_agent.normal_responses if child_agent else 0
        report["changed_files"] = [change["path"] for change in changes]
        report["artifact_directory"] = str(directory)
        (directory / "changes.json").write_text(
            json.dumps(changes, ensure_ascii=False), encoding="utf-8"
        )
        (directory / "patch.diff").write_text(describe_patch(changes), encoding="utf-8")
        return index, report, changes

    results = []
    with ThreadPoolExecutor(max_workers=parent.max_parallel_subagents) as executor:
        for result in executor.map(run_child, accepted):
            results.append(result)
    # All children have stopped; integrate sequentially, independently per child.
    for index, report, changes in results:
        if report["status"] == "failed":
            report["modification_status"] = "withheld"
        elif changes:
            try:
                parent.sandbox.apply_changes(changes)
                report["modification_status"] = "applied"
            except Exception as exc:
                report.update(
                    status="apply_failed",
                    modification_status="rejected",
                    error=f"{type(exc).__name__}: {exc}",
                )
        else:
            report["modification_status"] = "no_changes"
        reports[index] = report
        directory = parent.trajectory_dir / "subagents" / report["subtask_id"]
        (directory / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        parent.log("subtask_result", tool_call_id=tool_call_id, **report)
    parent.log("delegation_result", tool_call_id=tool_call_id, reports=reports)
    return json.dumps(reports, ensure_ascii=False)
