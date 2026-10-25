"""Summarize one saved think/subagent run, including all child model costs."""

import argparse
import json
from collections import Counter
from pathlib import Path

TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens")


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_trajectory(directory: Path, instance_id: str) -> list[dict] | None:
    paths = sorted(
        (directory / instance_id).glob("*/events.jsonl"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in paths:
        events = read_events(path)
        if not any(event["event_type"] == "run_completed" for event in events):
            continue
        if not any(event["event_type"] == "termination" for event in events):
            continue
        children = [
            event
            for child in sorted(path.parent.glob("subagents/*/events.jsonl"))
            for event in read_events(child)
        ]
        return [*children, *events]
    return None


def summarize_trajectory(events: list[dict]) -> dict:
    termination = next(
        event
        for event in reversed(events)
        if event["event_type"] == "termination" and event.get("agent_id") == "main"
    )
    usages = [
        event["message"]["data"].get("usage_metadata")
        for event in events
        if event["event_type"] == "model_response"
    ]
    tokens = None
    if usages and all(
        usage is not None and all(key in usage for key in TOKEN_FIELDS) for usage in usages
    ):
        tokens = {key: sum(usage[key] for usage in usages) for key in TOKEN_FIELDS}
    return {
        "termination_reason": termination["termination_reason"],
        "model_iterations": termination["model_iterations"],
        "decision_counts": termination.get("decision_counts"),
        "tool_call_iterations": sum(
            event["tool_call_iterations"]
            for event in events
            if event["event_type"] == "termination"
        ),
        "elapsed_seconds": termination["elapsed_seconds"],
        "tokens": tokens,
        "tool_requests_by_name": dict(
            Counter(event["name"] for event in events if event["event_type"] == "tool_call")
        ),
        "tool_results_by_status": dict(
            Counter(event["status"] for event in events if event["event_type"] == "tool_result")
        ),
    }


def load_run(output: Path, trajectory_dir: Path | None = None) -> dict:
    metadata = json.loads(output.with_suffix(".run.json").read_text())
    results = json.loads(output.with_suffix(".results.json").read_text())
    statuses = {}
    for field, status in (
        ("resolved_ids", "resolved"),
        ("unresolved_ids", "unresolved"),
        ("empty_patch_ids", "empty_patch"),
        ("error_ids", "evaluation_error"),
        ("incomplete_ids", "no_prediction"),
    ):
        for instance_id in results["report"].get(field, []):
            statuses[instance_id] = status
    directory = Path(trajectory_dir or metadata.get("trajectory_dir", "trajectory"))
    trajectories = {}
    for instance_id in results["instance_ids"]:
        events = load_trajectory(directory, instance_id)
        if events is not None:
            trajectories[instance_id] = summarize_trajectory(events)
    return {
        "metadata": metadata,
        "instance_ids": results["instance_ids"],
        "statuses": statuses,
        "trajectories": trajectories,
    }


def summarize_costs(rows: list[dict]) -> dict:
    """Aggregate one run on the common valid task set."""
    count = len(rows)
    totals = {
        key: sum(row[key] for row in rows)
        for key in ("model_iterations", "tool_call_iterations", "elapsed_seconds")
    }
    known_tokens = [row["tokens"] for row in rows if row["tokens"] is not None]
    requests: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    for row in rows:
        requests.update(row["tool_requests_by_name"])
        statuses.update(row["tool_results_by_status"])
    return {
        "task_count": count,
        "totals": totals,
        "means": {key: value / count if count else None for key, value in totals.items()},
        "reported_tokens": {
            "covered_tasks": len(known_tokens),
            "total_tasks": count,
            "totals": {key: sum(usage[key] for usage in known_tokens) for key in TOKEN_FIELDS},
        },
        "termination_reasons": dict(Counter(row["termination_reason"] for row in rows)),
        "tool_requests_by_name": dict(requests),
        "tool_results_by_status": dict(statuses),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize one saved agent run")
    parser.add_argument("--predictions", type=Path, default=Path("outputs/dev/predictions.jsonl"))
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/dev/summary.json"))
    args = parser.parse_args(argv)
    run = load_run(args.predictions, args.trajectory_dir)
    summary = {
        "metadata": run["metadata"],
        "statuses": run["statuses"],
        "costs": summarize_costs(list(run["trajectories"].values())),
        "tasks": run["trajectories"],
        "missing_trajectories": [
            task for task in run["instance_ids"] if task not in run["trajectories"]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved run summary: {args.output}")


if __name__ == "__main__":
    main()
