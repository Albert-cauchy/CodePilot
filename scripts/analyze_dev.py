"""Compare saved two-tool and five-tool development experiments."""

import argparse
import json
from collections import Counter
from pathlib import Path

COMPARISON_FIELDS = (
    "dataset",
    "split",
    "revision",
    "instance_ids",
    "model",
    "base_url",
    "temperature",
    "top_p",
    "thinking",
    "max_completion_tokens",
    "max_model_calls",
)
TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens")
VALID_STATUSES = {"resolved", "unresolved", "empty_patch"}


def load_trajectory(directory: Path, instance_id: str) -> list[dict] | None:
    """Select the latest completed attempt, excluding failed attempt costs."""
    paths = sorted(
        directory.glob(f"{instance_id}.attempt-*.jsonl"),
        key=lambda path: int(path.stem.rsplit(".attempt-", 1)[1]),
        reverse=True,
    )
    for path in paths:
        events = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        # A resumed command can append another attempt to the same file.
        boundaries = [
            i
            for i, event in enumerate(events)
            if event["event_type"] in {"run_completed", "run_failed"}
        ]
        for end in reversed(boundaries):
            if events[end]["event_type"] != "run_completed":
                continue
            if events[end]["instance_id"] != instance_id:
                continue
            start = max((i + 1 for i in boundaries if i < end), default=0)
            segment = events[start : end + 1]
            if any(event["event_type"] == "termination" for event in segment):
                return segment
    return None


def summarize_trajectory(events: list[dict]) -> dict:
    """Read attempt costs without treating finish or rejected requests as attempts."""
    termination = next(
        event for event in reversed(events) if event["event_type"] == "termination"
    )
    usages = [
        event["message"]["data"].get("usage_metadata")
        for event in events
        if event["event_type"] == "model_response"
    ]
    tokens = None
    if all(
        usage is not None and all(key in usage for key in TOKEN_FIELDS)
        for usage in usages
    ):
        tokens = {key: sum(usage[key] for usage in usages) for key in TOKEN_FIELDS}
    return {
        "termination_reason": termination["termination_reason"],
        "model_iterations": termination["model_iterations"],
        "tool_call_iterations": termination["tool_call_iterations"],
        "elapsed_seconds": termination["elapsed_seconds"],
        "tokens": tokens,
        "tool_requests_by_name": dict(
            Counter(
                event["name"] for event in events if event["event_type"] == "tool_call"
            )
        ),
        "tool_results_by_status": dict(
            Counter(
                event["status"]
                for event in events
                if event["event_type"] == "tool_result"
            )
        ),
    }


def load_run(output: Path, trajectory_dir: Path | None = None) -> dict:
    """Load saved metadata, evaluation statuses, and completed trajectories."""
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
    trajectories = {}
    directory = (
        trajectory_dir if trajectory_dir is not None else output.parent / "trajectories"
    )
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
    """Aggregate one configuration on the common valid task set."""
    count = len(rows)
    totals = {
        key: sum(row[key] for row in rows)
        for key in ("model_iterations", "tool_call_iterations", "elapsed_seconds")
    }
    known_tokens = [row["tokens"] for row in rows if row["tokens"] is not None]
    requests = Counter()
    statuses = Counter()
    for row in rows:
        requests.update(row["tool_requests_by_name"])
        statuses.update(row["tool_results_by_status"])
    return {
        "task_count": count,
        "totals": totals,
        "means": {
            key: value / count if count else None for key, value in totals.items()
        },
        "reported_tokens": {
            "covered_tasks": len(known_tokens),
            "total_tasks": count,
            "totals": {
                key: sum(usage[key] for usage in known_tokens) for key in TOKEN_FIELDS
            },
        },
        "termination_reasons": dict(Counter(row["termination_reason"] for row in rows)),
        "tool_requests_by_name": dict(requests),
        "tool_results_by_status": dict(statuses),
    }


def compare_runs(two: dict, five: dict) -> dict:
    """Pair valid tasks under matching experimental conditions."""
    if (
        two["metadata"]["tool_configuration"] != "two"
        or five["metadata"]["tool_configuration"] != "five"
    ):
        raise ValueError("Expected two-tool and five-tool run metadata")
    for field in COMPARISON_FIELDS:
        if two["metadata"][field] != five["metadata"][field]:
            raise ValueError(f"Run settings differ: {field}")

    selected_ids = list(dict.fromkeys([*two["instance_ids"], *five["instance_ids"]]))
    groups = {
        key: []
        for key in ("both_resolved", "two_only", "five_only", "neither_resolved")
    }
    paired = []
    excluded = []
    for instance_id in selected_ids:
        issues = {}
        for label, run in (("two", two), ("five", five)):
            status = run["statuses"].get(instance_id, "not_evaluated")
            if instance_id not in run["instance_ids"]:
                issues[label] = "not_selected"
            elif status not in VALID_STATUSES:
                issues[label] = status
            elif instance_id not in run["trajectories"]:
                issues[label] = "missing_completed_trajectory"
        if issues:
            excluded.append({"instance_id": instance_id, "reasons": issues})
            continue
        resolved_two = two["statuses"][instance_id] == "resolved"
        resolved_five = five["statuses"][instance_id] == "resolved"
        group = (
            "both_resolved"
            if resolved_two and resolved_five
            else "two_only"
            if resolved_two
            else "five_only"
            if resolved_five
            else "neither_resolved"
        )
        groups[group].append(instance_id)
        paired.append(
            {
                "instance_id": instance_id,
                "two_resolved": resolved_two,
                "five_resolved": resolved_five,
                "two": two["trajectories"][instance_id],
                "five": five["trajectories"][instance_id],
            }
        )

    count = len(paired)
    two_count = len(groups["both_resolved"]) + len(groups["two_only"])
    five_count = len(groups["both_resolved"]) + len(groups["five_only"])
    return {
        "selected_count": len(selected_ids),
        "paired_count": count,
        "provisional": bool(excluded) or not count,
        "groups": groups,
        "resolved_at_1": {
            "two": two_count / count if count else None,
            "five": five_count / count if count else None,
            "five_minus_two": (five_count - two_count) / count if count else None,
        },
        "costs": {
            label: summarize_costs([pair[label] for pair in paired])
            for label in ("two", "five")
        },
        "excluded": excluded,
        "paired_tasks": paired,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare saved paired development runs"
    )
    parser.add_argument(
        "--two-output", type=Path, default=Path("outputs/dev/two/predictions.jsonl")
    )
    parser.add_argument(
        "--five-output", type=Path, default=Path("outputs/dev/five/predictions.jsonl")
    )
    parser.add_argument("--two-trajectory-dir", type=Path)
    parser.add_argument("--five-trajectory-dir", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/dev/comparison.json")
    )
    args = parser.parse_args(argv)
    summary = compare_runs(
        load_run(args.two_output, args.two_trajectory_dir),
        load_run(args.five_output, args.five_trajectory_dir),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Paired tasks: {summary['paired_count']}/{summary['selected_count']}")
    print(f"Saved comparison: {args.output}")


if __name__ == "__main__":
    main()
