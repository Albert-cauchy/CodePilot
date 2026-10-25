import json

from scripts.analyze_dev import load_run


def test_single_run_summary_counts_child_and_final_analysis_costs(tmp_path):
    output = tmp_path / "predictions.jsonl"
    trajectory = tmp_path / "trajectory"
    run = trajectory / "task-1/attempt-1-example"
    child = run / "subagents/child-1"
    child.mkdir(parents=True)
    output.with_suffix(".run.json").write_text(json.dumps({"trajectory_dir": str(trajectory)}))
    output.with_suffix(".results.json").write_text(
        json.dumps({"instance_ids": ["task-1"], "report": {"resolved_ids": ["task-1"]}})
    )

    def response(actor, total):
        return {
            "event_type": "model_response",
            "agent_id": actor,
            "message": {
                "data": {
                    "usage_metadata": {
                        "input_tokens": total,
                        "output_tokens": 1,
                        "total_tokens": total + 1,
                    }
                }
            },
        }

    root_events = [
        response("main", 10),
        response("main", 30),
        {
            "event_type": "termination",
            "agent_id": "main",
            "model_iterations": 3,
            "tool_call_iterations": 1,
            "elapsed_seconds": 2,
            "termination_reason": "model_call_limit",
            "decision_counts": {"main": 1, "subagent": 1, "final_analysis": 1, "total": 3},
        },
        {"event_type": "run_completed", "instance_id": "task-1"},
    ]
    (run / "events.jsonl").write_text("\n".join(json.dumps(event) for event in root_events))
    (child / "events.jsonl").write_text(json.dumps(response("child-1", 20)))
    saved = load_run(output)
    row = saved["trajectories"]["task-1"]
    assert row["model_iterations"] == 3
    assert row["tokens"] == {"input_tokens": 60, "output_tokens": 3, "total_tokens": 63}
