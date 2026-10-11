import json
import sys
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from codepilot.benchmark import Task
from codepilot.patch import PatchSnapshot
from codepilot.submission import Prediction, SubmissionWriter
from scripts import run_benchmark


def manifest_file(tmp_path):
    manifest = {
        "dataset": "SWE-bench/SWE-bench_Verified",
        "split": "test",
        "revision": "pinned-revision",
        "sample_size": 2,
        "instance_ids": ["org__repo-2", "org__repo-1"],
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def setup_adapter(monkeypatch, manifest):
    loaded = []

    def load_task(instance_id):
        loaded.append(instance_id)
        return Task(instance_id, "org/repo", "base", "Fix")

    def adapter(**kwargs):
        assert kwargs == {
            "dataset_name": manifest["dataset"],
            "split": "test",
            "revision": manifest["revision"],
        }
        return SimpleNamespace(load_task=load_task)

    monkeypatch.setattr(run_benchmark, "SWEBenchAdapter", adapter)
    return loaded


def test_development_run_uses_only_manifest_ids_and_records_trajectory(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    path, manifest = manifest_file(tmp_path)
    loaded = setup_adapter(monkeypatch, manifest)
    received = []

    class Model:
        def invoke(self, messages):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "finish",
                        "args": {"summary": "done"},
                        "id": "finish-1",
                    }
                ],
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            )

    def create_model(**kwargs):
        assert kwargs["model_name"] == "Qwen/Qwen3.5-35B-A3B"
        assert kwargs["thinking"] is True
        return Model()

    def run_task(task, model, sandbox_factory, **kwargs):
        received.append(task.instance_id)
        assert kwargs["tool_configuration"] == "five"
        assert kwargs["max_model_calls"] == 3
        model.invoke([HumanMessage(content=task.problem_statement)])
        kwargs["logger"].log(
            "tool_result", name="finish", content="done", status="success"
        )
        return {"termination_reason": "voluntary_completion"}, PatchSnapshot(
            "diff", ("file",)
        )

    evaluated = []
    monkeypatch.setattr(run_benchmark, "create_model", create_model)
    monkeypatch.setattr(run_benchmark, "run_task", run_task)
    monkeypatch.setattr(
        run_benchmark,
        "evaluate_predictions",
        lambda adapter, tasks, predictions, workers: (
            evaluated.append(tasks)
            or {"resolved_ids": [task.instance_id for task in tasks]}
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_dev", "--task-file", str(path), "--limit", "1", "--max-model-calls", "3"],
    )
    run_benchmark.main(development=True)
    assert loaded == received == [manifest["instance_ids"][0]]
    assert [task.instance_id for task in evaluated[0]] == received
    output = tmp_path / "outputs/dev/five/predictions.jsonl"
    assert SubmissionWriter.read(output)[0].model_name_or_path == "Qwen/Qwen3.5-35B-A3B"
    metadata = json.loads(output.with_suffix(".run.json").read_text())
    assert metadata["revision"] == "pinned-revision"
    assert metadata["instance_ids"] == manifest["instance_ids"]
    assert metadata["max_model_calls"] == 3
    assert "max_tool_calls" not in metadata
    events = [
        json.loads(line)
        for line in (output.parent / "trajectories/org__repo-2.attempt-1.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [event["event_type"] for event in events] == [
        "model_input",
        "model_response",
        "tool_result",
        "run_completed",
    ]
    assert events[-1]["patch"] == "diff"
    assert output.with_suffix(".results.json").exists()

    # A second invocation resumes without invoking a model or running a sandbox.
    monkeypatch.setattr(
        run_benchmark, "create_model", lambda **kwargs: pytest.fail("model called")
    )
    run_benchmark.main(development=True)
    assert received == [manifest["instance_ids"][0]]


def test_development_eval_only_never_runs_agent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    path, manifest = manifest_file(tmp_path)
    loaded = setup_adapter(monkeypatch, manifest)
    output = tmp_path / "predictions.jsonl"
    SubmissionWriter.write(
        output, [Prediction(manifest["instance_ids"][0], "diff", "model")]
    )
    output.with_suffix(".run.json").write_text(
        json.dumps({"tool_configuration": "two"})
    )
    monkeypatch.setattr(
        run_benchmark, "create_model", lambda **kwargs: pytest.fail("model called")
    )
    monkeypatch.setattr(
        run_benchmark, "run_task", lambda *args, **kwargs: pytest.fail("agent called")
    )
    evaluated = []
    monkeypatch.setattr(
        run_benchmark,
        "evaluate_predictions",
        lambda source, tasks, predictions, workers: (
            evaluated.extend(tasks)
            or {"resolved_ids": [], "incomplete_ids": [manifest["instance_ids"][1]]}
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dev",
            "--eval-only",
            "--task-file",
            str(path),
            "--output",
            str(output),
        ],
    )
    run_benchmark.main(development=True)
    assert (
        loaded == [task.instance_id for task in evaluated] == manifest["instance_ids"]
    )
    assert (
        json.loads(output.with_suffix(".results.json").read_text())[
            "tool_configuration"
        ]
        == "two"
    )


def test_evaluator_receives_pinned_rows_and_filters_predictions(monkeypatch, tmp_path):
    from swebench.harness import run_evaluation
    from swebench.harness.utils import load_swebench_dataset

    monkeypatch.chdir(tmp_path)
    task = Task("org__repo-1", "org/repo", "base", "Fix")
    rows = [
        {"instance_id": task.instance_id, "patch": "gold for evaluator only"},
        {"instance_id": "unselected", "patch": "not needed"},
    ]
    adapter = SimpleNamespace(
        dataset_name="SWE-bench/SWE-bench_Verified",
        split="test",
        revision="pinned",
        _dataset=rows,
    )

    def evaluation(**kwargs):
        assert load_swebench_dataset(kwargs["dataset_name"]) == rows[:1]
        report = {
            "resolved_ids": [task.instance_id],
            "unresolved_ids": [],
            "empty_patch_ids": [],
            "error_ids": [],
            "incomplete_ids": [],
        }
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(report))
        return report_path

    monkeypatch.setattr(run_evaluation, "main", evaluation)
    result = run_benchmark.evaluate_predictions(
        adapter,
        [task],
        {task.instance_id: Prediction(task.instance_id, "diff", "model")},
        2,
    )
    assert result["resolved_ids"] == [task.instance_id]


@pytest.mark.parametrize(
    "change",
    [
        {"instance_ids": ["duplicate", "duplicate"]},
        {"revision": ""},
        {"sample_size": 3},
        {"instance_ids": []},
    ],
)
def test_invalid_development_manifests_are_rejected(tmp_path, change):
    path, manifest = manifest_file(tmp_path)
    path.write_text(json.dumps({**manifest, **change}))
    with pytest.raises(ValueError):
        run_benchmark.load_development_manifest(path)
