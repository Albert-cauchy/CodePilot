import json
import sys
from pathlib import Path
from types import SimpleNamespace

from codepilot.benchmark import Task
from codepilot.submission import Prediction, SubmissionWriter
from scripts import run_benchmark
from scripts.run_benchmark import evaluate_predictions


def test_evaluation_counts_only_selected_instances(monkeypatch, tmp_path, capsys):
    from swebench.harness import run_evaluation

    monkeypatch.chdir(tmp_path)
    tasks = [
        Task(f"org__repo-{number}", "org/repo", "abc", "fix")
        for number in range(1, 4)
    ]
    predictions = {
        task.instance_id: Prediction(task.instance_id, "diff", "test-model")
        for task in tasks[:1]
    }
    predictions[tasks[1].instance_id] = Prediction(
        tasks[1].instance_id, "", "test-model"
    )
    predictions["org__repo-4"] = Prediction(
        "org__repo-4", "diff", "test-model"
    )

    def fake_evaluation(**kwargs):
        assert kwargs["dataset_name"] == "SWE-bench/SWE-bench_Verified"
        assert kwargs["instance_ids"] == [task.instance_id for task in tasks]
        assert kwargs["max_workers"] == 2
        rows = [
            json.loads(line)
            for line in Path(kwargs["predictions_path"]).read_text().splitlines()
        ]
        assert {row["instance_id"] for row in rows} == {
            tasks[0].instance_id,
            tasks[1].instance_id,
        }

        report_path = tmp_path / "results.json"
        report_path.write_text(json.dumps({
            "resolved_ids": [tasks[0].instance_id],
            "unresolved_ids": [],
            "empty_patch_ids": [tasks[1].instance_id],
            "error_ids": [],
            "incomplete_ids": [tasks[2].instance_id],
        }))
        return report_path

    monkeypatch.setattr(run_evaluation, "main", fake_evaluation)
    adapter = SimpleNamespace(
        dataset_name="SWE-bench/SWE-bench_Verified",
        split="test",
    )

    evaluate_predictions(adapter, tasks, predictions, max_workers=2)

    output = capsys.readouterr().out
    assert f"{tasks[0].instance_id}: RESOLVED" in output
    assert f"{tasks[1].instance_id}: EMPTY_PATCH" in output
    assert f"{tasks[2].instance_id}: NO_PREDICTION" in output
    assert "SWE-bench resolved: 1/3" in output


def test_eval_only_uses_saved_predictions(monkeypatch, tmp_path):
    task = Task("org__repo-1", "org/repo", "abc", "fix")
    adapter = SimpleNamespace(
        iter_tasks=lambda: iter([task]),
        dataset_name="SWE-bench/SWE-bench_Verified",
        split="test",
    )
    output = tmp_path / "predictions.jsonl"
    SubmissionWriter.write(output, [Prediction(task.instance_id, "diff", "test")])
    monkeypatch.setattr(run_benchmark, "SWEBenchAdapter", lambda: adapter)
    monkeypatch.setattr(
        run_benchmark,
        "create_model",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("model called")),
    )
    called = []
    monkeypatch.setattr(
        run_benchmark,
        "evaluate_predictions",
        lambda source, tasks, predictions, workers: called.append(
            (tasks, predictions)
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_benchmark", "--eval-only", "--output", str(output)],
    )

    run_benchmark.main()

    assert called[0][0] == [task]
    assert task.instance_id in called[0][1]
