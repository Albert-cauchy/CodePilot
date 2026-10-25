import json

from codepilot.benchmark import Task
from codepilot.submission import SubmissionWriter


def test_submission_jsonl_schema(tmp_path):
    task = Task(
        instance_id="org__repo-1",
        repo="org/repo",
        base_commit="abc",
        problem_statement="fix",
    )
    prediction = SubmissionWriter.from_patch(
        task,
        "diff --git a/a.py b/a.py\n",
        model_name_or_path="my-agent",
    )

    path = SubmissionWriter.write(
        tmp_path / "predictions.jsonl",
        [prediction],
    )

    row = json.loads(path.read_text().strip())
    assert row == {
        "instance_id": "org__repo-1",
        "model_patch": "diff --git a/a.py b/a.py\n",
        "model_name_or_path": "my-agent",
    }
    assert SubmissionWriter.read(path) == [prediction]
