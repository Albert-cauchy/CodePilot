from codepilot.benchmark import SWEBenchAdapter, Task, TaskSource


def test_to_task_whitelists_only_safe_fields():
    raw = {
        "instance_id": "org__repo-1",
        "repo": "org/repo",
        "base_commit": "abc123",
        "problem_statement": "fix me",
        "patch": "GOLD ANSWER",
        "test_patch": "HIDDEN TEST",
        "FAIL_TO_PASS": ["x"],
        "PASS_TO_PASS": ["y"],
    }

    task = SWEBenchAdapter._to_task(raw)

    assert task == Task(
        instance_id="org__repo-1",
        repo="org/repo",
        base_commit="abc123",
        problem_statement="fix me",
    )
    assert not hasattr(task, "patch")
    assert not hasattr(task, "test_patch")
    assert not hasattr(task, "FAIL_TO_PASS")
    assert not hasattr(task, "PASS_TO_PASS")


def test_task_source_reads_only_inference_fields():
    source: TaskSource = SWEBenchAdapter(
        dataset=[
            {
                "instance_id": "org__repo-1",
                "repo": "org/repo",
                "base_commit": "abc123",
                "problem_statement": "fix me",
                "patch": "GOLD ANSWER",
            }
        ]
    )

    assert source.load_task("org__repo-1") == next(source.iter_tasks())
    assert "GOLD ANSWER" not in repr(source.load_task("org__repo-1"))
