from types import SimpleNamespace

from codepilot.benchmark import Task
from codepilot.sandbox import DockerSandbox, swebench_image_name


def test_remote_swebench_image_name():
    assert swebench_image_name("sympy__sympy-20590") == (
        "swebench/sweb.eval.x86_64.sympy_1776_sympy-20590:latest"
    )


def test_local_swebench_image_name_keeps_double_underscore():
    assert swebench_image_name(
        "sympy__sympy-20590",
        namespace=None,
    ) == "sweb.eval.x86_64.sympy__sympy-20590:latest"


def test_execute_preserves_patch_output():
    class FakeContainer:
        status = "running"

        def reload(self):
            pass

        def exec_run(self, *args, **kwargs):
            return SimpleNamespace(
                exit_code=0,
                output=(b"diff --git a/a.py b/a.py\n\n", b""),
            )

    task = Task("org__repo-1", "org/repo", "abc123", "fix me")
    sandbox = DockerSandbox(task, client=object())
    sandbox.container = FakeContainer()

    assert sandbox.execute("git diff").stdout.endswith("\n\n")
