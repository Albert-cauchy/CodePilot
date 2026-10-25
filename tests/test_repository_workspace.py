import subprocess

import pytest

from codepilot.sandbox.base import ExecResult, RepositorySandbox


class LocalRepository(RepositorySandbox):
    """Real subprocess adapter at the sandbox boundary; no Docker/data needed."""

    def __init__(self, root):
        self.root = root

    def start(self):
        pass

    def stop(self):
        pass

    def execute(self, command, *, timeout=120, check=False, workdir=None):
        output = subprocess.run(
            command,
            shell=True,
            cwd=workdir or self.root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        result = ExecResult(command, output.returncode, output.stdout, output.stderr, 0)
        if check and not result.ok:
            raise RuntimeError(result.output)
        return result

    def prepare_patch_snapshot(self):
        pass

    def git_diff(self, *, binary=True):
        return ""

    def changed_files(self):
        return []


def test_snapshot_keeps_parent_changes_and_applies_child_delta_atomically(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "a.txt").write_text("parent's unfinished change\n")
    (parent / "untracked.txt").write_text("new parent file\n")
    sandbox = LocalRepository(parent)
    baseline = sandbox.snapshot()
    child = tmp_path / "child"
    baseline.materialize(child)
    assert (child / "a.txt").read_text() == "parent's unfinished change\n"
    assert (child / "untracked.txt").read_text() == "new parent file\n"
    (child / "a.txt").write_text("child fix\n")
    (child / "b.txt").write_text("new child file\n")
    changes = LocalRepository(child).changes_since(baseline)
    # A conflicting parent file prevents BOTH child changes from being applied.
    (parent / "a.txt").write_text("concurrent parent update\n")
    with pytest.raises(Exception, match="conflict"):
        sandbox.apply_changes(changes)
    assert not (parent / "b.txt").exists()
    assert (parent / "a.txt").read_text() == "concurrent parent update\n"
    (parent / "a.txt").write_text("parent's unfinished change\n")
    sandbox.apply_changes(changes)
    assert (parent / "a.txt").read_text() == "child fix\n"
    assert (parent / "b.txt").read_text() == "new child file\n"
    assert (parent / "untracked.txt").read_text() == "new parent file\n"


def test_controller_does_not_import_task_files_named_after_standard_library_modules(tmp_path):
    (tmp_path / "base64.py").write_text(
        "from pathlib import Path\nPath('unauthorized.txt').write_text('escaped')\n"
        "raise RuntimeError('repository module executed')\n"
    )
    baseline = LocalRepository(tmp_path).snapshot()
    assert "base64.py" in baseline.entries
    assert not (tmp_path / "unauthorized.txt").exists()
