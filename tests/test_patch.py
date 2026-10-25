import subprocess
from types import SimpleNamespace

from codepilot.patch import PatchManager
from codepilot.sandbox import DockerSandbox


class FakeSandbox:
    def __init__(self):
        self.prepared = False

    def prepare_patch_snapshot(self):
        self.prepared = True

    def git_diff(self, *, binary=True):
        assert self.prepared
        assert binary is True
        return "diff --git a/a.py b/a.py\n"

    def changed_files(self):
        assert self.prepared
        return ["a.py"]


def test_patch_manager_collect():
    patch = PatchManager(FakeSandbox()).collect()
    assert patch.changed_files == ("a.py",)
    assert not patch.is_empty
    assert "diff --git" in patch.text


class LocalGitSandbox(DockerSandbox):
    def __init__(self, workdir, initial_head):
        self.local_workdir = workdir
        self._initial_head = initial_head

    def execute(self, command, **kwargs):
        result = subprocess.run(
            command,
            cwd=self.local_workdir,
            shell=True,
            check=kwargs.get("check", False),
            capture_output=True,
            text=True,
        )
        return SimpleNamespace(stdout=result.stdout.rstrip())


def test_patch_includes_committed_unstaged_and_untracked_files(tmp_path):
    def git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    (tmp_path / "committed.txt").write_text("before\n")
    (tmp_path / "unstaged.txt").write_text("before\n")
    git("add", ".")
    git("commit", "-qm", "initial")
    initial_head = git("rev-parse", "HEAD")

    (tmp_path / "committed.txt").write_text("after\n")
    git("add", "committed.txt")
    git("commit", "-qm", "agent commit")
    (tmp_path / "unstaged.txt").write_text("after\n")
    (tmp_path / "untracked.txt").write_text("new\n")

    patch = PatchManager(
        LocalGitSandbox(tmp_path, initial_head)
    ).collect()

    assert patch.changed_files == (
        "committed.txt",
        "unstaged.txt",
        "untracked.txt",
    )
    assert "diff --git a/untracked.txt b/untracked.txt" in patch.text
