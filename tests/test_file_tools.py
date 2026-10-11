import subprocess

import pytest
from pydantic import ValidationError

from codepilot.sandbox.base import BaseSandbox, ExecResult, SandboxError
from tools.edit import create_edit_tool
from tools.read import create_read_tool
from tools.write import create_write_tool


class LocalSandbox(BaseSandbox):
    def __init__(self, root):
        self.root = root

    def start(self):
        pass

    def stop(self):
        pass

    def execute(self, command, *, timeout=120, check=False, workdir=None):
        result = subprocess.run(
            command,
            shell=True,
            cwd=workdir or self.root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return ExecResult(command, result.returncode, result.stdout, result.stderr, 0)


def test_write_and_read_execute_in_sandbox_without_shell_interpolation(tmp_path):
    sandbox = LocalSandbox(tmp_path)
    path = "nested/file ' $(touch injected).txt"
    contents = "first\r\n`touch injected`\r\nlast\n"
    create_write_tool(sandbox).invoke({"path": path, "content": contents})
    assert (tmp_path / path).read_bytes() == contents.encode()
    assert not (tmp_path / "injected").exists()
    assert create_read_tool(sandbox).invoke(
        {"path": path, "offset": 2, "limit": 1}
    ) == (f"{path} (3 lines total)\n2: `touch injected`")
    create_write_tool(sandbox).invoke({"path": path, "content": ""})
    assert (tmp_path / path).read_bytes() == b""


@pytest.mark.parametrize("arguments", [{"offset": 0}, {"limit": 0}, {"offset": 1.5}])
def test_read_rejects_invalid_ranges(tmp_path, arguments):
    with pytest.raises(ValidationError):
        create_read_tool(LocalSandbox(tmp_path)).invoke({"path": "file", **arguments})


def test_read_empty_and_past_end(tmp_path):
    (tmp_path / "file").write_text("")
    read = create_read_tool(LocalSandbox(tmp_path))
    assert read.invoke({"path": "file"}) == "file (0 lines total)"
    (tmp_path / "file").write_text("one\ntwo\n")
    assert read.invoke({"path": "file", "offset": 9}) == "file (2 lines total)"


def test_edits_all_match_original_not_previous_replacements(tmp_path):
    (tmp_path / "file").write_text("first\nsecond\n")
    create_edit_tool(LocalSandbox(tmp_path)).invoke(
        {
            "path": "file",
            "edits": [
                {"old_text": "first", "new_text": "second and longer"},
                {"old_text": "second", "new_text": "last"},
            ],
        }
    )
    assert (tmp_path / "file").read_text() == "second and longer\nlast\n"


@pytest.mark.parametrize(
    "text,edits,error",
    [
        (
            "alpha beta",
            [
                {"old_text": "alpha", "new_text": "changed"},
                {"old_text": "missing", "new_text": "oops"},
            ],
            "Could not find the exact text",
        ),
        (
            "repeat repeat",
            [{"old_text": "repeat", "new_text": "x"}],
            "multiple matches",
        ),
        ("aaaa", [{"old_text": "aaa", "new_text": "x"}], "multiple matches"),
        (
            "alpha beta",
            [
                {"old_text": "alpha beta", "new_text": "x"},
                {"old_text": "beta", "new_text": "y"},
            ],
            "overlap",
        ),
        (
            "‘text’\n“text”",
            [
                {"old_text": "'text'", "new_text": "x"},
                {"old_text": "absent", "new_text": "y"},
            ],
            "Could not find",
        ),
        (
            "“text”\n„text‟",
            [{"old_text": '"text"', "new_text": "x"}],
            "multiple matches",
        ),
    ],
)
def test_edit_failure_writes_nothing(tmp_path, text, edits, error):
    path = tmp_path / "file"
    path.write_bytes(text.encode())
    with pytest.raises(SandboxError, match=error):
        create_edit_tool(LocalSandbox(tmp_path)).invoke(
            {"path": "file", "edits": edits}
        )
    assert path.read_bytes() == text.encode()


def test_normalized_match_preserves_unedited_bytes(tmp_path):
    path = tmp_path / "file"
    path.write_bytes("prefix\r\n“smart”\u00a0— value  \r\nsuffix\r\n".encode())
    create_edit_tool(LocalSandbox(tmp_path)).invoke(
        {
            "path": "file",
            "edits": [
                {"old_text": '"smart" - value\n', "new_text": "replacement\n"},
            ],
        }
    )
    assert path.read_bytes() == b"prefix\r\nreplacement\nsuffix\r\n"


def test_exact_match_wins_even_when_normalization_would_be_ambiguous(tmp_path):
    path = tmp_path / "file"
    path.write_text('"text"\n“text”\n')
    create_edit_tool(LocalSandbox(tmp_path)).invoke(
        {
            "path": "file",
            "edits": [
                {"old_text": '"text"', "new_text": "exact"},
            ],
        }
    )
    assert path.read_text() == "exact\n“text”\n"


def test_edit_can_delete_and_rejects_empty_old_text(tmp_path):
    path = tmp_path / "file"
    path.write_text("hello\n")
    edit = create_edit_tool(LocalSandbox(tmp_path))
    with pytest.raises(ValidationError):
        edit.invoke({"path": "file", "edits": [{"old_text": "", "new_text": "x"}]})
    edit.invoke({"path": "file", "edits": [{"old_text": "hello\n", "new_text": ""}]})
    assert path.read_text() == ""
