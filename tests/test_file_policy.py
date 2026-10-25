import subprocess

from codepilot.sandbox.file_policy import RestrictedWorkspace


def test_shell_cannot_modify_other_files_but_can_create_rename_and_delete_authorized_files(
    tmp_path,
):
    backing = tmp_path / "private"
    backing.mkdir()
    (backing / "allowed.txt").write_text("before")
    (backing / "other.txt").write_text("protected")
    (backing / "alias.txt").symlink_to("other.txt")
    with RestrictedWorkspace(backing, ["allowed.txt", "new.txt", "renamed.txt"]) as mount:
        result = subprocess.run(
            [
                "/bin/sh",
                "-c",
                "echo fixed > allowed.txt; echo new > new.txt; "
                "mv new.txt renamed.txt; rm allowed.txt; "
                "echo forbidden > other.txt; echo forbidden > alias.txt",
            ],
            cwd=mount,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert (mount / "other.txt").read_text() == "protected"
        assert (mount / "renamed.txt").read_text() == "new\n"
        assert not (mount / "allowed.txt").exists()
        assert not (mount / "new.txt").exists()
    assert (backing / "other.txt").read_text() == "protected"
