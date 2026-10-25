"""Standard-library repository snapshots and transactional delta application.

Executed by the controller in the sandbox, never exposed as a model tool.
"""

import base64
import os
import stat
import tempfile
from pathlib import Path, PurePosixPath


def repository_path(root, name):
    path = PurePosixPath(name)
    if (
        not name
        or name == "."
        or path.is_absolute()
        or str(path) != name
        or any(part in ("..", ".git") for part in path.parts)
        or any(char in name for char in "\x00\r\n")
    ):
        raise ValueError("Invalid repository file path: " + repr(name))
    target = root / name
    for parent in target.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("Symbolic link parent: " + name)
    return target


def file_entry(path):
    if path.is_symlink():
        return {"kind": "symlink", "target": os.readlink(path)}
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError("Not a regular file: " + str(path))
    return {
        "kind": "file",
        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        "mode": stat.S_IMODE(path.stat().st_mode) & 0o777,
    }


def capture(root):
    entries = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name != ".git")
        for name in list(dirs):
            path = Path(directory) / name
            if path.is_symlink():
                names.append(name)
                dirs.remove(name)
        for name in sorted(names):
            if name == ".git":
                continue
            path = Path(directory) / name
            entries[path.relative_to(root).as_posix()] = file_entry(path)
    return entries


def write_entry(path, entry):
    path.parent.mkdir(parents=True, exist_ok=True)
    if entry["kind"] == "symlink":
        path.symlink_to(entry["target"])
    elif entry["kind"] == "file":
        path.write_bytes(base64.b64decode(entry["data"], validate=True))
        path.chmod(entry["mode"])
    else:
        raise ValueError("Unknown file kind")


def apply(root, changes):
    # Validate every precondition BEFORE moving any parent file.
    for change in changes:
        target = repository_path(root, change["path"])
        if file_entry(target) != change["before"]:
            raise ValueError("Parent file conflict: " + change["path"])
    created_dirs = []
    with tempfile.TemporaryDirectory(prefix=".codepilot-apply-", dir=root) as stage:
        staging = Path(stage)
        for index, change in enumerate(changes):
            if change["after"] is not None:
                write_entry(staging / (str(index) + ".new"), change["after"])
        applied = []
        try:
            for index, change in enumerate(changes):
                target = repository_path(root, change["path"])
                missing = []
                parent = target.parent
                while not parent.exists():
                    missing.append(parent)
                    parent = parent.parent
                for parent in reversed(missing):
                    parent.mkdir()
                    created_dirs.append(parent)
                backup = staging / (str(index) + ".old")
                if change["before"] is not None:
                    os.replace(target, backup)
                applied.append((target, backup, change))
                if change["after"] is not None:
                    os.replace(staging / (str(index) + ".new"), target)
        except BaseException:
            for target, backup, change in reversed(applied):
                if target.is_symlink() or target.exists():
                    target.unlink()
                if change["before"] is not None:
                    os.replace(backup, target)
            for directory in reversed(created_dirs):
                directory.rmdir()
            raise
    return [change["path"] for change in changes]


def execute_operation(payload):
    root = Path.cwd()
    if payload["operation"] == "snapshot":
        return capture(root)
    if payload["operation"] == "apply":
        return apply(root, payload["changes"])
    raise ValueError("Unknown repository operation")
