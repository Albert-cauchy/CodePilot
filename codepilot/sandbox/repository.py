"""Controller-owned repository content, independent of Git's index or HEAD."""

import inspect
import json
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from . import _repository_operations as operations


@dataclass(frozen=True)
class RepositorySnapshot:
    entries: dict[str, dict[str, Any]]

    def materialize(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name, entry in self.entries.items():
            target = operations.repository_path(directory, name)
            operations.write_entry(target, entry)

    def validate_scope(self, names: list[str]) -> tuple[str, ...]:
        scope = []
        for name in names:
            path = PurePosixPath(name)
            operations.repository_path(Path("/repository"), name)
            if any(char in name for char in "*?[]"):
                raise ValueError("Use exact file paths, not globs")
            if any(key.startswith(name + "/") for key in self.entries):
                raise ValueError("Directory authorization is not allowed: " + name)
            for ancestor in (path, *path.parents):
                entry = self.entries.get(str(ancestor))
                if entry and entry["kind"] == "symlink":
                    raise ValueError("Symbolic-link authorization is not allowed: " + name)
                if ancestor != path and entry:
                    raise ValueError("Authorized path has a file parent: " + name)
            scope.append(name)
        if len(set(scope)) != len(scope):
            raise ValueError("Duplicate authorized file")
        return tuple(scope)

    def delta(self, current: "RepositorySnapshot") -> list[dict[str, Any]]:
        return [
            {"path": name, "before": self.entries.get(name), "after": current.entries.get(name)}
            for name in sorted(self.entries.keys() | current.entries.keys())
            if self.entries.get(name) != current.entries.get(name)
        ]


def repository_command(payload: dict) -> str:
    script = inspect.getsource(operations) + (
        "\nimport json, sys\nprint(json.dumps(execute_operation(json.loads(sys.argv[1]))))\n"
    )
    return f"python -I -S -c {shlex.quote(script)} {shlex.quote(json.dumps(payload))}"
