from __future__ import annotations

from dataclasses import dataclass

from codepilot.sandbox.base import RepositorySandbox


@dataclass(frozen=True, slots=True)
class PatchSnapshot:
    text: str
    changed_files: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class PatchManager:
    """Collect the model's worktree changes without touching gold data."""

    def __init__(self, sandbox: RepositorySandbox) -> None:
        self.sandbox = sandbox

    def collect(self) -> PatchSnapshot:
        self.sandbox.prepare_patch_snapshot()
        return PatchSnapshot(
            text=self.sandbox.git_diff(binary=True),
            changed_files=tuple(self.sandbox.changed_files()),
        )

    def require_non_empty(self) -> PatchSnapshot:
        patch = self.collect()
        if patch.is_empty:
            raise ValueError("Agent produced an empty patch.")
        return patch
