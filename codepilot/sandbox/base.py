from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class SandboxError(RuntimeError):
    """Base sandbox exception."""


@dataclass(frozen=True, slots=True)
class ExecResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    @property
    def output(self) -> str:
        if self.stdout and self.stderr:
            return f"{self.stdout}\n{self.stderr}"
        return self.stdout or self.stderr


class SandboxCommandError(SandboxError):
    def __init__(self, result: ExecResult) -> None:
        self.result = result
        super().__init__(
            f"Command failed with exit code {result.exit_code}: "
            f"{result.command}\n{result.output}"
        )


class BaseSandbox(ABC):
    """Execution interface used by the agent's shell tool."""

    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def execute(
        self,
        command: str,
        *,
        timeout: int | None = 120,
        check: bool = False,
        workdir: str | None = None,
    ) -> ExecResult:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


class RepositorySandbox(BaseSandbox):
    """Repository operations used after the agent finishes a task."""

    @abstractmethod
    def prepare_patch_snapshot(self) -> None:
        ...

    @abstractmethod
    def git_diff(self, *, binary: bool = True) -> str:
        ...

    @abstractmethod
    def changed_files(self) -> list[str]:
        ...
