from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True, slots=True)
class Task:
    """Inference-safe task data; never include benchmark answers or test patches."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str


class TaskSource(ABC):
    """Read tasks without exposing the underlying dataset format to the agent."""

    @abstractmethod
    def load_task(self, instance_id: str) -> Task:
        ...

    @abstractmethod
    def iter_tasks(self) -> Iterator[Task]:
        ...
