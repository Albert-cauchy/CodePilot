from __future__ import annotations

from collections.abc import Iterator

from datasets import Dataset, load_dataset

from .base import Task, TaskSource


class SWEBenchAdapter(TaskSource):
    """Adapter from a SWE-bench Hugging Face split to CodePilot Task objects."""

    def __init__(
        self,
        dataset_name: str = "SWE-bench/SWE-bench_Verified",
        split: str = "test",
        *,
        dataset: Dataset | None = None,
        revision: str | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.revision = revision
        self._dataset = (
            dataset
            if dataset is not None
            else load_dataset(dataset_name, split=split, revision=revision)
        )
        self._index = {
            str(item["instance_id"]): i for i, item in enumerate(self._dataset)
        }

    def __len__(self) -> int:
        return len(self._dataset)

    def __contains__(self, instance_id: str) -> bool:
        return instance_id in self._index

    def list_instance_ids(self) -> list[str]:
        return list(self._index)

    def load_task(self, instance_id: str) -> Task:
        try:
            index = self._index[instance_id]
        except KeyError as exc:
            raise KeyError(f"Unknown SWE-bench instance: {instance_id!r}") from exc

        return self._to_task(self._dataset[index])

    def iter_tasks(self) -> Iterator[Task]:
        for raw in self._dataset:
            yield self._to_task(raw)

    @staticmethod
    def _to_task(raw: dict) -> Task:
        """Whitelist only inference-safe fields."""
        return Task(
            instance_id=str(raw["instance_id"]),
            repo=str(raw["repo"]),
            base_commit=str(raw["base_commit"]),
            problem_statement=str(raw["problem_statement"]),
        )
