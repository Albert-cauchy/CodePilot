from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from codepilot.benchmark import Task
from codepilot.patch import PatchSnapshot


@dataclass(frozen=True, slots=True)
class Prediction:
    instance_id: str
    model_patch: str
    model_name_or_path: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class SubmissionWriter:
    """Write predictions in the JSONL schema consumed by SWE-bench."""

    @staticmethod
    def read(path: str | Path) -> list[Prediction]:
        input_path = Path(path)
        if not input_path.exists():
            return []

        with input_path.open(encoding="utf-8") as f:
            return [
                Prediction(**json.loads(line))
                for line in f
                if line.strip()
            ]

    @staticmethod
    def from_patch(
        task: Task,
        patch: PatchSnapshot | str,
        *,
        model_name_or_path: str,
    ) -> Prediction:
        text = patch.text if isinstance(patch, PatchSnapshot) else patch
        return Prediction(
            instance_id=task.instance_id,
            model_patch=text,
            model_name_or_path=model_name_or_path,
        )

    @staticmethod
    def write(path: str | Path, predictions: Iterable[Prediction]) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")

        with temporary.open("w", encoding="utf-8") as f:
            for prediction in predictions:
                f.write(
                    json.dumps(
                        prediction.to_dict(),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            f.flush()
            os.fsync(f.fileno())

        os.replace(temporary, output)
        return output
