"""Small manual smoke-test utility for PatchManager/SubmissionWriter.

This does NOT run an agent. It only shows how your future AgentCore should hand
off the final worktree diff to SWE-bench.
"""

from pathlib import Path

from codepilot.benchmark import SWEBenchAdapter
from codepilot.patch import PatchManager
from codepilot.sandbox import DockerSandbox
from codepilot.submission import SubmissionWriter


INSTANCE_ID = "sympy__sympy-20590"


def main() -> None:
    task = SWEBenchAdapter().load_task(INSTANCE_ID)

    with DockerSandbox(task) as sandbox:
        print(
            "Container is ready. This script makes no code edits, so the "
            "collected patch should be empty."
        )
        patch = PatchManager(sandbox).collect()
        print(f"changed files: {list(patch.changed_files)}")
        print(f"patch bytes: {len(patch.text.encode('utf-8'))}")

        prediction = SubmissionWriter.from_patch(
            task,
            patch,
            model_name_or_path="codepilot-infra-smoke-test",
        )
        output = SubmissionWriter.write(
            Path("outputs") / "smoke_predictions.jsonl",
            [prediction],
        )
        print(f"wrote: {output}")


if __name__ == "__main__":
    main()
